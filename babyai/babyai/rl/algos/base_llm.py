from abc import ABC, abstractmethod
import torch
import numpy as np
from tqdm import tqdm
from collections import deque
import torch.nn.functional as F

from babyai.rl.format import default_preprocess_obss
from babyai.rl.utils import DictList, ParallelEnv
from babyai.rl.utils.supervised_losses import ExtraInfoCollector
import babyai.utils
from torch.distributions import Categorical
import logging

logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt


class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, lm_server, llm_scoring_module_key, num_frames_per_proc, discount, lr, gae_lambda,
                 entropy_coef, value_loss_coef, max_grad_norm, reshape_reward, subgoals, nbr_obs, aux_info):
        """
        Initializes a `BaseAlgo` instance.
        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        lm_server : Lamorel Caller
        llm_scoring_module_key : str
            the key of the module function to ask scroing from
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        aux_info : list
            a list of strings corresponding to the name of the extra information
            retrieved from the environment for supervised auxiliary losses
        """
        # Store parameters

        self.env = envs
        self.lm_server = lm_server
        self.llm_scoring_module_key = llm_scoring_module_key
        # Useful filter to avoid computing score of each candidate when using additional heads directly
        if llm_scoring_module_key == "__score":
            self.filter_candidates_fn = lambda candidates: candidates
        elif llm_scoring_module_key == "policy_head":
            self.filter_candidates_fn = lambda candidates: None
        else:
            raise NotImplementedError()
        # self.acmodel.train()
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.reshape_reward = reshape_reward
        self.aux_info = aux_info

        # Store helpers values

        self.device = torch.device("cpu")
        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        # Initialize experience values
        self.nbr_obs = nbr_obs
        self.obs_queue = [deque([], maxlen=self.nbr_obs) for _ in range(self.num_procs)]
        self.acts_queue = [deque([], maxlen=self.nbr_obs-1) for _ in range(self.num_procs)]
        self.subgoals = subgoals

        shape = (self.num_frames_per_proc, self.num_procs)

        logging.info("resetting environment")
        self.obs, self.infos = self.env.reset()
        logging.info("reset environment")
        for i in range(self.num_procs):
            self.obs_queue[i].append(self.infos[i]['descriptions'])
        self.obss = [None] * (shape[0])

        self.prompts = [None] * (shape[0])

        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)

        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)

        self.values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.rewards_bonus = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)

        if self.aux_info:
            self.aux_info_collector = ExtraInfoCollector(self.aux_info, shape, self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return_bonus = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_reshaped_return_bonus = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

    @classmethod
    def generate_prompt(cls, goal, subgoals, deque_obs, deque_actions):

        ldo = len(deque_obs)
        lda = len(deque_actions)

        head_prompt = "Possible action of the agent:"
        for sg in subgoals:
            head_prompt += " {},".format(sg)
        head_prompt = head_prompt[:-1]

        g = " \n Goal of the agent: {}".format(goal)
        obs = ""
        for i in range(ldo):
            obs += " \n Observation {}: ".format(i)
            for d_obs in deque_obs[i]:
                obs += "{}, ".format(d_obs)
            obs += "\n Action {}: ".format(i)
            if i < lda:
                obs += "{}".format(deque_actions[i])
        return head_prompt + g + obs


    def collect_experiences(self, debug=False):
        """Collects rollouts and computes advantages.
        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.
        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """

        for i in tqdm(range(self.num_frames_per_proc), ascii=" " * 9 + ">", ncols=100):
            # Do one agent-environment interaction

            prompt = [self.generate_prompt(goal=self.obs[j]['mission'], subgoals=self.subgoals[j],
                                           deque_obs=self.obs_queue[j], deque_actions=self.acts_queue[j])
                      for j in range(self.num_procs)]

            output = self.lm_server.custom_module_fns(module_function_keys=[self.llm_scoring_module_key, 'value'],
                                                      contexts=prompt,
                                                      candidates=self.filter_candidates_fn(self.subgoals))
            # output = self.lm_server.score(contexts=prompt, candidates=self.subgoals,
            #                               additional_module_function_keys=['value'])
            scores = torch.stack([_o[self.llm_scoring_module_key] for _o in output]).squeeze()
            scores_max = torch.max(scores, dim=1)[0]
            """print("scores: {}".format(scores.shape))
            print("scores_max: {}".format(scores_max.shape))"""
            values = torch.stack([_o["value"][0] for _o in output])

            proba_dist = []
            for j in range(len(scores)):
                if self.llm_scoring_module_key == "__score":
                    # rescaled scores to avoid the flattening effect of softmax
                    # softmax([1e-9, 1e-100, 1e-9])~[0.33, 0.33, 0.33]
                    # softmax([1e-9, 1e-100, 1e-9]*1e9)~[0.4223, 0.1554, 0.4223]
                    if scores_max[j] < 1e-45 or torch.isnan(scores_max[j]):
                        proba_dist.append(F.softmax(torch.ones_like(scores[j]), dim=-1).unsqueeze(dim=0))
                    else:
                        proba_dist.append(F.softmax(scores[j]/scores_max[j], dim=-1).unsqueeze(dim=0))
                else:
                    proba_dist.append(F.softmax(scores[j], dim=-1).unsqueeze(dim=0))

            proba_dist = torch.cat(proba_dist, dim=0)
            dist = Categorical(probs=proba_dist)
            action = dist.sample()
            a = action.cpu().numpy()

            for j in range(self.num_procs):
                self.acts_queue[j].append(self.subgoals[j][int(a[j])])

            if len(self.subgoals[0]) > 6:
                # only useful when we test the impact of the number of actions
                real_a = np.copy(a)
                real_a[real_a > 6] = 6
                obs, reward, done, self.infos = self.env.step(real_a)
            else:
                obs, reward, done, self.infos = self.env.step(a)

            for j in range(self.num_procs):
                if done[j]:
                    # reinitialise memory of past observations and actions
                    self.obs_queue[j].clear()
                    self.acts_queue[j].clear()
                self.obs_queue[j].append(self.infos[j]['descriptions'])

            info = self.infos

            if self.aux_info:
                env_info = self.aux_info_collector.process(env_info)
                # env_info = self.process_aux_info(env_info)

            if debug:
                babyai.utils.viz(self.env)
                print(babyai.utils.info(reward, heading="Reward"))
                print(babyai.utils.info(info, "Subtasks"))

            # Update experiences values

            self.obss[i] = self.obs
            self.obs = obs

            self.prompts[i] = prompt

            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)

            self.actions[i] = action
            self.values[i] = values.squeeze()

            if self.reshape_reward is not None:
                rewards_shaped = torch.tensor([
                    self.reshape_reward(subgoal_proba=None, reward=reward_, policy_value=None, llm_0=None)
                    for reward_ in reward
                ], device=self.device)
                self.rewards[i] = rewards_shaped[:, 0]
                self.rewards_bonus[i] = rewards_shaped[:, 1]
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)

            log_prob = dist.log_prob(action)

            if len(log_prob.shape) > 1:
                log_prob = log_prob.sum(dim=-1)
            self.log_probs[i] = log_prob

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_reshaped_return_bonus += self.rewards_bonus[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_reshaped_return_bonus.append(self.log_episode_reshaped_return_bonus[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_reshaped_return_bonus *= self.mask
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences
        prompt = [self.generate_prompt(goal=self.obs[i]['mission'], subgoals=self.subgoals[i],
                                       deque_obs=self.obs_queue[i], deque_actions=self.acts_queue[i])
                  for i in range(self.num_procs)]
        output = self.lm_server.custom_module_fns(module_function_keys=['value'], contexts=prompt)
        next_value = torch.stack([_o["value"] for _o in output]).squeeze()

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i + 1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i + 1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i + 1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Flatten the data correctly, making sure that
        # each episode's data is a continuous chunk

        exps = DictList()
        exps.prompt = np.array([self.prompts[i][j]
                       for j in range(self.num_procs)
                       for i in range(self.num_frames_per_proc)])
        exps.subgoal = np.array([self.subgoals[j]
                        for j in range(self.num_procs)
                        for i in range(self.num_frames_per_proc)])
        # In commments below T is self.num_frames_per_proc, P is self.num_procs,
        # D is the dimensionality

        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)

        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        if self.aux_info:
            exps = self.aux_info_collector.end_collection(exps)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        log = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "reshaped_return_bonus_per_episode": self.log_reshaped_return_bonus[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames,
            "episodes_done": self.log_done_counter,
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_reshaped_return_bonus = self.log_reshaped_return_bonus[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, log

    @abstractmethod
    def update_parameters(self):
        pass
