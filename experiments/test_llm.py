from abc import ABC, abstractmethod
import torch
import numpy as np
from tqdm import tqdm
from collections import deque
import torch.nn.functional as F

from babyai.rl.utils import DictList, ParallelEnv
from babyai.rl.utils.supervised_losses import ExtraInfoCollector
import babyai.utils
from torch.distributions import Categorical
import logging

logger = logging.getLogger(__name__)


class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, lm_server, number_epsiodes, reshape_reward, subgoals):
        """
        Initializes a `BaseAlgo` instance.
        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        llm : torch.Module
            the language model
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
        """
        # Store parameters

        self.env = envs
        self.lm_server = lm_server
        self.reshape_reward = reshape_reward

        # Store helpers values

        self.device = torch.device("cpu")
        self.number_episodes = number_epsiodes
        self.num_procs = len(envs)

        # Initialize experience values

        self.obs_queue = [deque([], maxlen=3) for _ in range(self.num_procs)]
        self.acts_queue = [deque([], maxlen=2) for _ in range(self.num_procs)]
        self.subgoals = subgoals

        logging.info("resetting environment")
        self.obs, self.infos = self.env.reset()
        logging.info("reset environment")
        for i in range(self.num_procs):
            self.obs_queue[i].append(self.infos[i]['descriptions'])

        self.mask = torch.ones(self.num_procs, device=self.device)

        self.rewards = []
        self.rewards_bonus = []

        self.prompts = []
        self.images = []
        self.actions = []
        self.vals = []    # values
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


    def generate_prompt_english(self, goal, subgoals, deque_obs, deque_actions):

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

    def generate_prompt_french(self, goal, subgoals, deque_obs, deque_actions):

        ldo = len(deque_obs)
        lda = len(deque_actions)
        head_prompt = "Actions possibles pour l'agent:"
        for sg in subgoals:
            head_prompt += " {},".format(sg)
        head_prompt = head_prompt[:-1]

        # translate goal in French
        dico_traduc_det = {"the": "la",
                           'a': 'une'}
        dico_traduc_names = {"box": "boîte",
                             "ball": "balle",
                             "key": "clef"}
        dico_traduc_adjs = {'red': 'rouge',
                            'green': 'verte',
                            'blue': 'bleue',
                            'purple': 'violette',
                            'yellow': 'jaune',
                            'grey': 'grise'}

        det = ''
        name = ''
        adj = ''

        for k in dico_traduc_det.keys():
            if k in goal:
                det = dico_traduc_det[k]
        for k in dico_traduc_names.keys():
            if k in goal:
                name = dico_traduc_names[k]
        for k in dico_traduc_adjs.keys():
            if k in goal:
                adj = dico_traduc_adjs[k]
        trad_goal = 'aller à ' + det + ' ' + name + ' ' + adj

        g = " \n But de l'agent: {}".format(trad_goal)
        obs = ""
        for i in range(ldo):
            obs += " \n Observation {}: ".format(i)
            for d_obs in deque_obs[i]:
                obs += "{}, ".format(d_obs)
            obs += "\n Action {}: ".format(i)
            if i < lda:
                obs += "{}".format(deque_actions[i])
        return head_prompt + g + obs

    @classmethod
    def prompt_modifier(cls, prompt: str, dict_changes: dict) -> str:
        """use a dictionary of equivalence to modify the prompt accordingly
        ex:
        prompt= 'green box red box', dict_changes={'box':'tree'}
        promp_modifier(prompt, dict_changes)='green tree red tree' """

        for key, value in dict_changes.items():
            prompt = prompt.replace(key, value)
        return prompt

    def generate_trajectories(self, dict_modifier, language='english', im_learning=False, debug=False):
        """Generates trajectories and calculates relevant metrics.
        Runs several environments concurrently.
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

        if language == "english":
            generate_prompt = self.generate_prompt_english
            subgoals = self.subgoals
        elif language == "french":
            dico_traduc_act = {'turn left': "tourner à gauche",
                               "turn right": "tourner à droite",
                               "go forward": "aller tout droit",
                               "pick up": "attraper",
                               "drop": "lâcher",
                               "toggle": "basculer",
                               "eat": "manger",
                               "dance": "dancer",
                               "sleep": "dormir",
                               "do nothing": "ne rien faire",
                               "cut": "couper",
                               "think": "penser"}
            generate_prompt = self.generate_prompt_french
            subgoals = [[BaseAlgo.prompt_modifier(sg, dico_traduc_act) for sg in sgs] for sgs in self.subgoals]

        nbr_frames = self.num_procs
        pbar = tqdm(range(self.number_episodes), ascii=" " * 9 + ">", ncols=100)
        while self.log_done_counter < self.number_episodes:
            # Do one agent-environment interaction
            nbr_frames += self.num_procs
            prompt = [BaseAlgo.prompt_modifier(generate_prompt(goal=self.obs[j]['mission'], subgoals=subgoals[j],
                                                           deque_obs=self.obs_queue[j],
                                                           deque_actions=self.acts_queue[j]), dict_modifier)
                      for j in range(self.num_procs)]


            """
            self.images.append(self.env.render(mode="rgb_array"))"""

            if im_learning:
                output = self.lm_server.score(contexts=prompt, candidates=subgoals)
                scores = torch.stack(output)
            else:
                output = self.lm_server.score(contexts=prompt, candidates=subgoals,
                                          additional_module_function_keys=['value'])
                vals = torch.stack([_o["value"][0] for _o in output]).cpu().numpy()
                scores = torch.stack([_o["__score"] for _o in output])
            scores_max = torch.max(scores, dim=1)[0]

            proba_dist = []
            for j in range(len(scores)):
                # rescaled scores to avoid the flattening effect of softmax
                # softmax([1e-9, 1e-100, 1e-9])~[0.33, 0.33, 0.33]
                # softmax([1e-9, 1e-100, 1e-9]*1e9)~[0.4223, 0.1554, 0.4223]
                if scores_max[j] < 1e-45:
                    proba_dist.append(F.softmax(torch.ones_like(scores[j]), dim=-1).unsqueeze(dim=0))
                else:
                    proba_dist.append(F.softmax(scores[j] / scores_max[j], dim=-1).unsqueeze(dim=0))

            proba_dist = torch.cat(proba_dist, dim=0)
            dist = Categorical(probs=proba_dist)
            action = dist.sample()
            # action = proba_dist.argmax(dim=1)
            a = action.cpu().numpy()

            for j in range(self.num_procs):
                self.actions.append(subgoals[j][int(a[j])])
                self.acts_queue[j].append(subgoals[j][int(a[j])])

            obs, reward, done, self.infos = self.env.step(a)

            for j in range(self.num_procs):
                if not im_learning:
                    self.vals.append(vals[j][0])
                self.prompts.append(prompt[j])
                if done[j]:
                    # reinitialise memory of past observations and actions
                    self.obs_queue[j].clear()
                    self.acts_queue[j].clear()
                self.obs_queue[j].append(self.infos[j]['descriptions'])

            info = self.infos

            if debug:
                babyai.utils.viz(self.env)
                print(babyai.utils.info(reward, heading="Reward"))
                print(babyai.utils.info(info, "Subtasks"))

            self.obs = obs

            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)

            if self.reshape_reward is not None:
                rewards_shaped = torch.tensor([
                    self.reshape_reward(subgoal_proba=None, reward=reward_, policy_value=None, llm_0=None)
                    for reward_ in reward
                ], device=self.device)
                self.rewards.append(rewards_shaped[:, 0])
                self.rewards_bonus.append(rewards_shaped[:, 1])
            else:
                self.rewards.append(torch.tensor(reward, device=self.device))

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[-1]
            self.log_episode_reshaped_return_bonus += self.rewards_bonus[-1]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    pbar.update(1)
                    self.log_return.append(self.log_episode_return[i].item())
                    if self.log_episode_return[i].item() > 0:
                        print(self.obs[i]['mission'])
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_reshaped_return_bonus.append(self.log_episode_reshaped_return_bonus[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_reshaped_return_bonus *= self.mask
            self.log_episode_num_frames *= self.mask

        pbar.close()

        exps = DictList()
        exps.prompts = np.array(self.prompts)
        # exps.images = np.stack(self.images)

        # In commments below T is self.num_frames_per_proc, P is self.num_procs,
        # D is the dimensionality

        # for all tensors below, T x P -> P x T -> P * T
        exps.actions = np.array(self.actions)

        exps.vals = np.array(self.vals)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        log = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "reshaped_return_bonus_per_episode": self.log_reshaped_return_bonus[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "episodes_done": self.log_done_counter,
            "nbr_frames": nbr_frames
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_reshaped_return_bonus = self.log_reshaped_return_bonus[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, log
