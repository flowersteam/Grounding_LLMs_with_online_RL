from .base_ppo_agent import BasePPOAgent

import babyai.utils
from babyai.rl.utils import DictList

import os
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm
from collections import deque
import logging

class LLMPPOAgent(BasePPOAgent):
    def __init__(self, envs, lm_server, llm_scoring_module_key, nbr_llms=None, num_frames_per_proc=None, discount=0.99,
                 lr=7e-4, beta1=0.9, beta2=0.999, gae_lambda=0.95, entropy_coef=0.01, value_loss_coef=0.5,
                 max_grad_norm=0.5, adam_eps=1e-5, clip_eps=0.2, epochs=4, batch_size=64, reshape_reward=None,
                 name_experiment=None, saving_path_model=None, saving_path_logs=None, number_envs=None, subgoals=None,
                 nbr_obs=3, id_expe=None, template_test=1, aux_info=None, debug=False):
        super().__init__(envs, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef, value_loss_coef,
                         max_grad_norm, reshape_reward, aux_info, device=torch.device("cpu"))

        self.lm_server = lm_server
        self.llm_scoring_module_key = llm_scoring_module_key
        # Useful filter to avoid computing score of each candidate when using additional heads directly
        if llm_scoring_module_key == "score":
            self.filter_candidates_fn = lambda candidates: candidates
        elif llm_scoring_module_key == "policy_head":
            self.filter_candidates_fn = lambda candidates: None
        else:
            raise NotImplementedError()

        self.nbr_obs = nbr_obs
        self.obs_queue = [deque([], maxlen=self.nbr_obs) for _ in range(self.num_procs)]
        self.acts_queue = [deque([], maxlen=self.nbr_obs - 1) for _ in range(self.num_procs)]
        self.subgoals = subgoals
        shape = (self.num_frames_per_proc, self.num_procs)
        logging.info("resetting environment")
        self.obs, self.infos = self.env.reset()
        logging.info("reset environment")
        for i in range(self.num_procs):
            self.obs_queue[i].append(self.infos[i]['descriptions'])

        self.prompts = [None] * (shape[0])
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)

        self.nbr_llms = nbr_llms

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.debug = debug

        self.beta1 = beta1
        self.beta2 = beta2
        self.adam_eps = adam_eps

        self.name_experiment = name_experiment
        self.saving_path_model = saving_path_model
        self.saving_path_logs = saving_path_logs
        self.number_envs = number_envs

        self.id_expe = id_expe
        self.template_test = template_test
        self.number_updates = 0

        self.experiment_path = os.path.join(self.saving_path_logs, id_expe)

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
            scores = torch.stack([_o[self.llm_scoring_module_key] for _o in output]).squeeze()
            dist = Categorical(logits=scores)
            values = torch.stack([_o["value"][0] for _o in output])
            
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

    def update_parameters(self):
        # Collect experiences
        exps, logs = self.collect_experiences(debug=self.debug)
        # print(exps.action)
        # action_counts = exps.action.unique(return_counts=True)
        # pi_l_action_counts = exps.pi_l_action.unique(return_counts=True)
        '''
        exps is a DictList with the following keys ['prompt', 'action', 'value', 'reward',
         'advantage', 'returnn', 'log_prob'] and ['collected_info', 'extra_predictions'] if we use aux_info
        exps.prompt is a (n_procs * n_frames_per_proc) of prompt
        if we use aux_info: exps.collected_info and exps.extra_predictions are DictLists with keys
        being the added information. They are either (n_procs * n_frames_per_proc) 1D tensors or
        (n_procs * n_frames_per_proc) x k 2D tensors where k is the number of classes for multiclass classification
        '''
        lm_server_update_first_call = True
        for _ in tqdm(range(self.epochs), ascii=" " * 9 + "<", ncols=100):
            # Initialize log values

            log_entropies = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            log_losses = []

            # Create minibatch of size self.batch_size*self.nbr_llms
            # each llm receive a batch of size batch_size
            for inds in self._get_batches_starting_indexes():
                # inds is a numpy array of indices that correspond to the beginning of a sub-batch
                # there are as many inds as there are batches

                exps_batch = exps[inds]

                # return the list of dict_return calculate by each llm
                list_dict_return = self.lm_server.update(exps_batch.prompt,
                                                         self.filter_candidates_fn(exps_batch.subgoal),
                                                         exps=dict(exps_batch),
                                                         lr=self.lr,
                                                         beta1=self.beta1,
                                                         beta2=self.beta2,
                                                         adam_eps=self.adam_eps,
                                                         clip_eps=self.clip_eps,
                                                         entropy_coef=self.entropy_coef,
                                                         value_loss_coef=self.value_loss_coef,
                                                         max_grad_norm=self.max_grad_norm,
                                                         nbr_llms=self.nbr_llms,
                                                         id_expe=self.id_expe,
                                                         lm_server_update_first_call=lm_server_update_first_call,
                                                         saving_path_model=self.saving_path_model,
                                                         experiment_path=self.experiment_path,
                                                         number_updates=self.number_updates,
                                                         scoring_module_key=self.llm_scoring_module_key,
                                                         template_test=self.template_test)

                lm_server_update_first_call = False

                log_losses.append(np.mean([d["loss"] for d in list_dict_return]))
                log_entropies.append(np.mean([d["entropy"] for d in list_dict_return]))
                log_policy_losses.append(np.mean([d["policy_loss"] for d in list_dict_return]))
                log_value_losses.append(np.mean([d["value_loss"] for d in list_dict_return]))
                log_grad_norms.append(np.mean([d["grad_norm"] for d in list_dict_return]))

        # Log some values

        logs["entropy"] = np.mean(log_entropies)
        logs["policy_loss"] = np.mean(log_policy_losses)
        logs["value_loss"] = np.mean(log_value_losses)
        logs["grad_norm"] = np.mean(log_grad_norms)
        logs["loss"] = np.mean(log_losses)

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.
        Returns
        -------
        batches_starting_indexes : list of lists of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = np.arange(0, self.num_frames)
        indexes = np.random.permutation(indexes)

        num_indexes = self.batch_size
        batches_starting_indexes = [indexes[i:i + num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes

    def generate_trajectories(self, dict_modifier, n_tests, language='english', im_learning=False, debug=False):
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
            generate_prompt = self.generate_prompt
            subgoals = self.subgoals
        elif language == "french":
            generate_prompt = self.generate_prompt_french
            subgoals = [[LLMPPOAgent.prompt_modifier(sg, self.dict_translation_action) for sg in sgs] for sgs in self.subgoals]

        nbr_frames = self.num_procs
        pbar = tqdm(range(n_tests), ascii=" " * 9 + ">", ncols=100)
        while self.log_done_counter < n_tests:
            # Do one agent-environment interaction
            nbr_frames += self.num_procs
            prompt = [self.prompt_modifier(generate_prompt(goal=self.obs[j]['mission'], subgoals=subgoals[j],
                                                           deque_obs=self.obs_queue[j],
                                                           deque_actions=self.acts_queue[j]), dict_modifier)
                      for j in range(self.num_procs)]

            if im_learning:
                output = self.lm_server.custom_module_fns(
                    module_function_keys=[self.llm_scoring_module_key],
                    contexts=prompt,
                    candidates=self.filter_candidates_fn(self.subgoals))
                scores = torch.stack([_o[self.llm_scoring_module_key] for _o in output])
            else:
                output = self.lm_server.custom_module_fns(
                    module_function_keys=[self.llm_scoring_module_key, 'value'],
                    contexts=prompt,
                    candidates=self.filter_candidates_fn(self.subgoals))
                scores = torch.stack([_o[self.llm_scoring_module_key] for _o in output])
                vals = torch.stack([_o["value"][0] for _o in output]).cpu().numpy()

            dist = Categorical(logits=scores)
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