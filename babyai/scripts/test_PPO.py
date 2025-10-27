from abc import ABC, abstractmethod
import torch
import numpy
from tqdm import tqdm

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

    def __init__(self, envs, acmodel, number_episodes, reshape_reward, preprocess_obss,
                 aux_info, sampling_temperature=1):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        aux_info : list
            a list of strings corresponding to the name of the extra information
            retrieved from the environment for supervised auxiliary losses

        """
        # Store parameters

        self.env = envs
        self.acmodel = acmodel
        self.number_episodes = number_episodes
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward
        self.aux_info = aux_info
        self.sampling_temperature = sampling_temperature

        # Store helpers values

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_procs = len(envs)

        # Initialize experience values

        logging.info("resetting environment")
        self.obs = self.env.reset()
        logging.info("reset environment")

        self.memory = torch.zeros(self.num_procs, self.acmodel.memory_size, device=self.device)

        self.mask = torch.ones(self.num_procs, device=self.device)

        self.rewards = []
        self.rewards_bonus = []

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

    def prompt_modifier(self, prompt: str, dict_changes: dict) -> str:
        """use a dictionary of equivalence to modify the prompt accordingly
        ex:
        prompt= 'green box red box', dict_changes={'box':'tree'}
        promp_modifier(prompt, dict_changes)='green tree red tree' """

        for key, value in dict_changes.items():
            prompt = prompt.replace(key, value)
        return prompt
    def generate_trajectories(self, dict_modifier):
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
        # TODO change the goal in obs
        pbar = tqdm(range(self.number_episodes), ascii=" " * 9 + ">", ncols=100)
        while self.log_done_counter < self.number_episodes:
            # Do one agent-environment interaction
            for o in self.obs:
                o['mission'] = self.prompt_modifier(o['mission'], dict_modifier)
            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                model_results = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                dist = model_results['dist']
                value = model_results['value']
                memory = model_results['memory']
                extra_predictions = model_results['extra_predictions']

            if self.sampling_temperature != 1:
                dist = Categorical(logits=dist.logits/self.sampling_temperature)
            action = dist.sample()
            # action = dist.probs.argmax(dim=1)

            a = action.cpu().numpy()
            real_a = numpy.copy(a)
            real_a[real_a > 6] = 6
            obs, reward, done, env_info = self.env.step(real_a)

            if isinstance(env_info, tuple) and len(env_info) == 2:
                info, pi_l_actions = env_info
            else:
                info = env_info
            if self.aux_info:
                env_info = self.aux_info_collector.process(env_info)
                # env_info = self.process_aux_info(env_info)

            # Update experiences values
            self.obs = obs
            self.memory = memory
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)

            if self.reshape_reward is not None:
                rewards_shaped = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_, info_)
                    for obs_, action_, reward_, done_, info_ in zip(obs, action, reward, done, info)
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

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        log = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "reshaped_return_bonus_per_episode": self.log_reshaped_return_bonus[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "episodes_done": self.log_done_counter,
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_reshaped_return_bonus = self.log_reshaped_return_bonus[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return log
