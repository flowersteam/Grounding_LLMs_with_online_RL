from babyai.bot import Bot
from babyai.rl.utils import DictList
from collections import deque
from tqdm import tqdm
import numpy as np

from experiments.agents.base_agent import BaseAgent

class BotAgent(BaseAgent):
    def __init__(self, envs, subgoals):
        """An agent based on BabyAI's GOFAI bot."""
        self.env = envs.envs[0]
        self.subgoals = subgoals[0]
        self.logs = {
            "return_per_episode": [],
        }
        self.obs, self.infos = self.env.reset()
        self.bot = Bot(self.env)

        self.obs_queue = deque([], maxlen=3)
        self.acts_queue = deque([], maxlen=2)

        self.obs_queue.append(self.infos['descriptions'])

        self.prompts = []
        self.actions = []

        self.log_done_counter = 0

    def act(self, action_choosen=None):
        actions = self.bot.replan(action_choosen)
        return actions

    def generate_trajectories(self, dict_modifier, n_tests, language='english'):
        assert language == "english"

        nbr_frames = 1
        pbar = tqdm(range(n_tests), ascii=" " * 9 + ">", ncols=100)
        previous_action = None
        while self.log_done_counter < n_tests:
            nbr_frames += 1
            prompt = self.prompt_modifier(self.generate_prompt(goal=self.obs['mission'], subgoals=self.subgoals,
                                                          deque_obs=self.obs_queue,
                                                          deque_actions=self.acts_queue), dict_modifier)

            action = self.act(previous_action)
            # previous_action = action
            self.actions.append(self.subgoals[int(action)])
            self.acts_queue.append(self.subgoals[int(action)])
            self.prompts.append(prompt)

            self.obs, reward, done, self.infos = self.env.step(action)

            if done:
                self.log_done_counter += 1
                pbar.update(1)
                self.logs["return_per_episode"].append(reward)
                self.obs_queue.clear()
                self.acts_queue.clear()
                self.obs, infos = self.env.reset()
                self.bot = Bot(self.env)
            self.obs_queue.append(self.infos['descriptions'])
        pbar.close()

        exps = DictList()
        exps.prompts = np.array(self.prompts)
        exps.actions = np.array(self.actions)

        self.logs["episodes_done"] = self.log_done_counter
        self.logs["nbr_frames"] = nbr_frames
        self.log_done_counter = 0
        return exps, self.logs

    def update_parameters(self):
        pass

