from babyai.bot import Bot
import numpy as np
from tqdm import tqdm

class BotAgent:
    def __init__(self, envs, nbr_envs, size_action_space, number_episodes):
        """An agent based on a GOFAI bot."""
        self.envs = envs
        self.nbr_envs = nbr_envs
        self.size_action_space = size_action_space
        self.number_episodes = number_episodes
        self.returns = [0 for _ in range(self.nbr_envs)]
        self.logs = {
            "return_per_episode": [],
        }
        obs, infos = self.envs.reset()
        self.bots = [Bot(env) for env in self.envs]
        self.on_reset()

    def on_reset(self, env):
        return Bot(env)

    def act(self, obs=None, action_choosen=None, update_internal_state=True, *args, **kwargs):
        actions = [bot.replan(action_choosen) for bot in self.bots]
        return actions

    def generate_trajectories(self, dict_modifier, language='english'):
        episodes_done = 0

        pbar = tqdm(range(self.number_episodes), ascii=" " * 9 + ">", ncols=100)
        while episodes_done < self.number_episodes:

            actions = self.act(obs)

            obs, rewards, dones, infos = self.envs.step(actions)

            for j in range(self.nbr_envs):
                self.returns[j] += rewards[j]
                if dones[j]:
                    episodes_done += 1
                    pbar.update(1)
                    self.logs["return_per_episode"].append(self.returns[j])
                    self.returns[j] = 0
                    self.bots[j] = self.on_reset(self.envs[j])
        pbar.close()

        self.logs["episodes_done"] = episodes_done
        return None, self.logs
