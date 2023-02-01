import numpy as np
from tqdm import tqdm

class Random_agent:
    def __init__(self, envs, nbr_envs, size_action_space, number_episodes):
        self.envs = envs
        obs, infos = self.envs.reset()
        self.nbr_envs = nbr_envs
        self.size_action_space = size_action_space
        self.number_episodes = number_episodes
        self.returns = [0 for _ in range(self.nbr_envs)]
        self.logs = {
            "return_per_episode": [],
        }

    def generate_trajectories(self, dict_modifier, language='english'):
        episodes_done = 0
        pbar = tqdm(range(self.number_episodes), ascii=" " * 9 + ">", ncols=100)
        while episodes_done < self.number_episodes:

            actions = np.random.randint(low=0, high=self.size_action_space, size=(self.nbr_envs,))

            if self.size_action_space > 6:
                # only useful when we test the impact of the number of actions
                real_a = np.copy(actions)
                real_a[real_a > 6] = 6
                obs, rewards, dones, infos = self.envs.step(real_a)
            else:
                obs, rewards, dones, infos = self.envs.step(actions)

            for j in range(self.nbr_envs):
                self.returns[j] += rewards[j]
                if dones[j]:
                    episodes_done += 1
                    pbar.update(1)
                    self.logs["return_per_episode"].append(self.returns[j])
                    self.returns[j] = 0
        pbar.close()

        self.logs["episodes_done"] = episodes_done
        return None, self.logs
