import numpy as np
from tqdm import tqdm

from experiments.agents.base_agent import BaseAgent

class Random_agent(BaseAgent):
    def __init__(self, envs, subgoals):
        super().__init__(envs)
        self.env.reset()
        self.subgoals = subgoals
        self.returns = [0 for _ in range(self.env.num_envs)]
        self.logs = {
            "return_per_episode": [],
        }

    def generate_trajectories(self, dict_modifier, n_tests, language='english'):
        episodes_done = 0
        pbar = tqdm(range(n_tests), ascii=" " * 9 + ">", ncols=100)
        while episodes_done < n_tests:
            actions = np.random.randint(low=0, high=len(self.subgoals[0]), size=(self.env.num_envs,))

            if len(self.subgoals[0]) > 6:
                # only useful when we test the impact of the number of actions
                real_a = np.copy(actions)
                real_a[real_a > 6] = 6
                obs, rewards, dones, infos = self.env.step(real_a)
            else:
                obs, rewards, dones, infos = self.env.step(actions)

            for j in range(self.env.num_envs):
                self.returns[j] += rewards[j]
                if dones[j]:
                    episodes_done += 1
                    pbar.update(1)
                    self.logs["return_per_episode"].append(self.returns[j])
                    self.returns[j] = 0
        pbar.close()

        self.logs["episodes_done"] = episodes_done
        return None, self.logs

    def update_parameters(self):
        pass
