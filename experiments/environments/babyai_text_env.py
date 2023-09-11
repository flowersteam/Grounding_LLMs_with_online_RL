from .base_env import BaseEnv

import gym
import babyai_text
import babyai.utils as utils
from babyai.paral_env_simple import ParallelEnv

class BabyAITextEnv(BaseEnv):
    def __init__(self, config_dict):
        super().__init__(config_dict)
        envs = []
        self._action_space = [a.replace("_", " ") for a in config_dict["action_space"]]
        for i in range(config_dict["number_envs"]):
            env = gym.make(config_dict["task"])
            env.seed(100 * config_dict["seed"] + i)
            envs.append(env)

        self._env = ParallelEnv(envs)

    def __prepare_infos(self, obs, infos):
        for _obs, info in zip(obs, infos):
            info["possible_actions"] = self._action_space
            info["goal"] = f"Goal of the agent: {_obs['mission']}"
        return list(infos)

    def __generate_obs(self, obs, infos):
        return [info["descriptions"] for info in infos]
    def reset(self):
        obs, infos = self._env.reset()
        return self.__generate_obs(obs, infos), self.__prepare_infos(obs, infos)
    def step(self, actions_id, actions_command):
        obs, rews, dones, infos = self._env.step(actions_id)
        return self.__generate_obs(obs, infos), \
                [rew * 20.0 for rew in rews], \
                dones, \
                self.__prepare_infos(obs, infos)

    def get_template_prompts(self, template_test):
        head_prompt = "Possible action of the agent:"
        for sg in subgoals:
            head_prompt += " {},".format(sg)
        head_prompt = head_prompt[:-1]

        if template_test == 1:
            # expected answers: go forward, turn left, turn left, toggle
            templated_prompts = [
                ' \n Goal of the agent: go to the green ball \n Observation 0: A wall 2 step left, A purple key 1 step left and 2 steps forward, A yellow key 1 step left and 1 step forward, A green ball 3 steps forward, A grey ball 1 step right and 5 steps forward, A green key 1 step right and 2 steps forward, A grey ball 1 step right and 1 step forward, A green key 2 steps right and 4 steps forward, A red box 2 steps right and 2 steps forward, \n Action 0: ',
                ' \n Goal of the agent: go to the green ball \n Observation 0: A wall 2 step left, A purple key 1 step left and 2 steps forward, A yellow key 1 step left and 1 step forward, A green ball 3 steps forward, A grey ball 1 step right and 5 steps forward, A green key 1 step right and 2 steps forward, A grey ball 1 step right and 1 step forward, A green key 2 steps right and 4 steps forward, A red box 2 steps right and 2 steps forward, \n Action 0: go forward \n Observation 1: A purple key 1 step left and 1 step forward, A yellow key 1 step left, A green ball 2 steps forward, A grey ball 1 step right and 4 steps forward, A green key 1 step right and 1 step forward, A grey ball 1 step right, A green key 2 steps right and 3 steps forward, A red box 2 steps right and 1 step forward, \n Action 1: turn right \n Observation 2: A wall 2 step right, A green key 3 steps left and 2 steps forward, A green ball 2 steps left, A red box 1 step left and 2 steps forward, A green key 1 step left and 1 step forward, A grey ball 1 step forward, \n Action 2: ',
                ' \n Goal of the agent: open the purple door \n Observation 0: You see a wall 3 steps forward, You see a wall 3 steps left, You see a yellow key 1 step right and 1 step forward, You see a locked purple door 2 steps right and 3 steps forward, You see a purple ball 3 steps right and 1 step forward, You see a green box 3 steps right, You see a purple key 2 steps left \n Action 0: ',
                ' \n Goal of the agent: open the purple door \n Observation 0: You see a wall 3 steps forward, You see a wall 3 steps left, You see a yellow key 1 step right and 1 step forward, You see a locked purple door 2 steps right and 3 steps forward, You see a purple ball 3 steps right and 1 step forward, You see a green box 3 steps right, You see a purple key 2 steps left \n Action 0: turn left \n Observation 1: You see a wall 3 steps forward, You see a wall 3 steps right, You see a purple key 2 steps forward \n Action 1: go forward \n Observation 2: You see a wall 2 steps forward, You see a wall 3 steps right, You see a purple key 1 step forward \n Action 2: ',
                ' \n Goal of the agent: open the purple door \n Observation 0: You carry a purple key, You see a wall 3 steps forward, You see a wall 5 steps left, You see a yellow key 1 step left and 1 step forward, You see a locked purple door 3 steps forward, You see a purple ball 1 step right and 1 step forward, You see a green box 1 step right \n Action 0: go forward \n Observation 1: You carry a purple key, You see a wall 2 steps forward, You see a wall 5 steps left, You see a yellow key 1 step left, You see a locked purple door 2 steps forward, You see a purple ball 1 step right \n Action 1: go forward \n Observation 2: You carry a purple key, You see a wall 1 step forward, You see a wall 5 steps left, You see a locked purple door 1 step forward \n Action 2: ',
                ' \n Goal of the agent: pick up green box \n Observation 0: You see a wall 2 steps forward, You see a wall 2 steps left, You see a yellow ball 1 step left and 1 step forward, You see a green box 2 steps right \n Action 0: ',
                ' \n Goal of the agent: pick up green box \n Observation 0: You see a wall 2 steps forward, You see a wall 2 steps left, You see a yellow ball 1 step left and 1 step forward, You see a green box 2 steps right \n Action 0: turn right \n Observation 1: You see a wall 2 steps left, You see a blue key 1 step right, You see a red ball 2 steps right and 1 step forward, You see a green box 2 steps forward \n Action 1: go forward \n Observation 2: You see a wall 2 steps left, You see a red ball 2 steps right, You see a green box 1 step forward \n Action 2: ',
                ' \n Goal of the agent: put blue ball next to red box \n Observation 0: You carry a blue ball, You see a wall 5 steps forward, You see a wall 2 steps left, You see a grey key 1 step right and 2 steps forward, You see a red box 3 steps forward \n Action 0: go forward \n Observation 1: You carry a blue ball, You see a wall 4 steps forward, You see a wall 2 steps left, You see a grey key 1 step right and 1 step forward, You see a red box 2 steps forward \n Action 1: ',
                ' \n Goal of the agent: pick up the blue ball then go to the red box \n Observation 0: You see a wall 3 steps forward, You see a wall 4 steps right, You see a purple key 2 steps forward, You see a red box 2 steps right, You see a blue ball 2 steps left \n Action 0: ',
                ' \n Goal of the agent: go to the red box after you pick up the blue ball \n Observation 0: You see a wall 3 steps forward, You see a wall 4 steps right, You see a purple key 2 steps forward, You see a red box 2 steps right, You see a blue ball 2 steps left \n Action 0: ',
                ' \n Goal of the agent: pick up the green key then pick up the the red box \n Observation 0: You carry a green key, You see a wall 4 steps forward, You see a wall 4 steps left, You see a red box 1 step left, You see a purple ball 2 steps left and 1 step forward \n Action 0:  ']
        elif template_test == 2:
            # expected answers: go forward, turn left
            templated_prompts = [
                ' \n Goal of the agent: go to the green ball \n Observation 0: A wall 2 step left, A purple key 1 step left and 2 steps forward, A yellow key 1 step left and 1 step forward, A green ball 3 steps forward, A grey ball 1 step right and 5 steps forward, A green key 1 step right and 2 steps forward, A grey ball 1 step right and 1 step forward, A green key 2 steps right and 4 steps forward, A red box 2 steps right and 2 steps forward, \n Action 0: ',
                ' \n Goal of the agent: go to the green ball \n Observation 0: A wall 2 step left, A purple key 1 step left and 2 steps forward, A yellow key 1 step left and 1 step forward, A green ball 3 steps forward, A grey ball 1 step right and 5 steps forward, A green key 1 step right and 2 steps forward, A grey ball 1 step right and 1 step forward, A green key 2 steps right and 4 steps forward, A red box 2 steps right and 2 steps forward, \n Action 0: go forward \n Observation 1: A purple key 1 step left and 1 step forward, A yellow key 1 step left, A green ball 2 steps forward, A grey ball 1 step right and 4 steps forward, A green key 1 step right and 1 step forward, A grey ball 1 step right, A green key 2 steps right and 3 steps forward, A red box 2 steps right and 1 step forward, \n Action 1: turn right \n Observation 2: A wall 2 step right, A green key 3 steps left and 2 steps forward, A green ball 2 steps left, A red box 1 step left and 2 steps forward, A green key 1 step left and 1 step forward, A grey ball 1 step forward, \n Action 2: ']

        for j in range(len(templated_prompts)):
            templated_prompts[j] = head_prompt + templated_prompts[j]
        return templated_prompts