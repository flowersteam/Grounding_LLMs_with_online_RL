from .base_env import BaseEnv
import alfworld.agents.environment as environment
import yaml

class AlfWorldEnv(BaseEnv):
    def __init__(self, config_dict):
        super().__init__(config_dict)
        config = yaml.safe_load(open(config_dict["config_file"]))  # "configs/base_config.yaml"
        env = environment.AlfredTWEnv(config, "train" if config_dict["train_mode"] else "eval")
        self._env = env.init_env(batch_size=config_dict["number_envs"])

    def __append_action_space_to_info(self, infos):
        _infos = [{
            "goal": self.__curent_goals[i]
        } for i in range(self.n_parallel)]

        for key, value in infos.items():
            if key == "admissible_commands":
                _key = "possible_actions"
            else:
                _key = key

            for i in range(self.n_parallel):
                _infos[i][_key] = value[i]

        return _infos

    def __prepare_obs(self, obs):
        return [[o] for o in obs]
    def reset(self):
        obs, infos = self._env.reset()
        self.__curent_goals = obs
        return ([["You are in the middle of the room"] for _ in range(self.n_parallel)],
                self.__append_action_space_to_info(infos))
    def step(self, actions_id, actions_command):
        obs, rews, dones, infos = self._env.step(actions_command)
        return self.__prepare_obs(obs), \
                rews, \
                dones, \
                self.__append_action_space_to_info(infos)

    # def get_template_prompts(self, template_test):
    #     if template_test == 1:
    #         # expected answers: go forward, turn left, turn left, toggle
    #         templated_prompts = [
    #             ' \n Goal of the agent: go to the green ball \n Observation 0: A wall 2 step left, A purple key 1 step left and 2 steps forward, A yellow key 1 step left and 1 step forward, A green ball 3 steps forward, A grey ball 1 step right and 5 steps forward, A green key 1 step right and 2 steps forward, A grey ball 1 step right and 1 step forward, A green key 2 steps right and 4 steps forward, A red box 2 steps right and 2 steps forward, \n Action 0: ',
    #             ' \n Goal of the agent: go to the green ball \n Observation 0: A wall 2 step left, A purple key 1 step left and 2 steps forward, A yellow key 1 step left and 1 step forward, A green ball 3 steps forward, A grey ball 1 step right and 5 steps forward, A green key 1 step right and 2 steps forward, A grey ball 1 step right and 1 step forward, A green key 2 steps right and 4 steps forward, A red box 2 steps right and 2 steps forward, \n Action 0: go forward \n Observation 1: A purple key 1 step left and 1 step forward, A yellow key 1 step left, A green ball 2 steps forward, A grey ball 1 step right and 4 steps forward, A green key 1 step right and 1 step forward, A grey ball 1 step right, A green key 2 steps right and 3 steps forward, A red box 2 steps right and 1 step forward, \n Action 1: turn right \n Observation 2: A wall 2 step right, A green key 3 steps left and 2 steps forward, A green ball 2 steps left, A red box 1 step left and 2 steps forward, A green key 1 step left and 1 step forward, A grey ball 1 step forward, \n Action 2: ',
    #             ' \n Goal of the agent: open the purple door \n Observation 0: You see a wall 3 steps forward, You see a wall 3 steps left, You see a yellow key 1 step right and 1 step forward, You see a locked purple door 2 steps right and 3 steps forward, You see a purple ball 3 steps right and 1 step forward, You see a green box 3 steps right, You see a purple key 2 steps left \n Action 0: ',
    #             ' \n Goal of the agent: open the purple door \n Observation 0: You see a wall 3 steps forward, You see a wall 3 steps left, You see a yellow key 1 step right and 1 step forward, You see a locked purple door 2 steps right and 3 steps forward, You see a purple ball 3 steps right and 1 step forward, You see a green box 3 steps right, You see a purple key 2 steps left \n Action 0: turn left \n Observation 1: You see a wall 3 steps forward, You see a wall 3 steps right, You see a purple key 2 steps forward \n Action 1: go forward \n Observation 2: You see a wall 2 steps forward, You see a wall 3 steps right, You see a purple key 1 step forward \n Action 2: ',
    #             ' \n Goal of the agent: open the purple door \n Observation 0: You carry a purple key, You see a wall 3 steps forward, You see a wall 5 steps left, You see a yellow key 1 step left and 1 step forward, You see a locked purple door 3 steps forward, You see a purple ball 1 step right and 1 step forward, You see a green box 1 step right \n Action 0: go forward \n Observation 1: You carry a purple key, You see a wall 2 steps forward, You see a wall 5 steps left, You see a yellow key 1 step left, You see a locked purple door 2 steps forward, You see a purple ball 1 step right \n Action 1: go forward \n Observation 2: You carry a purple key, You see a wall 1 step forward, You see a wall 5 steps left, You see a locked purple door 1 step forward \n Action 2: ',
    #             ' \n Goal of the agent: pick up green box \n Observation 0: You see a wall 2 steps forward, You see a wall 2 steps left, You see a yellow ball 1 step left and 1 step forward, You see a green box 2 steps right \n Action 0: ',
    #             ' \n Goal of the agent: pick up green box \n Observation 0: You see a wall 2 steps forward, You see a wall 2 steps left, You see a yellow ball 1 step left and 1 step forward, You see a green box 2 steps right \n Action 0: turn right \n Observation 1: You see a wall 2 steps left, You see a blue key 1 step right, You see a red ball 2 steps right and 1 step forward, You see a green box 2 steps forward \n Action 1: go forward \n Observation 2: You see a wall 2 steps left, You see a red ball 2 steps right, You see a green box 1 step forward \n Action 2: ',
    #             ' \n Goal of the agent: put blue ball next to red box \n Observation 0: You carry a blue ball, You see a wall 5 steps forward, You see a wall 2 steps left, You see a grey key 1 step right and 2 steps forward, You see a red box 3 steps forward \n Action 0: go forward \n Observation 1: You carry a blue ball, You see a wall 4 steps forward, You see a wall 2 steps left, You see a grey key 1 step right and 1 step forward, You see a red box 2 steps forward \n Action 1: ',
    #             ' \n Goal of the agent: pick up the blue ball then go to the red box \n Observation 0: You see a wall 3 steps forward, You see a wall 4 steps right, You see a purple key 2 steps forward, You see a red box 2 steps right, You see a blue ball 2 steps left \n Action 0: ',
    #             ' \n Goal of the agent: go to the red box after you pick up the blue ball \n Observation 0: You see a wall 3 steps forward, You see a wall 4 steps right, You see a purple key 2 steps forward, You see a red box 2 steps right, You see a blue ball 2 steps left \n Action 0: ',
    #             ' \n Goal of the agent: pick up the green key then pick up the the red box \n Observation 0: You carry a green key, You see a wall 4 steps forward, You see a wall 4 steps left, You see a red box 1 step left, You see a purple ball 2 steps left and 1 step forward \n Action 0:  ']
    #     elif template_test == 2:
    #         # expected answers: go forward, turn left
    #         templated_prompts = [
    #             ' \n Goal of the agent: go to the green ball \n Observation 0: A wall 2 step left, A purple key 1 step left and 2 steps forward, A yellow key 1 step left and 1 step forward, A green ball 3 steps forward, A grey ball 1 step right and 5 steps forward, A green key 1 step right and 2 steps forward, A grey ball 1 step right and 1 step forward, A green key 2 steps right and 4 steps forward, A red box 2 steps right and 2 steps forward, \n Action 0: ',
    #             ' \n Goal of the agent: go to the green ball \n Observation 0: A wall 2 step left, A purple key 1 step left and 2 steps forward, A yellow key 1 step left and 1 step forward, A green ball 3 steps forward, A grey ball 1 step right and 5 steps forward, A green key 1 step right and 2 steps forward, A grey ball 1 step right and 1 step forward, A green key 2 steps right and 4 steps forward, A red box 2 steps right and 2 steps forward, \n Action 0: go forward \n Observation 1: A purple key 1 step left and 1 step forward, A yellow key 1 step left, A green ball 2 steps forward, A grey ball 1 step right and 4 steps forward, A green key 1 step right and 1 step forward, A grey ball 1 step right, A green key 2 steps right and 3 steps forward, A red box 2 steps right and 1 step forward, \n Action 1: turn right \n Observation 2: A wall 2 step right, A green key 3 steps left and 2 steps forward, A green ball 2 steps left, A red box 1 step left and 2 steps forward, A green key 1 step left and 1 step forward, A grey ball 1 step forward, \n Action 2: ']
    #
    #     return templated_prompts