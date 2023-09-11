class BaseEnv:
    def __init__(self, config_dict):
        self.n_parallel = config_dict["number_envs"]

    def step(self, actions_id, actions_command):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()