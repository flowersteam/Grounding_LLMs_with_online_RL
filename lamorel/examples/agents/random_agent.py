import numpy as np

class RandomAgent:

    def build_state(self, obs, info):
        state = {
            'obs': obs,
            'inventory': info['inv'],
            'valid_actions': info['valid'],
        }
        return state

    def act(self, state):
        return np.random.choice(state['valid_actions'])
