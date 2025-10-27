from collections import namedtuple
import numpy as np
import json
import sys

State = namedtuple('State', ('obs'))  #, 'description', 'inventory'))
Transition = namedtuple('Transition', ('state', 'act', 'reward', 'next_state', 'next_acts', 'done'))


def sample(rng: np.random.RandomState, data: list, k: int):
    """ Chooses k unique random elements from a list. """
    return [data[i] for i in rng.choice(len(data), k, replace=False)]


class ReplayMemory(object):
    def __init__(self, capacity, seed=20210824):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.rng = np.random.RandomState(seed)

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return sample(self.rng, self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class PrioritizedReplayMemory(object):
    def __init__(self, capacity=100000, priority_fraction=0.0, seed=20210824):
        # Stored
        self.capacity = capacity
        self.priority_fraction = priority_fraction
        self.seed = seed
        
        # Calculated at init
        self.alpha_capacity = int(capacity * priority_fraction)
        self.beta_capacity = capacity - self.alpha_capacity

        # Declared
        self.alpha_memory, self.beta_memory = [], []
        self.alpha_position, self.beta_position = 0, 0

        # Initialized
        self.rng = np.random.RandomState(seed)

    def push(self, is_prior=False, *args):
        """Saves a transition."""
        if self.priority_fraction == 0.0:
            is_prior = False
        if is_prior:
            if len(self.alpha_memory) < self.alpha_capacity:
                self.alpha_memory.append(None)
            self.alpha_memory[self.alpha_position] = Transition(*args)
            self.alpha_position = (self.alpha_position + 1) % self.alpha_capacity
        else:
            if len(self.beta_memory) < self.beta_capacity:
                self.beta_memory.append(None)
            self.beta_memory[self.beta_position] = Transition(*args)
            self.beta_position = (self.beta_position + 1) % self.beta_capacity

    def sample(self, batch_size):
        if self.priority_fraction == 0.0:
            from_beta = min(batch_size, len(self.beta_memory))
            res = sample(self.rng, self.beta_memory, from_beta)
        else:
            from_alpha = min(int(self.priority_fraction * batch_size), len(self.alpha_memory))
            from_beta = min(batch_size - int(self.priority_fraction * batch_size), len(self.beta_memory))
            res = sample(self.rng, self.alpha_memory, from_alpha) + sample(self.rng, self.beta_memory, from_beta)

        self.rng.shuffle(res)
        return res

    def __len__(self):
        return len(self.alpha_memory) + len(self.beta_memory)

    def serializeToJSON(self, filenameOut):
        print("Serializing to JSON... ")
        sys.stdout.flush()

        packed = {
            "capacity": self.capacity,
            "priority_fraction": self.priority_fraction, 
            "alpha_memory": self.alpha_memory,
            "alpha_position": self.alpha_position,
            "beta_memory": self.beta_memory,
            "beta_position": self.beta_position,    
        }

        print(packed)
        sys.stdout.flush()

        with open(filenameOut, 'w') as outfile:
            outfile.write(json.dumps(packed, cls=NpEncoder, indent=2))

        print("Completed...")
        sys.stdout.flush()


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NpEncoder, self).default(obj)