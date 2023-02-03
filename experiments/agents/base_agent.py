from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, envs):
        self.env = envs

    @abstractmethod
    def collect_experiences(self, debug=False):
        raise NotImplementedError()

    @abstractmethod
    def update_parameters(self):
        raise NotImplementedError()

