import logging
lamorel_logger = logging.getLogger('lamorel_logger')

from torch.nn import Module

class BaseModuleFunction(Module):
    def __init__(self):
        super().__init__()

    def initialize(self):
        '''
        Use this method to initialize your module operations
        '''
        raise NotImplementedError()

    def forward(self, forward_outputs, minibatch, tokenized_context,  **kwargs):
        raise NotImplementedError()
