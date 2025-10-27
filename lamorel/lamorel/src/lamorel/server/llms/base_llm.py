import torch
# Accelerate
from accelerate import Accelerator

accelerator = Accelerator()


class BaseLLM(torch.nn.Module):
    def __init__(self, args, devices, use_cpu=False):
        super().__init__()
        self.devices = devices
        self.device_id = self.devices[0]
        self.use_cpu = use_cpu
        if self.use_cpu:
            self.device = 'cpu'
        else:
            self.device = torch.device(f'cuda:{self.device_id}')  # use first device of map for tensors

    def register_module_functions(self, module_functions):
        raise NotImplementedError()

    def generate(self, contexts, **kwargs):
        raise NotImplementedError()

    def get_model_config(self):
        raise NotImplementedError()

    def forward(self, module_function_keys, contexts, candidates=None, require_grad=False, **kwargs):
        raise NotImplementedError()

    def get_trainable_module(self):
        return None