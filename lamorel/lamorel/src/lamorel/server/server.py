import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import sys

import logging
lamorel_logger = logging.getLogger('lamorel_logger')

from .llms import HF_LLM
from .llms.updaters import BaseUpdater
from .llms.module_functions import BaseModuleFunction, ScoreModuleFunction
from .dispatcher import Dispatcher
from .utils import InstructionsEnum

from accelerate import Accelerator

accelerator = Accelerator()


class Server:
    def __init__(self, config, llm_index, llm_group, llm_master, rl_llm_group, rl_llm_group_size, custom_updater_class,
                 custom_module_functions):
        assert dist.is_initialized(), "torch distributed must be used!"
        self._index = llm_index  # index of current process in the list of llm processes
        self._master_server_rank = llm_master
        self._is_main_server = accelerator.process_index == self._master_server_rank
        self._rl_llm_group = rl_llm_group
        self._rl_llm_group_size = rl_llm_group_size
        self._llm_group = llm_group
        self._llm_group_size = dist.get_world_size(group=self._llm_group)

        # Assign devices
        if 'cpu' in config.accelerate_args.keys():
            use_cpu = True
            devices = [0]
            lamorel_logger.info("Using CPU on process {} (index {}".format(accelerator.process_index, self._index, devices))
        else:
            use_cpu = False
            devices = self._compute_current_device_map(config, llm_index)
            lamorel_logger.info("Devices on process {} (index {}): {}".format(accelerator.process_index, self._index, devices))
        self._model = HF_LLM(config.llm_args, devices, use_cpu)
        self._dispatcher = Dispatcher(self._llm_group, self._rl_llm_group_size - 1, self._llm_group_size,
                                      self._is_main_server, self._master_server_rank, self._index)

        custom_module_functions["__score"] = ScoreModuleFunction(self._model.pad_token, config.llm_args.model_type)
        for k, _fn in custom_module_functions.items():
            assert isinstance(_fn, BaseModuleFunction)
            _fn.device = self._model.device
            _fn.llm_config = self._model.get_model_config()
            _fn.initialize()
        self._model.register_module_functions(custom_module_functions)

        if custom_updater_class is not None:
            self._updater = custom_updater_class(
                DDP(self._model, process_group=self._llm_group,
                    find_unused_parameters=not config.allow_subgraph_use_whith_gradient),
                config.updater_args
            )
            assert isinstance(self._updater, BaseUpdater)
        else:
            self._updater = BaseUpdater(self._model)
        self.run()

    def _compute_current_device_map(self, config, llm_index):
        # First compute which partition of the local GPUs our current llm process should use
        n_processes_per_machine = config.accelerate_args.num_processes // config.accelerate_args.num_machines
        n_shared_rl_processes = config.distributed_setup_args.n_rl_processes % n_processes_per_machine
        n_shared_llm_processes = n_processes_per_machine - n_shared_rl_processes
        if llm_index < n_shared_llm_processes:  # if current process is shared with rl processes
            _local_llm_index = (accelerator.process_index - n_shared_rl_processes) % n_processes_per_machine
        else:
            _local_llm_index = accelerator.process_index % n_processes_per_machine

        # Compute partitions of local GPUs
        if config.distributed_setup_args.n_llm_processes < n_processes_per_machine:
            n_devices_per_llm = torch.cuda.device_count() // config.distributed_setup_args.n_llm_processes
        else:
            n_devices_per_llm = torch.cuda.device_count() // n_processes_per_machine

        n_devices_per_llm = min(n_devices_per_llm, config.llm_args.model_parallelism_size)
        lamorel_logger.info(f"Using min(number of accessible gpus, model_parallelism_size) = {n_devices_per_llm} gpus")

        start_device = _local_llm_index * n_devices_per_llm
        devices = [i for i in range(start_device, start_device + n_devices_per_llm)]
        return devices


    def _process_calls(self, calls):
        instruction = calls[0]
        if instruction in [InstructionsEnum.FORWARD, InstructionsEnum.GENERATE, InstructionsEnum.UPDATE]:
            calls_data = calls[1]
            if calls_data is None:
                return (instruction, [None])
            else:
                llm_results = []
                for _call in calls_data:
                    if instruction == InstructionsEnum.GENERATE:
                        llm_results.append(self._model.generate(**_call))
                    elif instruction == InstructionsEnum.FORWARD:
                        llm_results.append(self._model(**_call))
                    elif instruction == InstructionsEnum.UPDATE:
                        llm_results.append([self._updater.perform_update(**_call)])
                return (instruction, llm_results)
        elif instruction == InstructionsEnum.CLOSE:
            lamorel_logger.info("Closing LLM server process {}".format(accelerator.process_index))
            sys.exit()
        else:
            raise NotImplementedError('Unknown provided instruction.')

    def run(self):
        lamorel_logger.info("Launching LLM server process {}".format(accelerator.process_index))
        while True:
            #### Receive calls from RL processes and dispatch them over LLMs ####
            method_calls = [None for _ in range(self._rl_llm_group_size)]
            if self._is_main_server:
                dist.gather_object(
                    obj=None, object_gather_list=method_calls, dst=accelerator.process_index, group=self._rl_llm_group
                )
                method_calls = method_calls[:-1]  # remove last one coming from current process
                assert len(set([call["instruction"] for call in method_calls])) <= 1  # check all calls are the same
            calls_to_process = self._dispatcher.dispatch(method_calls)
            current_process_results = self._process_calls(calls_to_process)
            if current_process_results[1] is not None:  # expected answer from caller
                gathered_results = self._dispatcher.gather(current_process_results)
                if self._is_main_server:
                    assert len(gathered_results) == self._rl_llm_group_size-1
                    if method_calls[0]["instruction"] in [InstructionsEnum.FORWARD, InstructionsEnum.GENERATE]:
                        for idx, _call in enumerate(method_calls):
                            if 'candidates' in _call:
                                if "__score" in method_calls[0]["module_function_keys"]:
                                    for i in range(len(_call["contexts"])):
                                        assert len(gathered_results[idx][i]["__score"]) == len(_call["candidates"][i])
                            else: # enough generations
                                assert len(_call["contexts"]) == len(gathered_results[idx])

                    dist.broadcast_object_list(object_list=gathered_results + [None], src=self._master_server_rank,
                                               group=self._rl_llm_group)


