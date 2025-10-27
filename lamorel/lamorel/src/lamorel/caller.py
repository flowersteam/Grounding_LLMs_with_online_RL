import torch.distributed as dist
import typing
from accelerate import Accelerator
from .server import Server
from .server.utils import InstructionsEnum

import logging
lamorel_logger = logging.getLogger('lamorel_logger')

accelerator = Accelerator()



class Caller:
    '''
    This class should be called by each process.
    It will instantiate the different distributed groups.
    If the current process belongs to the LLM's processes, it will launch the LLM and wait for requests.
    '''
    def __init__(self, config, custom_updater_class=None, custom_module_functions={}):
        assert dist.is_initialized(), "torch distributed must be used!"
        self.__config = config
        self.__grad_fn_model = None

        # Set log level
        numeric_log_level = getattr(logging, config.log_level.upper(), None)
        if not isinstance(numeric_log_level, int):
            raise ValueError('Invalid log level: %s' % config.log_level)
        lamorel_logger.setLevel(numeric_log_level)

        # Initialize distributed groups
        # RL processes are considered as the first n processes
        rl_processes = list(range(config.distributed_setup_args.n_rl_processes))
        llm_processes = list(range(
            config.distributed_setup_args.n_rl_processes,
            config.distributed_setup_args.n_rl_processes + config.distributed_setup_args.n_llm_processes))

        lamorel_logger.info("Init rl group for process {}".format(accelerator.process_index))
        self._rl_group = dist.new_group(
            ranks=rl_processes,
            backend='gloo'
        )
        lamorel_logger.info("Init llm group for process {}".format(accelerator.process_index))
        self._llm_group = dist.new_group(
            ranks=llm_processes,
            backend='gloo'
        )
        lamorel_logger.info("Init rl-llm group for process {}".format(accelerator.process_index))
        self._llm_master_process = rl_processes[-1] + 1  # First LLM process is considered as master
        self._rl_llm_group = dist.new_group(
            ranks=rl_processes + [self._llm_master_process],
            backend='gloo'
        )

        if accelerator.process_index in llm_processes:
            Server(
                config,
                llm_processes.index(accelerator.process_index),
                self._llm_group,
                self._llm_master_process,
                self._rl_llm_group,
                len(rl_processes) + 1,
                custom_updater_class,
                custom_module_functions
            )

    def get_rl_dist_group(self):
        '''
        Use this group for communication between RL processes
        :return: torch distributed group
        '''
        return self._rl_group

    def score(self, contexts: typing.List[str], candidates: typing.List[typing.List[str]],
              additional_module_function_keys: typing.List[str] = [], **kwargs):
        module_function_keys = ["__score"]
        module_function_keys.extend(additional_module_function_keys)
        result = self.__call_model(InstructionsEnum.FORWARD, True, module_function_keys=module_function_keys,
                                   contexts=contexts, candidates=candidates, **kwargs)
        if additional_module_function_keys == []:
            result = [_r['__score'] for _r in result]
        return result

    def generate(self, contexts: list, **kwargs):
        return self.__call_model(InstructionsEnum.GENERATE, True, contexts=contexts, **kwargs)

    def update(self, contexts: typing.List[str], candidates: typing.List[typing.List[str]], **kwargs):
        result = self.__call_model(InstructionsEnum.UPDATE, True, contexts=contexts, candidates=candidates, **kwargs)
        if not isinstance(result, list):
            result = [result]
        return result

    def close(self):
        self.__call_model(InstructionsEnum.CLOSE, False)

    def custom_module_fns(self, module_function_keys : typing.List[str], contexts: typing.List[str], **kwargs):
        return self.__call_model(InstructionsEnum.FORWARD, True, module_function_keys=module_function_keys,
                                 contexts=contexts, **kwargs)

    def __call_model(self, instruction, expect_answer, **kwargs):
        dist.gather_object(
            obj={
                "instruction": instruction,
                **kwargs
            }, object_gather_list=None, dst=self._llm_master_process, group=self._rl_llm_group
        )

        if expect_answer:
            results = [None for _ in range(dist.get_world_size(group=self._rl_llm_group))]
            dist.broadcast_object_list(object_list=results, src=self._llm_master_process, group=self._rl_llm_group)
            return results[accelerator.process_index]
        else:
            return None


