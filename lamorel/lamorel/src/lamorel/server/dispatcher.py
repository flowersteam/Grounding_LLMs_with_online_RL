import torch.distributed as dist
from copy import copy, deepcopy
from enum import Enum
import numpy as np
import torch
from .utils import InstructionsEnum

class DispatchMode(Enum):
    calls = 0,
    batchs = 1


class Dispatcher:
    def __init__(self, process_group, n_callers, n_handlers, is_current_process_master, master_rank,
                 current_process_index):
        self._process_group = process_group
        self._n_callers = n_callers
        self._n_handlers = n_handlers
        self._is_master = is_current_process_master
        self._master_rank = master_rank
        self._current_process_index = current_process_index

        if self._n_callers >= self._n_handlers:
            self._mode = DispatchMode.calls
            self._n_calls_per_handler = self._n_callers // self._n_handlers
        else:
            self._mode = DispatchMode.batchs
            self._current_calls_infos = []
            self._n_handlers_per_call = self._n_handlers // self._n_callers

    def __dispatch_calls(self, calls):
        '''
        Dispatch `n_callers` calls over `n_handlers` llms
        '''
        if self._is_master:
            scattered_calls = []
            for i in range(0, len(calls), self._n_calls_per_handler):
                _current_handler_calls = []
                for j in range(self._n_calls_per_handler):
                    _call = {
                        k: v for k, v in calls[i+j].items()
                        if k not in ["instruction"]
                    }
                    if calls[i+j]["instruction"] in [InstructionsEnum.UPDATE]:
                        _call["_current_batch_ids"] = [i for i in range(len(_call["contexts"]))]
                    _current_handler_calls.append(_call)
                scattered_calls.append(_current_handler_calls)
        else:
            scattered_calls = [None for _ in range(self._n_handlers)]
        return scattered_calls

    def __gather_calls(self, calls):
        '''
        Gather `n_callers` results from `n_handlers` llms
        '''
        gathered_calls = []
        if self._is_master:
            for _calls in calls:
                gathered_calls.extend(_calls)
        return gathered_calls

    def __dispatch_batches(self, calls):
        '''
        When `n_callers` < `n_handlers` we can dispatch each call over multiple LLMs.
        We first check for each call whether multiple entries are sent (each caller can send a batch of entries) and dispatch them.
        If we still have enough LLMs, we check if the `score` method is called (i.e. with multiple candidates). If so,
        we dispatch the candidates over multiple LLMs to score minibatches in parallel.
        '''
        scattered_calls = [None for _ in range(self._n_handlers)]
        if self._is_master:
            i = 0
            for call in calls:
                _batch_size = len(call["contexts"])
                self._current_calls_infos.append({  # Useful for for gathering
                    'batch_size': _batch_size
                })
                dispatched_call = {
                    k: v for k, v in call.items()
                    if k not in ["instruction", "contexts", "candidates"]
                }
                if self._n_handlers_per_call / _batch_size > 1:  # Each batch entry should be splitted over multiple handlers
                    _ids = np.arange(i, i + self._n_handlers_per_call)
                    batches_handlers = np.array_split(_ids, _batch_size)
                    for j in range(len(call["contexts"])):
                        _dispatched_call = copy(dispatched_call)
                        _dispatched_call["contexts"] = [call["contexts"][j]]
                        _batch_handlers = batches_handlers[j]
                        if "candidates" in call:
                            _call_batch_ids = np.arange(len(call["candidates"][j]))
                            _minibatches = np.array_split(_call_batch_ids, len(_batch_handlers))
                            for _handler in _batch_handlers:
                                _call = deepcopy(_dispatched_call)
                                _call["candidates"] = [[call["candidates"][j][_idx] for _idx in _minibatches[_handler]]]
                                scattered_calls[_handler] = [_call]
                        else:
                            scattered_calls[_batch_handlers[0]] = [_dispatched_call] # TODO also parallelize generation
                else:  # Each handler handles multiple batch entry
                    _ids = np.arange(_batch_size)
                    batch_chunks = np.array_split(_ids, self._n_handlers_per_call)
                    for j in range(i, i + self._n_handlers_per_call):
                        _dispatched_call = copy(dispatched_call)
                        _dispatched_call["contexts"] = [call["contexts"][_idx] for _idx in batch_chunks[j]]
                        if "candidates" in call:
                            if call["candidates"] is not None:
                                _dispatched_call["candidates"] = [call["candidates"][_idx] for _idx in batch_chunks[j]]
                            else:
                                _dispatched_call["candidates"] = None
                        if call["instruction"] in [InstructionsEnum.UPDATE]:
                            _dispatched_call["_current_batch_ids"] = batch_chunks[j].tolist()
                        scattered_calls[j] = [_dispatched_call]

                i += self._n_handlers_per_call
        return scattered_calls

    def __gather_batches(self, calls):
        '''
        Gather dispatched batches (dispatched entries but also maybe dispatched scoring candidates)
        '''
        # TODO handle generation parallelization once implemented
        gathered_calls = []
        if self._is_master:
            for i in range(0, self._n_handlers, self._n_handlers_per_call):
                current_call_results = []
                _batch_size = self._current_calls_infos[i]["batch_size"]
                if self._n_handlers_per_call / _batch_size > 1:  # Each batch entry has been split over multiple handlers
                    _ids = np.arange(i, i + self._n_handlers_per_call)
                    batches_handlers = np.array_split(_ids, _batch_size)  # Infer number of batches
                    for _batch_handlers in batches_handlers:
                        current_context_entry_results = {}
                        for _handler in _batch_handlers:
                            _result = calls[_handler][0]  # [0] as a single call is handled by a handler
                            if _result is not None: # concat results to reconstruct entry
                                if isinstance(_result[0], dict): # [0] as a single context entry is handled by each handler
                                    for _k, _v in _result[0].items():
                                        if _k in current_context_entry_results:
                                            current_context_entry_results[_k] = \
                                                torch.cat((current_context_entry_results[_k], _result[0][_k]), dim=0)
                                        else:
                                            current_context_entry_results[_k] = _result[0][_k]
                                else:
                                    current_context_entry_results['__generate'] = _result[0]  # [0] as a single context entry is handled by each handler
                        current_call_results.append(current_context_entry_results)
                else:
                    for j in range(i, i + self._n_handlers_per_call):
                        current_call_results.extend(calls[j][0]) # [0] as a single context entry is handled by each handler
                gathered_calls.append(current_call_results)
        self._current_calls_infos = []
        return gathered_calls


    def dispatch(self, calls):
        instruction = None
        if self._is_master:
            instruction = calls[0]["instruction"]

        if instruction in [InstructionsEnum.FORWARD, InstructionsEnum.GENERATE, InstructionsEnum.UPDATE]:
            if self._mode == DispatchMode.calls:
                _scattered_calls = self.__dispatch_calls(calls)
            else:
                _scattered_calls = self.__dispatch_batches(calls)
            scattered_calls = [(instruction, _call) for _call in _scattered_calls]
        else:
            scattered_calls = [(instruction,) for _ in range(self._n_handlers)]
        #### Scatter over processes ####
        # TODO Open an issue on torch distributed for `scatter_object_list` not working
        # calls_to_process = [{} for _ in range(n_calls_per_llm)]
        # dist.scatter_object_list(scatter_object_output_list=calls_to_process,
        #                          scatter_object_input_list=scattered_calls,
        #                          src=self._master_server_rank, group=self._llm_group)
        # Get calls to be processed from current process
        dist.broadcast_object_list(scattered_calls, src=self._master_rank, group=self._process_group)
        return scattered_calls[self._current_process_index]

    def gather(self, handler_results):
        instruction = handler_results[0]

        if self._is_master:
            handlers_results = [None for _ in range(self._n_handlers)]
        else:
            handlers_results = None
        dist.gather_object(handler_results, handlers_results, dst=self._master_rank, group=self._process_group)

        if self._is_master:
            results = [_handler_results[1] for _handler_results in handlers_results]
            if instruction in [InstructionsEnum.FORWARD, InstructionsEnum.GENERATE, InstructionsEnum.UPDATE]:
                if self._mode == DispatchMode.calls:
                    return self.__gather_calls(results)
                else:
                    call_results = self.__gather_batches(results)
                    if instruction == InstructionsEnum.GENERATE:
                        call_results = list(map(
                            lambda call_batch: [_results['__generate'] for _results in call_batch],
                            call_results
                        ))
                    return call_results
            else:
                raise NotImplementedError()
        else:
            return None