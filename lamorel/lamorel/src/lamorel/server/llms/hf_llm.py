import torch
from math import ceil

import logging
lamorel_logger = logging.getLogger('lamorel_logger')

from .utils import load_hf_model_and_tokenizer
from .base_llm import BaseLLM

# Accelerate
from accelerate import Accelerator
accelerator = Accelerator()

from contextlib import nullcontext


class HF_LLM(BaseLLM):
    """
    This class is a wrapper around the language model, and handles distributing
    and/or compiling the model.
    For each task the default is text in, text out.
    """

    def __init__(self, args, devices, use_cpu):
        super().__init__(args, devices, use_cpu)
        print("Parallelising HF LLM on {} devices".format(len(self.devices)))
        # Load model and tokenizer
        self._LLM_tokenizer, self._LLM_model, num_layers = load_hf_model_and_tokenizer(
            args.model_type, args.model_path, args.pretrained)

        if use_cpu:
            # Current version of the lib does not support parallelization with cpu
            self._LLM_model.to('cpu')
        else:
            # Set model parallelism
            layers_per_device = ceil(num_layers / len(self.devices))
            device_map = {
                _device: list(range(i * layers_per_device, min((i + 1) * layers_per_device, num_layers)))
                for i, _device in enumerate(self.devices)
            }
            self._LLM_model.parallelize(device_map)

        # Set minibatch generation
        self._scoring_minibatch_size = args.minibatch_size
        if args.model_type == "causal":
            self.__minibatch_generator = self.__build_decoder_minibatch
            self.__input_encoder = lambda input: None
            self.model_type = "causal"
        elif args.model_type == "seq2seq":
            self.__minibatch_generator = self.__build_encoder_decoder_minibatch
            self.__input_encoder = lambda input: self.__get_input_embedding_from_encoder(input)
            self.model_type = "seq2seq"
        else:
            raise NotImplementedError()

        if self._LLM_tokenizer.pad_token is not None:
            self.pad_token = 0  # self._LLM_tokenizer(self._LLM_tokenizer.pad_token)
        else:
            self.pad_token = 0  # self._LLM_tokenizer(" ")
        self.__synchronize_gpus_after_scoring = args.synchronize_gpus_after_scoring
        self.__empty_cuda_cache_after_scoring = args.empty_cuda_cache_after_scoring

    def get_model_config(self):
        return self._LLM_model.config

    def register_module_functions(self, module_functions):
        self._module_functions = torch.nn.ModuleDict(module_functions)

    def __pad_sequence(self, sequence, size):
        sequence_size = len(sequence["input_ids"])
        ids = sequence["input_ids"] + [
            self.pad_token
            for _ in range(size - sequence_size)]
        mask = sequence["attention_mask"] + [0 for _ in range(size - sequence_size)]
        sequence["input_ids"] = ids
        sequence["attention_mask"] = mask
        return sequence

    def get_trainable_module(self):
        return self._LLM_model

    def __concat_sequences(self, sequence_a, sequence_b, sequence_size=None):
        if sequence_size is None: sequence_size = len(sequence_a['input_ids']) + len(sequence_b['input_ids'])
        return self.__pad_sequence({
            "input_ids": sequence_a['input_ids'] + sequence_b['input_ids'],
            "attention_mask": sequence_a['attention_mask'] + sequence_b['attention_mask']
        }, sequence_size)

    def __build_decoder_minibatch(self, outputs, inputs, inputs_representation=None):
        '''
            Concat state and output
        '''
        batch = {
            "input_ids": [],
            "attention_mask": []
        }
        output_max_size = max([len(_i['input_ids']) + len(_o['input_ids']) for _i, _o in zip(inputs, outputs)])
        for input, output in zip(inputs, outputs):
            concatenation = self.__concat_sequences(input, output, output_max_size)
            batch["input_ids"].append(concatenation["input_ids"])
            batch["attention_mask"].append(concatenation["attention_mask"])

        return batch

    def __build_encoder_decoder_minibatch(self, outputs, inputs=None, inputs_representation=None):
        '''
            Set state as encoder's input and action as decoder's input
        '''
        assert inputs is not None or inputs_representation is not None
        batch = {
            "decoder_input_ids": [],
            "decoder_attention_mask": []
        }

        output_max_size = max([len(o['input_ids']) for o in outputs])
        if inputs is not None:
            input_max_size = max([len(i['input_ids']) for i in inputs])
        for idx, output in enumerate(outputs):
            _output = self.__concat_sequences(
                {"input_ids": [self.pad_token], "attention_mask": [1]},
                output,
                output_max_size + 1
            )
            batch["decoder_input_ids"].append(_output["input_ids"])
            batch["decoder_attention_mask"].append(_output["attention_mask"])
            if inputs_representation is None:
                if idx == 0:
                    batch["input_ids"] = []
                    batch["attention_mask"] = []
                _input = self.__pad_sequence(inputs[idx], input_max_size)
                batch["input_ids"].append(_input["input_ids"])
                batch["attention_mask"].append(_input["attention_mask"])

        if inputs_representation is not None:
            batch["encoder_outputs"] = inputs_representation

        return batch

    def __get_input_embedding_from_encoder(self, input):
        input_batch = {}
        for key, value in input.items():
            input_batch[key] = torch.tensor([value], device=self.device)
        return self._LLM_model.encoder(**input_batch, return_dict=False)[0]

    def generate(self, contexts, **kwargs):
        generations = []
        for text_input in contexts:
            encoded_input = self._LLM_tokenizer.encode(text_input, return_tensors='pt').to(self.device)
            results = self._LLM_model.generate(
                encoded_input,
                return_dict_in_generate=True,
                output_scores=True,
                **kwargs
            )
            if self.model_type == "causal":  # hence input should be removed from result
                generated_sequences = results.sequences[:, encoded_input.shape[-1]:]
            else:
                generated_sequences = results.sequences[:, 1:]
            _generated_texts = self._LLM_tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)
            probabilities = torch.stack(results.scores, dim=1).softmax(-1)
            texts_probabilities = torch.gather(probabilities, 2, generated_sequences[:, :, None]).squeeze(-1)
            _scores = texts_probabilities.prod(-1)

            generations.append([
                {
                    "text": _text,
                    "score": _score.detach().cpu().numpy()
                }
                for _text, _score in zip(_generated_texts, _scores)
            ])
        return generations

    def forward(self, module_function_keys, contexts, candidates=None, require_grad=False, **kwargs):
        llm_scores = []
        if candidates is None:
            candidates = [[""] for _ in range(len(contexts))]
        _w = 0
        for input, outputs in zip(contexts, candidates):
            if len(outputs) == 0:
                llm_scores.append([])
                break
            _w += 1
            lamorel_logger.debug(f"Tokenizing the {_w}-th batch")
            # Tokenize
            input = self._LLM_tokenizer(input)
            outputs = [self._LLM_tokenizer(output) for output in outputs]

            lamorel_logger.debug(f"Encoding the {_w}-th batch")

            with torch.no_grad() if not require_grad else nullcontext():
                input_representation = self.__input_encoder(input)  # Encode input just once and copy it in minibatches

                batch_results = {k: [] for k in module_function_keys}
                for step in range(len(outputs) // self._scoring_minibatch_size + 1):
                    step_idx = step * self._scoring_minibatch_size
                    current_minibatch_size = min(self._scoring_minibatch_size, len(outputs) - step_idx)
                    if current_minibatch_size <= 0: break

                    lamorel_logger.debug(f"Building the {step + 1}-th minibatch")
                    # Build mini batch
                    current_minibatch_outputs = outputs[step_idx:step_idx + current_minibatch_size]
                    if input_representation is not None:
                        _input_representation = tuple([input_representation.expand(current_minibatch_size, -1, -1)])
                    else:
                        _input_representation = None
                    minibatch_inputs = self.__minibatch_generator(
                        current_minibatch_outputs,
                        inputs=[input for _ in range(current_minibatch_size)],
                        inputs_representation=_input_representation
                    )

                    if not module_function_keys == ['__score']:
                        minibatch_inputs["output_hidden_states"] = True

                    # Transform it to tensors on the right device
                    lamorel_logger.debug(f"Putting it on device {self.device}")
                    minibatch = {}
                    for key, value in minibatch_inputs.items():
                        if key == "encoder_outputs":
                            minibatch[key] = value
                        else:
                            minibatch[key] = torch.tensor(value, device=self.device)

                    lamorel_logger.debug(f"Calling forward on process {accelerator.process_index}")
                    _outputs = self._LLM_model(**minibatch)  # Get scores before softmax
                    lamorel_logger.debug(f"Forward succeeded on process {accelerator.process_index}")

                    for _key in module_function_keys:
                        lamorel_logger.debug(f"Computing {_key} function")
                        _fn = self._module_functions[_key]
                        if _key == "__score":
                            results = _fn(_outputs, minibatch, input)
                        else:
                            results = _fn(_outputs, minibatch=minibatch, tokenized_context=input, **kwargs)
                        batch_results[_key].append(results)

                for k, _ in batch_results.items():
                    batch_results[k] = torch.cat(batch_results[k])
                llm_scores.append(batch_results)

        if self.__synchronize_gpus_after_scoring:
            lamorel_logger.debug(f"Synchronizing GPUs on process {accelerator.process_index}")
            for device in self.devices:
                torch.cuda.synchronize(device)
        if self.__empty_cuda_cache_after_scoring:
            lamorel_logger.debug(f"Emptying CUDA cache on process {accelerator.process_index}")
            torch.cuda.empty_cache()

        lamorel_logger.debug(f"Scoring finished on process {accelerator.process_index}")
        return llm_scores

    # def single_candidate_fns(self, module_function_keys, contexts, candidates, **kwargs):
    #     assert sum([len(_candidates) for _candidates in candidates]) == len(contexts)  # 1 candidate per context
    #     batch_scores = {k: [] for k in module_function_keys}
    #     for step in range(len(contexts) // self._scoring_minibatch_size + 1):
    #         step_idx = step * self._scoring_minibatch_size
    #         current_minibatch_size = min(self._scoring_minibatch_size, len(contexts) - step_idx)
    #         if current_minibatch_size <= 0: break
    #
    #         # Build mini batch
    #         inputs = [self._LLM_tokenizer(_c) for _c in contexts[step_idx:step_idx + current_minibatch_size]]
    #         outputs = [self._LLM_tokenizer(_c[0]) for _c in candidates[step_idx:step_idx + current_minibatch_size]]
    #         minibatch_inputs = self.__minibatch_generator(
    #             outputs,
    #             inputs=inputs
    #         )
    #
    #         # Transform it to tensors on the right device
    #         minibatch = {}
    #         for key, value in minibatch_inputs.items():
    #             if key == "encoder_outputs":
    #                 minibatch[key] = tuple([value[0].to(self.device)])
    #             else:
    #                 minibatch[key] = torch.tensor(value, device=self.device)
    #
    #         _outputs = self._LLM_model(**minibatch)  # Get scores before softmax
    #
    #         for _key in module_function_keys:
    #             lamorel_logger.debug(f"Computing {_key} function")
    #             _fn = self._module_functions[_key]
    #             if _key == "__score":
    #                 results = _fn(_outputs, minibatch, input)
    #             else:
    #                 results = _fn(_outputs, **kwargs)
    #             batch_scores[_key].append(results)
    #
    #     for k, _ in batch_scores:
    #         batch_scores[k] = torch.cat(batch_scores[k])
    #
    #     if self.__synchronize_gpus_after_scoring:
    #         for device in self.devices:
    #             torch.cuda.synchronize(device)
    #     if self.__empty_cuda_cache_after_scoring:
    #         torch.cuda.empty_cache()
    #
    #     return batch_scores
