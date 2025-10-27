"""
This script run a simple agent in a ScienceWorld environment, with dummy calls to an API
to perform inference on the provided data.
"""
import transformers
# transformers.models.gpt2.modeling_gpt2.GPT2Block = None
import torch

from lamorel import Caller, lamorel_init
lamorel_init()

import hydra
from accelerate import Accelerator

accelerator = Accelerator()

from lamorel import BaseUpdater, BaseModuleFunction

class ValueModuleFn(BaseModuleFunction):
    def __init__(self, model_type):
        super().__init__()
        self._model_type = model_type

    def initialize(self):
        llm_hidden_size = self.llm_config.to_dict()[self.llm_config.attribute_map['hidden_size']]
        self.value_head_op = torch.nn.Sequential(
            torch.nn.Linear(llm_hidden_size, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, 1),
            torch.nn.Sigmoid(),
        ).to(self.device)

    def forward(self, forward_outputs, minibatch, tokenized_context, **kwargs):
        if self._model_type == "causal":
            model_head = forward_outputs['hidden_states'][0][0, len(tokenized_context["input_ids"])-1, :]
        else:
            model_head = forward_outputs['encoder_last_hidden_state'][0, len(tokenized_context["input_ids"]) - 1, :]

        value = self.value_head_op(model_head)
        return value

class TestUpdater(BaseUpdater):
    def perform_update(self, contexts, candidates, _current_batch_ids, **kwargs):
        if not hasattr(self, 'loss_fn'):
            self.loss_fn = torch.nn.BCELoss()
        if not hasattr(self, 'optimizer'):
            # self.optimizer = torch.optim.Adam(self._llm_module._LLM_model.parameters())
            self.optimizer = torch.optim.Adam(self._llm_module.module._module_functions["value"].parameters())
            # self.optimizer = torch.optim.Adam(self._llm_module.parameters())

        output = self._llm_module(['value'], contexts=contexts, require_grad=True)
        values = torch.stack([_o["value"] for _o in output]).to('cpu')
        loss = self.loss_fn(values, kwargs["labels"][_current_batch_ids, :])
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return {"loss": loss}

# This will be overriden by lamorel's launcher if used
@hydra.main(config_path='config', config_name='config')
def main(config_args):

    # lm server
    lm_server = Caller(config_args.lamorel_args, custom_updater_class=TestUpdater,
                       custom_module_functions={'value': ValueModuleFn(config_args.lamorel_args.llm_args.model_type)})

    train_dataset = {
        "x": [
             "This sentence contains test",
             "This is just test",
             "Yet another test",
             "A test this sentence is",
             "This sentence is random",
             "This is just nothing",
             "Yet another trial",
             "Don't focus on examples"
        ],
        "y": [
            [1],
            [1],
            [1],
            [1],
            [0],
            [0],
            [0],
            [0],
        ]
    }

    test_dataset = {
        "x": [
             "What about this test",
             "Or a test like this",
             "And this",
             "Here is a random sentence"
        ],
        "y": [
            1,
            1,
            0,
            0,
        ]
    }

    print("### RESULTS BEFORE TRAINING ###")
    tests = lm_server.custom_module_fns(module_function_keys=['value'],
                                        contexts=test_dataset["x"])
    print(tests)

    print("### TRAINING ###")
    for i in range(100):
        loss = lm_server.update(
            contexts=train_dataset["x"],
            candidates=None,
            labels=torch.tensor(train_dataset["y"], dtype=torch.float32),
        )
        print(f"Loss at step {i}: {torch.mean(torch.stack([l['loss'] for l in loss]))}")

    print("### RESULTS AFTER TRAINING ###")
    tests = lm_server.custom_module_fns(module_function_keys=['value'],
                                        contexts=test_dataset["x"])
    print(tests)
    lm_server.close()

if __name__ == '__main__':
    main()
