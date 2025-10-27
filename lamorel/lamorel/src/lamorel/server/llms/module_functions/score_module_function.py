from . import BaseModuleFunction
import torch

class ScoreModuleFunction(BaseModuleFunction):
    def __init__(self, pad_token, model_type):
        super().__init__()
        self._pad_token = pad_token
        self._model_type = model_type

    def initialize(self):
        pass

    def forward(self, forward_outputs, minibatch, tokenized_context,  **kwargs):
        if self._model_type == "causal":  # hence input should be removed from result
            logits = forward_outputs["logits"][:, len(tokenized_context["input_ids"]) - 1:-1, :]
            output_tokens = minibatch["input_ids"][:, len(tokenized_context["input_ids"]):]
        else:
            logits = forward_outputs["logits"][:, :-1, :]  # skip </s> token appended by tokenizer
            output_tokens = minibatch["decoder_input_ids"][:, 1:]  # skip pad token

        probs = logits.softmax(-1)  # compute softmax over vocabulary
        tokens_probs = \
            torch.gather(probs, 2, output_tokens[:, :, None]).squeeze(-1)  # filter with sequence tokens

        # Compute mask to assign probability 1 to padding tokens
        mask = torch.ones(tokens_probs.shape, dtype=torch.bool, device=self.device)
        for i, _output in enumerate(output_tokens):
            for j, _token in enumerate(_output):
                if _token != self._pad_token:
                    mask[i, j] = False
        masked_token_probs = tokens_probs.masked_fill(mask, 1.0)  # apply mask
        minibatch_probs = masked_token_probs.prod(-1)  # compute final sequences' probability

        return minibatch_probs.cpu()