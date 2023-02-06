import transformers
from transformers.modeling_outputs import ModelOutput
from transformers import top_k_top_p_filtering

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import custom_fwd, custom_bwd

import numpy as np
import time

from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise

from tqdm.auto import tqdm
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FrozenBNBLinear(nn.Module):
    def __init__(self, weight, absmax, code, bias=None):
        assert isinstance(bias, nn.Parameter) or bias is None
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.register_buffer("weight", weight.requires_grad_(False))
        self.register_buffer("absmax", absmax.requires_grad_(False))
        self.register_buffer("code", code.requires_grad_(False))
        self.adapter = None
        self.bias = bias

    def forward(self, input):
        output = DequantizeAndLinear.apply(input, self.weight, self.absmax, self.code, self.bias)
        if self.adapter:
            output += self.adapter(input)
        return output

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "FrozenBNBLinear":
        weights_int8, state = quantize_blockise_lowmemory(linear.weight)
        return cls(weights_int8, *state, linear.bias)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"


class DequantizeAndLinear(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input: torch.Tensor, weights_quantized: torch.ByteTensor,
                absmax: torch.FloatTensor, code: torch.FloatTensor, bias: torch.FloatTensor):
        weights_deq = dequantize_blockwise(weights_quantized, absmax=absmax, code=code)
        ctx.save_for_backward(input, weights_quantized, absmax, code)
        ctx._has_bias = bias is not None
        return F.linear(input, weights_deq, bias)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        assert not ctx.needs_input_grad[1] and not ctx.needs_input_grad[2] and not ctx.needs_input_grad[3]
        input, weights_quantized, absmax, code = ctx.saved_tensors
        # grad_output: [*batch, out_features]
        weights_deq = dequantize_blockwise(weights_quantized, absmax=absmax, code=code)
        grad_input = grad_output @ weights_deq
        grad_bias = grad_output.flatten(0, -2).sum(dim=0) if ctx._has_bias else None
        return grad_input, None, None, None, grad_bias


class FrozenBNBEmbedding(nn.Module):
    def __init__(self, weight, absmax, code):
        super().__init__()
        self.num_embeddings, self.embedding_dim = weight.shape
        self.register_buffer("weight", weight.requires_grad_(False))
        self.register_buffer("absmax", absmax.requires_grad_(False))
        self.register_buffer("code", code.requires_grad_(False))
        self.adapter = None

    def forward(self, input, **kwargs):
        with torch.no_grad():
            # note: both quantuized weights and input indices are *not* differentiable
            weight_deq = dequantize_blockwise(self.weight, absmax=self.absmax, code=self.code)
            output = F.embedding(input, weight_deq, **kwargs)
        if self.adapter:
            output += self.adapter(input)
        return output

    @classmethod
    def from_embedding(cls, embedding: nn.Embedding) -> "FrozenBNBEmbedding":
        weights_int8, state = quantize_blockise_lowmemory(embedding.weight)
        return cls(weights_int8, *state)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_embeddings}, {self.embedding_dim})"


def quantize_blockise_lowmemory(matrix: torch.Tensor, chunk_size: int = 2 ** 20):
    assert chunk_size % 4096 == 0
    code = None
    chunks = []
    absmaxes = []
    flat_tensor = matrix.view(-1)
    for i in range((matrix.numel() - 1) // chunk_size + 1):
        input_chunk = flat_tensor[i * chunk_size: (i + 1) * chunk_size].clone()
        quantized_chunk, (absmax_chunk, code) = quantize_blockwise(input_chunk, code=code)
        chunks.append(quantized_chunk)
        absmaxes.append(absmax_chunk)

    matrix_i8 = torch.cat(chunks).reshape_as(matrix)
    absmax = torch.cat(absmaxes)
    return matrix_i8, (absmax, code)


def convert_to_int8(model):
    """Convert linear and embedding modules to 8-bit with optional adapters"""
    for module in list(model.modules()):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and name not in ['summary']:  # no quantization of value head
                print('name: {}, child:{}'.format(name, child))
                setattr(
                    module,
                    name,
                    FrozenBNBLinear(
                        weight=torch.zeros(child.out_features, child.in_features, dtype=torch.uint8),
                        absmax=torch.zeros((child.weight.numel() - 1) // 4096 + 1),
                        code=torch.zeros(256),
                        bias=child.bias,
                    ),
                )
            elif isinstance(child, nn.Embedding):
                setattr(
                    module,
                    name,
                    FrozenBNBEmbedding(
                        weight=torch.zeros(child.num_embeddings, child.embedding_dim, dtype=torch.uint8),
                        absmax=torch.zeros((child.weight.numel() - 1) // 4096 + 1),
                        code=torch.zeros(256),
                    )
                )


@dataclass
class CausalLMOutputWithCrossAttentions(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    value: Optional[torch.FloatTensor] = None


class ValueHead(nn.Module):
    """The ValueHead class implements a head for GPTJ that returns a scalar for each output token."""

    def __init__(self, config):
        super().__init__()

        self.detach_head = False
        self.summary_type = config.summary_type if hasattr(config, "summary_type") else "last"
        if self.summary_type == "attn":
            raise NotImplementedError

        self.summary = nn.Identity()
        if hasattr(config, "summary_use_proj") and config.summary_use_proj:
            if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = nn.Linear(config.hidden_size, num_classes)

        self.activation = nn.Identity()
        if hasattr(config, "summary_activation") and config.summary_activation == "tanh":
            self.activation = nn.Tanh()

        self.first_dropout = nn.Identity()
        if hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(config.summary_first_dropout)

        self.last_dropout = nn.Identity()
        if hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(config.summary_last_dropout)

        self.flatten = nn.Flatten()

    def forward(self, hidden_states, cls_index=None):
        if self.detach_head:
            output = hidden_states.detach()
        else:
            output = hidden_states
        # output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        # output = self.last_dropout(output)

        return output


class GPTJBlock(transformers.models.gptj.modeling_gptj.GPTJBlock):
    def __init__(self, config):
        super().__init__(config)

        convert_to_int8(self.attn)
        convert_to_int8(self.mlp)


class GPTJModel(transformers.models.gptj.modeling_gptj.GPTJModel):
    def __init__(self, config):
        super().__init__(config)
        convert_to_int8(self)


class GPTJForCausalLM(transformers.models.gptj.modeling_gptj.GPTJForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        transformers.models.gptj.modeling_gptj.GPTJBlock = GPTJBlock
        convert_to_int8(self)


class GPTJForCausalLMWithValueModel(transformers.models.gptj.modeling_gptj.GPTJForCausalLM):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"h\.\d+\.attn\.bias"]

    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        self.v_head = ValueHead(config)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        loss = None
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        value = self.v_head(hidden_states).squeeze(-1)

        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # print("shift_labels: {}".format(shift_labels))
            # print("shift logit shape:{}".format(shift_logits.shape))
            # print("shift logit transpose shape:{}".format(torch.transpose(shift_logits, 1, 2).shape))
            # print("shift labels shape:{}".format(shift_labels.shape))
            # print("shift labels view shape:{}".format(shift_labels.view(-1).shape))
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(torch.transpose(shift_logits, 1, 2), shift_labels)

            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            if loss is not None:
                outputs = {'loss': loss, 'lm_logits': lm_logits, 'transformers': transformer_outputs[1:],
                           'value': value}
                return outputs
            else:
                # outputs = (lm_logits,) + transformer_outputs[1:] + (value,)
                outputs = {'lm_logits': lm_logits, 'transformers': transformer_outputs[1:], 'value': value}
                return outputs

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            value=value,
        )


class GPTJForCausalLMWithValueModel_quantized(GPTJForCausalLMWithValueModel):
    def __init__(self, config):
        super().__init__(config)
        convert_to_int8(self)


def respond_to_batch(model, queries, txt_len=20, top_k=0, top_p=1.0):
    """Sample text from language model."""
    input_ids = queries['input_ids']
    attention_mask = queries['attention_mask']
    for i in range(txt_len):
        # Get Logits
        outputs = model(input_ids, attention_mask=attention_mask)
        next_token_logits = outputs['lm_logits'][:, -1, :]
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        # Sample
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), device=device)], dim=-1)
    return input_ids[:, -txt_len:]


def perplexity(model, inputs_ids, attention_mask, len_subgoals):
    target_ids = inputs_ids.clone()
    target_ids[:, :-len_subgoals] = -100
    with torch.no_grad():
        # t1 = time.time()
        outputs = model(inputs_ids, attention_mask=attention_mask, labels=target_ids)
        # t2 = time.time()
        # print("Forward pass duration: {}".format(t2-t1))
    neg_log_likelihood = outputs['loss']
    # print("nll: {}".format(neg_log_likelihood))
    if inputs_ids.shape[0] == 1:
        return torch.exp(neg_log_likelihood).unsqueeze(dim=0)
    else:
        return torch.exp(neg_log_likelihood)


def ranking_subgoals(model, prompt, attention_mask, subgoals_tokenized):
    perplexity_matrix = []  # final dimension: nbr_prompts x nbr_subgoals

    for s in subgoals_tokenized.keys():
        # print("prompt.shape: {}".format(prompt.shape))
        # print("attention_mask.shape: {}".format(attention_mask.shape))
        # print("sg tokenized shape: {}".format(subgoals_tokenized[s]['input_ids'].shape))
        # print("repeat_shape: {}".format(torch.repeat_interleave(subgoals_tokenized[s]['input_ids'], prompt.shape[0], dim=0).shape))
        # print("ones_shape: {}".format(torch.ones((subgoals_tokenized[s]['input_ids'].shape[0], prompt.shape[0]), device=device).shape))

        input_ids = torch.cat(
            [prompt, torch.repeat_interleave(subgoals_tokenized[s]['input_ids'], prompt.shape[0], dim=0)], dim=1)
        update_attention_mask = torch.cat(
            [attention_mask, torch.ones((prompt.shape[0], subgoals_tokenized[s]['input_ids'].shape[1]), device=device)],
            dim=-1)
        # print("input_ids.shape: {}".format(input_ids.shape))
        # print("update_attention_mask.shape: {}".format(update_attention_mask.shape))
        ppl = perplexity(model, input_ids, update_attention_mask,
                         subgoals_tokenized[s]['input_ids'].shape[1]).unsqueeze(
            dim=0)
        # print(ppl)
        perplexity_matrix.append(ppl)

    # print(torch.cat(perplexity_matrix, dim=0))
    perplexity_matrix = torch.transpose(torch.cat(perplexity_matrix, dim=0), 0,
                                        1)  # before transpose dimension are: nbr_subgoals x nbr_prompts

    return perplexity_matrix.cpu().detach().numpy()


def choosing_subgoals(model, prompt, attention_mask, subgoal_tokenized, eps):
    ppl_matrix = ranking_subgoals(model, prompt, attention_mask, subgoal_tokenized)

    subgoal_id_array = np.ones(prompt.shape[0])
    # print("The ppl matrix: {}".format(ppl_matrix))
    for i in range(prompt.shape[0]):
        if np.random.rand() > eps:
            subgoal_id_array[i] = np.argmin(ppl_matrix[i])
        else:
            inv_ppl = 1 / ppl_matrix[i]
            proba = np.exp(inv_ppl)/np.sum(np.exp(inv_ppl))
            subgoal_id_array[i] = np.random.choice(np.arange(ppl_matrix.shape[1]), p=proba)
    return subgoal_id_array


# config = transformers.GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")
# config = transformers.GPTJConfig.from_json_file('storage/models/GPTJ/config.json')

# tokenizer = transformers.AutoTokenizer.from_pretrained('storage/models/GPTJ')
# tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-j-6B')

"""tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", padding_side="left")
tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

# transformers.models.gptj.modeling_gptj.GPTJBlock = GPTJBlock
gpt = GPTJForCausalLMWithValueModel_quantized.from_pretrained("hivemind/gpt-j-6B-8bit", low_cpu_mem_usage=True)


gpt.to(device)"""

"""query_txt_1 = "My most favourite movie is"
query_txt_2 = "My most favourite movies"
query_txt_3 = "My most favourite movie is Red"
queries_txt = [query_txt_1, query_txt_2, query_txt_3]

queries = tokenizer(queries_txt, return_tensors='pt', padding=True).to(device)

print("input_ids: {}".format(queries["input_ids"]))
print("attention_mask: {}".format(queries["attention_mask"]))

print(queries)
responses = respond_to_batch(gpt, queries, txt_len=10)

for i in range(responses.shape[0]):
    response_txt = tokenizer.decode(responses[i])
    query_txt = queries_txt[i]
    print(query_txt + response_txt)"""

"""prompt = [
    "Possible action of the agent: go forward, turn right, turn left \n Goal of the agent: Go to green ball \n Past observations and actions \n Observation 1: A green key is 2 step left and 1 step in front, a grey box is 2 step left, a purple box is 1 step left and 1 step in front, a green box is 2 step in front, a blue key is 1 step right and 2 step in front, a green ball is 2 step right and 2 step in front \n Action 1: ",
    "Possible action of the agent: go forward, turn right, turn left \n Goal of the agent: Go to purple box \n Past observations and actions \n Observation 1: A green key is 2 step left and 1 step in front, a grey box is 2 step left, a purple box is 1 step left and 1 step in front, a green box is 2 step in front, a blue key is 1 step right and 2 step in front, a green ball is 2 step right and 2 step in front \n Action 1: "]

subgoals = {0: "go forward",
            1: "turn right",
            2: "turn left"}
subgoals_tokenized = {0: tokenizer(["go forward"], return_tensors='pt').to(device),
                      1: tokenizer(["turn right"], return_tensors='pt').to(device),
                      2: tokenizer(["turn left"], return_tensors='pt').to(device)}
prompt_inputs = tokenizer(prompt, return_tensors='pt', padding=True).to(device)

eps = 0
sbg = choosing_subgoals(gpt,
                        prompt=prompt_inputs['input_ids'],
                        attention_mask=prompt_inputs['attention_mask'],
                        subgoal_tokenized=subgoals_tokenized,
                        eps=eps)

for s in sbg:
    print(subgoals[s])"""
