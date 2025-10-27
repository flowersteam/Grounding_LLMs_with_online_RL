from enum import Enum

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig

class ModelTypesEnum(Enum):
    causal = AutoModelForCausalLM
    seq2seq = AutoModelForSeq2SeqLM


def load_hf_model_and_tokenizer(type, path, pretrained):
    print("Loading model {}".format(path))
    tokenizer = AutoTokenizer.from_pretrained(path)

    # Select class according to type
    model_class = ModelTypesEnum[type].value
    if pretrained:
        model = model_class.from_pretrained(path)
    else:
        config = AutoConfig.from_pretrained(path)
        model = model_class.from_config(config)

    if ModelTypesEnum[type] == ModelTypesEnum.causal:
        n_layers = model.config.n_layer
    elif ModelTypesEnum[type] == ModelTypesEnum.seq2seq:
        n_layers = len(model.encoder.block)
    else:
        raise NotImplementedError()

    return tokenizer, model, n_layers
