from abc import abstractmethod, abstractproperty
import torch.nn as nn
import torch.nn.functional as F

class ACModel:
    recurrent = False

    @abstractmethod
    def __init__(self, obs_space, action_space):
        pass

    @abstractmethod
    def forward(self, obs):
        pass

class RecurrentACModel(ACModel):
    recurrent = True

    @abstractmethod
    def forward(self, obs, memory):
        pass

    @property
    @abstractmethod
    def memory_size(self):
        pass

class ETModel(nn.Module):
    def __init__(self, args, embs_ann, vocab_out, pad, seg):
        '''
        Abstract model
        '''
        nn.Module.__init__(self)
        self.args = args
        self.vocab_out = vocab_out
        self.pad, self.seg = pad, seg
        self.visual_tensor_shape = data_util.read_dataset_info(
            args.data['train'][0])['feat_shape'][1:]

        # create language and action embeddings
        self.embs_ann = nn.ModuleDict({})
        for emb_name, emb_size in embs_ann.items():
            self.embs_ann[emb_name] = nn.Embedding(emb_size, args.demb)

        # dropouts
        self.dropout_vis = nn.Dropout(args.dropout['vis'], inplace=True)
        self.dropout_lang = nn.Dropout2d(args.dropout['lang'])

    def init_weights(self, init_range=0.1):
        '''
        init linear layers in embeddings
        '''
        for emb_ann in self.embs_ann.values():
            emb_ann.weight.data.uniform_(-init_range, init_range)


    def forward(self, vocab, **inputs):
        '''
        forward the model for multiple time-steps (used for training)
        '''
        raise NotImplementedError()