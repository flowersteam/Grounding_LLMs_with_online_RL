import os
import torch
import math
from torch import nn


class PosEncoding(nn.Module):
    '''
    Transformer-style positional encoding with wavelets
    '''
    def __init__(self, d_model, max_len=1250):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe[None])

    def forward(self, lang, frames, actions, lens_lang, lens_frames, pos=None):
        if pos is None:
            enc = self.pe[:, :lang.shape[1] + frames.shape[1]]
        else:
            enc = [[] for _ in range(len(lang))]
            for batch_idx in range(pos.shape[0]):
                for pos_idx in range(lang.shape[1] + frames.shape[1]):
                    enc[batch_idx].append(self.pe[0, pos[batch_idx, pos_idx]])
            enc = torch.stack([torch.stack(pos_batch) for pos_batch in enc])
        enc = enc / math.sqrt(self.d_model)
        lang = lang + enc[:, :lang.shape[1]]
        for i in range(frames.shape[0]):
            frames[i] = frames[i] + enc[0, lens_lang[i]: lens_lang[i] + frames.shape[1]]
        # use the same position indices for actions as for the frames
        for i in range(actions.shape[0]):
            actions[i] = actions[i] + enc[0, lens_lang[i]: lens_lang[i] + actions.shape[1]]
        return lang, frames, actions


class LearnedEncoding(nn.Module):
    '''
    Learned additive encoding implemented on top of nn.Embedding
    '''
    def __init__(self, d_model, vocab_size, init_range=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.emb.weight.data.uniform_(-init_range, init_range)

    def forward(self, x, tokens):
        tokens_emb = self.emb(tokens)
        return x + tokens_emb


class PosLearnedEncoding(nn.Module):
    '''
    Learned additive positional encoding implemented on top of nn.Embedding
    '''
    def __init__(self, d_model, max_pos=1250, init_range=0.1):
        super().__init__()
        self.emb = nn.Embedding(max_pos, d_model)
        self.emb.weight.data.uniform_(-init_range, init_range)

    def forward(self, lang, frames, actions, lens_lang, lens_frames):
        pos_lang = torch.stack([torch.arange(0, lang.shape[1])] * lang.shape[0])
        pos_frames = torch.stack([torch.arange(0, frames.shape[1]) + l for l in lens_lang])
        # use the same position indices for actions as for the frames
        pos_actions = torch.stack([torch.arange(0, actions.shape[1]) + l for l in lens_lang])
        lang += self.emb(pos_lang.to(lang.device))
        frames += self.emb(pos_frames.to(frames.device))
        actions += self.emb(pos_actions.to(actions.device))
        return lang, frames, actions


class TokenLearnedEncoding(nn.Module):
    '''
    Learned additive img/word/action token encoding implemented on top of nn.Embedding
    '''
    def __init__(self, d_model, vocab_size=3, init_range=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.emb.weight.data.uniform_(-init_range, init_range)

    def forward(self, lang, frames, actions):
        token_lang = torch.ones(lang.shape[:2], device=lang.device, dtype=torch.long) * 0
        token_lang_emb = self.emb(token_lang)
        lang += token_lang_emb
        token_frames = torch.ones(frames.shape[:2], device=frames.device, dtype=torch.long) * 1
        token_frames_emb = self.emb(token_frames)
        frames += token_frames_emb
        token_actions = torch.ones(actions.shape[:2], device=actions.device, dtype=torch.long) * 2
        token_actions_emb = self.emb(token_actions)
        actions += token_actions_emb
        return lang, frames, actions

class PosLangEncoding(nn.Module):
    '''
    Transformer-style positional encoding with wavelets
    '''
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe[None])

    def forward(self, x, pos=None):
        if pos is None:
            enc = self.pe[:, :x.shape[1]]
        else:
            enc = [[] for _ in range(len(x))]
            for batch_idx in range(pos.shape[0]):
                for pos_idx in range(pos.shape[1]):
                    enc[batch_idx].append(self.pe[0, pos[batch_idx, pos_idx]])
            enc = torch.stack([torch.stack(pos_batch) for pos_batch in enc])
        x = x + enc / math.sqrt(self.d_model)
        return x

class InstrLangEncoding(PosLangEncoding):
    '''
    Relative position in an instruction (a sentence) encoding with wavelets
    '''
    def forward(self, x, tokens_mask):
        counts = torch.zeros_like(tokens_mask)[:, 0].long()
        instrs = torch.zeros_like(tokens_mask).long()
        # offset the tokens by 1
        tokens_mask[:, 1:] = tokens_mask.clone()[:, :-1]
        for i in range(tokens_mask.shape[1] - 1):
            instrs[:, i] = counts
            counts += (tokens_mask[:, i + 1] == True)
        instrs[:, -1] = instrs[:, -2]
        pe_tokens = self.pe[0, instrs]
        x = x + pe_tokens / math.sqrt(self.d_model)
        return x


class DatasetLearnedEncoding(nn.Module):
    '''
    Learned additive dataset id encoding implemented on top of nn.Embedding
    '''
    def __init__(self, d_model, datasets, init_range=0.1):
        super().__init__()
        self.datasets = {dataset: i for i, dataset in enumerate(datasets)}
        self.emb = nn.Embedding(len(datasets), d_model)
        self.emb.weight.data.uniform_(-init_range, init_range)

    def forward(self, lang, vocab):
        dataset_ids = torch.ones(lang.shape[0], device=lang.device, dtype=torch.long)
        dataset_emb = self.emb(dataset_ids * self.datasets[vocab.name])
        lang_enc = lang + dataset_emb[:, None]
        return lang_enc
