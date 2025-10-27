import os
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from nn.encodings import PosLangEncoding, InstrLangEncoding


class EncoderLang(nn.Module):
    def __init__(self, num_layers, args,
                 subgoal_token='<<instr>>', goal_token='<<goal>>'):
        '''
        transformer encoder for language inputs
        '''
        super(EncoderLang, self).__init__()
        self.subgoal_token = subgoal_token
        self.goal_token = goal_token

        # transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            args.demb, args.encoder_heads, args.demb,
            args.dropout['transformer']['encoder'])
        if args.encoder_lang['shared']:
            enc_transformer = nn.TransformerEncoder(
                encoder_layer, num_layers)
            self.enc_transformers = enc_transformer
        else:
            self.enc_transformers = nn.TransformerEncoder(
                    encoder_layer, num_layers)

        # encodings
        self.enc_pos = PosLangEncoding(args.demb) if args.encoder_lang['pos_enc'] else None
        self.enc_instr = InstrLangEncoding(args.demb) if args.encoder_lang['instr_enc'] else None
        self.enc_layernorm = nn.LayerNorm(args.demb)
        self.enc_dropout = nn.Dropout(args.dropout['lang'], inplace=True)

    def forward(self, lang_pad, embedder, vocab, pad):
        '''
        pass embedded inputs through embeddings and encode them using a transformer
        '''
        # pad the input language sequences and embed them with a linear layer
        mask_pad = (lang_pad == pad)
        emb_lang = embedder(lang_pad)
        # add positional encodings
        mask_token = EncoderLang.mask_token(
            lang_pad, vocab, {self.subgoal_token, self.goal_token})
        emb_lang = self.encode_inputs(emb_lang, mask_token, mask_pad)
        # pass the inputs through the encoder
        hiddens = EncoderLang.encoder(
            self.enc_transformers, emb_lang, mask_pad, vocab)
        lengths = (lang_pad != pad).sum(dim=1)
        return hiddens, lengths

    @staticmethod
    def mask_token(lang_pad, vocab, tokens):
        '''
        returns mask of the tokens
        '''
        tokens_mask = torch.zeros_like(lang_pad).long()
        for token in tokens:
            tokens_mask += lang_pad == vocab.word2index(token)
        return tokens_mask.bool()

    @staticmethod
    def encoder(encoders, emb_lang, mask_pad, mask_attn=None):
        '''
        compute encodings for all tokens using a normal flat encoder
        '''
        # skip mask: mask padded words
        if mask_attn is None:
            # attention mask: all tokens can attend to all others
            mask_attn = torch.zeros(
                (mask_pad.shape[1], mask_pad.shape[1]), device=mask_pad.device).float()
        # encode the inputs
        output = encoders(
            emb_lang.transpose(0, 1),
            mask_attn,
            mask_pad).transpose(0, 1)
        return output

    def encode_inputs(self, emb_lang, mask_token, mask_pad):
        '''
        add positional encodings, apply layernorm and dropout
        '''
        emb_lang = self.enc_pos(emb_lang) if self.enc_pos else emb_lang
        emb_lang = self.enc_instr(emb_lang, mask_token) if self.enc_instr else emb_lang
        emb_lang = self.enc_dropout(emb_lang)
        emb_lang = self.enc_layernorm(emb_lang)
        return emb_lang
