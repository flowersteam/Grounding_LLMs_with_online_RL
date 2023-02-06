import torch
from torch import nn
import nn.model_util as model_util
from nn.encodings import PosEncoding, PosLearnedEncoding, TokenLearnedEncoding


class EncoderVL(nn.Module):
    def __init__(self, args):
        '''
        transformer encoder for language, frames and action inputs
        '''
        super(EncoderVL, self).__init__()

        # transofmer layers
        encoder_layer = nn.TransformerEncoderLayer(
            args.demb, args.encoder_heads, args.demb,
            args.dropout['transformer']['encoder'])
        self.enc_transformer = nn.TransformerEncoder(
            encoder_layer, args.encoder_layers)

        # how many last actions to attend to
        self.num_input_actions = args.num_input_actions

        # encodings
        self.enc_pos = PosEncoding(args.demb) if args.enc['pos'] else None
        self.enc_pos_learn = PosLearnedEncoding(args.demb) if args.enc['pos_learn'] else None
        self.enc_token = TokenLearnedEncoding(args.demb) if args.enc['token'] else None
        self.enc_layernorm = nn.LayerNorm(args.demb)
        self.enc_dropout = nn.Dropout(args.dropout['emb'], inplace=True)

    def forward(self,
                emb_lang,
                emb_frames,
                emb_actions,
                lengths_lang,
                lengths_frames,
                lengths_actions,
                length_frames_max,
                attn_masks=True):
        '''
        pass embedded inputs through embeddings and encode them using a transformer
        '''
        # emb_lang is processed on each GPU separately so they size can vary
        length_lang_max = lengths_lang.max().item()
        emb_lang = emb_lang[:, :length_lang_max]
        # create a mask for padded elements
        length_mask_pad = length_lang_max + length_frames_max * (
            2 if lengths_actions.max() > 0 else 1)
        mask_pad = torch.zeros(
            (len(emb_lang), length_mask_pad), device=emb_lang.device).bool()
        for i, (len_l, len_f, len_a) in enumerate(
                zip(lengths_lang, lengths_frames, lengths_actions)):
            # mask padded words
            mask_pad[i, len_l: length_lang_max] = True
            # mask padded frames
            mask_pad[i, length_lang_max + len_f:
                      length_lang_max + length_frames_max] = True
            # mask padded actions
            mask_pad[i, length_lang_max + length_frames_max + len_a:] = True

        # encode the inputs
        emb_all = self.encode_inputs(
            emb_lang, emb_frames, emb_actions, lengths_lang, lengths_frames, mask_pad)

        # create a mask for attention (prediction at t should not see frames at >= t+1)
        if attn_masks:
            # assert length_frames_max == max(lengths_actions)
            mask_attn = model_util.generate_attention_mask(
                length_lang_max, length_frames_max,
                emb_all.device, self.num_input_actions)
        else:
            # allow every token to attend to all others
            mask_attn = torch.zeros(
                (mask_pad.shape[1], mask_pad.shape[1]),
                device=mask_pad.device).float()

        # encode the inputs
        output = self.enc_transformer(
            emb_all.transpose(0, 1), mask_attn, mask_pad).transpose(0, 1)
        return output, mask_pad

    def encode_inputs(self, emb_lang, emb_frames, emb_actions,
                      lengths_lang, lengths_frames, mask_pad):
        '''
        add encodings (positional, token and so on)
        '''
        if self.enc_pos is not None:
            emb_lang, emb_frames, emb_actions = self.enc_pos(
                emb_lang, emb_frames, emb_actions, lengths_lang, lengths_frames)
        if self.enc_pos_learn is not None:
            emb_lang, emb_frames, emb_actions = self.enc_pos_learn(
                emb_lang, emb_frames, emb_actions, lengths_lang, lengths_frames)
        if self.enc_token is not None:
            emb_lang, emb_frames, emb_actions = self.enc_token(
                emb_lang, emb_frames, emb_actions)
        emb_cat = torch.cat((emb_lang, emb_frames, emb_actions), dim=1)
        emb_cat = self.enc_layernorm(emb_cat)
        emb_cat = self.enc_dropout(emb_cat)
        return emb_cat
