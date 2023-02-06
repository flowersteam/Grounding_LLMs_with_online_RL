import gc
import torch
from torch import nn
from torch.nn import functional as F

from babyai import base
from nn.enc_lang_QA import EncoderLang_QA
from nn.enc_visual import FeatureFlat, SimpleEncoder
from nn.enc_vl import EncoderVL
# from alfred.nn.encodings import DatasetLearnedEncoding
from nn.dec_QA import QAClassifier

class Model(base.Model):
    def __init__(self, args, emb_ann_size, numb_action, pad):
        '''
        transformer agent
        '''
        super().__init__(args, emb_ann_size, numb_action, pad)

        # pre-encoder for language tokens
        self.encoder_lang = EncoderLang_QA(args.encoder_lang['layers'], args)

        # dataset id learned encoding (applied after the encoder_lang)
        self.dataset_enc = None

        # decoder parts
        encoder_output_size = args.demb
        self.dec_QA = QAClassifier(encoder_output_size, args['vocab_path'])

        # final touch
        self.init_weights()
        self.reset()

    def forward(self, vocab, **inputs):
        '''
        forward the model for multiple time-steps (used for training)
        '''
        # embed language
        indexes = torch.squeeze((inputs['questions'] == 1).nonzero(as_tuple=False)[:, 1:], dim=1)
        indexes_3d = torch.unsqueeze(torch.unsqueeze(indexes, dim=1), dim=1)
        output = {}
        emb_lang, lengths_lang = self.embed_lang(inputs['questions'], vocab)
        emb_lang = self.dataset_enc(emb_lang, vocab) if self.dataset_enc else emb_lang

        decoder_input = emb_lang.reshape(-1, self.args.demb)
        answer_flat = self.dec_QA(decoder_input)  # B*language_seq x voc_size
        answers = answer_flat.view(
            *emb_lang.shape[:2], *answer_flat.shape[1:])  # B x language_seq x voc_size

        indices = torch.mul(indexes_3d, torch.ones((answers.shape[0], 1, answers.shape[2]), device=torch.device("cuda"))).type(torch.LongTensor).cuda()  # B x 1 x voc_size
        answers = torch.gather(answers, 1, indices)  # B x 1 x voc_size
        answers = answers.reshape(-1, answers.shape[2])  # B x voc_size

        output.update({'answers': answers})
        return output

    def embed_lang(self, lang_pad, vocab):
        '''
        take a list of annotation tokens and extract embeddings with EncoderLang
        '''
        assert lang_pad.max().item() < len(vocab)
        embedder_lang = self.emb_ann
        emb_lang, lengths_lang = self.encoder_lang(
            lang_pad, embedder_lang, vocab, self.pad)
        if self.args.detach_lang_emb:
            emb_lang = emb_lang.clone().detach()
        return emb_lang, lengths_lang


    def reset(self):
        '''
        reset internal states (used for real-time execution during eval)
        '''
        self.frames_traj = torch.zeros(1, 0, *self.visual_tensor_shape)
        self.action_traj = torch.zeros(1, 0).long()



    def compute_batch_loss(self, model_out, gt_dict):
        '''
        loss function for Seq2Seq agent
        '''
        losses = dict()

        # answer classes loss
        answer_pred = model_out['answers'].view(-1, model_out['answers'].shape[-1])
        answer_gt = gt_dict['answers'].view(-1)
        answer_loss = F.cross_entropy(answer_pred, answer_gt, reduction='mean')
        losses['answers'] = answer_loss

        # prediction of <<no answer>> loss
        no_answer_pred = model_out['no_answers'].view(-1, model_out['no_answers'].shape[-1])
        no_answer_gt = gt_dict['no_answers'].view(-1)
        no_answer_loss = F.cross_entropy(no_answer_pred, no_answer_gt, reduction='mean')
        losses['no_answers'] = no_answer_loss

        return losses


    def init_weights(self, init_range=0.1):
        '''
        init embeddings uniformly
        '''
        super().init_weights(init_range)

