import time
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

        # encoder and visual embeddings
        self.encoder_vl = EncoderVL(args)
        # pre-encoder for language tokens
        self.encoder_lang = EncoderLang_QA(args.encoder_lang['layers'], args)

        # Simple image encoder
        self.im_encoder = SimpleEncoder()
        # feature embeddings
        self.vis_feat = FeatureFlat(
            input_shape=self.visual_tensor_shape,
            output_size=args.demb)
        # dataset id learned encoding (applied after the encoder_lang)
        self.dataset_enc = None
        """if args.enc['dataset']:
            self.dataset_enc = DatasetLearnedEncoding(args.demb, args.data['train'])"""
        # embeddings for actions
        self.emb_action = nn.Embedding(numb_action, args.demb)
        # dropouts
        self.dropout_action = nn.Dropout2d(args.dropout['transformer']['action'])

        # decoder parts
        encoder_output_size = args.demb
        self.dec_action = nn.Linear(
            encoder_output_size, args.demb)
        self.dec_QA = QAClassifier(encoder_output_size, args['vocab_path'])

        # skip connection for object predictions
        self.object_feat = FeatureFlat(
            input_shape=self.visual_tensor_shape,
            output_size=args.demb)

        # resize encoded language
        """with open(args['vocab_path'], 'rb') as filehandle:
            # read the data as binary data stream
            vocab_list = pkl.load(filehandle)['answer']
        self.num_classes = len(vocab_list)
        self.emb_enc_lang = nn.Linear(args.demb, self.num_classes)"""

        # final decoder
        """ self.classifier_lang = nn.Linear(self.num_classes*2, self.num_classes)"""

        """# progress monitoring heads
        if self.args.progress_aux_loss_wt > 0:
            self.dec_progress = nn.Linear(encoder_output_size, 1)
        if self.args.subgoal_aux_loss_wt > 0:
            self.dec_subgoal = nn.Linear(encoder_output_size, 1)
        """
        # final touch
        self.init_weights()
        self.reset()

    def forward(self, vocab, **inputs):
        '''
        forward the model for multiple time-steps (used for training)
        '''
        # embed language
        output = {}
        emb_lang, lengths_lang = self.embed_lang(inputs['questions'], vocab)
        emb_lang = self.dataset_enc(emb_lang, vocab) if self.dataset_enc else emb_lang
        # embed frames and actions
        frames_encoded = self.im_encoder(inputs['frames'].view(-1, *inputs['frames'].shape[-3:]).float())
        frames_encoded = frames_encoded.view(*inputs['frames'].shape[:2], *frames_encoded.shape[-3:])
        emb_frames, emb_object = self.embed_frames(frames_encoded)
        lengths_frames = inputs['length_frames']
        emb_actions = self.embed_actions(inputs['actions'])
        t1 = time.time()
        assert emb_frames.shape == emb_actions.shape
        lengths_actions = lengths_frames.clone()
        length_frames_max = inputs['length_frames_max']
        # concatenate language, frames and actions and add encodings
        encoder_out, _ = self.encoder_vl(
            emb_lang, emb_frames, emb_actions, lengths_lang,
            lengths_frames, lengths_actions, length_frames_max)
        # use outputs corresponding to visual frames for prediction only
        encoder_out_visual = encoder_out[
            :, lengths_lang.max().item():
            lengths_lang.max().item() + length_frames_max]
        # print(encoder_out_visual.shape)
        # get the output actions
        decoder_input = encoder_out_visual.reshape(-1, self.args.demb)
        # print(decoder_input.shape)
        # get the output objects
        emb_object_flat = emb_object.view(-1, self.args.demb)
        decoder_input = decoder_input + emb_object_flat
        answer_flat = self.dec_QA(decoder_input)
        answers = answer_flat.view(
            *encoder_out_visual.shape[:2], *answer_flat.shape[1:])
        # answers = torch.sum(answers, dim=1) bad idea
        answers = answers[:, -1, :]
        # decoder_input = torch.cat([answers, encoder_out_language_emb_question], dim=1)
        # answers = self.classifier_lang(decoder_input)

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

    def embed_frames(self, frames_pad):
        '''
        take a list of frames tensors, pad it, apply dropout and extract embeddings
        '''
        self.dropout_vis(frames_pad)
        frames_4d = frames_pad.view(-1, *frames_pad.shape[2:])
        frames_pad_emb = self.vis_feat(frames_4d).view(
            *frames_pad.shape[:2], -1)
        frames_pad_emb_skip = self.object_feat(
            frames_4d).view(*frames_pad.shape[:2], -1)
        return frames_pad_emb, frames_pad_emb_skip

    def embed_actions(self, actions):
        '''
        embed previous actions
        '''
        emb_actions = self.emb_action(actions)
        emb_actions = self.dropout_action(emb_actions)
        return emb_actions

    def reset(self):
        '''
        reset internal states (used for real-time execution during eval)
        '''
        self.frames_traj = torch.zeros(1, 0, *self.visual_tensor_shape)
        self.action_traj = torch.zeros(1, 0).long()

    def step(self, input_dict, vocab, prev_action=None):
        '''
        forward the model for a single time-step (used for real-time execution during eval)
        '''
        frames = input_dict['frames']
        device = frames.device
        if prev_action is not None:
            prev_action_int = vocab['action_low'].word2index(prev_action)
            prev_action_tensor = torch.tensor(prev_action_int)[None, None].to(device)
            self.action_traj = torch.cat(
                (self.action_traj.to(device), prev_action_tensor), dim=1)
        self.frames_traj = torch.cat(
            (self.frames_traj.to(device), frames[None]), dim=1)
        # at timestep t we have t-1 prev actions so we should pad them
        action_traj_pad = torch.cat(
            (self.action_traj.to(device),
             torch.zeros((1, 1)).to(device).long()), dim=1)
        model_out = self.forward(
            vocab=vocab['word'],
            lang=input_dict['lang'],
            lengths_lang=input_dict['lengths_lang'],
            length_lang_max=input_dict['length_lang_max'],
            frames=self.frames_traj.clone(),
            lengths_frames=torch.tensor([self.frames_traj.size(1)]),
            length_frames_max=self.frames_traj.size(1),
            action=action_traj_pad)
        step_out = {}
        for key, value in model_out.items():
            # return only the last actions, ignore the rest
            step_out[key] = value[:, -1:]
        return step_out

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
        self.dec_action.bias.data.zero_()
        self.dec_action.weight.data.uniform_(-init_range, init_range)
        self.emb_action.weight.data.uniform_(-init_range, init_range)

