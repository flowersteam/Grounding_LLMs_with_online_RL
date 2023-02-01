#!/usr/bin/env python3

"""
Script to train agent through imitation learning using demonstrations.
"""

import os
import csv
import copy
import gym
import time
import datetime
import numpy as np
import sys
import logging
import torch
import pickle as pkl

import babyai.utils as utils
from babyai.arguments import ArgumentParser
from babyai.trainer_l_class import TrainerClass

from attrdict import AttrDict

# Parse arguments
parser = ArgumentParser()
parser.add_argument("--demos", default=None,
                    help="demos filename (REQUIRED or demos-origin or multi-demos required)")
parser.add_argument("--demos-origin", required=False,
                    help="origin of the demonstrations: human | agent (REQUIRED or demos or multi-demos required)")
parser.add_argument("--episodes", type=int, default=0,
                    help="number of episodes of demonstrations to use"
                         "(default: 0, meaning all demos)")
parser.add_argument("--multi-env", nargs='*', default=None,
                    help="name of the environments used for validation/model loading")
parser.add_argument("--multi-demos", nargs='*', default=None,
                    help="demos filenames for envs to train on (REQUIRED when multi-env is specified)")
parser.add_argument("--multi-episodes", type=int, nargs='*', default=None,
                    help="number of episodes of demos to use from each file (REQUIRED when multi-env is specified)")
parser.add_argument("--save-interval", type=int, default=1,
                    help="number of epochs between two saves (default: 1, 0 means no saving)")

parser.add_argument("--include-done", action="store_true", default=False,
                    help="predict termination")
parser.add_argument("--full-obs", action="store_true", default=False,
                    help="use full observations of the environment")
parser.add_argument("--ego", action="store_true", default=False,
                    help="use egocentric full observations")

parser.add_argument("--extra-actions", type=int, default=0,
                    help="number of extra actions to add to the action space")
parser.add_argument("--QA", action="store_true", default=False,
                    help="train a QA predictor instead of a full policy")
parser.add_argument("--oversample", type=int, default=1,
                    help="how many times positive examples are oversampled for the done classifier")

parser.add_argument("--model-number", type=int, default=0,
                    help="number of the model that will be saved")

def main(args, attr):
    # Verify the arguments when we train on multiple environments
    # No need to check for the length of len(args.multi_env) in case, for some reason, we need to validate on other envs
    if args.multi_env is not None:
        assert len(args.multi_demos) == len(args.multi_episodes)

    args.model = args.model or TrainerClass.default_model_name(args)
    utils.configure_logging(args.model)
    logger = logging.getLogger(__name__)

    l_class = TrainerClass(args, attr)

    # Log command, availability of CUDA, and model
    logger.info(args)
    logger.info("CUDA available: {}".format(torch.cuda.is_available()))

    # il_learn.train(il_learn.train_demos, writer, csv_writer, status_path, header)
    log = l_class.train()
    print('At the last epoch train CE {} SR {} valid CE {} and the SR is {}'.format(
        log["loss_cross_entropy_train"][-1],
        log["success_pred_train"][-1],
        log["loss_cross_entropy_valid"][-1],
        log["success_pred_valid"][-1]))


if __name__ == "__main__":
    args = parser.parse_args()

    attr = AttrDict()

    # TRANSFORMER settings
    # size of transformer embeddings
    attr['demb'] = 768
    # number of heads in multi-head attention
    attr['encoder_heads'] = 12
    # number of layers in transformer encoder
    attr['encoder_layers'] = 2
    # how many previous actions to use as input
    attr['num_input_actions'] = 1
    # which encoder to use for language encoder (by default no encoder)
    attr['encoder_lang'] = {
        'shared': True,
        'layers': 2,
        'pos_enc': True,
        'instr_enc': False,
    }
    # which decoder to use for the speaker model
    attr['decoder_lang'] = {
        'layers': 2,
        'heads': 12,
        'demb': 768,
        'dropout': 0.1,
        'pos_enc': True,
    }

    attr['detach_lang_emb'] = False

    # DROPOUT
    attr['dropout'] = {
        # dropout rate for language (goal + instr)
        'lang': 0.0,
        # dropout rate for Resnet feats
        'vis': 0.3,
        # dropout rate for processed lang and visual embeddings
        'emb': 0.0,
        # transformer model specific dropouts
        'transformer': {
            # dropout for transformer encoder
            'encoder': 0.1,
            # remove previous actions
            'action': 0.0,
        },
    }

    # ENCODINGS
    attr['enc'] = {
        # use positional encoding
        'pos': True,
        # use learned positional encoding
        'pos_learn': False,
        # use learned token ([WORD] or [IMG]) encoding
        'token': False,
        # dataset id learned encoding
        'dataset': False,
    }
    attr['vocab_path'] = 'storage/demos/{}_agent_done_QG_no_answer_biased_vocab.pkl'.format(args.env)
    print(args)
    print(attr)
    main(args, attr)
