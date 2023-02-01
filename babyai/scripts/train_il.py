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
from babyai.arguments import ArgumentParser
import babyai.utils as utils
from babyai.imitation import ImitationLearning


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
parser.add_argument("--done-classifier", action="store_true", default=False,
                    help="train a binary termination predictor instead of a full policy")
parser.add_argument("--oversample", type=int, default=1,
                    help="how many times positive examples are oversampled for the done classifier")

def main(args):
    # Verify the arguments when we train on multiple environments
    # No need to check for the length of len(args.multi_env) in case, for some reason, we need to validate on other envs
    if args.multi_env is not None:
        assert len(args.multi_demos) == len(args.multi_episodes)

    args.model = args.model or ImitationLearning.default_model_name(args)
    utils.configure_logging(args.model)
    logger = logging.getLogger(__name__)

    il_learn = ImitationLearning(args)

    # Define logger and Tensorboard writer
    header = (["update", "frames", "fps", "duration", "entropy", "train_loss", "train_accuracy"]
              + ["validation_loss", "validation_accuracy"])
    if args.multi_env is None:
        header.extend(["validation_reward", "validation_rate"])
    else:
        header.extend(["validation_reward_{}".format(env) for env in args.multi_env])
        header.extend(["validation_rate_{}".format(env) for env in args.multi_env])
    writer = None
    if args.tb:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(utils.get_log_dir(args.model))
    if args.wb:
        import wandb
        wandb.init(project="ella", name=args.model)
        wandb.config.update(args)
        writer = wandb

    # Define csv writer
    csv_writer = None
    csv_path = os.path.join(utils.get_log_dir(args.model), 'log.csv')
    first_created = not os.path.exists(csv_path)
    # we don't buffer data going in the csv log, cause we assume
    # that one update will take much longer that one write to the log
    csv_writer = csv.writer(open(csv_path, 'a', 1))
    if first_created:
        csv_writer.writerow(header)

    # Get the status path
    status_path = os.path.join(utils.get_log_dir(args.model), 'status.json')

    # Log command, availability of CUDA, and model
    logger.info(args)
    logger.info("CUDA available: {}".format(torch.cuda.is_available()))
    logger.info(il_learn.acmodel)

    il_learn.train(il_learn.train_demos, writer, csv_writer, status_path, header)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
