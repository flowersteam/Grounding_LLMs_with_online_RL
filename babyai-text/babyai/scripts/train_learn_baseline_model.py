#!/usr/bin/env python3

"""
Training code for the LEARN model (Goyal et al., 2019)
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
import wandb
from babyai.arguments import ArgumentParser
import babyai.utils as utils

from learn_baseline import LEARNBaseline
from babyai.arguments import ArgumentParser
import babyai.utils as utils


parser = ArgumentParser()

parser.add_argument("--demos", default=None,
                    help="demos filename (REQUIRED or demos-origin or multi-demos required)")
parser.add_argument("--demos-origin", required=False,
                    help="origin of the demonstrations: human | agent (REQUIRED or demos or multi-demos required)")
parser.add_argument("--episodes", type=int, default=0,
                    help="number of high-level episodes of demonstrations to use"
                        "(default: 0, meaning all demos)")
parser.add_argument("--save-interval", type=int, default=1,
                    help="number of epochs between two saves (default: 1, 0 means no saving)")


def main(args):

    args.model = args.model or LEARNBaseline.default_model_name(args)
    utils.configure_logging(args.model)
    logger = logging.getLogger(__name__)

    learn_baseline = LEARNBaseline(args)

    header = (["update", "frames", "fps", "duration", "train_loss", "train_accuracy", "train_precision", "train_recall"] 
        + ["validation_loss", "validation_accuracy", "validation_precision", "validation_recall"])

    writer = None
    if args.wb:
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
    logger.info(learn_baseline.model)

    learn_baseline.train(learn_baseline.train_demos, writer, csv_writer, status_path, header)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)