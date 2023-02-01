#!/usr/bin/env python3

"""
Pre-training code for the subtask prediction model (relevance classifier).
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

from subtask_prediction import SubtaskPrediction
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
parser.add_argument("--low-level-demos", default=None,
                    help="low-level demos filename")
parser.add_argument("--ll-episodes", type=int, default=0,
                    help="number of low-level episodes of demonstrations to use"
                        "(default: 0, meaning all demos)")
parser.add_argument("--save-interval", type=int, default=1,
                    help="number of epochs between two saves (default: 1, 0 means no saving)")
parser.add_argument("--denoise", action="store_true",
                    help="whether or not to denoise the data")
parser.add_argument("--denoise-k", type=int, default=1,
                    help="how many examples of each instruction to use")
parser.add_argument("--denoise-total", type=int, default=100,
                    help="total number of instructions in the denoised dataset")
parser.add_argument("--augment", action="store_true",
                    help="whether or not to augment the data")
parser.add_argument("--augment-total", type=int, default=100,
                    help="total number of instructions in the augmented dataset")
parser.add_argument("--wait-finetune", type=int, default=50,
                    help="how long to wait to fine-tune")
parser.add_argument("--ones", action="store_true", default=False,
                    help="whether to ignore labels")

def main(args):

    args.model = args.model or SubtaskPrediction.default_model_name(args)
    utils.configure_logging(args.model)
    logger = logging.getLogger(__name__)

    subtask_prediction = SubtaskPrediction(args)

    header = (["update", "frames", "fps", "duration", "train_loss", "train_accuracy", "train_precision", "train_recall"]
              + ["validation_loss", "validation_accuracy"]
              + ["ground_truth_validation_accuracy", "ground_truth_validation_precision", "ground_truth_validation_recall"])

    writer = None
    if args.wb:
        import wandb
        wandb.init(project="ella")
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
    logger.info(subtask_prediction.model)

    subtask_prediction.train(subtask_prediction.train_demos, writer, csv_writer, status_path, header)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)