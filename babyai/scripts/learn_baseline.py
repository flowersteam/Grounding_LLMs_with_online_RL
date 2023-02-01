"""
Training interface for the LEARN model (Goyal et al., 2019)
"""

import copy
import gym
import time
import datetime
import numpy as np
import sys
import itertools
import torch
import torch.nn as nn
import babyai.utils as utils
import os
import json
import logging
import wandb
from tqdm import tqdm
from gym import spaces
from learn_baseline_model import LEARNBaselineModel
from instruction_handler import InstructionHandler
from gym_minigrid.minigrid import MiniGridEnv

logger = logging.getLogger(__name__)


class LEARNBaseline(object):

    def __init__(self, args, online_args=False):
        
        self.args = args

        utils.seed(self.args.seed)

        self.model = LEARNBaselineModel({"instr":100, "num_actions":len(MiniGridEnv.Actions)}, dropout=args.dropout)

        if getattr(self.args, 'pretrained_model', None):
            logger.info("loading pretrained model")
            self.model = utils.load_model(args.pretrained_model, raise_not_found=True)

        demos_path = utils.get_demos_path(args.demos, args.env, args.demos_origin, valid=False)
        demos_path_valid = utils.get_demos_path(args.demos, args.env, args.demos_origin, valid=True)
        demos = utils.load_demos(demos_path)
        if args.episodes:
            if args.episodes > len(demos):
                raise ValueError("there are only {} demos".format(len(demos)))
            demos = demos[:args.episodes]
        
        self.train_demos = utils.demos.transform_demos_learn(demos)

        logger.info('loading instruction handler')
        self.instr_handler = InstructionHandler(demos, load_bert=False, save_path=os.path.join(os.path.splitext(demos_path)[0], "ih"))
        logger.info('loaded instruction handler')

        val_demos = utils.load_demos(demos_path_valid)
        if args.val_episodes > len(val_demos):
            logger.info('Using all the available {} demos to evaluate valid. accuracy'.format(len(val_demos)))
        val_demos = val_demos[:self.args.val_episodes]

        self.val_demos = utils.demos.transform_demos_learn(val_demos)

        self.instr_preprocessor = utils.InstructionOnlyPreprocessor(args.model, getattr(self.args, 'pretrained_model', None))
        self.instr_preprocessor.vocab.save()

        self.model.train()
        if torch.cuda.is_available():
            self.model.cuda()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.args.lr, eps=self.args.optim_eps)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.95)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def default_model_name(args):
        named_envs = args.env
        # Define model name
        suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        instr = args.instr_arch if args.instr_arch else "noinstr"
        model_name_parts = {
            'envs': named_envs,
            'arch': args.arch,
            'instr': instr,
            'seed': args.seed,
            'suffix': suffix}
        default_model_name = "{envs}_{arch}_{instr}_seed{seed}_{suffix}".format(**model_name_parts)
        if getattr(args, 'pretrained_model', None):
            default_model_name = args.pretrained_model + '_pretrained_' + default_model_name    
        return default_model_name
    
    def run_epoch_recurrence(self, demos, batch_size, is_training=False):
        indices = list(range(len(demos)))
        if is_training:
            np.random.shuffle(indices)
        
        batch_size = min(self.args.batch_size, len(demos))
        offset = 0

        if not is_training:
            self.model.eval()
        
        log = {"loss": [], "accuracy": [], "correct_true": 0, "predict_true": 0, "target_true": 0, "precision": 0, "recall": 0, "frames": 0}

        start_time = time.time()
        frames = 0

        for batch_index in range(len(indices) // batch_size):
            batch = [demos[i] for i in indices[offset : offset + batch_size]]
            frames += len(batch)

            _log = self.run_epoch_recurrence_one_batch(batch, is_training=is_training)

            log["loss"].append(_log["loss"])
            log["accuracy"].append(_log["accuracy"])
            log["frames"] = frames

            log["correct_true"] += _log["correct_true"]
            log["target_true"] += _log["target_true"]
            log["predict_true"] += _log["predict_true"]

            offset += batch_size
        
        if log["predict_true"] > 0:
            log["precision"] = log["correct_true"] / log["predict_true"]
        if log["target_true"] > 0:
            log["recall"] = log["correct_true"] / log["target_true"]
        
        if not is_training:
            self.model.train()
        
        return log

    def run_epoch_recurrence_one_batch(self, batch, is_training=False):
        missions, action_freqs, labels = list(zip(*batch))
        
        missions = self.instr_preprocessor(missions, device=self.device)
        action_freqs = torch.tensor(action_freqs).to(self.device).float()
        labels = torch.tensor(labels).to(self.device).long()
        
        predictions, logits = self.model(missions, action_freqs)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        accuracy = float((predictions == labels).sum()) / len(labels)

        if is_training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        log = {}
        log["loss"] = float(loss)
        log["accuracy"] = float(accuracy)

        log["correct_true"] = int(((predictions == labels).int() & labels.int()).sum())
        log["predict_true"] = int(predictions.sum())
        log["target_true"] = int(labels.sum())

        return log

    def train(self, train_demos, writer, csv_writer, status_path, header, reset_status=False):
        
        # Load the status
        def initial_status():
            return {'i': 0,
                    'num_frames': 0}

        status = initial_status()
        if os.path.exists(status_path) and not reset_status:
            with open(status_path, 'r') as src:
                status = json.load(src)
        elif not os.path.exists(os.path.dirname(status_path)):
            # Ensure that the status directory exists
            os.makedirs(os.path.dirname(status_path))
        
        # If the batch size is larger than the number of demos, we need to lower the batch size
        if self.args.batch_size > len(train_demos):
            self.args.batch_size = len(train_demos)
            logger.info("Batch size too high. Setting it to the number of train demos ({})".format(len(train_demos)))

        # Model saved initially to avoid "Model not found Exception" during first validation step
        utils.save_model(self.model, self.args.model, writer)

        # best mean return to keep track of performance on validation set
        best_success_rate, i = 0, 0
        total_start_time = time.time()

        while status['i'] < getattr(self.args, 'epochs', int(1e9)):
            
            status['i'] += 1
            i = status['i']
            update_start_time = time.time()

            log = self.run_epoch_recurrence(train_demos, batch_size=self.args.batch_size, is_training=True)
            # Learning rate scheduler
            self.scheduler.step()

            status['num_frames'] += log['frames']

            update_end_time = time.time()

            # Print logs
            if status['i'] % self.args.log_interval == 0:
                total_ellapsed_time = int(time.time() - total_start_time)
                fps = log['frames'] / (update_end_time - update_start_time)
                duration = datetime.timedelta(seconds=total_ellapsed_time)

                for key in log:
                    log[key] = np.mean(log[key])
                
                train_data = [status['i'], status['num_frames'], fps, total_ellapsed_time,
                              log["loss"], log["accuracy"], log["precision"], log["recall"]]
                
                logger.info("U {} | M {:06} | MPS {:04.0f} | D {} | Loss {:.3f} | Accuracy {:.3f} | Precision {:.3f} | Recall {:.3f}".format(*train_data))

                # Log the gathered data only when we don't evaluate the validation metrics. It will be logged anyways
                # afterwards when status['i'] % self.args.val_interval == 0
                if status['i'] % self.args.val_interval != 0:
                    # instantiate a validation_log with empty strings when no validation is done
                    validation_data = [''] * len([key for key in header if 'valid' in key])
                    assert len(header) == len(train_data + validation_data)
                    if self.args.tb:
                        for key, value in zip(header, train_data):
                            writer.add_scalar(key, float(value), status['num_frames'])
                    elif self.args.wb:
                        writer.log({key: float(value) for key, value in zip(header, train_data)},\
                            step=status['num_frames'])
                    csv_writer.writerow(train_data + validation_data)

                if status['i'] % self.args.val_interval == 0:
                    val_log = self.run_epoch_recurrence(self.val_demos, batch_size=self.args.batch_size, is_training=False)
                    validation_precision = np.mean(val_log["precision"])
                    validation_recall = np.mean(val_log["recall"])
                    validation_loss = np.mean(val_log["loss"])
                    validation_accuracy = np.mean(val_log["accuracy"])
                    success_rate = validation_accuracy

                    if status['i'] % self.args.log_interval == 0:
                        validation_data = [validation_loss, validation_accuracy, validation_precision, validation_recall]
                        logger.info(("Validation: Loss {: .3f} | Accuracy {: .3f} | Val Precision {: .3f} | Val Recall {: .3f}"
                                    ).format(*validation_data))
                        
                        assert len(header) == len(train_data + validation_data)
                        if self.args.wb:
                            writer.log({key: float(value) for key, value in zip(header, train_data + validation_data)},\
                                step=status['num_frames'])
                            csv_writer.writerow(train_data + validation_data)
                
                    if np.mean(success_rate) > best_success_rate:
                        best_success_rate = np.mean(success_rate)
                        with open(status_path, 'w') as dst:
                            json.dump(status, dst)
                        # Saving the model
                        logger.info("Saving best model")

                        if torch.cuda.is_available():
                            self.model.cpu()
                        utils.save_model(self.model, self.args.model + "_best", writer)
                        self.instr_preprocessor.vocab.save(utils.get_vocab_path(self.args.model + "_best"))
                        if torch.cuda.is_available():
                            self.model.cuda()
            
            if status['i'] % self.args.save_interval == 0:
                logger.info("Saving current model")
                if torch.cuda.is_available():
                    self.model.cpu()
                utils.save_model(self.model, self.args.model, writer)
                self.instr_preprocessor.vocab.save()
                if torch.cuda.is_available():
                    self.model.cuda()
                with open(status_path, 'w') as dst:
                    json.dump(status, dst)
