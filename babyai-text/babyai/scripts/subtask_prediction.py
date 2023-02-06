"""
Code for the subtask prediction model (relevance classifier).
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
from tqdm import tqdm
from gym import spaces
from subtask_prediction_model import SubtaskPredictionModel
from instruction_handler import InstructionHandler

logger = logging.getLogger(__name__)


class SubtaskDataset(object):
    """Dataset for online subtask decomposition collection"""

    def __init__(self):
        self.denoised_demos = {}

    def add_demos(self, demos):
        """demos: list of observed decompositions [(instr, [(t, subtask),...])]
                  t (time of completion) is currently ignored
        """

        for instr, subtasks in demos:
            # ignore timestep
            subtasks_flat = [idx for sub in subtasks for idx in sub[1]]
            if instr in self.denoised_demos:
                if len(self.denoised_demos[instr].intersection(subtasks_flat)) > 0:
                    self.denoised_demos[instr] = self.denoised_demos[instr].intersection(
                        subtasks_flat)
                else:
                    self.denoised_demos[instr] = set(subtasks_flat)
            else:
                self.denoised_demos[instr] = set(subtasks_flat)

    def get_stats(self):
        num_subs = list([len(v)
                         for v in self.denoised_demos.values() if len(v) > 0])
        if len(num_subs) > 0:
            return (len(num_subs), np.mean(num_subs), np.std(num_subs), min(num_subs), max(num_subs))
        else:
            return (0, 0, 0, 0)

    def get_demos(self):
        return [(instr, [(-1, list(subtasks))]) for instr, subtasks in self.denoised_demos.items()]


class SubtaskPrediction(object):

    def __init__(self, args, online_args=False):

        self.args = args
        self.online_args = online_args

        utils.seed(self.args.seed)

        if online_args:
            if getattr(self.args, 'subtask_pretrained_model', None):
                logger.info("loading pretrained model")
                self.model = utils.load_model(
                    args.subtask_pretrained_model, raise_not_found=True)
            else:
                self.model = SubtaskPredictionModel(
                    {"instr": 100}, instr_dim=args.instr_dim, arch=args.subtask_arch, lang_model=args.instr_arch)

        else:
            if getattr(self.args, 'pretrained_model', None):
                logger.info("loading pretrained model")
                self.model = utils.load_model(
                    args.pretrained_model, raise_not_found=True)
            else:
                self.model = SubtaskPredictionModel(
                    {"instr": 100}, instr_dim=args.instr_dim, arch=args.arch, lang_model=args.instr_arch)

        if online_args:
            demos_path_valid = utils.get_demos_path(
                args.subtask_hl_demos, args.env, args.demos_origin, valid=True)
            ll_demos_path = utils.get_demos_path(
                args.demos, None, None, valid=False)
            logger.info('loading low-level demos')
            ll_demos = utils.load_demos(ll_demos_path)
            logger.info('loaded low-level demos')
            if args.episodes:
                if args.episodes > len(ll_demos):
                    raise ValueError(
                        "there are only {} low-level demos".format(len(ll_demos)))
                ll_demos = ll_demos[:args.episodes]
        else:
            demos_path = utils.get_demos_path(
                args.demos, args.env, args.demos_origin, valid=False)
            demos_path_valid = utils.get_demos_path(
                args.demos, args.env, args.demos_origin, valid=True)
            ll_demos_path = utils.get_demos_path(
                args.low_level_demos, args.env, args.demos_origin, valid=False)
            logger.info('loading low-level demos')
            ll_demos = utils.load_demos(ll_demos_path)
            logger.info('loaded low-level demos')
            if args.ll_episodes:
                if args.ll_episodes > len(ll_demos):
                    raise ValueError(
                        "there are only {} low-level demos".format(len(ll_demos)))
                ll_demos = ll_demos[:args.ll_episodes]

        logger.info('loading instruction handler')
        self.instr_handler = InstructionHandler(
            ll_demos, load_bert=False, save_path=os.path.join(os.path.splitext(ll_demos_path)[0], "ih"))
        logger.info('loaded instruction handler')

        if not online_args:
            # Load training data
            logger.info('loading demos')
            self.train_demos = utils.load_demos(demos_path)
            logger.info('loaded demos')
            self.train_demos = list(
                tuple(zip(self.train_demos[0], self.train_demos[1])))
            if args.episodes:
                if args.episodes > len(self.train_demos):
                    raise ValueError(
                        "there are only {} train demos".format(len(self.train_demos)))
                self.train_demos = self.train_demos[:args.episodes]

            if args.denoise:
                self.train_demos = self.denoise_demos(
                    self.train_demos, args.denoise_k, args.denoise_total)
                logger.info(
                    f"denoised demos -> {len(self.train_demos)} each with {args.denoise_k} instances")

            if args.augment:
                self.train_demos.extend(self.augment_demos(args.augment_total))
                logger.info(f"augmented demos -> {args.augment_total}")

        self.val_demos = utils.load_demos(demos_path_valid)
        self.val_demos = list(tuple(zip(self.val_demos[0], self.val_demos[1])))
        if online_args:
            if args.subtask_val_episodes > len(self.val_demos):
                logger.info('Using all the available {} demos to evaluate valid. accuracy'.format(
                    len(self.val_demos)))
            self.val_demos = self.val_demos[:self.args.subtask_val_episodes]
        else:
            if args.val_episodes > len(self.val_demos):
                logger.info('Using all the available {} demos to evaluate valid. accuracy'.format(
                    len(self.val_demos)))
            self.val_demos = self.val_demos[:self.args.val_episodes]

        if online_args:
            self.instr_preprocessor = utils.InstructionOnlyPreprocessor(
                args.subtask_model, getattr(self.args, 'subtask_pretrained_model', None))
            self.instr_preprocessor.vocab.save()
        else:
            self.instr_preprocessor = utils.InstructionOnlyPreprocessor(
                args.model, getattr(self.args, 'pretrained_model', None))
            self.instr_preprocessor.vocab.save()

        self.model.train()
        if torch.cuda.is_available():
            self.model.cuda()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), self.args.lr, eps=self.args.optim_eps)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100, gamma=0.9)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        if online_args:
            self.status = {"i": 0, "num_frames": 0}

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
        default_model_name = "{envs}_SP_{arch}_{instr}_seed{seed}_{suffix}".format(
            **model_name_parts)
        if getattr(args, 'pretrained_model', None):
            default_model_name = args.pretrained_model + '_pretrained_' + default_model_name
        return default_model_name

    def augment_demos(self, total):
        demos = []
        for i, instr1 in enumerate(self.instr_handler.missions):
            for j, instr2 in enumerate(self.instr_handler.missions):
                demos.append((instr1 + " and " + instr2, [(-1, [i, j])]))
        np.random.shuffle(demos)
        demos = demos[:total]
        return demos

    def denoise_demos(self, demos, k, total):
        denoised = {}
        num_used = {}
        for instr, subtasks in demos:
            if instr not in num_used or num_used[instr] < k:
                subtasks_flat = [idx for sub in subtasks for idx in sub[1]]
                denoised[instr] = denoised.get(instr, set(
                    subtasks_flat)).intersection(subtasks_flat)
                num_used[instr] = num_used.get(instr, 0) + 1
        num_subs = list([len(v) for i, v in denoised.items()
                         if num_used[i] == k and len(v) > 0])
        logger.info("denoised {} +- {}, ({}, {})".format(np.mean(num_subs), np.std(num_subs),
                                                         min(num_subs), max(num_subs)))
        return [(instr, [(-1, list(subtasks))]) for instr, subtasks in denoised.items()
                if len(subtasks) > 0 and num_used[instr] == k][:total]

    def run_epoch_recurrence(self, demos, batch_size, is_training=False, truth=False, ones=False):

        if ones:
            demos = utils.demos.transform_demos_subtasks_cross_ones(
                demos, self.instr_handler)
        else:
            demos = utils.demos.transform_demos_subtasks_cross(
                demos, self.instr_handler)

        indices = list(range(len(demos)))
        if is_training:
            np.random.shuffle(indices)

        offset = 0

        if not is_training:
            self.model.eval()

        log = {"loss": [], "accuracy": [], "correct_true": 0, "predict_true": 0,
               "target_true": 0, "precision": 0, "recall": 0, "frames": 0}

        start_time = time.time()
        frames = 0
        for batch_index in range(len(indices) // batch_size):
            batch = [demos[i] for i in indices[offset: offset + batch_size]]
            frames += len(batch)

            _log = self.run_epoch_recurrence_one_batch(
                batch, is_training=is_training, truth=truth)

            log["loss"].append(_log["loss"])
            log["accuracy"].append(_log["accuracy"])
            log["frames"] = frames

            if truth:
                log["correct_true"] += _log["correct_true"]
                log["target_true"] += _log["target_true"]
                log["predict_true"] += _log["predict_true"]

            offset += batch_size

        if truth:
            if log["predict_true"] > 0:
                log["precision"] = log["correct_true"] / log["predict_true"]
            if log["target_true"] > 0:
                log["recall"] = log["correct_true"] / log["target_true"]

        if not is_training:
            self.model.train()

        return log

    def run_epoch_recurrence_one_batch(self, batch, is_training=False, truth=False):

        missions, subtasks, labels = zip(*batch)

        missions = self.instr_preprocessor(missions, device=self.device)
        try:
            subtasks = self.instr_preprocessor(subtasks, device=self.device)
        except:
            import pdb
            pdb.set_trace()
            subtasks = self.instr_preprocessor(subtasks, device=self.device)
        labels = torch.tensor(labels).to(self.device).float()

        preds = self.model(missions, subtasks)
        predictions = preds.round()
        loss_fn = nn.BCELoss()
        loss = loss_fn(preds, labels)
        accuracy = float((predictions == labels).sum()) / len(labels)

        if is_training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        log = {}
        log["loss"] = float(loss)
        log["accuracy"] = float(accuracy)

        if truth:
            log["correct_true"] = int(
                ((predictions == labels).int() & labels.int()).sum())
            log["predict_true"] = int(predictions.sum())
            log["target_true"] = int(labels.sum())

        return log

    def online_update(self, train_demos, header, writer, validate=True):

        logger.info("Dataset size: {}".format(len(train_demos)))
        total_start_time = time.time()

        for u in range(self.args.subtask_updates):

            self.status['i'] += 1
            i = self.status['i']
            update_start_time = time.time()

            log = self.run_epoch_recurrence(
                train_demos, batch_size=self.args.subtask_batch_size, is_training=True, truth=True)

            update_end_time = time.time()

            if u == self.args.subtask_updates-1:
                total_ellapsed_time = int(time.time() - total_start_time)
                fps = log['frames'] / (update_end_time - update_start_time)
                duration = datetime.timedelta(seconds=total_ellapsed_time)

                for key in log:
                    log[key] = np.mean(log[key])

                train_data = [self.status['i'], self.status['num_frames'], fps, total_ellapsed_time,
                              log["loss"], log["accuracy"], log["precision"], log["recall"]]

                logger.info(
                    "U {} | M {:06} | MPS {:04.0f} | D {} | Loss {:.3f} | Accuracy {:.3f} | Precision {:.3f} | Recall {:.3f}".format(*train_data))

                if validate:
                    val_log = self.run_epoch_recurrence(
                        self.val_demos, batch_size=self.args.subtask_batch_size, is_training=False)
                    truth_validation_accuracy = 0  # Deprecated but retained not to mess up logging
                    truth_validation_precision = 0
                    truth_validation_recall = 0
                    validation_loss = np.mean(val_log["loss"])
                    validation_accuracy = np.mean(val_log["accuracy"])
                    success_rate = validation_accuracy

                    validation_data = [validation_loss, validation_accuracy,
                                       truth_validation_accuracy, truth_validation_precision, truth_validation_recall]
                    logger.info(("Validation: Loss {: .3f} | Accuracy {: .3f} | GT Accuracy {: .3f} | GT Precision {: .3f} | GT Recall {: .3f}"
                                 ).format(*validation_data))

                    subtask_log = {key: float(value) for key, value in zip(
                        header, train_data + validation_data)}
                else:
                    subtask_log = {key: float(value)
                                   for key, value in zip(header, train_data)}

                logger.info("Saving current model")
                if torch.cuda.is_available():
                    self.model.cpu()
                utils.save_model(self.model, self.args.subtask_model, writer)
                self.instr_preprocessor.vocab.save()
                if torch.cuda.is_available():
                    self.model.cuda()

        return subtask_log

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
            logger.info("Batch size too high. Setting it to the number of train demos ({})".format(
                len(train_demos)))

        # Model saved initially to avoid "Model not found Exception" during first validation step
        utils.save_model(self.model, self.args.model, writer)

        # best mean return to keep track of performance on validation set
        best_success_rate, i = 0, 0
        total_start_time = time.time()

        while status['i'] < getattr(self.args, 'epochs', int(1e9)):

            if self.args.augment:
                if status['i'] < self.args.wait_finetune:
                    train_demos_truncated = train_demos[-self.args.augment_total:]
                else:
                    train_demos_truncated = train_demos[:-
                                                        self.args.augment_total]
            else:
                train_demos_truncated = train_demos

            status['i'] += 1
            i = status['i']
            update_start_time = time.time()

            log = self.run_epoch_recurrence(
                train_demos_truncated, batch_size=self.args.batch_size, is_training=True, truth=True, ones=self.args.ones)

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

                logger.info(
                    "U {} | M {:06} | MPS {:04.0f} | D {} | Loss {:.3f} | Accuracy {:.3f} | Precision {:.3f} | Recall {:.3f}".format(*train_data))

                # Log the gathered data only when we don't evaluate the validation metrics. It will be logged anyways
                # afterwards when status['i'] % self.args.val_interval == 0
                if status['i'] % self.args.val_interval != 0:
                    # instantiate a validation_log with empty strings when no validation is done
                    validation_data = [
                        ''] * len([key for key in header if 'valid' in key])
                    assert len(header) == len(train_data + validation_data)
                    if self.args.tb:
                        for key, value in zip(header, train_data):
                            writer.add_scalar(key, float(
                                value), status['num_frames'])
                    elif self.args.wb:
                        writer.log({key: float(value) for key, value in zip(header, train_data)},
                                   step=status['num_frames'])
                    csv_writer.writerow(train_data + validation_data)

                if status['i'] % self.args.val_interval == 0:
                    val_log = self.run_epoch_recurrence(
                        self.val_demos, batch_size=self.args.batch_size, is_training=False, ones=self.args.ones)
                    truth_validation_accuracy = 0  # Deprecated but retained not to mess up logging
                    truth_validation_precision = 0
                    truth_validation_recall = 0
                    validation_loss = np.mean(val_log["loss"])
                    validation_accuracy = np.mean(val_log["accuracy"])
                    success_rate = validation_accuracy

                    if status['i'] % self.args.log_interval == 0:
                        validation_data = [validation_loss, validation_accuracy,
                                           truth_validation_accuracy, truth_validation_precision, truth_validation_recall]
                        logger.info(("Validation: Loss {: .3f} | Accuracy {: .3f} | GT Accuracy {: .3f} | GT Precision {: .3f} | GT Recall {: .3f}"
                                     ).format(*validation_data))

                        assert len(header) == len(train_data + validation_data)
                        if self.args.wb:
                            writer.log({key: float(value) for key, value in zip(header, train_data + validation_data)},
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
                        utils.save_model(
                            self.model, self.args.model + "_best", writer)
                        self.instr_preprocessor.vocab.save(
                            utils.get_vocab_path(self.args.model + "_best"))
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
