import copy
import gym
import time
import datetime
import numpy as np
import sys
import itertools
import torch
import pickle as pkl
import blosc
import multiprocessing
import os
import logging
import babyai.utils as utils

import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from nn.enc_visual import Resnet18
from tqdm import tqdm
from PIL import Image
from gym import spaces

from babyai.evaluate import batch_evaluate
from babyai.QA import Model
from babyai.l_class import Model

logger = logging.getLogger(__name__)
if torch.cuda.is_available():
    resnet = Resnet18('cuda')
else:
    resnet = Resnet18('cpu')

softmax = nn.Softmax(dim=1)


class TrainerClass(object):
    def __init__(self, args, attr):
        self.args = args
        self.attr = attr
        utils.seed(self.args.seed)

        demos_path = utils.get_demos_QG_path(args.demos, args.env, args.demos_origin, valid=False)
        demos_path_valid = utils.get_demos_QG_path(args.demos, args.env, args.demos_origin, valid=True)
        demos_voc = utils.get_demos_QG_voc_path(args.demos, args.env, args.demos_origin, valid=False)

        demos_path_l_class = str(demos_path).replace("QG", "QG_no_answer_biased_l_class")
        demos_path_valid_l_class = str(demos_path_valid).replace("QG", "QG_no_answer_biased_l_class")
        demos_voc_l_class = str(demos_voc).replace("QG", "QG_no_answer_biased")
        print(demos_path_l_class)
        print(demos_path_valid_l_class)
        print(demos_voc_l_class)
        logger.info('loading train demos language classifier')
        self.train_demos_l_class = utils.load_demos(demos_path_l_class)
        logger.info('loaded train demos language classifier')

        logger.info('loading valid demos language classifier')
        self.valid_demos_l_class = utils.load_demos(demos_path_valid_l_class)
        logger.info('loaded valid demos language classifier')

        logger.info('loading voc train demos language classifier')
        self.demos_voc_l_class = utils.load_voc(demos_voc_l_class)
        logger.info('loaded voc train demos language classifier')

        print(self.demos_voc_l_class['question'].to_dict()['index2word'])
        print(" ")
        print(self.demos_voc_l_class['answer'].to_dict()['index2word'])
        # Define episodic transformer for QA
        if self.args.QA:
            emb_size = len(self.demos_voc_l_class['question'])
            self.l_qa = Model(attr, emb_size, numb_action=0, pad=0)
        else:
            ValueError("no args.QA")

        if torch.cuda.is_available():
            self.l_qa.cuda()

        self.optimizer = torch.optim.Adam(self.l_qa.parameters(),
                                          self.args.lr,
                                          eps=self.args.optim_eps)
        self.scheduler_1 = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                           step_size=10,
                                                           gamma=0.1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_batch(self, indices, batch_size, train=True):

        if train:
            demo = self.train_demos_l_class
        else:
            demo = self.valid_demos_l_class

        offset = 0
        with tqdm(total=len(indices) // batch_size, desc="creation batch") as t:
            for batch_index in range(len(indices) // batch_size):
                batch_demo = {}
                for k in demo:
                    if k == 'questions':
                        batch_demo[k] = []
                        for i in indices[offset: offset + batch_size]:
                            len_q = len(demo[k][i])
                            for j in range(len_q):
                                batch_demo[k].append(demo[k][i][j])

                    else:
                        assert k == 'answers'
                        batch_demo[k] = []
                        for i in indices[offset: offset + batch_size]:
                            len_q = len(demo[k][i])
                            for j in range(len_q):
                                batch_demo[k].append(demo[k][i][j])
                    if k == 'answers':
                        batch_demo[k] = torch.tensor(batch_demo[k])

                # pad  and tensorize questions
                batch_demo['questions'] = pad_sequence(
                    [torch.tensor(x, dtype=torch.float32) for x in batch_demo['questions']],
                    batch_first=True,
                    padding_value=0).type(torch.IntTensor)

                assert batch_demo['questions'].shape[0] == batch_demo['answers'].shape[0]

                if train:
                    pkl.dump(batch_demo, open("storage/batch_train_l_class/batch_{}.pkl".format(batch_index), "wb"))
                else:
                    pkl.dump(batch_demo, open("storage/batch_valid_l_class/batch_{}.pkl".format(batch_index), "wb"))

                offset += batch_size
                t.update()

    def train(self):

        # Log dictionary
        log = {"loss_cross_entropy_train": [], "success_pred_train": [], "loss_cross_entropy_valid": [],
               "success_pred_valid": [], "confidence": []}
        generated = False
        print(' ')
        print('Batch generated {}'.format(generated))
        print(' ')
        unique_t = []
        count_t = []

        for e in range(self.args.epochs):
            print('lr {}'.format(self.optimizer.state_dict()['param_groups'][0]['lr']))
            # Train
            batch_size = min(self.args.batch_size, len(self.train_demos_l_class['questions']))
            if self.args.epoch_length == 0:
                indices = list(range(len(self.train_demos_l_class['questions'])))
            else:
                indices = np.random.choice(len(self.train_demos_l_class['questions']), self.args.epoch_length)
            self.l_qa.train()

            if not generated:
                np.random.shuffle(indices)
                self.generate_batch(indices, batch_size, train=True)

            with tqdm(total=len(indices) // batch_size, desc="train") as t:
                answer_loss_batch = 0
                success_pred_batch = 0
                for batch_index in range(len(indices) // batch_size):
                    # batch_index_overfit = 0
                    demo = pkl.load(open("storage/batch_train_l_class/batch_{}.pkl".format(batch_index), "rb"))
                    batch_demo = {}
                    for k, v in demo.items():
                        if k != 'length_frames_max' and k != 'env_ids' and k != 'missions':
                            batch_demo[k] = v.cuda()
                        else:
                            batch_demo[k] = v

                    answer_pred = self.l_qa.forward(self.demos_voc_l_class['question'], **batch_demo)
                    answer_loss = F.cross_entropy(answer_pred['answers'], batch_demo['answers'], reduction='mean')

                    # count the number of time a class is answered on a sub_batch
                    if batch_index == 0 or batch_index == (len(indices) // batch_size) - 1:
                        unique, return_counts = torch.unique(torch.argmax(answer_pred['answers'], dim=1),
                                                             return_counts=True)
                        unique_t.append(unique.cpu().detach().numpy())
                        d = {k: v for k, v in
                             zip(self.demos_voc_l_class['answer'].index2word(list(unique.cpu().detach().numpy())),
                                 return_counts.cpu().detach().numpy())}
                        print(d)
                        count_t.append(return_counts.cpu().detach().numpy())

                    success_pred_batch += (torch.argmax(answer_pred['answers'], dim=1)
                                           == batch_demo['answers']).sum().cpu().detach().numpy() / \
                                          batch_demo['answers'].shape[0]
                    answer_loss_batch += answer_loss.cpu().detach().numpy()

                    self.optimizer.zero_grad()
                    answer_loss.backward()
                    self.optimizer.step()

                    t.update()
                self.scheduler_1.step()
                # print('lr {}'.format(self.scheduler_1.get_last_lr()))
                # self.scheduler_seq.step()

                log["loss_cross_entropy_train"].append(answer_loss_batch / (len(indices) // batch_size))
                log["success_pred_train"].append(success_pred_batch / (len(indices) // batch_size))

            # Valid
            with torch.no_grad():
                if self.args.epoch_length == 0:
                    indices = list(range(len(self.valid_demos_l_class['questions'])))
                else:
                    indices = np.random.choice(len(self.valid_demos_l_class['questions']), self.args.epoch_length)

                batch_size = min(self.args.batch_size, len(self.valid_demos_l_class['questions']))

                self.l_qa.eval()

                if not generated:
                    np.random.shuffle(indices)
                    self.generate_batch(indices, batch_size, train=False)
                    generated = True

                with tqdm(total=len(indices) // batch_size, desc="valid") as t:
                    answer_loss_batch = 0
                    success_pred_batch = 0
                    '''table_confidence = np.zeros(4)'''
                    for batch_index in range(len(indices) // batch_size):
                        demo = pkl.load(open("storage/batch_valid_l_class/batch_{}.pkl".format(batch_index), "rb"))
                        batch_demo = {}
                        for k, v in demo.items():
                            if k != 'length_frames_max' and k != 'env_ids' and k != 'missions':
                                batch_demo[k] = v.cuda()
                            else:
                                batch_demo[k] = v

                        answer_pred = self.l_qa.forward(self.demos_voc_l_class['question'], **batch_demo)
                        answer_loss = F.cross_entropy(answer_pred['answers'], batch_demo['answers'], reduction='mean')

                        success_pred_batch += (torch.argmax(answer_pred['answers'], dim=1)
                                               == batch_demo['answers']).sum().cpu().detach().numpy() / \
                                              batch_demo['answers'].shape[0]

                        answer_loss_batch += answer_loss.cpu().detach().numpy()
                        t.update()

                    # self.scheduler_2.step(answer_loss_batch / (len(indices) // batch_size))
                    log["loss_cross_entropy_valid"].append(answer_loss_batch / (len(indices) // batch_size))
                    log["success_pred_valid"].append(success_pred_batch / (len(indices) // batch_size))


            logger.info(
                'Epoch {} train CE {} SR {} valid CE {} and the SR is {}'.format(e,
                                                                                 log["loss_cross_entropy_train"][-1],
                                                                                 log["success_pred_train"][-1],
                                                                                 log["loss_cross_entropy_valid"][-1],
                                                                                 log["success_pred_valid"][-1]))

            pkl.dump(log,
                     open('storage/models/{}_l_class/model_{}/log.pkl'.format(self.args.env, self.args.model_number), "wb"))
            pkl.dump(np.array(unique_t, dtype=object),
                     open('storage/models/{}_l_class/model_{}/unique.pkl'.format(self.args.env, self.args.model_number),
                              "wb"))
            pkl.dump(np.array(count_t, dtype=object),
                     open('storage/models/{}_l_class/model_{}/count.pkl'.format(self.args.env, self.args.model_number),
                              "wb"))
            torch.save(self.l_qa.state_dict(),
                       'storage/models/{}_l_class/model_{}/et_qa_{}.pt'.format(self.args.env, self.args.model_number, e))

        return log
