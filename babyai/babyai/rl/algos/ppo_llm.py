import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)
from tqdm import tqdm

from babyai.rl.algos.base_llm import BaseAlgo


class PPOAlgoLlm(BaseAlgo):
    """The class for the Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, lm_server, llm_scoring_module_key, nbr_llms=None,
                 num_frames_per_proc=None, discount=0.99, lr=7e-4, beta1=0.9, beta2=0.999,
                 gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5,
                 adam_eps=1e-5, clip_eps=0.2, epochs=4, batch_size=64,
                 reshape_reward=None, name_experiment=None, saving_path_model=None, saving_path_logs=None,
                 number_envs=None, subgoals=None, nbr_obs=3, id_expe=None, template_test=1, aux_info=None, debug=False):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, lm_server, llm_scoring_module_key, num_frames_per_proc, discount, lr, gae_lambda,
                         entropy_coef, value_loss_coef, max_grad_norm, reshape_reward, subgoals, nbr_obs, aux_info)

        self.nbr_llms = nbr_llms

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.debug = debug

        self.beta1 = beta1
        self.beta2 = beta2
        self.adam_eps = adam_eps

        self.name_experiment = name_experiment
        self.saving_path_model = saving_path_model
        self.saving_path_logs = saving_path_logs
        self.number_envs = number_envs

        self.id_expe = id_expe
        self.template_test = template_test
        self.number_updates = 0

        self.experiment_path = os.path.join(self.saving_path_logs, id_expe)

    def update_parameters(self):
        # Collect experiences
        exps, logs = self.collect_experiences(debug=self.debug)
        # print(exps.action)
        # action_counts = exps.action.unique(return_counts=True)
        # pi_l_action_counts = exps.pi_l_action.unique(return_counts=True)
        '''
        exps is a DictList with the following keys ['prompt', 'action', 'value', 'reward',
         'advantage', 'returnn', 'log_prob'] and ['collected_info', 'extra_predictions'] if we use aux_info
        exps.prompt is a (n_procs * n_frames_per_proc) of prompt
        if we use aux_info: exps.collected_info and exps.extra_predictions are DictLists with keys
        being the added information. They are either (n_procs * n_frames_per_proc) 1D tensors or
        (n_procs * n_frames_per_proc) x k 2D tensors where k is the number of classes for multiclass classification
        '''
        lm_server_update_first_call = True
        for _ in tqdm(range(self.epochs), ascii=" " * 9 + "<", ncols=100):
            # Initialize log values

            log_entropies = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            log_losses = []

            # Create minibatch of size self.batch_size*self.nbr_llms
            # each llm receive a batch of size batch_size
            for inds in self._get_batches_starting_indexes():
                # inds is a numpy array of indices that correspond to the beginning of a sub-batch
                # there are as many inds as there are batches

                exps_batch = exps[inds]

                # return the list of dict_return calculate by each llm
                list_dict_return = self.lm_server.update(exps_batch.prompt,
                                                         self.filter_candidates_fn(exps_batch.subgoal),
                                                         exps=dict(exps_batch),
                                                         lr=self.lr,
                                                         beta1=self.beta1,
                                                         beta2=self.beta2,
                                                         adam_eps=self.adam_eps,
                                                         clip_eps=self.clip_eps,
                                                         entropy_coef=self.entropy_coef,
                                                         value_loss_coef=self.value_loss_coef,
                                                         max_grad_norm=self.max_grad_norm,
                                                         nbr_llms=self.nbr_llms,
                                                         id_expe=self.id_expe,
                                                         lm_server_update_first_call=lm_server_update_first_call,
                                                         saving_path_model=self.saving_path_model,
                                                         experiment_path=self.experiment_path,
                                                         number_updates=self.number_updates,
                                                         scoring_module_key=self.llm_scoring_module_key,
                                                         template_test=self.template_test)

                lm_server_update_first_call = False

                log_losses.append(np.mean([d["loss"] for d in list_dict_return]))
                log_entropies.append(np.mean([d["entropy"] for d in list_dict_return]))
                log_policy_losses.append(np.mean([d["policy_loss"] for d in list_dict_return]))
                log_value_losses.append(np.mean([d["value_loss"] for d in list_dict_return]))
                log_grad_norms.append(np.mean([d["grad_norm"] for d in list_dict_return]))

        # Log some values

        logs["entropy"] = np.mean(log_entropies)
        logs["policy_loss"] = np.mean(log_policy_losses)
        logs["value_loss"] = np.mean(log_value_losses)
        logs["grad_norm"] = np.mean(log_grad_norms)
        logs["loss"] = np.mean(log_losses)

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.
        Returns
        -------
        batches_starting_indexes : list of lists of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = np.arange(0, self.num_frames)
        indexes = np.random.permutation(indexes)

        num_indexes = self.batch_size
        batches_starting_indexes = [indexes[i:i + num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes
