"""
This script run a simple agent in a BabyAI GoTo-Local environment.
"""
import os
import sys
import csv
import json
import logging

import time
import numpy as np
import torch
import gym
import babyai.utils as utils
import hydra
import test_llm

from babyai.paral_env_simple import ParallelEnv
from colorama import Fore
from lamorel import Caller, lamorel_init
from lamorel import BaseUpdater, BaseModuleFunction
from accelerate import Accelerator
from main import ValueModuleFn

from agents.drrn.drrn import DRRN_Agent
from agents.random_agent.random_agent import Random_agent
from agents.bot.bot import BotAgent

lamorel_init()
logger = logging.getLogger(__name__)
accelerator = Accelerator()


# TODO add the value of the true reward *20 who should receive the final reward?
def reward_function(subgoal_proba=None, reward=None, policy_value=None, llm_0=None):
    if reward > 0:
        return [20 * reward, 0]
    else:
        return [0, 0]


# TODO think about a correct value for the beta of the reward shaping part
def reward_function_shapped(subgoal_proba=None, reward=None, policy_value=None, llm_0=None):
    if reward > 0:
        return [20 * reward - np.log(subgoal_proba / policy_value), -np.log(subgoal_proba / policy_value)]
    else:
        return [-1 - np.log(subgoal_proba / policy_value), -1 - np.log(subgoal_proba / policy_value)]


"""dict_modifier_french = [{},
                        {
                            'clef': 'chaise',
                            'balle': 'table',
                            'boîte': 'voiture'
                        },
                        {
                            'rouge': 'vermilion',
                            'verte': 'jade',
                            'bleue': 'cyan',
                            'violette': 'mauve',
                            'jaune': 'dorée',
                            'gris': 'argent'
                        },
                        {
                            'clef': 'dax',
                            'balle': 'xolo',
                            'boîte': 'afze'
                        },
                        {
                            'rouge': 'faze',
                            'verte': 'jatu',
                            'bleue': 'croh',
                            'violette': 'vurst',
                            'jaune': 'gakul',
                            'grise': 'sil'
                        },
                        {
                            'clef': 'dax',
                            'balle': 'xolo',
                            'boîte': 'afze',
                            'rouge': 'faze',
                            'verte': 'jatu',
                            'bleue': 'croh',
                            'violette': 'vurst',
                            'jaune': 'gakul',
                            'grise': 'sil'
                        }]
dict_dict_modifier = {'english': dict_modifier_english, 'french': dict_modifier_french}
dict_modifier_name = ['no_modifications', 'other_name_same_categories', 'adj_synonym', 'no_meaning_nouns',
                      'no_meaning_adj', 'no_meaning_words', 'important_words_suppress']"""

dict_modifier_french = [{},
                        {
                            'clef': 'chaise',
                            'balle': 'table',
                            'boîte': 'voiture'
                        },
                        {
                            'rouge': 'vermilion',
                            'verte': 'jade',
                            'bleue': 'cyan',
                            'violette': 'mauve',
                            'jaune': 'dorée',
                            'gris': 'argent'
                        },
                        {
                            'clef': 'dax',
                            'balle': 'xolo',
                            'boîte': 'afze'
                        },
                        {
                            'rouge': 'faze',
                            'verte': 'jatu',
                            'bleue': 'croh',
                            'violette': 'vurst',
                            'jaune': 'gakul',
                            'grise': 'sil'
                        },
                        {
                            'clef': 'dax',
                            'balle': 'xolo',
                            'boîte': 'afze',
                            'rouge': 'faze',
                            'verte': 'jatu',
                            'bleue': 'croh',
                            'violette': 'vurst',
                            'jaune': 'gakul',
                            'grise': 'sil'
                        },
                        {"But de l'agent": "Je veux que l'agent fasse"},
                        {"But de l'agent": 'Tu dois faire'}]

dict_modifier_english = [{},
                         {
                             'key': 'chair',
                             'ball': 'table',
                             'box': 'car'
                         },
                         {
                             'red': 'vermilion',
                             'green': 'jade',
                             'blue': 'cyan',
                             'purple': 'violet',
                             'yellow': 'golden',
                             'grey': 'silver'
                         },
                         {
                             'key': 'dax',
                             'ball': 'xolo',
                             'box': 'afze'
                         },
                         {
                             'red': 'faze',
                             'green': 'jatu',
                             'blue': 'croh',
                             'purple': 'vurst',
                             'yellow': 'gakul',
                             'grey': 'sil'
                         },
                         {
                             'key': 'dax',
                             'ball': 'xolo',
                             'box': 'afze',
                             'red': 'faze',
                             'green': 'jatu',
                             'blue': 'croh',
                             'purple': 'vurst',
                             'yellow': 'gakul',
                             'grey': 'sil'
                         },
                         {'Goal of the agent': 'I would like the agent to'},
                         {'Goal of the agent': 'You have to'}]

dict_modifier_name = ['no_modification_test', 'other_name_same_categories', 'adj_synonym', 'no_meaning_nouns',
                      'no_meaning_adj', 'no_meaning_words', 'change_intro_first_personne_speaker',
                      'change_intro_first_personne_agent']

"""dict_modifier_english = [{}]
dict_modifier_french = [{}]
dict_modifier_name = ['no_modification_test']"""

dict_dict_modifier = {'english': dict_modifier_english, 'french': dict_modifier_french}


class updater(BaseUpdater):
    def perform_update(self, contexts, candidates, _current_batch_ids, **kwargs):
        if not hasattr(self, "is_loaded"):

            if "im_learning" in kwargs:
                self._llm_module.module._LLM_model.load_state_dict(torch.load(kwargs["saving_path_model"] + "/model.checkpoint"))
                self.is_loaded = True
                print("im")
            else:
                try:
                    self._llm_module.load_state_dict(torch.load(kwargs["saving_path_model"] +
                                                                "/" + kwargs["id_expe"] + "/last/model.checkpoint"))
                    self.is_loaded = True
                    print("Last")
                except:
                    self._llm_module.load_state_dict(torch.load(kwargs["saving_path_model"] +
                                                                "/" + kwargs["id_expe"] + "/backup/model.checkpoint"))
                    self.is_loaded = True
                    print("Backup")


def run_agent(args, algo, saving_path_logs, id_expe):
    if args.random_agent:
        format_str = ("Language: {} | Name dict: {} | Episodes Done: {} | Reward: {: .2f} +- {: .2f}  (Min: {: .2f} Max: {: .2f}) |\
        Success Rate: {: .2f} |")
    else:
        format_str = ("Language: {} | Name dict: {} | Episodes Done: {} | Reward: {: .2f} +- {: .2f}  (Min: {: .2f} Max: {: .2f}) |\
        Success Rate: {: .2f} | \nReshaped: {: .2f} +- {: .2f}  (Min: {: .2f} Max: {: .2f}) | Bonus: {: .2f} +- {: .2f}\
                                 (Min: {: .2f} Max: {: .2f})")

    dm = dict_dict_modifier[args.language]
    for d, d_name in zip(dm, dict_modifier_name):

        if args.modified_action_space:
            d_name += '_'
            for a in args.new_action_space:
                d_name += a + '_'
            d_name = d_name[:-1]

        if args.zero_shot:
            d_name += '_zero_shot'

        if args.im_learning:
            d_name += '_im'

        test_path = os.path.join(os.path.join(saving_path_logs, id_expe), 'test')
        experiment_path = os.path.join(test_path, args.name_environment)
        path_test_folder = os.path.join(experiment_path, 'return_per_episode')


        if args.get_example_trajectories:
            if d_name == 'no_modification_test':
                nbr_frames = 0
                k = 0
                if args.bot:
                    # trajectories generated by the bot given in BabyAI
                    np_path = os.path.join(path_test_folder, 'bot_trajectories')
                else:
                    # trajectories generated by the agent after its training
                    np_path = os.path.join(path_test_folder, 'trajectories')

                status_path = os.path.join(path_test_folder, 'status.json')
                if os.path.exists(status_path):
                    with open(status_path, 'r') as src:
                        status = json.load(src)
                else:
                    status = {'k': 0,
                              'nbr_frames': 0}

                while status['nbr_frames'] < args.num_steps:
                    exps, logs = algo.generate_trajectories(d, args.language)

                    np.save(np_path+'_prompts_{}'.format(status['k']), exps.prompts)
                    np.save(np_path+'_actions_{}'.format(status['k']), exps.actions)
                    # np.save(np_path+'_values_{}'.format(status['k']), exps.vals)
                    status['nbr_frames'] += logs['nbr_frames']
                    status['k'] += 1

                    with open(status_path, 'w') as dst:
                        json.dump(status, dst)
        else:
            if args.im_learning:
                exps, logs = algo.generate_trajectories(d, args.language, args.im_learning)
            else:
                exps, logs = algo.generate_trajectories(d, args.language)

            return_per_episode = utils.synthesize(logs["return_per_episode"])
            success_per_episode = utils.synthesize(
                [1 if r > 0 else 0 for r in logs["return_per_episode"]])
            if not args.random_agent:
                reshaped_return_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
                reshaped_return_bonus_per_episode = utils.synthesize(logs["reshaped_return_bonus_per_episode"])
            # num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            if args.random_agent:
                data = [args.language, d_name, logs['episodes_done'], *return_per_episode.values(),
                    success_per_episode['mean']]
            else:
                data = [args.language, d_name, logs['episodes_done'], *return_per_episode.values(),
                    success_per_episode['mean'],
                    *reshaped_return_per_episode.values(),
                    *reshaped_return_bonus_per_episode.values()]

            logger.info(Fore.YELLOW + format_str.format(*data) + Fore.RESET)
            np_path = os.path.join(path_test_folder, d_name)
            np.save(np_path, np.array(logs["return_per_episode"]))



# This will be overriden by lamorel's launcher if used
@hydra.main(config_path='config', config_name='config')
def main(config_args):
    """name_env = config_args.rl_script_args.name_environment
    for i in range(1000):
        env = gym.make(name_env)
        env.seed(int(i))
        obs, info = env.reset()
        print(obs['mission'])"""

    # lm server
    if config_args.lamorel_args.distributed_setup_args.n_llm_processes > 0:
        if config_args.rl_script_args.im_learning:
            lm_server = Caller(config_args.lamorel_args, custom_updater_class=updater)
        else:
            lm_server = Caller(config_args.lamorel_args, custom_updater_class=updater,
                           custom_module_functions={'value': ValueModuleFn(config_args.lamorel_args.llm_args.model_type)})

    id_expe = config_args.rl_script_args.name_experiment + \
              '_nbr_env_{}_'.format(config_args.rl_script_args.number_envs) + \
              '{}_'.format(config_args.rl_script_args.name_model) + \
              'pretrained_{}_'.format(config_args.lamorel_args.llm_args.pretrained)

    if not config_args.lamorel_args.llm_args.pretrained:
        id_expe += 'load_embedding_{}_'.format(config_args.rl_script_args.load_embedding) + \
                   'use_action_heads_{}_'.format(config_args.rl_script_args.use_action_heads)

    if config_args.rl_script_args.nbr_obs != 3:
        id_expe += 'nbr_obs_{}_'.format(config_args.rl_script_args.nbr_obs)

    id_expe += 'nbr_actions_{}_'.format(len(config_args.rl_script_args.action_space))

    # if config_args.rl_script_args.modified_action_space is not False we keep the same id_expe
    # we just create a file with test_name containing the modified action in
    # name_experiment/test/return_per_episode/test_name.npy
    for a in config_args.rl_script_args.action_space:
        id_expe += a + '_'

    id_expe += 'shape_reward_beta_{}_'.format(config_args.rl_script_args.reward_shaping_beta) + \
               'seed_{}'.format(config_args.rl_script_args.seed)

    # Env
    name_env = config_args.rl_script_args.name_environment
    seed = config_args.rl_script_args.seed
    envs = []
    subgoals = []
    number_envs = config_args.rl_script_args.number_envs
    if config_args.rl_script_args.modified_action_space:
        list_actions = [a.replace("_", " ") for a in config_args.rl_script_args.new_action_space]
    else:
        list_actions = [a.replace("_", " ") for a in config_args.rl_script_args.action_space]

    for i in range(number_envs):
        env = gym.make(name_env)
        env.seed(
            int(1e9 * seed + i))  # to be sure to not have the same seeds as in the train (100h max ~ 100000 episodes done in our settings)
        envs.append(env)
        subgoals.append(list_actions)
    envs = ParallelEnv(envs)

    if config_args.rl_script_args.reward_shaping_beta == 0:
        reshape_reward = reward_function
    else:
        reshape_reward = reward_function_shapped  # TODO ad the beta

    # create the folder for the agent
    model_path = os.path.join(config_args.rl_script_args.saving_path_model, id_expe)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    log_path = os.path.join(config_args.rl_script_args.saving_path_logs, id_expe)
    # create the folder for the tests results and return_per_episode
    test_path = os.path.join(log_path, 'test')
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    test_path_env = os.path.join(test_path, config_args.rl_script_args.name_environment)
    if not os.path.exists(test_path_env):
        os.makedirs(test_path_env)
        os.makedirs(os.path.join(test_path_env, 'return_per_episode'))

    if config_args.lamorel_args.distributed_setup_args.n_llm_processes > 0:
        if not config_args.rl_script_args.zero_shot:
            if config_args.rl_script_args.im_learning:
                saving_path_model = config_args.rl_script_args.im_path + '_seed_{}'
                saving_path_model = saving_path_model.format(config_args.rl_script_args.seed)
                lm_server.update([None for _ in range(config_args.lamorel_args.distributed_setup_args.n_llm_processes)],
                             [[None] for _ in range(config_args.lamorel_args.distributed_setup_args.n_llm_processes)],
                             im_learning=True, saving_path_model=saving_path_model)
            else:
                lm_server.update([None for _ in range(config_args.lamorel_args.distributed_setup_args.n_llm_processes)],
                             [[None] for _ in range(config_args.lamorel_args.distributed_setup_args.n_llm_processes)],
                             id_expe=id_expe, saving_path_model=config_args.rl_script_args.saving_path_model)

        algo = test_llm.BaseAlgo(envs, lm_server, config_args.rl_script_args.number_episodes, reshape_reward,
                                 subgoals)
    else:
        if config_args.rl_script_args.random_agent:
            algo = Random_agent(envs=envs,
                                nbr_envs=number_envs,
                                size_action_space=len(config_args.rl_script_args.action_space),
                                number_episodes=config_args.rl_script_args.number_episodes)
        elif config_args.rl_script_args.bot:
            algo = BotAgent(envs=envs,
                            subgoals=subgoals,
                            number_episodes=config_args.rl_script_args.number_episodes)
        else:
            if not config_args.rl_script_args.zero_shot:
                algo = DRRN_Agent(envs, subgoals, reshape_reward, config_args.rl_script_args.spm_path,
                                  max_steps=number_envs * 4,
                                  number_epsiodes_test=config_args.rl_script_args.number_episodes,
                                  saving_path=config_args.rl_script_args.saving_path_model + "/" + id_expe)
                algo.load()
            else:
                algo = DRRN_Agent(envs, subgoals, reshape_reward, config_args.rl_script_args.spm_path,
                                  max_steps=number_envs * 4,
                                  number_epsiodes_test=config_args.rl_script_args.number_episodes,
                                  saving_path=config_args.rl_script_args.saving_path_model + "/" + id_expe)

    run_agent(config_args.rl_script_args, algo, config_args.rl_script_args.saving_path_logs, id_expe)
    if config_args.lamorel_args.distributed_setup_args.n_llm_processes > 0:
        lm_server.close()


if __name__ == '__main__':
    main()
