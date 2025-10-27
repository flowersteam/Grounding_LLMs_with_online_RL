#!/usr/bin/env python3

"""
Generate a set of subtask decompositions -- via an oracle or a
low-level/termination policy.
"""

import argparse
import gym
import logging
import sys
import os
import time
import numpy as np
import torch
from tqdm import tqdm
import babyai.utils as utils
from babyai.utils.agent import ModelAgent, BotAgent
from instruction_handler import InstructionHandler
from babyai.hrl import HRLManyEnvs
from babyai.shaped_env import ParallelShapedEnv


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--low-level-demos", default=None,
                    help="demos filename (REQUIRED or demos-origin or multi-demos required)")
parser.add_argument("--multi-low-level-demos", nargs="*", default=None,
                    help="multiple demos filenames")
parser.add_argument("--low-level-demos-origin", required=False,
                    help="origin of the demonstrations: human | agent (REQUIRED or demos or multi-demos required)")
parser.add_argument("--ll-episodes", type=int, default=100,
                    help="number of low-level episodes to load")
parser.add_argument("--multi-ll-episodes", type=int, nargs="*", default=None,
                    help="number of low-level episodes to load")
parser.add_argument("--episodes", type=int, default=100,
                    help="number of episodes to generate demonstrations for")
parser.add_argument("--valid-episodes", type=int, default=100,
                    help="number of validation episodes to generate demonstrations for")
parser.add_argument("--seed", type=int, default=int(1e7),
                    help="start random seed")
parser.add_argument("--pi-l", default=None,
                    help="model to use for low-level policy")
parser.add_argument("--done-classifier", action="store_true", default=False,
                    help="whether the low-level policy is actually a binary termination classifier")
parser.add_argument("--debug", action="store_true", default=False,
                    help="debug mode")
parser.add_argument("--oracle", action="store_true", default=False,
                    help="use oracle breakdown rather than running pi_l")
parser.add_argument("--nonstrict", action="store_true", default=False,
                    help="don't be strict about articles in oracle decomposition")


args = parser.parse_args()
logger = logging.getLogger(__name__)

def generate_subtask_breakdowns(env_name, pi_l, instr_handler, seed, num_demos, max_t=64,
                                valid=False, debug=args.debug, oracle=args.oracle,
                                done_classifier=args.done_classifier, strict=not args.nonstrict):
    missions = []
    subtasks = []

    for s in tqdm(range(seed, seed+num_demos)):
        env = gym.make(args.env)
        env.seed(s)
        if oracle:
            henv = ParallelShapedEnv([env], instr_handler=instr_handler, reward_shaping=None)
        else:
            henv = HRLManyEnvs([env], hrl="none", pi_l=pi_l, instr_handler=instr_handler,\
                reward_shaping="pi_l_all", done_classifier=done_classifier)
        henv.reset()
        env.reset()
        if "BabyAI" in env_name:
            bot = BotAgent(env)
            missions.append(env.mission)
        else:
            missions.append(env.instr)
        subtasks.append([])

        obs = henv.gen_obs()[0]
        done = False
        if oracle:
            subtasks_text = instr_handler.get_oracle_stack(missions[-1], strict=strict, unlock="Unlock" in env_name)
            subtasks[-1].append((-1, [instr_handler.get_index(s) for s in subtasks_text]))
            if debug:
                input()
                print(env.mission, subtasks_text)
        else:
            for t in range(max_t):
                if done:
                    break
                # Get bot's recommended action
                a = bot.act(obs)['action'].value
                # Perform it
                obs, reward, done, env_info, extra_info = henv.step([a], extra_info=True, project=False)
                env.step(a)
                pi_l_actions = env_info[1]
                done_action = 1 if done_classifier else env.actions.done
                pi_l_done_instr_idx = [i for i in range(instr_handler.D_l_size()) if pi_l_actions[i] == done_action]
                if pi_l_done_instr_idx:
                    subtasks[-1].append((t, pi_l_done_instr_idx))

                if debug:
                    input()
                    pi_l_done_instr = [instr_handler.get_instruction(i) for i in range(instr_handler.D_l_size())\
                        if pi_l_actions[i] == done_action]
                    print(utils.info(pi_l_done_instr, "Done Instr"))

                    pi_l_done_top_probs = extra_info['dist'].probs[:,done_action].topk(4)
                    print(utils.info(pi_l_done_top_probs.values, "Top Done Probs"))
                    print(utils.info([instr_handler.get_instruction(i) for i in pi_l_done_top_probs.indices], "Top Done Instr"))

                    pi_l_value = extra_info['value']
                    pi_l_top_values = extra_info['value'].topk(4)
                    print(utils.info(pi_l_top_values.values, "Top Value Value"))  
                    print(utils.info([instr_handler.get_instruction(i) for i in pi_l_top_values.indices], "Top Value Instr"))
                    
                    utils.viz(henv, aux_info=utils.info(a, "Action") + utils.info(reward, "Reward") + utils.info(done, "Done"))
    
    logger.info("Saving demos...")
    suffix = "_oracle" if args.oracle else ""
    demos_path = utils.get_demos_path(None, args.env, 'agent_subtasks' + suffix, valid)
    utils.save_demos((missions, subtasks), demos_path)
    
    return (missions, subtasks)

def get_breakdown_stats(missions, subtasks):
    num_subtasks = np.array([sum([len(t[1]) for t in s]) for s in subtasks])
    logging.info(f"collected {len(missions)} missions")
    logging.info(f"average of # subtasks: {np.mean(num_subtasks)}")
    logging.info(f"variance of # subtasks: {np.var(num_subtasks)}")

logging.basicConfig(level='INFO', format="%(asctime)s: %(levelname)s: %(message)s")
logger.info(args)

if getattr(args, 'multi_low_level_demos'):
    low_level_demos = []
    for demos, episodes in zip(args.multi_low_level_demos, args.multi_ll_episodes):
        demos_path = utils.get_demos_path(demos, None, None, valid=False)
        logger.info('loading {} of {} demos'.format(episodes, demos))
        demos = utils.load_demos(demos_path)
        logger.info('loaded demos')
        if episodes > len(demos):
            raise ValueError("there are only {} low-level demos".format(len(low_level_demos)))
        low_level_demos.extend(demos[:episodes])
        logger.info('So far, {} demos loaded'.format(len(low_level_demos)))
    logger.info('loading instruction handler')
    instr_handler = InstructionHandler(low_level_demos, load_bert=False, save_path=None)
    logger.info("loaded pi_l model")
        
elif getattr(args, 'low_level_demos', None):
    low_level_demos_path = utils.get_demos_path(args.low_level_demos, None, None, valid=False)
    logger.info('loading demos')
    low_level_demos = utils.load_demos(low_level_demos_path)
    logger.info('loaded demos')
    if args.ll_episodes > len(low_level_demos):
        raise ValueError("there are only {} low-level demos".format(len(low_level_demos)))
    low_level_demos = low_level_demos[:args.ll_episodes]
    logger.info('loading instruction handler')
    instr_handler = InstructionHandler(low_level_demos, load_bert=False, save_path=os.path.join(os.path.splitext(low_level_demos_path)[0], "ih"))
    logger.info('loaded instruction handler')

if args.oracle:
    pi_l = None
else:
    logger.info("loading pi_l model")
    pi_l = ModelAgent(args.pi_l, None, done_classifier=args.done_classifier, argmax=True)
    logger.info("loaded pi_l model")

logger.info(f"collecting {args.episodes} training missions for {args.env}")
missions, subtasks = generate_subtask_breakdowns(args.env, pi_l, instr_handler, seed=args.seed, num_demos=args.episodes)
get_breakdown_stats(missions, subtasks)

logger.info(f"collecting {args.valid_episodes} validation missions for {args.env}")
valid_missions, valid_subtasks = generate_subtask_breakdowns(args.env, pi_l, instr_handler, seed=int(1e9), num_demos=args.valid_episodes, valid=True)
get_breakdown_stats(valid_missions, valid_subtasks)
