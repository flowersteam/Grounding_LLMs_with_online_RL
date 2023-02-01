#!/usr/bin/env python3

"""
Generate a set of agent demonstrations.

The agent can either be a trained model or the heuristic expert (bot).

Demonstration generation can take a long time, but it can be parallelized
if you have a cluster at your disposal. Provide a script that launches
make_agent_demos.py at your cluster as --job-script and the number of jobs as --jobs.
"""

import gym
import sys
import os

for p in sys.path:
    print(p)

import gym_minigrid.window

import time
import numpy as np

import torch
import pandas as pd
import matplotlib.pyplot as plt

from collections import deque

import babyai.utils as utils

from nn.GPTJ_with_value_head import GPTJForCausalLMWithValueModel_quantized, choosing_subgoals
import transformers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TILE_PIXELS = 32


def generate_traj(model, env, seed):
    """
    generate a simple trajectory with windows display
    """
    env_name = env
    # Generate environment
    env = gym.make(env)

    env.window = gym_minigrid.window.Window('gym_minigrid')
    env.window.show(block=False)

    utils.seed(seed)
    traj = np.zeros((env.width, env.height), dtype=np.int32)

    number_model = 1
    model_i = model.format(number_model)

    utils.seed(seed)

    agent = torch.load('storage/models/' + model_i + '/model.pt')
    obss_preprocessor = utils.ObssPreprocessor(model_i, env.observation_space, None)
    if torch.cuda.is_available():
        agent.eval()
        agent.cuda()

    memory = torch.zeros(1, agent.memory_size, device=device)
    done = False
    env.seed(seed)
    obs, infos = env.reset()
    print('mission: {}'.format(obs['mission']))
    print("graph: \n")
    for t in infos:
        print(t)
    for i in range(100):
        env.render('human')
    while not done:

        preprocessed_obs = obss_preprocessor([obs], device=device)
        model_results = agent(preprocessed_obs, memory)

        dist = model_results['dist']
        memory = model_results['memory']

        action = torch.argmax(dist.probs).cpu().numpy()
        print(action)
        new_obs, reward, done, infos = env.step(action)
        traj[env.agent_pos[0]][env.agent_pos[1]] += 1
        obs = new_obs

        print(" ")
        print("graph: \n")
        for t in infos:
            print(t)
        for i in range(100):
            env.render('human')


def generate_prompt(goal, deque_obs, deque_actions):
    ldo = len(deque_obs)
    lda = len(deque_actions)
    head_prompt = "Possible action of the agent: go forward, turn right, turn left "
    """modify_goal = "go face" + goal[5:]
    g = " \n Goal of the agent: {}".format(modify_goal)"""
    g = " \n Goal of the agent: {}".format(goal)
    obs = ""
    for i in range(ldo):
        obs += " \n Observation {}: ".format(i)
        for d_obs in deque_obs[i]:
            obs += "{}, ".format(d_obs)
        obs += "\n Action {}: ".format(i)
        if i < lda:
            obs += "{}".format(deque_actions[i])
    return head_prompt + g + obs


def traj_LLM(model, env, seed):
    """
    measure 0-shot performance of a LLM
    """
    env_name = env
    # Generate environment
    env = gym.make(env)

    env.window = gym_minigrid.window.Window('gym_minigrid')
    env.window.show(block=False)

    utils.seed(seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained('storage/models/GPTJ', padding_side="left")
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    subgoals = {0: "turn left",
                1: "turn right",
                2: "go forward"}
    subgoals_tokenized = {0: tokenizer(["turn left"], return_tensors='pt').to(device),
                          1: tokenizer(["turn right"], return_tensors='pt').to(device),
                          2: tokenizer(["go forward"], return_tensors='pt').to(device)}

    obs_queue = deque([], maxlen=3)
    acts_queue = deque([], maxlen=2)

    done = False

    env.seed(seed)
    obs, infos = env.reset()
    print('mission: {}'.format(obs['mission']))
    obs_queue.append(infos)
    """print("graph: \n")
    for t in infos:
        print(t)"""
    for i in range(100):
        env.render('human')
    while not done:

        prompt = [generate_prompt(goal=obs['mission'], deque_obs=obs_queue, deque_actions=acts_queue)]
        print(prompt)
        prompt_inputs = tokenizer(prompt, return_tensors='pt', padding=True).to(device)
        eps = 0
        sbg = choosing_subgoals(model,
                                prompt=prompt_inputs['input_ids'],
                                attention_mask=prompt_inputs['attention_mask'],
                                subgoal_tokenized=subgoals_tokenized,
                                eps=eps)

        action = sbg[0]
        acts_queue.append(subgoals[action])
        print("{} -> {}".format(sbg[0], subgoals[action]))
        new_obs, reward, done, infos = env.step(action)
        obs = new_obs
        obs_queue.append(infos)
        """print(" ")
        print("graph: \n")
        for t in infos:
            print(t)"""
        for i in range(100):
            env.render('human')
    env.close()


def perf_LLM(model, env, eps=0., random=False, min_seed=0, max_seed=100):
    """
    measure 0-shot performance of a LLM
    """
    env_name = env
    # Generate environment
    env = gym.make(env)

    success = 0
    chosen_moves = np.zeros(3)
    seed_tab = []
    average_reward = 0

    for seed in range(min_seed, max_seed):
        utils.seed(seed)

        if not random:
            tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", padding_side="left")
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
            subgoals = {0: "turn left",
                        1: "turn right",
                        2: "go forward"}
            subgoals_tokenized = {0: tokenizer(["turn left"], return_tensors='pt').to(device),
                                  1: tokenizer(["turn right"], return_tensors='pt').to(device),
                                  2: tokenizer(["go forward"], return_tensors='pt').to(device)}

            obs_queue = deque([], maxlen=3)
            acts_queue = deque([], maxlen=2)

        done = False

        env.seed(seed)
        obs, infos = env.reset()
        print('seed {}, mission: {}'.format(seed, obs['mission']))
        if not random:
            obs_queue.append(infos)

        while not done:
            if not random:
                prompt = [generate_prompt(goal=obs['mission'], deque_obs=obs_queue, deque_actions=acts_queue)]
                # print(prompt)
                prompt_inputs = tokenizer(prompt, return_tensors='pt', padding=True).to(device)
                sbg = choosing_subgoals(model,
                                        prompt=prompt_inputs['input_ids'],
                                        attention_mask=prompt_inputs['attention_mask'],
                                        subgoal_tokenized=subgoals_tokenized,
                                        eps=eps)

                action = int(sbg[0])
            else:
                action = np.random.randint(0, 3)
            chosen_moves[action] += 1
            if not random:
                acts_queue.append(subgoals[action])
                # print("{} -> {}".format(sbg[0], subgoals[action]))
            new_obs, reward, done, infos = env.step(action)

            obs = new_obs
            if not random:
                obs_queue.append(infos)

            if reward > 0:
                success += 1
                average_reward += reward
                seed_tab.append(seed)

        if seed % 10 == 0:
            print("success: {}".format(success/((seed-min_seed)+1)))
            print("average_reward: {}".format(average_reward/((seed-min_seed)+1)))
            print("chosen_moves: {}".format(chosen_moves/np.sum(chosen_moves)))
            # print(seed_tab)
            print(" ")

    return success/(max_seed-min_seed), average_reward/(max_seed-min_seed), chosen_moves/np.sum(chosen_moves), seed_tab


# generate_traj('QG_QA/PNL-adjusted-train_env-PNL_no_answer-lambda_24-model-2_10-seed_{}', 'BabyAI-PutNextLocal-v0', 1)

gpt = GPTJForCausalLMWithValueModel_quantized.from_pretrained("storage/models/GPTJ/GPTJForCausalLMWithValueModel_quantized")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpt.to(device)

"""success, average_reward, chosen_moves, seed_tab = perf_LLM(gpt, 'BabyAI-GoToLocal-v0',
                                                           eps=0.5,
                                                           random=False,
                                                           min_seed=0,
                                                           max_seed=150)
print("success: {}".format(success))
print("average_reward: {}".format(average_reward))
print("chosen_moves: {}".format(chosen_moves))
print(seed_tab)"""

for i in [0, 7, 9, 18, 19, 24, 39, 51, 58, 62, 67, 85, 86, 89, 98]:
    print("SEED:{}".format(i))
    traj_LLM(gpt, 'BabyAI-GoToLocal-v0', i)

"""success, chosen_moves, seed_tab = perf_LLM(None, 'BabyAI-GoToLocal-v0', eps=0., random=True, max_seed=100)
print("success: {}".format(success))
print("chosen_moves: {}".format(chosen_moves))
print(seed_tab)"""