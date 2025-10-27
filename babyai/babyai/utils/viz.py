"""
This file allows BabyAI environments / policies to be conveniently visualized 
in the terminal or in a Jupyter notebook.
"""

from os import system, name 
from IPython.display import clear_output
from gym_minigrid.wrappers import *
from babyai.utils.agent import ModelAgent, BotAgent
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore, Back, Style, init
from termcolor import colored
import time

COLORS = {
        'R': Fore.RED,
        'G': Fore.GREEN,
        'B': Fore.BLUE,
        'P': Fore.MAGENTA,
        'Y': Fore.YELLOW,
        'Q': Fore.WHITE,
        'K': Fore.BLACK,
        ' ': Fore.RESET,
        }
SYMBOLS = {
    'A': '@ ',
    'K': '* ',
    'W': '+ ',
    'B': '# ',
    '>': '> ',
    '<': '< ',
    'V': 'v ',
    '^': '^ '
}

def clear(): 
    clear_output(wait=True)
    if name == 'nt': 
        _ = system('cls') 
    else: 
        _ = system('clear') 


def emph(text, color="B"):
    ascii_color = COLORS.get(color, Fore.BLACK)
    return ascii_color + Style.BRIGHT + text + Fore.BLACK + Style.RESET_ALL


def info(text, heading=None, heading_color="B"):
    if heading:
        return f"\t{emph(heading)}: {text}"
    else:
        return f"\t{text}"


def viz(env, show_env_name=True, show_mission=True, aux_info=None, mode="colored_text"):
    """ Visualize a BabyAI environment, optionally displaying the env name, mission, and
        arbitrary auxiliary information.

        Modes: colored_text, image
    """
    if "Parallel" in env.spec.id or "hrl" in dir(env):
        mission = env.gen_obs()[0]['mission']
    else:
        mission = env.gen_obs()['mission']
    if mode == "image":
        print()
        print(info(env.spec.id, heading="Env"))
        print(info(mission, heading="Mission"))
        print(aux_info)
        plt.imshow(env.render(mode="rgb_array"))
        plt.show()
    elif mode == "colored_text":
        env_str = env.__str__()
        env_str_ll = env_str.split("\n")
        # Clear out wrapper from plain text representation
        width = len(env_str_ll[1])
        env_str_ll[0] = env_str_ll[0][-width:]
        env_str_ll[-1] = env_str_ll[-1][:width]
        # Output the representation in appropriate colors
        for l, line in enumerate(env_str_ll):
            for i in range(0, len(line)-1, 2):
                if line[i] == " ":
                    print(2*line[i], end="")
                else:
                    print(emph(SYMBOLS.get(line[i], 2*line[i]), color=line[i+1]), end="")
            if show_env_name and l == len(env_str_ll)//2 - 3:
                print(info(env.spec.id, heading="Env"), end="")
            elif show_mission and l == len(env_str_ll)//2 - 1:
                print(info(mission, heading="Mission"), end="")
            elif aux_info and l == len(env_str_ll)//2 + 1:
                print(aux_info, end="")
            print()


def watch(agent, env, max_t=None, pause=.5, mode="colored_text", hrl=False, bot=False, clear_screen=True):
    """ Visualize an agent working through a BabyAI environment.

        Modes: colored_text, image, static
    """
    env.reset()
    if agent == "BOT":
        bot = True
        if not hrl:
            agent = BotAgent(env)
        else:
            agent = BotAgent(env.envs[0])
    if mode == "static":
        obs = env.gen_obs()
        x = [env.render(mode="rgb_array", highlight=False)]
        done = False
        if hrl:
            obs = obs[0]
        print()
        print(info(env.spec.id, "Env"))
        print(info(obs['mission'], "Mission"))
        print(info("", "Actions"), end="")
        t = 0
        while not done:
            t += 1
            if not bot:
                a = agent.act(obs)['action'].item()
            else:
                a = agent.act(obs)['action'].value
            print(a, end=" ")
            if t % 25 == 0:
                print('\n\t\t', end="")
            if hrl:
                a = [a]
            obs, reward, done, env_info = env.step(a)
            if hrl:
                obs = obs[0]
            x.append(env.render(mode="rgb_array", highlight=False))
            if t == max_t:
                break
        all = np.array(x)
        all_max = all.max(axis=0).astype(int)
        all_mean = all.mean(axis=0).astype(int)
        out = (0.5 * all_max).astype(int) + (0.5 * all_mean).astype(int)
        print()
        print(info(done, "Done"))
        print(info(reward, "Reward"))
        plt.imshow(out)
    else:
        obs = env.gen_obs()
        if hrl:
            obs = obs[0]
        viz(env, mode=mode)
        time.sleep(pause)
        done = False
        t = 0
        while not done:
            t += 1
            if not bot:
                a = agent.act(obs)['action'].item()
            else:
                a = agent.act(obs)['action'].value
            if hrl:
                a = [a]
            obs, reward, done, env_info = env.step(a)
            if clear_screen:
                clear()
            viz(env, aux_info=info(a, "Action") + info(reward, "Reward") + info(done, "Done") + info(env_info, "Info"), mode=mode)
            time.sleep(pause)
            if t == max_t:
                break
            if hrl:
                obs = obs[0]