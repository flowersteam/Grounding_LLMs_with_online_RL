import gym
import torch
import numpy as np
from copy import deepcopy
from torch.multiprocessing import Process, Pipe

import logging
import babyai.utils as utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def multi_worker(conn, envs):
    """Target for a subprocess that handles a set of envs"""
    while True:
        cmd, data = conn.recv()
        # step(actions, stop_mask)
        if cmd == "step":
            ret = []
            for env, a, stopped in zip(envs, data[0], data[1]):
                if not stopped:
                    obs, reward, done, info = env.step(a)
                    if done:
                        obs, info = env.reset()
                    ret.append((obs, reward, done, info))
                else:
                    ret.append((None, 0, False, None))
            conn.send(ret)
        # reset()
        elif cmd == "reset":
            ret = []
            for env in envs:
                obs, info = env.reset()
                ret.append((obs, info))
            conn.send(ret)
        # render_one()
        elif cmd == "render_one":
            mode, highlight = data
            ret = envs[0].render(mode, highlight)
            conn.send(ret)
            # __str__()
        elif cmd == "__str__":
            ret = str(envs[0])
            conn.send(ret)
        else:
            raise NotImplementedError


def multi_worker_cont(conn, envs):
    """Target for a subprocess that handles a set of envs"""
    while True:
        cmd, data = conn.recv()
        # step(actions, stop_mask)
        if cmd == "step":
            ret = []
            for env, a, stopped in zip(envs, data[0], data[1]):
                if not stopped:
                    obs, reward, done, info = env.step(action=a)
                    if done:
                        obs, info = env.reset()
                    ret.append((obs, reward, done, info))
                else:
                    ret.append((None, 0, False, None))
            conn.send(ret)
        # reset()
        elif cmd == "reset":
            ret = []
            for env in envs:
                ret.append(env.reset())
            conn.send(ret)
        # render_one()
        elif cmd == "render_one":
            mode = data
            ret = envs[0].render(mode)
            conn.send(ret)
            # __str__()
        elif cmd == "__str__":
            ret = str(envs[0])
            conn.send(ret)
        else:
            raise NotImplementedError


class ParallelEnv(gym.Env):
    """Parallel environment that holds a list of environments and can
       evaluate a low-level policy for use in reward shaping.
    """

    def __init__(self,
                 envs,  # List of environments
                 ):
        assert len(envs) >= 1, "No environment provided"
        self.envs = envs
        self.num_envs = len(self.envs)
        self.device = torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu")
        self.spec = deepcopy(self.envs[0].unwrapped.spec)
        self.spec_id = f"ParallelShapedEnv<{self.spec.id}>"
        self.env_name = self.envs[0].unwrapped.spec.id
        self.action_space = self.envs[0].action_space

        if "BabyAI" in self.env_name:
            self.envs_per_proc = 64
        elif "BabyPANDA" in self.env_name:
            self.envs_per_proc = 1
        else:
            self.envs_per_proc = 64

        # Setup arrays to hold current observation and timestep
        # for each environment
        self.obss = []
        self.ts = np.array([0 for _ in range(self.num_envs)])

        # Spin up subprocesses
        self.locals = []
        self.processes = []
        self.start_processes()

    def __len__(self):
        return self.num_envs

    def __str__(self):
        self.locals[0].send(("__str__", None))
        return f"<ParallelShapedEnv<{self.locals[0].recv()}>>"

    def __del__(self):
        for p in self.processes:
            p.terminate()

    def gen_obs(self):
        return self.obss

    def render(self, mode="rgb_array", highlight=False):
        """Render a single environment"""
        if "BabyPANDA" in self.spec_id:
            self.locals[0].send(("render_one", mode))
        else:
            self.locals[0].send(("render_one", (mode, highlight)))
        return self.locals[0].recv()

    def start_processes(self):
        """Spin up the num_envs/envs_per_proc number of processes"""
        logger.info(f"spinning up {self.num_envs} processes")
        for i in range(0, self.num_envs, self.envs_per_proc):
            local, remote = Pipe()
            self.locals.append(local)
            if "BabyPANDA" in self.spec_id:
                p = Process(target=multi_worker_cont,
                            args=(remote, self.envs[i:i + self.envs_per_proc]))
            else:
                p = Process(target=multi_worker,
                            args=(remote, self.envs[i:i + self.envs_per_proc]))
            p.daemon = True
            p.start()
            remote.close()
            self.processes.append(p)
        logger.info("done spinning up processes")

    def request_reset_envs(self):
        """Request all processes to reset their envs"""
        logger.info("requesting resets")
        for local in self.locals:
            local.send(("reset", None))
        self.obss = []
        logger.info("requested resets")

        infos = []
        for local in self.locals:
            res = local.recv()

            for j in range(len(res)):
                infos.append(res[j][1])
                if res[j][0] is not None:
                    self.obss += [res[j][0]]
            # self.obss += local.recv()
        logger.info("completed resets")
        return infos

    def reset(self):
        """Reset all environments"""
        infos = self.request_reset_envs()
        return [obs for obs in self.obss], infos

    def request_step(self, actions, stop_mask):
        """Request processes to step corresponding to (primitive) actions
           unless stop mask indicates otherwise"""
        for i in range(0, self.num_envs, self.envs_per_proc):
            self.locals[i // self.envs_per_proc].send(
                ("step", [actions[i:i + self.envs_per_proc],
                          stop_mask[i:i + self.envs_per_proc]])
            )
        results = []
        for i in range(0, self.num_envs, self.envs_per_proc):
            res = self.locals[i // self.envs_per_proc].recv()
            for j in range(len(res)):
                results.append(res[j])
                if results[-1][0] != None:
                    self.obss[i + j] = results[-1][0]
        return zip(*results)

    def step(self, actions):
        """Complete a step and evaluate low-level policy / termination
           classifier as needed depending on reward shaping scheme.
           
           Returns:  obs: list of environment observations,
                     reward: np.array of extrinsic rewards,
                     done: np.array of booleans,
                     info: depends on self.reward_shaping. Output can be used
                           to shape the reward.
        """
        # Make sure input is numpy array
        if type(actions) != np.ndarray:
            if type(actions) == list or type(actions) == int:
                actions = np.array(actions)
            elif type(actions) == torch.Tensor:
                actions = actions.cpu().numpy()
            else:
                raise TypeError
        actions_to_take = actions.copy()

        # Make a step in the environment
        stop_mask = np.array([False for _ in range(self.num_envs)])
        obs, reward, done, info = self.request_step(actions_to_take, stop_mask)
        reward = np.array(reward)
        done_mask = np.array(done)

        self.ts += 1
        self.ts[done_mask] *= 0

        return [obs for obs in self.obss], reward, done_mask, info
