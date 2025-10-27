from torch.multiprocessing import Process, Pipe
import gym
from tqdm import tqdm
import logging
import torch
from tqdm import tqdm
logger = logging.getLogger(__name__)
import concurrent.futures

# For multiprocessing
def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step": 
            obs, reward, done, info = env.step(data)
            if done:
                obs = env.reset()
            conn.send((obs, reward, done, info))
        elif cmd == "reset":
            obs = env.reset()
            conn.send(obs)
        else:
            raise NotImplementedError

# For multithreading
def thread(env, cmd, *args):
    if cmd == "step":
        obs, reward, done, info = env.step(args[0])
        if done:
            obs = env.reset()
        return obs, reward, done, info
    elif cmd == "reset":
        obs = env.reset()
        return obs
    else:
        raise NotImplementedError

class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs, use_procs=False):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.use_procs = use_procs

        if self.use_procs:
            self.locals = []
            self.processes = []
            for env in tqdm(self.envs[1:]):
                local, remote = Pipe()
                self.locals.append(local)
                p = Process(target=worker, args=(remote, env))
                p.daemon = True
                p.start()
                remote.close()
                self.processes.append(p)

    def reset(self):
        if self.use_procs:
            for local in self.locals:
                local.send(("reset", None))
            proc_results = []
            for local in self.locals:
                proc_results.append(local.recv())
            results = [self.envs[0].reset()] + proc_results
            # results = [self.envs[0].reset()] + [local.recv() for local in self.locals]
        else:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(thread, self.envs[i], "reset") for i in range(len(self.envs))]
                results = [f.result() for f in futures]
        return results

    def step(self, actions):
        if self.use_procs:
            for local, action in zip(self.locals, actions[1:]):
                local.send(("step", action))
            obs, reward, done, info = self.envs[0].step(actions[0])
            if done:
                obs = self.envs[0].reset()
            results = zip(*[(obs, reward, done, info)] + [local.recv() for local in self.locals])
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
                futures = [executor.submit(thread, self.envs[i], "step", actions[i]) for i in range(len(self.envs))]
                results = [f.result() for f in futures]
            results = zip(*results)
        return results

    def render(self):
        raise NotImplementedError

    def __del__(self):
        if self.use_procs:
            for p in self.processes:
                p.terminate()