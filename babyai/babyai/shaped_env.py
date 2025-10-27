import gym
import torch
import numpy as np
from copy import deepcopy
from torch.multiprocessing import Process, Pipe
import torch.nn.functional as F
import logging
import babyai.utils

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
                    ret.append((obs, reward, done, {}))
                else:
                    ret.append((None, 0, False, None))
            conn.send(ret)
        # reset()
        elif cmd == "reset":
            ret = []
            for env in envs:
                obs, info = env.reset()
                ret.append((obs, {}))
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
                        obs = env.reset()
                    ret.append((obs, reward, done, {}))
                else:
                    ret.append((None, 0, False, None))
            conn.send(ret)
        # reset()
        elif cmd == "reset":
            ret = []
            for env in envs:
                obs, info = env.reset()
                ret.append((obs, {}))
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


class ParallelShapedEnv(gym.Env):
    """Parallel environment that holds a list of environments and can
       evaluate a low-level policy for use in reward shaping.
    """

    def __init__(self,
                 envs,  # List of environments
                 pi_l=None,  # Low-level policy or termination classifier
                 done_action=None,  # Output of pi_l indicating done
                 instr_handler=None,  # InstructionHandler for low-level demos
                 reward_shaping=None,  # Reward shaping type
                 subtask_cls=None,  # Subtask relevance classifier
                 subtask_cls_preproc=None,  # Instruction preprocessor
                 subtask_online_ds=None,  # Dataset for subtask classifier
                 subtask_discount=1,  # Discount for done subtask count
                 learn_baseline_cls=None,  # LEARN baseline classifier
                 learn_baseline_preproc=None,  # LEARN baseline classifier
                 ):
        assert len(envs) >= 1, "No environment provided"
        self.envs = envs
        self.num_envs = len(self.envs)
        self.device = torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu")
        self.spec = deepcopy(self.envs[0].spec)
        self.spec.id = f"ParallelShapedEnv<{self.spec.id}>"
        self.env_name = self.envs[0].unwrapped.spec.id
        self.action_space = self.envs[0].action_space
        self.pi_l = pi_l
        self.done_action = done_action
        self.instr_handler = instr_handler
        if self.instr_handler:
            self.num_subtasks = self.instr_handler.D_l_size()
        self.reward_shaping = reward_shaping
        self.subtask_cls = subtask_cls
        self.subtask_cls_preproc = subtask_cls_preproc
        self.subtask_online_ds = subtask_online_ds
        self.subtask_discount = float(subtask_discount)
        self.learn_baseline_cls = learn_baseline_cls
        self.learn_baseline_preproc = learn_baseline_preproc

        if "BabyAI" in self.env_name:
            self.envs_per_proc = 64
        elif "BabyPANDA" in self.env_name:
            self.envs_per_proc = 1
        else:
            self.envs_per_proc = 64

        if self.reward_shaping in ["subtask_oracle_ordered"]:
            # Setup stacks to hold oracle subtasks
            self.stacks = [[] for _ in range(self.num_envs)]

        if self.reward_shaping in ["subtask_classifier_static",
                                   "subtask_classifier_online",
                                   "subtask_classifier_static_unclipped",
                                   "subtask_classifier_online_unclipped"]:
            # Setup arrays to keep track of which subtasks are completed
            # during episode, and past bonuses
            self.pi_l_already_done_relevant = np.array(
                [[False for j in range(self.num_subtasks)]
                 for i in range(self.num_envs)]
            )
            self.pi_l_already_done_all = np.array(
                [[False for j in range(self.num_subtasks)]
                 for i in range(self.num_envs)]
            )
            self.past_pi_l_done_discounted = np.array(
                [0. for i in range(self.num_envs)]
            )
            # Setup list to keep track of instructions to store for dataset
            self.tasks_instr = ["" for _ in range(self.num_envs)]
            # Setup array to record of which environments have succeeded
            # at the high-level task
            self.tasks_succeeded = np.array([False for _ in range(self.num_envs)])

        if self.reward_shaping in ["learn_baseline"]:
            # Setup array to record unnormalized action frequencies
            assert "Discrete" in str(type(self.envs[0].action_space))
            self.num_actions = self.envs[0].action_space.n
            self.action_freqs = np.array(
                [[0 for j in range(self.num_actions)]
                 for i in range(self.num_envs)]
            )

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
        if "BabyPANDA" in self.spec.id:
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
            if "BabyPANDA" in self.spec.id:
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
        for local in self.locals:
            res = local.recv()

        infos = []
        for j in range(len(res)):
            infos.append(res[j][1])
            if res[j][0] is not None:
                self.obss += [res[j][0]]
        # self.obss += local.recv()
        logger.info("completed resets")
        return infos

    def reset(self):
        """Reset all environments"""
        self.request_reset_envs()
        return [obs for obs in self.obss]

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

    def reset_pi_l(self):
        """Clear pi_l's memory (in case it is a recurrent policy)"""
        self.pi_l.analyze_feedback(None, 1)
        self.pi_l.on_reset()

    def reset_pi_l_partial(self, reset_mask):
        """Clear pi_l's memory for certain environments based on reset mask"""
        self.pi_l.analyze_feedback(None,
                                   torch.tensor(reset_mask).to(self.device).int().unsqueeze(1))

    def pop_masked(self, stacks, mask, allow_zero=False):
        if allow_zero:
            stacks = [stacks[i][1:] if mask[i]
                      else stacks[i] for i in range(len(stacks))]
        else:
            stacks = [stacks[i][1:] if len(stacks[i]) > 1 and mask[i]
                      else stacks[i] for i in range(len(stacks))]
        return stacks

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

        # Oracle
        if self.reward_shaping in ["subtask_oracle_ordered"]:
            self.pi_l_obss = deepcopy(self.obss)
            self.out_of_instr = np.array([False for _ in range(self.num_envs)])
            for i in range(self.num_envs):
                # For every newly reset environment, get a new stack
                if self.ts[i] == 0:
                    old_mission = self.pi_l_obss[i]['mission']
                    self.stacks[i] = self.instr_handler.get_oracle_stack(
                        old_mission, unlock="Unlock" in self.env_name)
                # For every environment, set change the mission of the
                # observation to pi_l to what's at the top of the stack
                if len(self.stacks[i]) > 0:
                    self.pi_l_obss[i]['mission'] = self.stacks[i][0]
                else:
                    self.out_of_instr[i] = True
            # Run pi_l on these observations and determine which
            # predict termination (ignoring those where the stack's empty)
            pi_l_eval = self.pi_l.act_batch(self.pi_l_obss,
                                            stop_mask=self.out_of_instr)
            pi_l_actions = pi_l_eval['action'].cpu().numpy()
            pi_l_done = (pi_l_actions == self.done_action) * \
                        (1 - self.out_of_instr)

        # LEARN Baseline
        elif self.reward_shaping in ["learn_baseline"]:
            for i in range(self.num_envs):
                if self.ts[i] == 0:
                    self.action_freqs[i] *= 0
            task_text = [self.obss[i]["mission"] for i in range(self.num_envs)]

        # Subtask classifier, static or learned online
        elif self.reward_shaping in ["subtask_classifier_static",
                                     "subtask_classifier_online",
                                     "subtask_classifier_static_unclipped",
                                     "subtask_classifier_online_unclipped",
                                     ]:
            self.pi_l_obss = [deepcopy(self.obss[i])
                              for i in range(self.num_envs)
                              for _ in range(self.num_subtasks)]
            for i in range(self.num_envs):
                # For every newly reset environment, add to the dataset
                # if task was successful (and classifier is learned online),
                # and reset arrays
                if self.ts[i] == 0:
                    old_mission = self.tasks_instr[i]
                    if self.reward_shaping in ["subtask_classifier_online",
                                               "subtask_classifier_online_unclipped"]:
                        if self.tasks_succeeded[i] and \
                                self.pi_l_already_done_all[i].sum() > 0:
                            self.subtask_online_ds.add_demos([
                                (old_mission,
                                 [(-1, np.where(self.pi_l_already_done_all[i])[0])]
                                 )
                            ])
                    self.pi_l_already_done_relevant[i] *= False
                    self.pi_l_already_done_all[i] *= False
                    self.past_pi_l_done_discounted[i] *= 0
                    self.tasks_succeeded[i] = False
                    self.tasks_instr[i] = self.obss[i]["mission"]
                # For every (environment, subtask) pair, set the mission
                # of pi_l's observation to the subtask instruction
                for j in range(self.num_subtasks):
                    self.pi_l_obss[i * self.num_subtasks + j]["mission"] = \
                        self.instr_handler.get_instruction(j)
            pi_l_eval = self.pi_l.act_batch(self.pi_l_obss, stop_mask=None)
            pi_l_actions = pi_l_eval["action"].cpu().numpy()
            pi_l_done = pi_l_actions == self.done_action
            pi_l_done = pi_l_done.reshape((self.num_envs, self.num_subtasks))
            # Just keep the instructions that weren't already done and relevant
            pi_l_done *= np.invert(self.pi_l_already_done_relevant)
            if pi_l_done.sum() > 0:
                # Preprocess the instructions for the tasks and completed
                # subtasks
                task_idx, subtask_idx = np.where(pi_l_done)
                task_text = [self.obss[i]["mission"] for i in task_idx]
                subtask_text = self.instr_handler.missions[subtask_idx]
                task_preproc = self.subtask_cls_preproc(task_text)
                subtask_preproc = self.subtask_cls_preproc(subtask_text)
                if self.reward_shaping in ["subtask_classifier_online",
                                           "subtask_classifier_online_unclipped"]:
                    task_preproc = task_preproc.to(self.device)
                    subtask_preproc = subtask_preproc.to(self.device)
                # Run them through the subtask classifier
                predicted_subtasks = self.subtask_cls(task_preproc, subtask_preproc) \
                    .round().detach().cpu().numpy().astype(bool)
                # Record them
                self.pi_l_already_done_all |= pi_l_done
                # Overwrite pi_l_done with only the done and relevant subtasks
                # and record them
                pi_l_done &= False
                for j in range(len(task_idx)):
                    if predicted_subtasks[j]:
                        pi_l_done[task_idx[j], subtask_idx[j]] = True
                        self.pi_l_already_done_relevant[task_idx[j],
                                                        subtask_idx[j]] = True

        # Make a step in the environment
        stop_mask = np.array([False for _ in range(self.num_envs)])
        obs, reward, done, info = self.request_step(actions_to_take, stop_mask)
        reward = np.array(reward)
        done_mask = np.array(done)

        # Add reward shaping information to info
        if self.reward_shaping in ["subtask_oracle_ordered"]:
            self.stacks = self.pop_masked(self.stacks, pi_l_done, allow_zero=True)
            to_reset = done | pi_l_done
            self.reset_pi_l_partial(to_reset)
            info = (pi_l_done.astype(int),
                    torch.tensor(pi_l_actions).to(self.device))

        elif self.reward_shaping in ["learn_baseline"]:
            prev_action_freqs = torch.as_tensor(
                np.nan_to_num(np.divide(self.action_freqs, self.ts[:, None]),
                              posinf=0)).float().to(self.device)
            for i in range(self.num_envs):
                self.action_freqs[i][actions_to_take[i]] += 1
            cur_action_freqs = torch.as_tensor(
                np.divide(self.action_freqs, self.ts[:, None] + 1)).float().to(self.device)
            task_preproc = self.learn_baseline_preproc(task_text).to(self.device)
            prev_pred = F.softmax(self.learn_baseline_cls(task_preproc, \
                                                          prev_action_freqs)[1], dim=-1)
            cur_pred = F.softmax(self.learn_baseline_cls(task_preproc, \
                                                         cur_action_freqs)[1], dim=-1)
            prev_potential = prev_pred[:, 1] - prev_pred[:, 0]
            cur_potential = cur_pred[:, 1] - cur_pred[:, 0]
            info = (np.stack((prev_potential.detach().cpu().numpy(), \
                              cur_potential.detach().cpu().numpy()), axis=-1), None)

        elif self.reward_shaping in ["subtask_classifier_static",
                                     "subtask_classifier_online"]:
            # Reset all pi_l models for an environment if any subtasks
            # predict termination
            to_reset = (done | pi_l_done.sum(1) > 0).repeat(self.num_subtasks)
            self.reset_pi_l_partial(to_reset)

            pi_l_done_count_clipped = pi_l_done.sum(1).clip(0, 1)
            self.past_pi_l_done_discounted += pi_l_done_count_clipped
            info = (np.stack((pi_l_done_count_clipped, self.past_pi_l_done_discounted), axis=-1),
                    torch.tensor(pi_l_actions).to(self.device))
            self.tasks_succeeded = reward > 0
            self.past_pi_l_done_discounted *= 1. / self.subtask_discount

        elif self.reward_shaping in ["subtask_classifier_static_unclipped",
                                     "subtask_classifier_online_unclipped"]:
            # Reset all pi_l models for an environment if any subtasks
            # predict termination
            to_reset = (done | pi_l_done.sum(1) > 0).repeat(self.num_subtasks)
            self.reset_pi_l_partial(to_reset)

            pi_l_done_count = pi_l_done.sum(1)
            self.past_pi_l_done_discounted += pi_l_done_count
            info = (np.stack((pi_l_done_count, self.past_pi_l_done_discounted), axis=-1),
                    torch.tensor(pi_l_actions).to(self.device))
            self.tasks_succeeded = reward > 0
            self.past_pi_l_done_discounted *= 1. / self.subtask_discount

        self.ts += 1
        self.ts[done_mask] *= 0

        return [obs for obs in self.obss], reward, done_mask, info
