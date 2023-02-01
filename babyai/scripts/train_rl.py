#!/usr/bin/env python3

"""
Script to train the agent through reinforcement learning.
"""

import os
import logging
import csv
import json
import gym
import time
import datetime
import torch

import babyai
import babyai.utils as utils
import babyai.rl
from babyai.arguments import ArgumentParser
from babyai.model import ACModel
from babyai.utils.agent import ModelAgent
from gym_minigrid.wrappers import FullyObsImgDirWrapper, FullyObsImgEgoWrapper
from babyai.shaped_env import ParallelShapedEnv
from gym import spaces
from instruction_handler import InstructionHandler
from subtask_prediction import SubtaskPrediction, SubtaskDataset
from colorama import Fore, Back, Style

if __name__ == "__main__":

    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("--algo", default='ppo',
                        help="algorithm to use (default: ppo)")
    parser.add_argument("--discount", type=float, default=0.99,
                        help="discount factor (default: 0.99)")
    parser.add_argument("--reward-scale", type=float, default=20.,
                        help="Reward scale multiplier")
    parser.add_argument("--gae-lambda", type=float, default=0.99,
                        help="lambda coefficient in GAE formula (default: 0.99, 1 means no gae)")
    parser.add_argument("--value-loss-coef", type=float, default=0.5,
                        help="value loss term coefficient (default: 0.5)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="maximum norm of gradient (default: 0.5)")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="clipping epsilon for PPO (default: 0.2)")
    parser.add_argument("--ppo-epochs", type=int, default=4,
                        help="number of epochs for PPO (default: 4)")
    parser.add_argument("--save-interval", type=int, default=50,
                        help="number of updates between two saves (default: 50, 0 means no saving)")
    parser.add_argument("--full-obs", action="store_true", default=False,
                        help="use full observations of the environment")
    parser.add_argument("--ego", action="store_true", default=False,
                        help="use egocentric full observations")
    parser.add_argument("--pi-l", default=None,
                        help="model to use for low-level policy")
    parser.add_argument("--hrl", default=None,
                        help="either 'vanilla', 'shape', or a hierarchical rl type (deprecated)")
    parser.add_argument("--N", type=int, default=1,
                        help="hierarchical timestep")
    parser.add_argument("--T", type=int, default=0,
                        help="number of steps per instruction in HRL (0 means to termination)")
    parser.add_argument("--demos", default=None,
                        help="demos filename (REQUIRED or demos-origin or multi-demos required)")
    parser.add_argument("--demos-origin", required=False,
                        help="origin of the demonstrations: human | agent (REQUIRED or demos or multi-demos required)")
    parser.add_argument("--episodes", type=int, default=0,
                        help="number of episodes of demonstrations to use"
                            "(default: 0, meaning all demos)")
    parser.add_argument("--multi-demos", nargs='*', default=None,
                        help="demos filenames for envs to train on (REQUIRED when multi-env is specified)")
    parser.add_argument("--multi-episodes", type=int, nargs='*', default=None,
                        help="number of episodes of demos to use from each file (REQUIRED when multi-env is specified)")

    parser.add_argument("--sampling-temperature", type=float, default=1,
                        help="softmax temperature to use when sampling from action distribution")
    parser.add_argument("--oracle-rate", type=float, default=1,
                        help="rate at which hierarchical oracle option is non-null")
    parser.add_argument("--reward-shaping", type=str, default="multiply",
                        help="apply reward shaping")
    parser.add_argument("--pi-l-scale", type=float, default=1.,
                        help="Reshaped pi-l multiplier")
    parser.add_argument("--pi-l-scale-2", type=float, default=1.,
                        help="Another reshaped pi-l multiplier (e.g. for penalties)")

    parser.add_argument("--high-level-demos", default=None,
                        help="demos filename")
    parser.add_argument("--hl-episodes", type=int, default=0,
                        help="number of high-level episodes of demonstrations to use"
                            "(default: 0, meaning all demos)")
    parser.add_argument("--subtask-model", default=None,
                        help="model to use for subtask prediction")
    parser.add_argument("--subtask-arch", default=None,
                        help="architecture of subtask model")
    parser.add_argument("--subtask-pretrained-model", default=None,
                        help="pretrained subtask model")
    parser.add_argument("--subtask-hl-demos", default=None,
                        help="demos for online subtask training (only validation used)")
    parser.add_argument("--subtask-val-episodes", type=int, default=None,
                        help="number of validation demos to use for the subtask model")
    parser.add_argument("--subtask-batch-size", type=int, default=None,
                        help="batch size for subtask model")
    parser.add_argument("--subtask-update-rate", type=int, default=None,
                        help="rate at which subtask predictor is updated")
    parser.add_argument("--subtask-updates", type=int, default=None,
                        help="number of gradient steps")
    parser.add_argument("--subtask-discount", type=float, default=1.,
                        help="discount (un)applied when removing subtask bonuses")

    parser.add_argument("--done-classifier", action="store_true", default=False,
                        help="whether pi_l is actually a binary termination classifier")

    parser.add_argument("--number-actions", type=int, default=None,
                        help="nbr actions can be done more than 7, if more than 7 all additional actions are the action done in order to study the effect of useless actions")

    parser.add_argument("--learn-baseline", default=None,
                        help="model to use for LEARN baseline classifier")

    parser.add_argument("--debug", action="store_true", default=False,
                        help="whether to run RL in debug mode")

    args = parser.parse_args()

    utils.seed(args.seed)

    # Generate environments

    envs = []
    for i in range(args.procs):
        env = gym.make(args.env)

        env.seed(100 * args.seed + i)
        if args.full_obs:
            if args.ego:
                env = FullyObsImgEgoWrapper(env)
            else:
                env = FullyObsImgDirWrapper(env)
        envs.append(env)

    # Define model name
    suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    instr = args.instr_arch if args.instr_arch else "noinstr"
    mem = "mem" if not args.no_mem else "nomem"
    model_name_parts = {
        'env': args.env,
        'algo': args.algo,
        'arch': args.arch,
        'instr': instr,
        'mem': mem,
        'seed': args.seed,
        'info': '',
        'coef': '',
        'suffix': suffix}
    default_model_name = "{env}_{algo}_{arch}_{instr}_{mem}_seed{seed}{info}{coef}_{suffix}".format(**model_name_parts)
    if args.pretrained_model:
        default_model_name = args.pretrained_model + '_pretrained_' + default_model_name
    elif args.hrl:
        if args.pi_l is not None:
            default_model_name = args.pi_l + '_pi_l_' + default_model_name
    args.model = args.model.format(**model_name_parts) if args.model else default_model_name

    utils.configure_logging(args.model)
    logger = logging.getLogger(__name__)

    # Define logger and Tensorboard writer and CSV writer
    header = (["update", "episodes", "frames", "FPS", "duration"]
            + ["return_" + stat for stat in ['mean', 'std', 'min', 'max']]
            + ["success_rate"]
            + ["reshaped_return_" + stat for stat in ['mean', 'std', 'min', 'max']]
            + ["reshaped_return_bonus_" + stat for stat in ['mean', 'std', 'min', 'max']]
            + ["num_frames_" + stat for stat in ['mean', 'std', 'min', 'max']]
            + ["entropy", "value", "policy_loss", "value_loss", "loss", "grad_norm"])

    writer = None
    if args.tb:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(utils.get_log_dir(args.model))
    if args.wb:
        import wandb
        wandb.init(project="ella", name=args.model)
        wandb.config.update(args)
        writer = wandb

    csv_path = os.path.join(utils.get_log_dir(args.model), 'log.csv')
    first_created = not os.path.exists(csv_path)
    # we don't buffer data going in the csv log, cause we assume
    # that one update will take much longer that one write to the log
    csv_writer = csv.writer(open(csv_path, 'a', 1))
    if first_created:
        csv_writer.writerow(header)

    # Define obss preprocessor
    if 'emb' in args.arch:
        obss_preprocessor = utils.IntObssPreprocessor(args.model, envs[0].observation_space, args.pretrained_model)
    elif args.full_obs and not args.ego:
        obss_preprocessor = utils.ObssDirPreprocessor(args.model, envs[0].observation_space, args.pretrained_model)
    elif 'cont' in args.arch:
        obss_preprocessor = utils.ObssContPreprocessor(args.model, envs[0].observation_space, args.pretrained_model)
    else:
        obss_preprocessor = utils.ObssPreprocessor(args.model, envs[0].observation_space, args.pretrained_model)

    pi_l_agent = None
    instr_handler = None

    # Load the instruction handler from demonstrations
    if args.hrl is not None:
        if getattr(args, 'multi_demos', None):
            train_demos = []
            for demos, episodes in zip(args.multi_demos, args.multi_episodes):
                demos_path = utils.get_demos_path(demos, None, None, valid=False)
                logger.info('loading {} of {} demos'.format(episodes, demos))
                train_demos = utils.load_demos(demos_path)
                logger.info('loaded demos')
                if episodes > len(train_demos):
                    raise ValueError("there are only {} train demos in {}".format(len(train_demos), demos))
                train_demos.extend(train_demos[:episodes])
                logger.info('So far, {} demos loaded'.format(len(self.train_demos)))
            logger.info('Loaded all demos')
        elif getattr(args, 'demos', None):
            demos_path = utils.get_demos_path(args.demos, None, args.demos_origin, valid=False)
            demos_path_valid = utils.get_demos_path(args.demos, None, args.demos_origin, valid=True)
            logger.info('loading demos')
            train_demos = utils.load_demos(demos_path)
            logger.info('loaded demos')
            if args.episodes:
                if args.episodes > len(train_demos):
                    raise ValueError("there are only {} train demos".format(len(train_demos)))
                train_demos = train_demos[:args.episodes]
        logger.info('loading instruction handler')
        if args.hrl != "vanilla":
            instr_handler = InstructionHandler(train_demos, load_bert="projection" in args.hrl, save_path=os.path.join(os.path.splitext(demos_path)[0], "ih"))
        logger.info('loading instruction handler')

    if getattr(args, 'high_level_demos', None):
        hl_demos_path = utils.get_demos_path(args.high_level_demos, args.env, args.demos_origin, valid=False)
        logger.info('loading high-level demos')
        hl_demos = utils.load_demos(hl_demos_path)
        logger.info('loaded high-level demos')
        if args.hl_episodes:
            if args.hl_episodes > len(hl_demos):
                raise ValueError("there are only {} high-level demos".format(len(hl_demos)))
            hl_demos = hl_demos[:args.hl_episodes]

    # Load low-level model (low-level policy or termination classifier)
    if args.hrl is not None and args.hrl != "vanilla":
        pi_l_agent = ModelAgent(args.pi_l, None, argmax=True)
        logger.info("loaded pi_l models")

    # Initialize datasets / models used for shaping
    if args.reward_shaping in ["subtask_classifier_static"]:
        subtask_model = utils.load_model(args.subtask_model)
        subtask_model_preproc = utils.InstructionOnlyPreprocessor(args.subtask_model, load_vocab_from=args.subtask_model)
        subtask_dataset = None
    elif args.reward_shaping in ["subtask_classifier_online", "subtask_classifier_online_unclipped"]:
        args.subtask_model = args.model + "_subtask"
        subtask_prediction = SubtaskPrediction(args, online_args=True)
        subtask_model = subtask_prediction.model
        subtask_model_preproc = subtask_prediction.instr_preprocessor
        subtask_dataset = SubtaskDataset()
    else:
        subtask_model = None
        subtask_model_preproc = None
        subtask_dataset = None
    learn_baseline_cls = None
    learn_baseline_preproc = None
    if args.reward_shaping in ['learn_baseline']:
        learn_baseline_cls = utils.load_model(args.learn_baseline)
        if torch.cuda.is_available():
            learn_baseline_cls.cuda()
        learn_baseline_preproc = utils.InstructionOnlyPreprocessor(args.learn_baseline, load_vocab_from=args.learn_baseline)

    # Adjust action space if necessary
    if args.hrl is not None:
        if envs[0].action_space.__class__.__name__ == "Discrete":
            if args.number_actions is None:
                A = envs[0].action_space.n
            else:
                A = int(args.number_actions)
            action_space = spaces.Discrete(A)
            logger.info("setting hrl to {}; |A| = {}".format(args.hrl, action_space.n))
            if args.done_classifier:
                done_action = 1
            else:
                done_action = envs[0].actions.done
        else:
            A = envs[0].action_space.shape[0]
            action_space = envs[0].action_space
            done_action = 1

        # Create vectorized environment
        envs = ParallelShapedEnv(envs, pi_l=pi_l_agent, done_action=done_action,
                    instr_handler=instr_handler, reward_shaping=args.reward_shaping,
                    subtask_cls=subtask_model, subtask_cls_preproc=subtask_model_preproc,
                    subtask_online_ds=subtask_dataset, subtask_discount=args.subtask_discount,
                    learn_baseline_cls=learn_baseline_cls, learn_baseline_preproc=learn_baseline_preproc)
    else:
        action_space = envs[0].action_space

    # Define actor-critic model
    logger.info("loading ACModel")
    acmodel = utils.load_model(args.model, raise_not_found=False)
    if acmodel is None:
        if args.pretrained_model:
            acmodel = utils.load_model(args.pretrained_model, raise_not_found=True)
        else:
            acmodel = ACModel(obss_preprocessor.obs_space, action_space,
                            args.image_dim, args.memory_dim, args.instr_dim,
                            not args.no_instr, args.instr_arch, not args.no_mem, args.arch)
    logger.info("loaded ACModel")
    obss_preprocessor.vocab.save()
    utils.save_model(acmodel, args.model, writer)

    if torch.cuda.is_available():
        acmodel.cuda()

    # Set reward shaping function
    def bonus_penalty(_0, _1, reward, _2, info):
        if info[0] > 0:
            return [args.reward_scale * reward + args.pi_l_scale * max(info[0], 1), args.pi_l_scale * max(info[0], 1)]
        elif info[1] > 0:
            return [args.reward_scale * reward - args.pi_l_scale_2 * max(info[1], 1), -args.pi_l_scale_2 * max(info[1], 1)]
        else:
            return [args.reward_scale * reward, 0]

    if args.reward_shaping == "multiply":
        reshape_reward = lambda _0, _1, reward, _2, _3: [args.reward_scale * reward, 0]

    def subtask_shaping(_0, _1, reward, _2, info):
        if reward > 0:
            return [args.reward_scale * reward + args.pi_l_scale * info[0] - args.pi_l_scale * info[1],
                    args.pi_l_scale * info[0] - args.pi_l_scale * info[1]]
        else:
            return [args.pi_l_scale * info[0],
                    args.pi_l_scale * info[0]]

    def learn_baseline_shaping(_0, _1, reward, _2, info):
        return [args.reward_scale * reward + args.pi_l_scale * (args.subtask_discount * info[1] - info[0]),
                args.subtask_discount * info[1] - info[0]]

    if args.reward_shaping in ["subtask_oracle_ordered",
                               "subtask_classifier_static",
                               "subtask_classifier_online",
                               "subtask_classifier_static_unclipped",
                               "subtask_classifier_online_unclipped"]:
        reshape_reward = subtask_shaping

    elif args.reward_shaping in ["learn_baseline"]:
        reshape_reward = learn_baseline_shaping

    # Define actor-critic algorithm
    if args.algo == "ppo":
        algo = babyai.rl.PPOAlgo(envs, acmodel, args.frames_per_proc, args.discount, args.lr, args.beta1, args.beta2,
                                args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.ppo_epochs, args.batch_size, obss_preprocessor,
                                reshape_reward, use_penv=False, sampling_temperature=args.sampling_temperature,
                                debug=args.debug)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    # When using extra binary information, more tensors (model params) are initialized compared to when we don't use that.
    # Thus, there starts to be a difference in the random state. If we want to avoid it, in order to make sure that
    # the results of supervised-loss-coef=0. and extra-binary-info=0 match, we need to reseed here.

    utils.seed(args.seed)

    # Restore training status
    status_path = os.path.join(utils.get_log_dir(args.model), 'status.json')
    if os.path.exists(status_path):
        with open(status_path, 'r') as src:
            status = json.load(src)
    else:
        status = {'i': 0,
                'num_episodes': 0,
                'num_frames': 0}


    logger.info('COMMAND LINE ARGS:')
    logger.info(args)
    logger.info("CUDA available: {}".format(torch.cuda.is_available()))

    # Train model

    total_start_time = time.time()
    best_success_rate = 0
    best_mean_return = 0
    test_env_name = args.env

    logger.info("starting training")
    while status['num_frames'] < args.frames:

        # Update parameters
        update_start_time = time.time()
        logs = algo.update_parameters()
        update_end_time = time.time()

        status['num_frames'] += logs["num_frames"]
        status['num_episodes'] += logs['episodes_done']
        status['i'] += 1

        # Print logs
        if status['i'] % args.log_interval == 0:
            total_ellapsed_time = int(time.time() - total_start_time)
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = datetime.timedelta(seconds=total_ellapsed_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            success_per_episode = utils.synthesize(
                [1 if r > 0 else 0 for r in logs["return_per_episode"]])
            reshaped_return_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            reshaped_return_bonus_per_episode = utils.synthesize(logs["reshaped_return_bonus_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            data = [status['i'], status['num_episodes'], status['num_frames'],
                    fps, total_ellapsed_time,
                    *return_per_episode.values(),
                    success_per_episode['mean'],
                    *reshaped_return_per_episode.values(),
                    *reshaped_return_bonus_per_episode.values(),
                    *num_frames_per_episode.values(),
                    logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"],
                    logs["loss"], logs["grad_norm"]]


            format_str = ("\nUpdate: {} | Episodes Done: {} | Frames Seen: {:06} | FPS: {:04.0f} | Ellapsed: {}\
                           \nReward: {: .2f} +- {: .2f}  (Min: {: .2f} Max: {: .2f}) | Success Rate: {: .2f}\
                           \nReshaped: {: .2f} +- {: .2f}  (Min: {: .2f} Max: {: .2f}) | Bonus: {: .2f} +- {: .2f}  (Min: {: .2f} Max: {: .2f})\
                           \nFrames/Eps: {:.1f} +- {:.1f}  (Min: {}, Max {})\
                           \nEntropy: {: .3f} | Value: {: .3f} | Policy Loss: {: .3f} | Value Loss: {: .3f} | Loss: {: .3f} | Grad Norm: {: .3f}")

            logger.info(Fore.YELLOW + format_str.format(*data) + Fore.RESET)
            if args.tb:
                assert len(header) == len(data)
                for key, value in zip(header, data):
                    writer.add_scalar(key, float(value), status['num_frames'])
            if args.wb:
                writer.log({key: float(value) for key, value in zip(header, data)},\
                    step=status['num_frames'])

            csv_writer.writerow(data)

        if args.reward_shaping in ["subtask_classifier_online", "subtask_classifier_online_unclipped"] and \
            status['i'] % args.subtask_update_rate == 0:
            s_header = ['subtask_' + item for item in \
                ["update", "frames", "fps", "duration", "train_loss", "train_accuracy", "train_precision", "train_recall"]
              + ["validation_loss", "validation_accuracy"]
              + ["ground_truth_validation_accuracy", "ground_truth_validation_precision", "ground_truth_validation_recall"]]
            subtask_log = subtask_prediction.online_update(subtask_dataset.get_demos(), s_header, writer)
            if args.wb:
                writer.log(subtask_log, step=status['num_frames'])
                s_stats = subtask_dataset.get_stats()
                s_header = ["subtask_dataset_" + item for item in ["len", "mean", "std", "min", "max"]]
                if s_stats:
                    writer.log({key: val for key, val in zip(s_header, s_stats)})
                text = [[str(subtask_dataset.denoised_demos)]]
                wandb.log({"subtask_dataset": wandb.Table(data=text, columns=["Contents"])})

        # Save obss preprocessor vocabulary and model
        if args.save_interval > 0 and status['i'] % args.save_interval == 0:
            obss_preprocessor.vocab.save()
            with open(status_path, 'w') as dst:
                json.dump(status, dst)
                utils.save_model(acmodel, args.model, writer)

            save_model = False
            mean_return = return_per_episode["mean"]
            success_rate = success_per_episode["mean"]
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                save_model = True
            elif (success_rate == best_success_rate) and (mean_return > best_mean_return):
                best_mean_return = mean_return
                save_model = True
            if save_model:
                utils.save_model(acmodel, args.model + '_best', writer)
                obss_preprocessor.vocab.save(utils.get_vocab_path(args.model + '_best'))
                logger.info("Return {: .2f}; best model is saved".format(mean_return))
            else:
                logger.info("Return {: .2f}; not the best model; not saved".format(mean_return))
