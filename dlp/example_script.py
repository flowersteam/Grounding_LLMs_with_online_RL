"""
This script run a simple agent in a ScienceWorld environment, with dummy calls to an API
to perform inference on the provided data.
"""
import transformers
# transformers.models.gpt2.modeling_gpt2.GPT2Block = None
import torch

from lamorel import Caller, lamorel_init
lamorel_init()

import hydra
from scienceworld import ScienceWorldEnv
from agents.random_agent import RandomAgent
from pprint import pprint
import time
from accelerate import Accelerator

accelerator = Accelerator()


def reset_env(env, args, variation='train'):
    if variation == 'train':
        variation_idx = env.getRandomVariationTrain()
    elif variation == 'dev':
        variation_idx = env.getRandomVariationDev()
    elif variation == 'test':
        variation_idx = env.getRandomVariationTest()
    else:
        raise ValueError(f"Unsupported vatiation {variation}, must be one of 'train', 'dev' or 'test'")

    obs, info = env.resetWithVariation(variation_idx, args.simplification_str)
    reward, done = 0, False
    return obs, reward, done, info


def get_generated_sequence(info, lm_server):
    # Something a bit like goal generation, takes in the room description, adds some
    # an additional prompt and then gets a suggestion from the model.
    promt_suffix = "\nThis is an example of what I could do here:"
    prompt = info['look'] + promt_suffix
    print("Generating sequences from LLM")
    start_time = time.time()
    _result = lm_server.generate(contexts=[prompt], max_length=512)
    print("Generation done in {} seconds".format(time.time() - start_time))
    generated = _result[0][0]["text"].split('.')[0]

    return generated


def get_actions_reranked(obs, info, lm_server):
    # gets the valid actions from the info dict and returns a list of reranked
    # actions, from lower to higher negative log likelihood under the model
    valid_actions = list(info['valid'])
    print("Getting scores from LLM of {} actions".format(len(valid_actions)))
    start_time = time.time()
    scores = lm_server.score(contexts=[obs], candidates=[valid_actions])
    print("Scores computed in {} seconds".format(time.time() - start_time))
    return scores[0]


def run_agent(agent, env, args, lm_server):
    # provides example uses of HF models for
    #   - 1. generating text sequences before an episode
    #   - 2. ranking actions from within an episode, at each step

    obs, reward, done, info = reset_env(env, args)
    generated_goal = get_generated_sequence(info, lm_server)
    print(f"Generated goal: {generated_goal}")

    for step in range(1, args.num_steps + 1):
        print(f'Step number {step}')
        state = agent.build_state(obs, info)
        action = agent.act(state)
        obs, reward, done, info = env.step(action)
        # this will cuda oom on most machine in most cases after a few steps
        valid_actions_reranked = get_actions_reranked(obs, info, lm_server)
        print("Reranked actions according to NLL:")
        pprint(valid_actions_reranked)

        if step % args.max_episode_steps == 0:
            lm_server.update(contexts=["test", "test", "test", "test"], candidates=[["test"], ["test"], ["test"], ["test"]], labels=torch.tensor([[1, 1, 1, 1]]))
            print(f"Step {step}, resetting env")
            obs, reward, done, info = reset_env(env, args)
            generated_goal = get_generated_sequence(info, lm_server)
            print(f"Generated goal: {generated_goal}")

from lamorel import BaseUpdater
class TestUpdater(BaseUpdater):
    def perform_update(self, contexts, candidates, _current_batch_ids, **kwargs):
        if not hasattr(self, 'loss_fn'):
            self.loss_fn = torch.nn.L1Loss()
        if not hasattr(self, 'optimizer'):
            self.optimizer = torch.optim.Adam(self._trainable_module.parameters())

        output = self._score_fn(
            contexts=contexts, candidates=candidates, require_grad=True).to('cpu')
        loss = self.loss_fn(output, kwargs["labels"][:, _current_batch_ids])
        loss.backward()
        self.optimizer.step()

# This will be overriden by lamorel's launcher if used
@hydra.main(config_path='config', config_name='config')
def main(config_args):

    # lm server
    lm_server = Caller(config_args.lamorel_args, custom_updater_class=TestUpdater)

    # Env
    env = ScienceWorldEnv('', envStepLimit=config_args.rl_script_args.max_episode_steps, threadNum=accelerator.process_index)
    task_names = env.getTaskNames()
    env.load(task_names[config_args.rl_script_args.task_idx], 0, config_args.rl_script_args.simplification_str)
    agent = RandomAgent()

    run_agent(agent, env, config_args.rl_script_args, lm_server)
    lm_server.close()

if __name__ == '__main__':
    main()
