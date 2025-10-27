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
# from scienceworld import ScienceWorldEnv
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
    scores = lm_server.score(contexts=[obs], candidates=[valid_actions], additional_module_function_keys=['value'])
    value = lm_server.custom_module_fns(module_function_keys=['value'], contexts=[obs])
    print("Scores computed in {} seconds".format(time.time() - start_time))
    return scores[0]


def run_agent(agent, env, args, lm_server):
    # provides example uses of HF models for
    #   - 1. generating text sequences before an episode
    #   - 2. ranking actions from within an episode, at each step

    # obs, reward, done, info = reset_env(env, args)
    obs = "test obs"
    info = {"look": "test", "valid": ["test", "test2"], "inv": ""}
    generated_goal = get_generated_sequence(info, lm_server)
    print(f"Generated goal: {generated_goal}")

    for step in range(1, args.num_steps + 1):
        print(f'Step number {step}')
        state = agent.build_state(obs, info)
        action = agent.act(state)
        # obs, reward, done, info = env.step(action)
        # this will cuda oom on most machine in most cases after a few steps
        valid_actions_reranked = get_actions_reranked(obs, info, lm_server)
        print("Reranked actions according to NLL:")
        pprint(valid_actions_reranked)

        if step % args.max_episode_steps == 0:
            test = lm_server.update(
                contexts=["test", "test", "test", "test"],
                candidates=[["test"], ["test"], ["test"], ["test"]],
                labels=torch.tensor([[1, 1, 1, 1]]),
                saving_path=args.saving_path
            )
            print(f"Step {step}, resetting env")
            obs, reward, done, info = reset_env(env, args)
            generated_goal = get_generated_sequence(info, lm_server)
            print(f"Generated goal: {generated_goal}")

from lamorel import BaseUpdater, BaseModuleFunction

class ValueModuleFn(BaseModuleFunction):
    def __call__(self, forward_outputs, minibatch, tokenized_context, **kwargs):
        if self._llm_instance.model_type == "causal":
            model_head = forward_outputs['hidden_states'][0][0, len(tokenized_context["input_ids"])-1, :]
        else:
            model_head = forward_outputs['encoder_last_hidden_state'][0, len(tokenized_context["input_ids"]) - 1, :]
        if not hasattr(self, 'value_head_op'):
            self.value_head_op = torch.nn.Sequential(
                torch.nn.Linear(model_head.shape[0], 1024),
                torch.nn.Sigmoid(),
                torch.nn.Linear(1024, 1024),
                torch.nn.Sigmoid(),
                torch.nn.Linear(1024, 1)
            ).to(self._llm_instance.device)

        value = self.value_head_op(model_head)
        return value

class TestUpdater(BaseUpdater):
    def perform_update(self, contexts, candidates, _current_batch_ids, **kwargs):
        if not hasattr(self, 'loss_fn'):
            self.loss_fn = torch.nn.L1Loss()
        if not hasattr(self, 'optimizer'):
            self.optimizer = torch.optim.Adam(self._trainable_module.parameters())
            # if accelerator.process_index == 1 and kwargs["load_optimizer_ckpt"]:
            #     self.optimizer.load_state_dict(kwargs["saving_path"] + "/optimizer.checkpoint")

        output = self._forward_fn(['__score', 'value'],
            contexts=contexts, candidates=candidates, require_grad=True)
        scores = torch.stack([_o["__score"] for _o in output]).to('cpu')
        values = torch.stack([_o["value"] for _o in output]).to('cpu')
        loss = self.loss_fn(scores, kwargs["labels"][:, _current_batch_ids])
        loss.backward()
        self.optimizer.step()

        if accelerator.process_index == 1:
            torch.save(self._trainable_module.state_dict(), kwargs["saving_path"] + "/model.checkpoint")
            torch.save(self.optimizer.state_dict(), kwargs["saving_path"] + "/optimizer.checkpoint")
        return {"test": loss}

# This will be overriden by lamorel's launcher if used
@hydra.main(config_path='config', config_name='config')
def main(config_args):

    # lm server
    lm_server = Caller(config_args.lamorel_args, custom_updater_class=TestUpdater,
                       custom_module_functions={'value': ValueModuleFn()})

    # Env
    # env = ScienceWorldEnv('', envStepLimit=config_args.rl_script_args.max_episode_steps, threadNum=accelerator.process_index)
    # task_names = env.getTaskNames()
    # env.load(task_names[config_args.rl_script_args.task_idx], 0, config_args.rl_script_args.simplification_str)
    env = None
    agent = RandomAgent()

    run_agent(agent, env, config_args.rl_script_args, lm_server)
    lm_server.close()

if __name__ == '__main__':
    main()
