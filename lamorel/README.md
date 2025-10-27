# Language Models for Reinforcement Learning

A library allowing the deployment and use of LLMs for interactive agents.
I've nicknamed the module lamorel (language models for RL) but ofc it's just a placeholder for now.

## Installation
1. `cd lamorel`
2. `pip install .`

## Launch
You should specify a hydra config defining the experimental setup.
```
python -m lamorel_launcher.launch --config-path <path> --config-name <name> rl_script_args.path=<path> lamorel_args.accelerate_args.machine_rank=<rank> <additional arguments to override config>
```
*Warning: use absolute paths*

### Launch examples
Several examples of configurations can be found in [examples](examples).

#### Single machine and no GPU
- Config: [local_cpu_config.yaml](examples/configs/local_cpu_config.yaml)
- Launch command(s):
    - ```shell
        python -m lamorel_launcher.launch --config-path absolute/path/to/project/examples/configs --config-name local_cpu_config rl_script_args.path=absolute/path/to/project/examples/example_script.py
      ```

#### Single machine and GPU(s)
- Config: [local_gpu_config.yaml](examples/configs/local_gpu_config.yaml)
- Launch command(s):
    - RL process:
    ```shell
        python -m lamorel_launcher.launch --config-path absolute/path/to/project/examples/configs --config-name local_gpu_config rl_script_args.path=absolute/path/to/project/examples/example_script.py lamorel_args.accelerate_args.machine_rank=0
    ```
    - LLM process:
    ```shell
        python -m lamorel_launcher.launch --config-path absolute/path/to/project/examples/configs --config-name local_gpu_config rl_script_args.path=absolute/path/to/project/examples/example_script.py lamorel_args.accelerate_args.machine_rank=1
    ```  

If you don't want your LLM process to use all your GPUs (for instance if you plan to launch multiple LLM processes), set an appropriate value to `model_parallelism_size` in the config.

#### SLURM cluster
- Config: [slurm_cluster_config.yaml](examples/configs/single-node_slurm_cluster_config.yaml)
- Launch command(s):
    - ```shell
        sbatch examples/slurm/job.slurm
      ```

## Using lamorel in your RL script
```python
import hydra
from lamorel import Caller, lamorel_init
lamorel_init()

@hydra.main(config_path='../config', config_name='config')
def main(config_args):
    lm_server = Caller(config_args.lamorel_args)
    # Do whatever you want with your LLM
    lm_server.generate(contexts=["This is an examples prompt, continue it with"])
    lm_server.score(contexts=["This is an examples prompt, continue it with"], candidates=["a sentence", "another sentence"])
    lm_server.close()
if __name__ == '__main__':
    main()
```

### Example script

We provide an [example_script](examples/example_script.py) with a simple loop running a random agent in a non-vectorized version of [ScienceWorld](https://github.com/allenai/ScienceWorld).