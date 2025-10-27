from accelerate import Accelerator, InitProcessGroupKwargs
from datetime import timedelta
import os

import logging
lamorel_logger = logging.getLogger('lamorel_logger')

def init_distributed_setup():
    try:
        accelerator = Accelerator(
            kwargs_handlers=[
                InitProcessGroupKwargs(timeout=timedelta(minutes=2))
            ]
        )
        lamorel_logger.info(f"Successfully initialized distributed process on process {accelerator.process_index}")
    except Exception as e:
        lamorel_logger.error(f"Error: {e}")
        lamorel_logger.error(f'MASTER_ADDR: {os.environ["MASTER_ADDR"]}')
        lamorel_logger.error(f'MASTER_PORT: {os.environ["MASTER_PORT"]}')
        lamorel_logger.error(f'WORLD_SIZE: {os.environ["WORLD_SIZE"]}')
        lamorel_logger.error(f'RANK: {os.environ["RANK"]}')
        lamorel_logger.error(f'LOCAL_RANK: {os.environ["LOCAL_RANK"]}')
        raise e