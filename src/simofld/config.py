import logging
import sys
from .envs import logger as envs_logger
from .masl import logger as masl_logger
from .br import logger as br_logger
from .model import logger as model_logger

# fmt = "%(asctime)s - %(message)s"
handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('result.log', mode='w')]
logging.basicConfig(handlers=handlers)
# logging.basicConfig(level=logging.INFO)
envs_logger.setLevel(logging.INFO)
masl_logger.setLevel(logging.ERROR)
br_logger.setLevel(logging.DEBUG)
model_logger.setLevel(logging.INFO)