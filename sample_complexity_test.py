from jax.config import config

from parse_args import parse_args
from training_setup.setup import sample_complexity_setup, setup

config.update("jax_enable_x64", True)

kwargs = parse_args()

kwargs["plot"] = "False"

sample_complexity_setup(kwargs)