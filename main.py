import jax

from parse_args import parse_args
from training_setup.setup import setup

jax.config.update("jax_enable_x64", True)

kwargs = parse_args()

setup(kwargs)