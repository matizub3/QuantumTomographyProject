# Import JAX, a fast math library often used for machine learning and scientific computing.
import jax

# Import the function that reads training settings from the command line.
from parse_args import parse_args

# Import the main setup function that prepares and starts the training process.
from training_setup.setup import setup

# Tell JAX to use 64-bit numbers, which are slower but more precise for scientific calculations.
jax.config.update("jax_enable_x64", True)


# Read the experiment settings, such as model type, learning rate, number of samples, and training length.
kwargs = parse_args()

# Use those settings to create the model, data distribution, loss function, and begin training.
setup(kwargs)
