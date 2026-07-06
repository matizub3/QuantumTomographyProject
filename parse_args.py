# Import argparse so the program can read settings typed into the terminal.
import argparse

# Import product so we can create every possible combination of settings for experiments.
from itertools import product


# Define the normal argument parser for running one training experiment.
def parse_args():
    # Create the object that stores all possible command-line options.
    parser = argparse.ArgumentParser()

    # Choose which quantum/problem setup to train on.
    parser.add_argument("--problem", type = str, default="BH3")

    # Choose how many wells or modes/components are in the target distribution.
    parser.add_argument("-nw", "--num_wells", type = int, default = 2)

    # Choose which type of flow model to use for learning the distribution.
    parser.add_argument("-m", "--model", type = str, default = "CNF", choices=["CNF", "RealNVP", "CPF"])

    # Choose how many neurons are in each hidden layer of the neural network.
    parser.add_argument("-ls", "--layer_size", type = int, default = 20)

    # Choose how many hidden layers the neural network has.
    parser.add_argument("-nl", "--network_layers", type = int, default = 3)

    # Choose whether the model should force scale values to be positive.
    parser.add_argument("-ps", "--positive_scale", type = str, default = "False", choices=["True", "False"])

    # Choose the specific neural network architecture used inside the flow.
    parser.add_argument("-net", "--network", type = str, default = "FFJORDNet", choices=["ResNet", "FFJORDNet"])

    # Choose the activation function, which controls how the network bends or transforms data.
    parser.add_argument("-a", "--activation", type = str, default = "SINE", choices=["GELU", "SINE"])

    # Choose how many extra input encodings to add, which can help the model represent complicated patterns.
    parser.add_argument("-n_enc", "--num_encodings", type = int, default = 0)

    # Choose whether to rescale the input or target samples.
    parser.add_argument("--rescale", type = str, default = "False")

    # Choose how strongly to multiply the rescaling effect.
    parser.add_argument("--rescale_mult", type=float, default=1.0)

    # Set the starting value of lambda, a tunable weight used by some losses or distributions.
    parser.add_argument("-li", "--lambda_init", type = float, default = 1.0)

    # Choose whether lambda should be learned during training or kept fixed.
    parser.add_argument("-ll", "--learn_lambda", type=str, default="True", choices=["True", "False"])
    
    # Label this run as part of an ensemble, meaning multiple models trained separately.
    parser.add_argument("-en", "--ensemble_number", type = int, default = 1)

    # Choose whether the ODE solver uses adaptive step sizes or constant step sizes.
    parser.add_argument("-sc", "--step_controller", type = str, default = "Adaptive", choices=["Adaptive", "Constant"])

    # Choose the step size for the ODE solver if constant stepping is used.
    parser.add_argument("-ss", "--step_size", type = float, default = 0.1)

    # Set how large each optimizer update is during training.
    parser.add_argument("-lr", "--learning_rate", type = float, default = 4e-3)

    # Choose whether the learning rate should decay, meaning get smaller during training.
    parser.add_argument("-d", "--decay", type = str, default = "False", choices = ["True", "False"])

    # Set the fraction of training spent slowly increasing the learning rate at the beginning.
    parser.add_argument("-w", "--warmup", type = float, default = 0)

    # Set how many training steps or passes the model will run.
    parser.add_argument("-e", "--epochs", type = int, default = 500)

    # Set how often plots are saved during training.
    parser.add_argument("-pe", "--plot_epochs", type = int, default = 10)

    # Choose whether to save intermediate plots while the model is still training.
    parser.add_argument("-pi", "--plot_intermediate", type = str, default = "True")

    # Set how many batches the training data is split into.
    parser.add_argument("-b", "--batches", type = int, default = 1)

    # Set how many sample points are used to represent the target distribution.
    parser.add_argument("-n", "--num_samples", type = int, default = 10000)

    # Choose whether to draw new samples during training instead of reusing the same ones.
    parser.add_argument("-res", "--resample", type = str, default = "True", choices = ["True", "False"])

    # Choose whether to compute validation results to check how well the model generalizes.
    parser.add_argument("-v", "--validation", type = str, default = "True", choices = ["True", "False"])

    # Choose which loss functions to train with; the loss measures how wrong the model is.
    parser.add_argument("-l", "--losses", type = str, default = ["default"], nargs = "+")

    # Set how many fresh samples are created each time resampling happens.
    parser.add_argument("-spr", "--samples_per_resample", type = int, default = 100)

    # Set how many epochs pass before the training data is resampled.
    parser.add_argument("-re", "--resample_every", type = int, default = 10)

    # Choose the control variate, a trick that can reduce noisy randomness in training estimates.
    parser.add_argument("-cv", "--control_variate", type = str, default = "mean", choices = ["mean", "1", "none"])

    # Set the resolution of the plotting/evaluation grid.
    parser.add_argument("-gs", "--grid_size", type = int, default = 100)

    # Choose whether the model learns the Q function or Wigner function representation.
    parser.add_argument("-r", "--representation", type = str, default = "Q", choices = ["Q", "W"])

    # Choose whether the target distribution should be treated as symmetric.
    parser.add_argument("-s", "--symmetric", type = str, default = "True")

    # Choose whether training is split into multiple stages.
    parser.add_argument("-st", "--split_training", type = str, default = "False")

    # Add optional noise to the data or target distribution.
    parser.add_argument("--noise", type = float, default = 0)

    # Read the actual command-line arguments typed by the user.
    args = parser.parse_args()

    # Convert the parsed arguments into a dictionary so the rest of the code can access them easily.
    kwargs = vars(args)

    # Return the dictionary of experiment settings.
    return kwargs


# Create every possible combination of hyperparameters for a grid search.
def generate_grid_combinations(kwargs):
    # Separate the dictionary into a list of setting names and a list of possible values.
    keys, values = zip(*kwargs.items())

    # Build one dictionary for every possible combination of the provided values.
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    # Return the full list of experiment settings.
    return combinations


# Create a list of experiment settings by matching the first value of each argument, second value of each argument, etc.
def generate_kwarg_list(kwargs):
    # Separate the dictionary into a list of setting names and a list of possible values.
    keys, values = zip(*kwargs.items())

    # Find the longest argument list so shorter one-value lists can be repeated to match it.
    list_len = max([len(v) for v in values])

    # Make sure every setting has either one value or the same number of values as the longest setting.
    for i in range(len(values)):
        # If a setting has only one value, repeat it for every experiment.
        if len(values[i]) == 1:
            values[i] = values[i]*list_len

        # If a setting has more than one value but too few values, stop because the experiment list is unclear.
        elif len(values[i]) < list_len:
            raise ValueError(f"Argument {keys[i]} has fewer values ({len(values[i])}) than the maximum number of values {list_len}")
    
    # Combine matching positions into separate experiment dictionaries.
    combinations = [dict(zip(keys, v)) for v in zip(*values)]

    # Return the list of matched experiment settings.
    return combinations


# Define a special argument parser for ablation studies, where many settings can be tested at once.
def parse_args_ablation():
    # Create the object that stores all possible command-line options for many experiments.
    parser = argparse.ArgumentParser()

    # Choose one or more quantum/problem setups to test.
    parser.add_argument("--problem", type = str, nargs="+", default=["BH3"])

    # Choose one or more numbers of wells or modes/components to test.
    parser.add_argument("-nw", "--num_wells", type = int, nargs="+", default = [2])

    # Choose one or more model types to compare.
    parser.add_argument("-m", "--model", type = str, nargs="+", default = ["CNF"], choices=["CNF", "RealNVP", "CPF"])

    # Choose one or more neural network layer sizes to compare.
    parser.add_argument("-ls", "--layer_size", type = int, nargs="+", default = [20])

    # Choose one or more numbers of neural network layers to compare.
    parser.add_argument("-nl", "--network_layers", type = int, nargs="+", default = [3])

    # Choose whether the scale parameter should be forced positive in each experiment.
    parser.add_argument("-ps", "--positive_scale", type = str, nargs="+", default = ["False"], choices=["True", "False"])

    # Choose one or more network architectures to compare inside the flow.
    parser.add_argument("-net", "--network", type = str, nargs="+", default = ["FFJORDNet"], choices=["ResNet", "FFJORDNet"])

    # Choose one or more activation functions to compare.
    parser.add_argument("-a", "--activation", type = str, nargs="+", default = ["SINE"], choices=["GELU", "SINE"])

    # Choose one or more numbers of extra input encodings to compare.
    parser.add_argument("-n_enc", "--num_encodings", type = int, nargs="+", default = [0])

    # Choose whether to rescale the input or target samples.
    parser.add_argument("--rescale", type = str, default = "False")

    # Choose how strongly to multiply the rescaling effect.
    parser.add_argument("--rescale_mult", type=float, default=1.0)

    # Choose one or more starting lambda values to compare.
    parser.add_argument("-li", "--lambda_init", type = float, nargs="+", default = [1.0])

    # Choose whether lambda should be learned during training or kept fixed.
    parser.add_argument("-ll", "--learn_lambda", type=str, default="True", choices=["True", "False"])

    # Label this run as part of an ensemble, meaning multiple models trained separately.
    parser.add_argument("-en", "--ensemble_number", type = int, default = 1)

    # Choose whether the ODE solver uses adaptive step sizes or constant step sizes.
    parser.add_argument("-sc", "--step_controller", type = str, default = "Adaptive", choices=["Adaptive", "Constant"])

    # Choose the step size for the ODE solver if constant stepping is used.
    parser.add_argument("-ss", "--step_size", type = float, default = 0.1)

    # Set how large each optimizer update is during training.
    parser.add_argument("-lr", "--learning_rate", type = float, default = 4e-3)

    # Choose whether the learning rate should decay over time.
    parser.add_argument("-d", "--decay", type = str, default = "False", choices = ["True", "False"])

    # Set the fraction of training spent warming up the learning rate.
    parser.add_argument("-w", "--warmup", type = float, default = 0)

    # Set how many training epochs each experiment runs.
    parser.add_argument("-e", "--epochs", type = int, default = 500)

    # Set how often plots are saved during training.
    parser.add_argument("-pe", "--plot_epochs", type = int, default = 10)

    # Choose whether to save plots while training is still happening.
    parser.add_argument("-pi", "--plot_intermediate", type = str, default = "True")

    # Set how many batches the training data is split into.
    parser.add_argument("-b", "--batches", type = int, default = 1)

    # Set how many samples are used to represent the target distribution.
    parser.add_argument("-n", "--num_samples", type = int, default = 10000)

    # Choose whether to draw new samples during training.
    parser.add_argument("-res", "--resample", type = str, default = "True", choices = ["True", "False"])

    # Choose whether to compute validation results.
    parser.add_argument("-v", "--validation", type = str, default = "True", choices = ["True", "False"])

    # Choose which loss functions to use during training.
    parser.add_argument("-l", "--losses", type = str, default = ["default"], nargs = "+")

    # Set how many fresh samples are created each time resampling happens.
    parser.add_argument("-spr", "--samples_per_resample", type = int, default = 100)

    # Set how often the training data is resampled.
    parser.add_argument("-re", "--resample_every", type = int, default = 10)

    # Choose whether experiments use the Q function or Wigner function representation.
    parser.add_argument("-r", "--representation", type = str, default = "Q", choices = ["Q", "W"])

    # Choose whether the target distribution should be treated as symmetric.
    parser.add_argument("-s", "--symmetric", type = str, default = "True")

    # Choose whether training is split into multiple stages.
    parser.add_argument("-st", "--split_training", type = str, default = "False")

    # Add optional noise to the data or target distribution.
    parser.add_argument("--noise", type = float, default = 0)

    # Read the actual command-line arguments typed by the user.
    args = parser.parse_args()

    # Convert the parsed arguments into a dictionary.
    kwargs = vars(args)

    if kwargs["grid_search"] == "True":

        kwargs.pop("grid_search")

        # Return every possible combination of the provided argument values.
        return generate_grid_combinations(kwargs)

    else:
        # Remove grid_search from the experiment settings because it controls how experiments are generated.
        kwargs.pop("grid_search")

        return generate_kwarg_list(kwargs)
