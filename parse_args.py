import argparse
from itertools import product

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type = str, default="BH3")
    parser.add_argument("-nw", "--num_wells", type = int, default = 2)

    parser.add_argument("-m", "--model", type = str, default = "CNF", choices=["CNF", "RealNVP", "CPF"])
    parser.add_argument("-ls", "--layer_size", type = int, default = 20)
    parser.add_argument("-nl", "--network_layers", type = int, default = 3)
    parser.add_argument("-ps", "--positive_scale", type = str, default = "False", choices=["True", "False"])
    parser.add_argument("-net", "--network", type = str, default = "FFJORDNet", choices=["ResNet", "FFJORDNet"])
    parser.add_argument("-a", "--activation", type = str, default = "SINE", choices=["GELU", "SINE"])
    parser.add_argument("-n_enc", "--num_encodings", type = int, default = 0)
    parser.add_argument("--rescale", type = str, default = "False")

    parser.add_argument("-en", "--ensemble_number", type = int, default = 1)

    parser.add_argument("-sc", "--step_controller", type = str, default = "Adaptive", choices=["Adaptive", "Constant"])
    parser.add_argument("-ss", "--step_size", type = float, default = 0.1)

    parser.add_argument("-lr", "--learning_rate", type = float, default = 4e-3)
    parser.add_argument("-d", "--decay", type = str, default = "False", choices = ["True", "False"])
    parser.add_argument("-w", "--warmup", type = float, default = 0)
    parser.add_argument("-e", "--epochs", type = int, default = 500)
    parser.add_argument("-pe", "--plot_epochs", type = int, default = 10)
    parser.add_argument("-pi", "--plot_intermediate", type = str, default = "True")

    parser.add_argument("-b", "--batches", type = int, default = 1)
    parser.add_argument("-n", "--num_samples", type = int, default = 10000)
    parser.add_argument("-res", "--resample", type = str, default = "True", choices = ["True", "False"])
    parser.add_argument("-v", "--validation", type = str, default = "True", choices = ["True", "False"])
    parser.add_argument("-l", "--losses", type = str, default = ["default"], nargs = "+")
    parser.add_argument("-spr", "--samples_per_resample", type = int, default = 100)
    parser.add_argument("-re", "--resample_every", type = int, default = 10)
    parser.add_argument("-cv", "--control_variate", type = str, default = "mean", choices = ["mean", "1", "none"])
    parser.add_argument("-gs", "--grid_size", type = int, default = 100)

    parser.add_argument("-r", "--representation", type = str, default = "Q", choices = ["Q", "W"])
    parser.add_argument("-s", "--symmetric", type = str, default = "True")

    parser.add_argument("-st", "--split_training", type = str, default = "False")
    parser.add_argument("--noise", type = float, default = 0)

    args = parser.parse_args()

    kwargs = vars(args)

    return kwargs

def generate_grid_combinations(kwargs):
    keys, values = zip(*kwargs.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    return combinations

def generate_kwarg_list(kwargs):
    keys, values = zip(*kwargs.items())

    list_len = max([len(v) for v in values])

    for i in range(len(values)):
        if len(values[i]) == 1:
            values[i] = values[i]*list_len
        elif len(values[i]) < list_len:
            raise ValueError(f"Argument {keys[i]} has fewer values ({len(values[i])}) than the maximum number of values {list_len}")
    
    combinations = [dict(zip(keys, v)) for v in zip(*values)]

    return combinations

def parse_args_ablation():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type = str, nargs="+", default=["BH3"])
    parser.add_argument("-nw", "--num_wells", type = int, nargs="+", default = [2])

    parser.add_argument("-m", "--model", type = str, nargs="+", default = ["CNF"], choices=["CNF", "RealNVP", "CPF"])
    parser.add_argument("-ls", "--layer_size", type = int, nargs="+", default = [20])
    parser.add_argument("-nl", "--network_layers", type = int, nargs="+", default = [3])
    parser.add_argument("-ps", "--positive_scale", type = str, nargs="+", default = ["False"], choices=["True", "False"])
    parser.add_argument("-net", "--network", type = str, nargs="+", default = ["FFJORDNet"], choices=["ResNet", "FFJORDNet"])
    parser.add_argument("-a", "--activation", type = str, nargs="+", default = ["SINE"], choices=["GELU", "SINE"])
    parser.add_argument("-n_enc", "--num_encodings", type = int, nargs="+", default = [0])
    parser.add_argument("--rescale", type = str, default = "False")

    parser.add_argument("-en", "--ensemble_number", type = int, default = 1)

    parser.add_argument("-sc", "--step_controller", type = str, default = "Adaptive", choices=["Adaptive", "Constant"])
    parser.add_argument("-ss", "--step_size", type = float, default = 0.1)

    parser.add_argument("-lr", "--learning_rate", type = float, default = 4e-3)
    parser.add_argument("-d", "--decay", type = str, default = "False", choices = ["True", "False"])
    parser.add_argument("-w", "--warmup", type = float, default = 0)
    parser.add_argument("-e", "--epochs", type = int, default = 500)
    parser.add_argument("-pe", "--plot_epochs", type = int, default = 10)
    parser.add_argument("-pi", "--plot_intermediate", type = str, default = "True")

    parser.add_argument("-b", "--batches", type = int, default = 1)
    parser.add_argument("-n", "--num_samples", type = int, default = 10000)
    parser.add_argument("-res", "--resample", type = str, default = "True", choices = ["True", "False"])
    parser.add_argument("-v", "--validation", type = str, default = "True", choices = ["True", "False"])
    parser.add_argument("-l", "--losses", type = str, default = ["default"], nargs = "+")
    parser.add_argument("-spr", "--samples_per_resample", type = int, default = 100)
    parser.add_argument("-re", "--resample_every", type = int, default = 10)

    parser.add_argument("-r", "--representation", type = str, default = "Q", choices = ["Q", "W"])
    parser.add_argument("-s", "--symmetric", type = str, default = "True")

    parser.add_argument("-st", "--split_training", type = str, default = "False")
    parser.add_argument("--noise", type = float, default = 0)

    args = parser.parse_args()

    kwargs = vars(args)

    if kwargs["grid_search"] == "True":
        kwargs.pop("grid_search")
        return generate_grid_combinations(kwargs)
    else:
        kwargs.pop("grid_search")
        return generate_kwarg_list(kwargs)