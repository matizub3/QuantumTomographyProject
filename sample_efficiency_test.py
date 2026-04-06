from parse_args import parse_args
from training_setup.loss_setup import Q_loss_init, W_loss_init
from training_setup.setup import setup_model_and_filename
from itertools import product


def run_ablation(model, target, target_params, kwargs):
    if kwargs["representation"] == "Q":
        loss_controller = Q_loss_init(model, target, target_params, kwargs, training_samples = training_samples)

        model_params = train(
            model,
            model_params,
            target,
            target_params,
            loss_controller,
            kwargs["epochs"],
            kwargs["learning_rate"],
            filename = "Q/"+filename,
            plot_ranges = plot_ranges,
            plot_epochs = kwargs["plot_epochs"],
            plot_intermediate=(kwargs["plot_intermediate"] == "True"),
            decay=(kwargs["decay"] == "True"),
            warmup = kwargs["warmup"],
        )

    elif kwargs["representation"] == "W":
        loss_controller = W_loss_init(model, target, target_params, kwargs, training_samples = training_samples)

        if kwargs["split_training"] == "True":
            print("HERE")
            pass

        model_params = train(
            model,
            model_params,
            target,
            target_params,
            loss_controller,
            kwargs["epochs"],
            kwargs["learning_rate"],
            filename = "Wigner/"+filename,
            plot_ranges = plot_ranges,
            plot_epochs = kwargs["plot_epochs"],
            plot_intermediate=(kwargs["plot_intermediate"] == "True"),
            decay=(kwargs["decay"] == "True"),
            warmup = kwargs["warmup"],
        )


default_kwargs = {
    "problem": "cat",
    "num_wells": 1,

    "model": "CNF",
    "layer_size": 20,
    "network_layers": 3,
    "positive_scale": "False",
    "network": "FFJORDNet",
    "activation": "SINE",
    "num_encodings": 0,
    "rescale": "True",                          # SHOULD WE DO THIS??

    "ensemble_number": 1,

    "step_controller": "Adaptive",
    "step_size": 0.1,

    "learning_rate": 4e-3,
    "decay": "True",
    "warmup": 0.1,
    "epochs": 1000,
    "plot_epochs": 100,
    "plot_intermediate": "True",

    "batches": 1,
    "num_samples": 1000,
    "resample": "True",
    "validation": "False",
    "losses": "KL_efficient",
    "samples_per_resample": 100,
    "resample_every": 10,

    "representation": "Q",
    "symmetric": "True",

    "split_training": "False",
    "noise": 0.0,
}

iterable_kwargs = {}
grid_search = False

iterable_keys = iterable_kwargs.keys()

if grid_search:
    iterable = product(*iterable_kwargs.values())
else:
    iterable = zip(*iterable_kwargs.values())

metric_names = []

kwargs_list =[]
metrics_list = []

for kwargs in iterable:
    # Update the default kwargs with the iterable kwargs
    new_kwargs = {
        key: default_kwargs[key] if not key in kwargs else kwargs[key]
        for key in default_kwargs
    }


    model, model_params, target, target_params, _, _ = setup_model_and_filename(new_kwargs)