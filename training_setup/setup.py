import os
import pickle
from jax import random
from jax import numpy as jnp
import jax
from matplotlib import pyplot as plt
import numpy as np
from plotting import plot_complexity_losses

from training_setup.distribution_setup import Q_distribution_init, W_distribution_init, distribution_name_init
from training_setup.flow_setup import Q_flow_setup, W_flow_setup, flow_init, flow_name_init
from training_setup.loss_setup import Q_loss_init, W_loss_init, loss_name_init
from training import train

jax.config.update("jax_enable_x64", True)

def setup_model_and_filename(kwargs):
    model = kwargs["model"]


    # Set up the distribution

    filename = ""
    filename = distribution_name_init(filename, kwargs)

    if kwargs["representation"] == "Q":
        target, plot_ranges = Q_distribution_init(kwargs)
    else:
        target, plot_ranges = W_distribution_init(kwargs)


    # Sample from the distribution and plot the samples

    if target is None:
        #weighted std
        rescale = 1
        target_samples = jnp.zeros((kwargs["num_samples"],2))
        target_params = None
    else:
        target_params = target.init(random.PRNGKey(0),jnp.zeros((1,target.input_dim)))
        target_samples = target.sample(target_params,kwargs["num_samples"],random.PRNGKey(0),sampler = "mcmc")

        for i in range(target_samples.shape[1]//2):
            plt.hist2d(target_samples[:,2*i],target_samples[:,2*i+1],bins = 100)
            plt.show()
        
        rescale = 1
        if kwargs["rescale"] == "True":
            rescale = target_samples.std(axis = 0)


    # Set up the flow model

    filename = flow_name_init(filename, kwargs)

    flow = flow_init(kwargs, rescale=rescale)

    if kwargs["representation"] == "Q":
        model, model_params = Q_flow_setup(flow, target_samples)
    else:
        model, model_params = W_flow_setup(flow, target_samples, kwargs)



    # Set up the loss functions

    filename = loss_name_init(filename, kwargs)
    
    if not np.isclose(kwargs["learning_rate"], 4e-3):
        filename += "_lr="+str(kwargs["learning_rate"])

    if kwargs["decay"] == "True":
        filename += "_cosine_decay"
    
    if kwargs["warmup"] > 0:
        filename += "_warmup="+str(kwargs["warmup"])

    # Train the model

    print(filename)

    return model, model_params, target, target_params, plot_ranges, filename

def run_training(kwargs, model, model_params, target, target_params, plot_ranges, filename, training_samples = None, key = None):
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
            key = key,
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
            key = key,
        )

    return model_params

def setup(kwargs):
    model, model_params, target, target_params, plot_ranges, filename = setup_model_and_filename(kwargs)

    return run_training(kwargs, model, model_params, target, target_params, plot_ranges, filename)

def L1_loss(Q_true, Q_pred, xvec, yvec):
    return jnp.sum(jnp.abs(Q_true - Q_pred)) * (xvec[0,1] - xvec[0,0]) * (yvec[1,0] - yvec[0,0])

def KL_rev_loss(Q_true, Q_pred, xvec, yvec):
    return jnp.sum(Q_true * jnp.log(Q_true/Q_pred)) * (xvec[0,1] - xvec[0,0]) * (yvec[1,0] - yvec[0,0])

def KL_loss(Q_true, Q_pred, xvec, yvec):
    return jnp.sum(Q_pred * jnp.log(Q_pred/Q_true)) * (xvec[0,1] - xvec[0,0]) * (yvec[1,0] - yvec[0,0])

def Q_fid(Q_true, Q_pred, xvec, yvec):
    return jnp.sum(jnp.sqrt(Q_true * Q_pred)) * (xvec[0,1] - xvec[0,0]) * (yvec[1,0] - yvec[0,0])

def sample_complexity_setup(kwargs):
    model, model_params, target, target_params, plot_ranges, filename = setup_model_and_filename(kwargs)

    test_X = jnp.arange(-5, 5, 0.02)
    test_Y = jnp.arange(-5, 5, 0.02)
    test_X, test_Y = jnp.meshgrid(test_X, test_Y)

    test_points = jnp.stack([test_X.flatten(), test_Y.flatten()], axis = 1)

    run_data = []
    losses_mean = {
        "L1": [],
        "KL_rev": [],
        "KL": [],
        "Fid": [],
    }

    losses_std = {
        "L1": [],
        "KL_rev": [],
        "KL": [],
        "Fid": [],
    }

    rng = random.PRNGKey(0)

    range_start = 5
    range_end = 15
    range_step = 3

    for grid_size in range(range_start, range_end, range_step):
        run_data.append(grid_size)

        temporary_losses = {
            "L1": [],
            "KL_rev": [],
            "KL": [],
            "Fid": [],
        }
        
        for i in range(1):
            new_filename = filename + f"_grid_size={grid_size}_run={i}"
            kwargs["grid_size"] = grid_size
            trained_model_params = run_training(kwargs, model, model_params, target, target_params, plot_ranges, new_filename, key = rng)
            key, rng = random.split(rng)

            trained_Q = model.Q(trained_model_params, test_points, None)
            target_Q = target.Q(target_params, test_points, None)

            temporary_losses["L1"].append(L1_loss(target_Q, trained_Q, test_X, test_Y))
            temporary_losses["KL_rev"].append(KL_rev_loss(target_Q, trained_Q, test_X, test_Y))
            temporary_losses["KL"].append(KL_loss(target_Q, trained_Q, test_X, test_Y))
            temporary_losses["Fid"].append(Q_fid(target_Q, trained_Q, test_X, test_Y))

        for key in temporary_losses.keys():
            losses_mean[key].append(np.mean(temporary_losses[key]))
            losses_std[key].append(np.std(temporary_losses[key]))

    # Make directory if it doesn't exist
    os.makedirs("sample_complexity_results", exist_ok=True)
    pickle.dump((run_data, losses_mean, losses_std), open(f"sample_complexity_results/{filename}_{range_start}_{range_end}_{range_step}.dat", "wb"))

