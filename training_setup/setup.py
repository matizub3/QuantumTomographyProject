# Import tools for file paths, saving Python objects, and adding timestamps to output filenames.
import os
import pickle
from datetime import datetime

# Import JAX tools for fast numerical computing, random numbers, and array math.
from jax import random
from jax import numpy as jnp
import jax

# Import plotting and standard numerical tools used for visualizing samples and averaging results.
from matplotlib import pyplot as plt
import numpy as np
from plotting import plot_complexity_losses

# Import helper functions that choose and initialize the target Q or Wigner distribution.
from training_setup.distribution_setup import Q_distribution_init, W_distribution_init, distribution_name_init

# Import helper functions that choose and initialize the normalizing flow model.
from training_setup.flow_setup import Q_flow_setup, W_flow_setup, flow_init, flow_name_init

# Import helper functions that choose the loss function used to train the model.
from training_setup.loss_setup import Q_loss_init, W_loss_init, loss_name_init

# Import the main training loop that updates the model parameters.
from training import train

# Tell JAX to use 64-bit numbers, which are more precise for scientific calculations.
jax.config.update("jax_enable_x64", True)


# Build the target distribution, flow model, model parameters, plot range, and experiment filename.
def setup_model_and_filename(kwargs):
    # Read which model type the user chose, such as CNF, RealNVP, or CPF.
    model = kwargs["model"]


    # Set up the distribution

    # Start with an empty filename that will be expanded to describe the experiment.
    filename = ""

    # Add information about the chosen target distribution to the filename.
    filename = distribution_name_init(filename, kwargs)

    # If using the Q representation, create the target Q distribution and plot ranges.
    if kwargs["representation"] == "Q":
        target, plot_ranges = Q_distribution_init(kwargs)

    # Otherwise, create the target Wigner distribution and plot ranges.
    else:
        target, plot_ranges = W_distribution_init(kwargs)


    # Sample from the distribution and plot the samples

    # If there is no target distribution, create placeholder samples and skip target parameters.
    if target is None:
        # Use no rescaling when there is no real target distribution to sample from.
        rescale = 1

        # Create fake zero samples so the model still has the expected sample shape.
        target_samples = jnp.zeros((kwargs["num_samples"],2))

        # Store no target parameters because there is no target object.
        target_params = None

    # If there is a real target distribution, initialize it and draw samples from it.
    else:
        # Initialize the target distribution parameters using a fixed random seed for repeatability.
        target_params = target.init(random.PRNGKey(0),jnp.zeros((1,target.input_dim)))

        # Draw sample points from the target distribution using MCMC, which samples by taking random steps.
        target_samples = target.sample(target_params,kwargs["num_samples"],random.PRNGKey(0),sampler = "mcmc")

        # Plot each 2D pair of sample coordinates so we can visually inspect the target distribution.
        for i in range(target_samples.shape[1]//2):
            plt.hist2d(target_samples[:,2*i],target_samples[:,2*i+1],bins = 100)
            plt.show()
        
        # Start with no rescaling unless the user turns rescaling on.
        rescale = 1

        # If rescaling is enabled, estimate a good scale for the data so the model trains more stably.
        if kwargs["rescale"] == "True":

            # Multiply the scale by the user-chosen rescale multiplier.
            rescale = kwargs["rescale_mult"] * target_samples.std(axis = 0)


    # Set up the flow model

    # Add information about the chosen flow model to the filename.
    filename = flow_name_init(filename, kwargs)

    # Create the normalizing flow model, optionally using the rescale value from the target samples.
    flow = flow_init(kwargs, rescale=rescale)

    # If using the Q representation, wrap the flow so it represents a Q function.
    if kwargs["representation"] == "Q":
        model, model_params = Q_flow_setup(flow, target_samples)

    # If using the Wigner representation, wrap the flow so it represents a Wigner function.
    else:
        model, model_params = W_flow_setup(flow, target_samples, kwargs)



    # Set up the loss functions

    # Add information about the chosen loss function to the filename.
    filename = loss_name_init(filename, kwargs)
    
    # If the learning rate is not the default, record it in the filename.
    if not np.isclose(kwargs["learning_rate"], 4e-3):
        filename += "_lr="+str(kwargs["learning_rate"])

    # If learning rate decay is enabled, record that in the filename.
    if kwargs["decay"] == "True":
        filename += "_cosine_decay"
    
    # If learning rate warmup is enabled, record the warmup amount in the filename.
    if kwargs["warmup"] > 0:
        filename += "_warmup="+str(kwargs["warmup"])

    # If training a Wigner model, record lambda settings because they affect the Wigner loss.
    if kwargs["representation"] == "W":
        filename += f"_li={kwargs['lambda_init']}"
        filename += f"_ll={kwargs['learn_lambda']}"

    # Train the model

    # Create a unique timestamp so each run saves to a different filename.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Add the timestamp to the filename to avoid overwriting old experiments.
    filename += f"_run={timestamp}"

    # Print the final filename so the user knows where this run will be saved.
    print(filename)

    # Return everything needed to train and evaluate the model.
    return model, model_params, target, target_params, plot_ranges, filename


# Train either a Q model or Wigner model using the objects created during setup.
def run_training(kwargs, model, model_params, target, target_params, plot_ranges, filename, training_samples = None, key = None):
    # If using the Q representation, create the Q loss and train the Q model.
    if kwargs["representation"] == "Q":
        # Create the loss controller, which knows how to measure Q-model training error.
        loss_controller = Q_loss_init(model, target, target_params, kwargs, training_samples = training_samples)

        # Run the training loop and update the model parameters.
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

    # If using the Wigner representation, create the Wigner loss and train the Wigner model.
    elif kwargs["representation"] == "W":
        # Create the loss controller, which knows how to measure Wigner-model training error.
        loss_controller = W_loss_init(model, target, target_params, kwargs, training_samples = training_samples)

        # Placeholder for a future split-training method, which is not implemented yet.
        if kwargs["split_training"] == "True":
            print("HERE")
            pass

        # Run the training loop and update the model parameters.
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

    # Return the trained model parameters.
    return model_params


# Main setup function that creates the model and immediately starts training it.
def setup(kwargs):
    # Create the model, parameters, target distribution, plot ranges, and filename.
    model, model_params, target, target_params, plot_ranges, filename = setup_model_and_filename(kwargs)

    # Train the model and return the final trained parameters.
    return run_training(kwargs, model, model_params, target, target_params, plot_ranges, filename)


# Compute L1 loss, which measures total absolute difference between true and predicted Q values.
def L1_loss(Q_true, Q_pred, xvec, yvec):
    return jnp.sum(jnp.abs(Q_true - Q_pred)) * (xvec[0,1] - xvec[0,0]) * (yvec[1,0] - yvec[0,0])


# Compute reverse KL loss, which measures how much information is lost when Q_pred approximates Q_true.
def KL_rev_loss(Q_true, Q_pred, xvec, yvec):
    return jnp.sum(Q_true * jnp.log(Q_true/Q_pred)) * (xvec[0,1] - xvec[0,0]) * (yvec[1,0] - yvec[0,0])


# Compute forward KL loss, which measures how different the predicted distribution is from the true one.
def KL_loss(Q_true, Q_pred, xvec, yvec):
    return jnp.sum(Q_pred * jnp.log(Q_pred/Q_true)) * (xvec[0,1] - xvec[0,0]) * (yvec[1,0] - yvec[0,0])


# Compute Q fidelity, which is high when the predicted and true Q functions overlap well.
def Q_fid(Q_true, Q_pred, xvec, yvec):
    return jnp.sum(jnp.sqrt(Q_true * Q_pred)) * (xvec[0,1] - xvec[0,0]) * (yvec[1,0] - yvec[0,0])


# Run a sample-complexity experiment to see how training quality changes with grid size.
def sample_complexity_setup(kwargs):
    # Create the model, parameters, target distribution, plot ranges, and base filename.
    model, model_params, target, target_params, plot_ranges, filename = setup_model_and_filename(kwargs)

    # Create x-coordinates for a dense testing grid from -5 to 5.
    test_X = jnp.arange(-5, 5, 0.02)

    # Create y-coordinates for a dense testing grid from -5 to 5.
    test_Y = jnp.arange(-5, 5, 0.02)

    # Turn the x and y coordinate lists into a 2D grid of test points.
    test_X, test_Y = jnp.meshgrid(test_X, test_Y)

    # Flatten the grid into a list of 2D points that can be passed into the model.
    test_points = jnp.stack([test_X.flatten(), test_Y.flatten()], axis = 1)

    # Store the grid sizes tested in this experiment.
    run_data = []

    # Store the average value of each evaluation metric across repeated runs.
    losses_mean = {
        "L1": [],
        "KL_rev": [],
        "KL": [],
        "Fid": [],
    }

    # Store the standard deviation of each metric to measure how much results vary across runs.
    losses_std = {
        "L1": [],
        "KL_rev": [],
        "KL": [],
        "Fid": [],
    }

    # Create a random key so repeated training runs can use controlled randomness.
    rng = random.PRNGKey(0)

    # Set the smallest grid size tested in the sample-complexity experiment.
    range_start = 5

    # Set the largest grid size boundary, not including this value.
    range_end = 15

    # Set how much the grid size increases after each experiment.
    range_step = 3

    # Loop over different grid sizes to test how many grid points are needed for good training.
    for grid_size in range(range_start, range_end, range_step):
        # Record the current grid size.
        run_data.append(grid_size)

        # Store the losses for repeated runs at this one grid size.
        temporary_losses = {
            "L1": [],
            "KL_rev": [],
            "KL": [],
            "Fid": [],
        }
        
        # Repeat training for this grid size; currently this only runs once.
        for i in range(1):
            # Create a filename that records the grid size and repeat number.
            new_filename = filename + f"_grid_size={grid_size}_run={i}"

            # Update the training settings to use the current grid size.
            kwargs["grid_size"] = grid_size

            # Train the model using the current grid size and random key.
            trained_model_params = run_training(kwargs, model, model_params, target, target_params, plot_ranges, new_filename, key = rng)

            # Split the random key so the next run gets new randomness.
            key, rng = random.split(rng)

            # Evaluate the trained model's Q function on the dense test grid.
            trained_Q = model.Q(trained_model_params, test_points, None)

            # Evaluate the true target Q function on the same dense test grid.
            target_Q = target.Q(target_params, test_points, None)

            # Save the L1 error for this run.
            temporary_losses["L1"].append(L1_loss(target_Q, trained_Q, test_X, test_Y))

            # Save the reverse KL error for this run.
            temporary_losses["KL_rev"].append(KL_rev_loss(target_Q, trained_Q, test_X, test_Y))

            # Save the forward KL error for this run.
            temporary_losses["KL"].append(KL_loss(target_Q, trained_Q, test_X, test_Y))

            # Save the fidelity score for this run.
            temporary_losses["Fid"].append(Q_fid(target_Q, trained_Q, test_X, test_Y))

        # Average each metric across repeated runs and store the result.
        for key in temporary_losses.keys():
            losses_mean[key].append(np.mean(temporary_losses[key]))
            losses_std[key].append(np.std(temporary_losses[key]))

    # Create the folder for sample-complexity results if it does not already exist.
    os.makedirs("sample_complexity_results", exist_ok=True)

    # Save the grid sizes, mean losses, and loss standard deviations to a file.
    pickle.dump((run_data, losses_mean, losses_std), open(f"sample_complexity_results/{filename}_{range_start}_{range_end}_{range_step}.dat", "wb"))
