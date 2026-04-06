import pickle
import jax
from jax import random
import traceback

from video import figures_to_video
jax.config.update("jax_enable_x64", True)

import optax

from matplotlib import pyplot as plt

import time

import os

from flow_IO import save_params

def print_losses(loss_controller, losses, validation_losses):
    if validation_losses == []:
        for loss_fn, loss_value in zip(loss_controller.losses, losses[-1]):
            print(f"{loss_fn.name} = {loss_value}")
    else:
        for loss_fn, loss_value, validation in zip(loss_controller.losses, losses[-1], validation_losses[-1]):
            print(f"{loss_fn.name} = {loss_value}, Validation = {validation}")

def plot_losses(loss_controller, losses, validation_losses, filename, filetype):
    plt.clf()
    fig = plt.figure()
    fig.set_size_inches(6.2,4)

    for i, loss_fn in enumerate(loss_controller.losses):
        plt.plot([row[i] for row in losses], color = loss_fn.color, label = f"{loss_fn.name} Loss")

    if validation_losses != []:
        for i, loss_fn in enumerate(loss_controller.losses):
            plt.plot([row[i] for row in validation_losses], color = loss_fn.color, linestyle = "-.", label = f"{loss_fn.name} Validation")

    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/losses/{filename+filetype}")

    plt.clf()

def save_losses(loss_controller, losses, validation_losses, filename):
    #Convert loss information into dictionary to be saved
    loss_dict = {}
    for i, loss_fn in enumerate(loss_controller.losses):
        loss_dict[loss_fn.name] = [float(row[i]) for row in losses]
    
    if validation_losses != []:
        for i, loss_fn in enumerate(loss_controller.losses):
            loss_dict[f"{loss_fn.name} Validation"] = [float(row[i]) for row in validation_losses]
    
    with open(f"loss_evolution_results/{filename}.dat", "wb") as f:
        pickle.dump(loss_dict, f)

def train_with_safety(
        model, 
        model_params, 
        target,
        target_params,
        loss_controller,
        num_epochs,
        learning_rate,
        filename = "None",
        plot_ranges = [(-7,7,0.3),(-7,7,0.3)],
        plot_epochs = 100,
        plot_intermediate = False,
        decay = False,
        warmup = 0,
        key = random.PRNGKey(0),
    ):

    if target is not None and target_params is None:
        target_params = target.init(0,0)

    print("STARTING TRAINING")

    warmup = int(num_epochs * warmup)

    if decay:
        main_schedule = optax.cosine_decay_schedule(
            init_value=learning_rate, 
            decay_steps=(num_epochs-warmup)*loss_controller.num_batches,
        )
    else:
        main_schedule = optax.constant_schedule(learning_rate)

    # Define the number of steps for the warmup phase
    warmup_steps = warmup * loss_controller.num_batches
        
    # Create a linear warmup schedule from 0 (or a small value) to the initial learning rate
    warmup_schedule = optax.linear_schedule(
        init_value=0.0,  # or a small fraction of the learning_rate for a smoother start
        end_value=learning_rate,
        transition_steps=warmup_steps,
    )
        
    # Combine the warmup and main schedules
    schedule = optax.join_schedules(
        schedules=[warmup_schedule, main_schedule],
        boundaries=[warmup_steps],
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_norm=1e2),
        optax.adamw(learning_rate = schedule, weight_decay = 0.0001),
    )
    opt_state = optimizer.init(model_params)

    losses = []
    validation_losses = []

    sampler_states = None

    for epoch in range(num_epochs):
        
        start_time = time.time()

        if loss_controller.has_validation:
            validation, key = loss_controller.get_validation_losses(key, model_params, target_params)
            validation_losses.append(validation)

        model_params, opt_state, loss, key, sampler_states = loss_controller.train_losses(
            key, 
            model_params, 
            target_params, 
            optimizer, 
            opt_state,
            sampler_states,
        )

        losses.append(loss)

        end_time = time.time()

        print()
        print(f"Epoch {epoch}, Time Elapsed = {end_time-start_time:.2f}s")
        print_losses(loss_controller, losses, validation_losses)

        if epoch % plot_epochs == 0 and plot_intermediate:
            save_params(model_params,f"flow_params/{filename}.flow")
            
            model.plot(
                model_params,
                plot_ranges,
                target,
                target_params,
                plot_difference = True,
                save = True,
                filename = f"plots/distributions/{filename}",
                filetype = ".png",
            )

            plot_losses(loss_controller, losses, validation_losses, filename, filetype = ".png")
            save_losses(loss_controller, losses, validation_losses, filename)

    save_params(model_params,f"flow_params/{filename}.flow")

    model.plot(
        model_params,
        plot_ranges,
        target,
        target_params,
        plot_difference = True,
        save = True,
        filename = f"plots/distributions/{filename}",
        filetype = ".png",
    )
    
    plot_losses(loss_controller, losses, validation_losses, filename, filetype = ".png")
    save_losses(loss_controller, losses, validation_losses, filename)

    return model_params


def train(
        model, 
        model_params, 
        target,
        target_params,
        loss_controller,
        num_epochs,
        learning_rate,
        filename = "None",
        plot_ranges = [(-7,7,0.1),(-7,7,0.1)],
        plot_epochs = 100,
        plot_intermediate = False,
        decay = False,
        warmup = 0,
        key = random.PRNGKey(0),
    ):

    if key is None:
        key = random.PRNGKey(0)

    path_in = filename[:filename.rfind("/")+1]
    file_prefix = filename[filename.rfind("/")+1:]
    print(f"plots/distributions/{path_in}")
    print(file_prefix)

    try:
        params = train_with_safety(
            model, 
            model_params, 
            target,
            target_params,
            loss_controller,
            num_epochs,
            learning_rate,
            filename = filename,
            plot_ranges = plot_ranges,
            plot_epochs = plot_epochs,
            plot_intermediate = plot_intermediate,
            decay = decay,
            warmup = warmup,
            key = key,
        )
    
    except Exception as e:
        traceback.print_exc()

    # Execute video.py with the following arguments:
    # path_in = filename up to and including the last /
    # file_prefix = filename after the last /
    # --delete_in
    # --fps = 10

    figures_to_video(f"plots/distributions/{path_in}", file_prefix, fps = 10, delete_in = True)

    return params