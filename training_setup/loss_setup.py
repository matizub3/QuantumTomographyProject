import jax
from jax import random
import jax.numpy as jnp

import numpy as np

from losses.Q_KL import KL_sample_efficient, KL_sample_model_control_variance, KL_sample_model_reparam, KL_sample_target, KL_sample_uniform
from losses.Q_KL_rev import KL_rev_sample_target, KL_rev_sample_model_reparam, KL_rev_sample_model_control_variance, KL_rev_sample_uniform
from losses.W_loss import W_L1_sample_efficient, W_L1_sample_model, W_L1_sample_target, W_L1_sample_uniform

from losses.multiple_loss import MultipleLoss
jax.config.update("jax_enable_x64", True)

from losses.Q_L1 import L1_sample_model_control_variance, L1_sample_model_reparam, L1_sample_target, L1_sample_uniform


def loss_name_init(filename, kwargs):
    if kwargs["representation"] == "W":
        if kwargs["problem"] == "QST_CGAN_W_Neg":
            print("Using QST_CGAN_W_Neg loss")
            filename += "_QST_CGAN_W_Neg"

        if kwargs["losses"] == ["default"]:
            kwargs["losses"] = ["L1_target"]
        


        if kwargs["losses"][0] == "L1_model_mcmc":
            filename += "_L1_model_mcmc"
        
        if kwargs["losses"][0] == "L1_model_average":
            filename += "_L1_model_average"


        if kwargs["losses"][0] == "L1_efficient":
            filename += "_L1_efficient"




    elif kwargs["representation"] == "Q":
        if kwargs["losses"] == ["default"]:
            kwargs["losses"] = ["L1_target"]
        


        if kwargs["losses"][0] == "L1_model_reparam":
            filename += "_L1_model_reparam"
        
        if kwargs["losses"][0] == "L1_model_control":
            filename += "_L1_model_control"

        if kwargs["losses"][0] == "L1_model":
            filename += "_L1_model"

        if kwargs["losses"][0] == "L1_uniform":
            filename += "_L1_uniform"



        if kwargs["losses"][0] == "KL_target":
            filename += "_KL_target"
        
        if kwargs["losses"][0] == "KL_model_reparam":
            filename += "_KL_model_reparam"
        
        if kwargs["losses"][0] == "KL_model_control":
            filename += "_KL_model_control"

        if kwargs["losses"][0] == "KL_uniform":
            filename += "_KL_uniform"
            if kwargs["control_variate"] == "1":
                filename += "_cv=1"
            elif kwargs["control_variate"] == "mean":
                filename += "_cv=mean"

        if kwargs["losses"][0] == "KL_efficient":
            filename += "_KL_efficient"
            if kwargs["control_variate"] == "1":
                filename += "_cv=1"

        
        if kwargs["losses"][0] == "KL_rev_target":
            filename += "_KL_rev_target"
        
        if kwargs["losses"][0] == "KL_rev_model_reparam":
            filename += "_KL_rev_model_reparam"
        
        if kwargs["losses"][0] == "KL_rev_model_control":
            filename += "_KL_rev_model_control"

        if kwargs["losses"][0] == "KL_rev_uniform":
            filename += "_KL_rev_uniform"

    if kwargs["num_samples"] != 10000:
        filename += f"_num_samples={kwargs['num_samples']}"

    if kwargs["samples_per_resample"] != 100:
        filename += f"_spr={kwargs['samples_per_resample']}"
    
    if kwargs["resample_every"] != 10:
        filename += f"_re={kwargs['resample_every']}"
    
    if kwargs["epochs"] != 500:
        filename += f"_epochs={kwargs['epochs']}"
    
    return filename


def Q_loss_init(model, target, target_params, kwargs, training_samples = None):
    losses = []

    if kwargs["losses"] == ["default"]:
        kwargs["losses"] = ["KL_target"]

    print(kwargs["losses"])

    for loss in kwargs["losses"]:
        if loss == "L1_target":
            losses.append(
                L1_sample_target(
                    rng = random.PRNGKey(0),
                    model = model,
                    target = target,
                    target_params = target_params,
                    num_samples = kwargs["num_samples"],
                    batch_size = kwargs["num_samples"]//kwargs["batches"],
                    resample = True if kwargs["resample"] == "True" else False,
                    has_validation = True if kwargs["validation"] == "True" else False,
                    grad_loss = True if kwargs["losses"][0] == "L1_target" else False,
                    name = "L1_target",
                    color = "tab:blue",
                )
            )

        if loss == "L1_model_reparam":
            losses.append(
                L1_sample_model_reparam(
                    rng = random.PRNGKey(0),
                    model = model,
                    target = target,
                    num_samples = kwargs["num_samples"],
                    batch_size = kwargs["num_samples"]//kwargs["batches"],
                    has_validation = True if kwargs["validation"] == "True" else False,
                    grad_loss = True if kwargs["losses"][0] == "L1_model_reparam" else False,
                    name = "L1_model_reparam",
                    color = "tab:orange",
                )
            )

        if loss == "L1_model_control":
            losses.append(
                L1_sample_model_control_variance(
                    rng = random.PRNGKey(0),
                    model = model,
                    target = target,
                    num_samples = kwargs["num_samples"],
                    batch_size = kwargs["num_samples"]//kwargs["batches"],
                    has_validation = True if kwargs["validation"] == "True" else False,
                    grad_loss = True if kwargs["losses"][0] == "L1_model_control" else False,
                    name = "L1_model_control",
                    color = "tab:green",
                )
            )

        if loss == "L1_model":
            losses.append(
                L1_sample_model_control_variance(
                    rng = random.PRNGKey(0),
                    model = model,
                    target = target,
                    num_samples = kwargs["num_samples"],
                    batch_size = kwargs["num_samples"]//kwargs["batches"],
                    has_validation = True if kwargs["validation"] == "True" else False,
                    grad_loss = True if kwargs["losses"][0] == "L1_model" else False,
                    name = "L1_model",
                    control_variance = False,
                    color = "tab:green",
                )
            )

        if loss == "L1_uniform":
            losses.append(
                L1_sample_uniform(
                    rng = random.PRNGKey(0),
                    model = model,
                    target = target,
                    target_params = target_params,
                    x_range = (-5,5,32),
                    y_range = (-3,3,32),
                    has_validation = False if training_samples is None else True,
                    train_samples = training_samples,
                    grad_loss = True if kwargs["losses"][0] == "L1_uniform" else False,
                    name = "L1_uniform",
                    color = "tab:blue",
                )
            )

        if loss == "KL_target":
            losses.append(
                KL_sample_target(
                    rng = random.PRNGKey(0),
                    model = model,
                    target = target,
                    target_params = target_params,
                    num_samples = kwargs["num_samples"],
                    batch_size = kwargs["num_samples"]//kwargs["batches"],
                    resample = True if kwargs["resample"] == "True" else False,
                    has_validation = True if kwargs["validation"] == "True" else False,
                    grad_loss = True if kwargs["losses"][0] == "KL_target" else False,
                    name = "KL_target",
                    color = "tab:blue",
                )
            )

        if loss == "KL_model_reparam":
            losses.append(
                KL_sample_model_reparam(
                    rng = random.PRNGKey(0),
                    model = model,
                    target = target,
                    num_samples = kwargs["num_samples"],
                    batch_size = kwargs["num_samples"]//kwargs["batches"],
                    has_validation = True if kwargs["validation"] == "True" else False,
                    grad_loss = True if kwargs["losses"][0] == "KL_model_reparam" else False,
                    name = "KL_model_reparam",
                    color = "tab:orange",
                )
            )

        if loss == "KL_model_control":
            losses.append(
                KL_sample_model_control_variance(
                    rng = random.PRNGKey(0),
                    model = model,
                    target = target,
                    num_samples = kwargs["num_samples"],
                    batch_size = kwargs["num_samples"]//kwargs["batches"],
                    has_validation = True if kwargs["validation"] == "True" else False,
                    grad_loss = True if kwargs["losses"][0] == "KL_model_control" else False,
                    name = "KL_model_control",
                    color = "tab:green",
                )
            )

        if loss == "KL_uniform":
            losses.append(
                KL_sample_uniform(
                    model = model,
                    target = target,
                    target_params = target_params,
                    x_range = (-5,5,kwargs["grid_size"]),
                    y_range = (-5,5,kwargs["grid_size"]),
                    has_validation = False,
                    grad_loss = True if kwargs["losses"][0] == "KL_uniform" else False,
                    name = "KL_uniform",
                    color = "tab:green",
                    control_variance=kwargs["control_variate"],
                )
            )

        if loss == "KL_efficient":
            losses.append(
                KL_sample_efficient(
                    model = model,
                    target = target,
                    samples_per_resample = kwargs["samples_per_resample"],
                    resample_every = kwargs["resample_every"],
                    batch_size = kwargs["num_samples"],
                    has_validation = False,
                    grad_loss = True if kwargs["losses"][0] == "KL_efficient" else False,
                    name = "KL_efficient",
                    color = "tab:green",
                    control_variate=kwargs["control_variate"],
                )
            )

        if loss == "KL_rev_target":
            losses.append(
                KL_rev_sample_target(
                    rng = random.PRNGKey(0),
                    model = model,
                    target = target,
                    target_params = target_params,
                    num_samples = kwargs["num_samples"],
                    batch_size = kwargs["num_samples"]//kwargs["batches"],
                    resample = True if kwargs["resample"] == "True" else False,
                    has_validation = True if kwargs["validation"] == "True" else False,
                    grad_loss = True if kwargs["losses"][0] == "KL_rev_target" else False,
                    name = "KL_rev_target",
                    color = "tab:blue",
                )
            )

        if loss == "KL_rev_model_reparam":
            losses.append(
                KL_rev_sample_model_reparam(
                    rng = random.PRNGKey(0),
                    model = model,
                    target = target,
                    num_samples = kwargs["num_samples"],
                    batch_size = kwargs["num_samples"]//kwargs["batches"],
                    has_validation = True if kwargs["validation"] == "True" else False,
                    grad_loss = True if kwargs["losses"][0] == "KL_rev_model_reparam" else False,
                    name = "KL_rev_model_reparam",
                    color = "tab:orange",
                )
            )

        if loss == "KL_rev_model_control":
            losses.append(
                KL_rev_sample_model_control_variance(
                    rng = random.PRNGKey(0),
                    model = model,
                    target = target,
                    num_samples = kwargs["num_samples"],
                    batch_size = kwargs["num_samples"]//kwargs["batches"],
                    has_validation = True if kwargs["validation"] == "True" else False,
                    grad_loss = True if kwargs["losses"][0] == "KL_rev_model_control" else False,
                    name = "KL_rev_model_control",
                    color = "tab:green",
                )
            )

        if loss == "KL_rev_uniform":
            losses.append(
                KL_rev_sample_uniform(
                    model = model,
                    target = target,
                    target_params = target_params,
                    x_range = (-5,5,32),
                    y_range = (-3,3,32),
                    has_validation = False,
                    grad_loss = True if kwargs["losses"][0] == "KL_rev_uniform" else False,
                    name = "KL_rev_uniform",
                    color = "tab:green",
                )
            )

    loss_controller = MultipleLoss(losses)

    return loss_controller


def W_loss_init(model, target, target_params, kwargs, training_samples = None):
    if kwargs["problem"] == "QST_CGAN_W_Neg":
        return get_qst_cgan_w_neg_loss(model, target, target_params, kwargs)

    losses = []

    if kwargs["losses"] == ["default"]:
        kwargs["losses"] = ["L1_target"]

    print(kwargs["losses"])

    for loss in kwargs["losses"]:
        if loss == "L1_target":
            losses.append(
                W_L1_sample_target(
                    rng = random.PRNGKey(0),
                    model = model,
                    target = target,
                    target_params = target_params,
                    num_samples = kwargs["num_samples"],
                    batch_size = kwargs["num_samples"]//kwargs["batches"],
                    resample = True if kwargs["resample"] == "True" else False,
                    has_validation = True if kwargs["validation"] == "True" else False,
                    grad_loss = True if kwargs["losses"][0] == "L1_target" else False,
                    name = "L1_target",
                    color = "tab:blue",
                )
            )

        if loss == "L1_model_mcmc":
            losses.append(
                W_L1_sample_model(
                    model = model,
                    target = target,
                    num_samples = kwargs["num_samples"],
                    batch_size = kwargs["num_samples"]//kwargs["batches"],
                    has_validation = True if kwargs["validation"] == "True" else False,
                    grad_loss = True if kwargs["losses"][0] == "L1_model_mcmc" else False,
                    name = "L1_model_mcmc",
                    sample_method = "mcmc",
                    color = "tab:green",
                )
            )

        if loss == "L1_model_average":
            losses.append(
                W_L1_sample_model(
                    model = model,
                    target = target,
                    num_samples = kwargs["num_samples"],
                    batch_size = kwargs["num_samples"]//kwargs["batches"],
                    has_validation = True if kwargs["validation"] == "True" else False,
                    grad_loss = True if kwargs["losses"][0] == "L1_model_average" else False,
                    name = "L1_model_average",
                    sample_method = "average",
                    color = "tab:orange",
                )
            )

        if loss == "L1_efficient":
            losses.append(
                W_L1_sample_efficient(
                    model = model, 
                    target = target,
                    samples_per_resample = kwargs["samples_per_resample"],
                    resample_every = kwargs["resample_every"],
                    batch_size = kwargs["num_samples"],
                    has_validation = True if kwargs["validation"] == "True" else False,
                    grad_loss = True if kwargs["losses"][0] == "L1_efficient" else False,
                    name = "L1_efficient",
                    sample_method = "average",
                    color = "tab:orange",
                )
            )

    loss_controller = MultipleLoss(losses)

    return loss_controller


def get_qst_cgan_w_neg_loss(model, target, target_params, kwargs):
    print("Using QST_CGAN_W_Neg loss")

    xvec = np.load("baselines/data/xvector.npy")
    yvec = np.load("baselines/data/yvector.npy")

    X,Y = np.meshgrid(xvec,yvec)
    samples = np.stack([X.flatten(),Y.flatten()], axis = -1)

    w_experimental = np.load("baselines/data/wigner_data.npy").flatten()

    #only samples with an L2 distance of 2 or less from the origin
    sample_mask = (samples[:,0]**2 + samples[:,1]**2) <= 2**2

    samples = jnp.array(samples[sample_mask])
    w_experimental = jnp.array(w_experimental[sample_mask])

    loss = W_L1_sample_uniform(
        model = model,
        target = target,
        target_params = target_params,
        samples = samples,
        target_W = w_experimental,
        grad_loss = True,
    )

    return MultipleLoss([loss])