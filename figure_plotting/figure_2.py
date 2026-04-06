import pickle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from jax.random import PRNGKey,uniform,split

from os.path import exists

from Q_flows.continuous import get_FFJORD
from Q_flows.flow_util import FFJORDNet, sine
from distributions.Q_function import Q, QFlow
from distributions.Q_targets import Q_BEC, Q_GKP, Q_Binomial, Q_CatState, Q_Num
from distributions.W_function import QWigner, Wigner
from distributions.W_targets import W_GKP, W_Binomial, W_CatState, W_Num, W_n_particle
from flow_IO import load_params

def get_inputs(x_range, y_range):
    x = jnp.arange(x_range[0],x_range[1],x_range[2])
    y = jnp.arange(y_range[0],y_range[1],y_range[2])

    x,y = jnp.meshgrid(x,y)

    xFlat = x.flatten()
    yFlat = y.flatten()

    key = PRNGKey(0)

    ins =  jnp.concatenate((jnp.expand_dims(xFlat,axis=1),jnp.expand_dims(yFlat,axis=1)),axis=1)

    return x, y, ins

def plot_2dim(model, axis, model_params, x_range, y_range, c_range, title, yaxis_label = None, xaxis_ticks = None, yaxis_ticks = None):
    x, y, ins = get_inputs(x_range, y_range)

    if isinstance(model, Q):
        probs_pred = model.Q(model_params, ins, PRNGKey(0))
        c_range = c_range["Q"]
        colors = sns.color_palette("icefire", n_colors=256)[128:]
        cmap = LinearSegmentedColormap.from_list('icefire_half', colors)
    elif isinstance(model, Wigner):
        print(model)
        probs_pred = model.w(model_params, ins, PRNGKey(0))
        c_range = c_range["Wigner"]
        cmap = sns.color_palette("icefire", as_cmap=True)
    probs_pred = jnp.reshape(probs_pred,x.shape)

    print(f"Max prob: {jnp.max(probs_pred)}")
    print(f"Min prob: {jnp.min(probs_pred)}")

    print(f"Total prob: {jnp.sum(probs_pred)*x_range[2]*y_range[2]}")

    img = axis.imshow(
        probs_pred,
        extent = (
            x_range[0],
            x_range[1],
            y_range[0],
            y_range[1],
        ), 
        vmin=c_range[0], 
        vmax=c_range[1], 
        aspect = "auto",
        cmap=cmap,
    )

    if title:
        axis.set_title(title, fontsize=14)

    if yaxis_label:
        axis.set_ylabel(yaxis_label, fontsize=14)

    axis.set_xticks(xaxis_ticks)
    axis.set_yticks(yaxis_ticks)
    axis.tick_params(labelsize=12)

    return img
        
def plot(models, model_params, x_ranges, y_ranges, titles, filename):
    fig, axes = plt.subplots(len(models), len(models[0]), squeeze=False,
                             constrained_layout=False, figsize=(2*len(models[0])+0.5, 2.5*len(models)))
    
    fig.subplots_adjust(wspace=0.02, hspace=0.02, right=0.85)
    c_ranges = {"Q": (0, 0.15), "Wigner": (-0.3, 0.3)}
    q_img = None
    wigner_img = None

    # use  "cool" for W and "magma" for Q
    
    for i in range(len(models)):
        for j in range(len(models[i])):
            print(i,j)
            img = plot_2dim(
                models[i][j],
                axes[i][j],
                model_params[i][j],
                x_ranges[j],
                y_ranges[i],
                c_ranges,
                titles[j] if i==0 else None,
                yaxis_label = ("True" if i%2 == 0 else "Fit") if j == 0 else None,
                xaxis_ticks = [],#[-4, -2, 0, 2, 4] if i == len(models)-1 else [],
                yaxis_ticks = [],#[-4, -2, 0, 2, 4] if j == 0 else [],
            )
            if i < 2 and q_img is None:
                q_img = img
            elif i >= 2 and wigner_img is None:
                wigner_img = img
    
    # Add two colorbars
    cax1 = fig.add_axes([0.86, 0.51, 0.02, 0.36])
    cbar1 = fig.colorbar(q_img, cax=cax1)
    cbar1.set_label('Q Function', rotation=270, labelpad=15, fontsize=14)
    cbar1.ax.tick_params(labelsize=12)

    cax2 = fig.add_axes([0.86, 0.12, 0.02, 0.36])
    cbar2 = fig.colorbar(wigner_img, cax=cax2)
    cbar2.set_label('Wigner Function', rotation=270, labelpad=15, fontsize=14)
    cbar2.ax.tick_params(labelsize=12)
    
    plt.savefig(filename, bbox_inches='tight')
    print(f"Result file saved to {filename}")

def find_rescale(params):
    if isinstance(params, list) or isinstance(params, tuple):
        for item in params:
            rescale = find_rescale(item)

            if rescale is not None:
                return rescale
            
        return None
    
    elif isinstance(params, dict):
        for key in params:
            if key == "rescale":
                return params[key]
            
            rescale = find_rescale(params[key])

            if rescale is not None:
                return rescale
            
        return None
    
    elif isinstance(params, jax.Array):
        return None
    
    else:
        raise ValueError(f"Expected, list, dict, tuple, or jax Array, got type {type(params)}")


def get_model_and_params(target, model_param_file):
    model_params = load_params(model_param_file)
    
    rescale = find_rescale(model_params)

    if rescale is None:
        target_samples = target.sample((),1000,PRNGKey(0),sampler = "mcmc")
        
        rescale = target_samples.std(axis = 0)

    else:
        print("RESCALE FOUND:", rescale)

    flow = get_FFJORD(
        num_layers = 5,
        internal_layer_size = 30,
        scale_layer=True,
        adaptive = True,
        dt = 0.1,
        model = FFJORDNet,
        combine_ty = False,
        activation = sine,
        num_encodings = 0,
        rescale = rescale,
    )

    if isinstance(target, Q):
        model = QFlow(
            flow_init = flow,
            input_dim = 2,
        )
    elif isinstance(target, Wigner):
        model = QWigner(
            flow_init = flow,
            input_dim = 2,
            positive_scale = False,
        )

    return model, model_params

def get_models_and_params(targets, model_param_files):
    models = []
    model_params = []

    for i in range(len(targets)):
        models.append([])
        models.append([])
        model_params.append([])
        model_params.append([])

        for j in range(len(targets[i])):
            model, params = get_model_and_params(targets[i][j], model_param_files[i][j])

            models[2*i].append(targets[i][j])
            models[2*i+1].append(model)
            model_params[2*i].append(())
            model_params[2*i+1].append(params)

    return models, model_params

models, model_params = get_models_and_params(
    [
        [
            Q_CatState(1,2), 
            Q_BEC(phi = jnp.array([1]), n=10, num_wells=1, normalized=True,),
            Q_Num(0),
            Q_Binomial(5,2,mu=0),
            Q_GKP(0.3,0,20,30),
        ],
        [
            W_CatState(-1,2,2), 
            W_n_particle(10),
            W_Num(0),
            W_Binomial(5,2,mu=0),
            W_GKP(0.3,0,20,30),
        ],
    ],
    [
        [
            "flow_params/Q/CNF_cat_FFJORD_SINE_Adaptive_layer_size=30_network_layers=5_KL_efficient_cosine_decay_warmup=0.1.flow", 
            "flow_params/Q/CNF_10_FFJORD_SINE_Adaptive_layer_size=30_network_layers=5_KL_efficient_cosine_decay_warmup=0.1.flow", 
            "flow_params/Q/CNF_num_0_FFJORD_SINE_Adaptive_layer_size=30_network_layers=5_KL_efficient_cosine_decay_warmup=0.1.flow", 
            "flow_params/Q/CNF_binom_0_FFJORD_SINE_Adaptive_layer_size=30_network_layers=5_KL_efficient_cosine_decay_warmup=0.1.flow",
            "flow_params/Q/CNF_GKP_FFJORD_SINE_Adaptive_layer_size=30_network_layers=5_KL_efficient_epochs=1000_cosine_decay_warmup=0.1.flow",
        ],
        [
            "flow_params/Wigner/CNF_cat_FFJORD_SINE_Adaptive_layer_size=30_network_layers=5_L1_efficient_cosine_decay_warmup=0.1.flow", 
            "flow_params/Wigner/CNF_10_FFJORD_SINE_Adaptive_layer_size=30_network_layers=5_L1_efficient_epochs=4000_cosine_decay_warmup=0.1.flow", 
            "flow_params/Wigner/CNF_num_0_FFJORD_SINE_Adaptive_layer_size=30_network_layers=5_L1_efficient_cosine_decay_warmup=0.1.flow", 
            "flow_params/Wigner/CNF_binom_0_FFJORD_SINE_Adaptive_layer_size=30_network_layers=5_L1_efficient_epochs=1000_cosine_decay_warmup=0.1.flow",
            "flow_params/Wigner/CNF_GKP_FFJORD_SINE_Adaptive_layer_size=30_network_layers=5_L1_efficient_epochs=4000_cosine_decay_warmup=0.1.flow",
        ],
    ]
)

plot(
    models,
    model_params,
    x_ranges = [(-5,5,0.1)]*5,#[(-5,5,0.02)]*5,
    y_ranges = [(-5,5,0.1)]*4,#[(-5,5,0.02)]*4,
    titles = ["Cat", "Fock", "Num", "Binomial", "GKP"],
    filename = "figure_plotting/figure2.pdf",
)