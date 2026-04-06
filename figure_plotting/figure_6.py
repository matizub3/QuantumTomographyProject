import pickle
import random

from matplotlib.colors import LinearSegmentedColormap
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import jax
from jax.random import PRNGKey,uniform,split, normal

import numpy as np
import seaborn as sns

from os.path import exists

from Q_flows.continuous import get_FFJORD
from Q_flows.flow_util import FFJORDNet, sine
from distributions.Q_function import Q, QFlow
from distributions.Q_targets import Q_BEC, Q_GKP, Q_Binomial, Q_CatState, Q_Num, Q_Rotation, Q_Tensor_Product
from distributions.W_function import QWigner, Wigner
from distributions.W_targets import W_GKP, W_Binomial, W_CatState, W_Num, W_Tensor_Product, W_n_particle
from flow_IO import load_params
from plotting import plot_Q_2dim


def plot_data(axes, pred, exact, x_range, y_range):
    cmap = sns.color_palette("icefire", as_cmap=True)
    img1 = axes[0].imshow(pred, extent = (x_range[0],x_range[1],y_range[0],y_range[1]), aspect = "auto", vmin = -0.25, vmax = 0.25, cmap = cmap)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    if exact != None:
        img2 = axes[1].imshow(exact,extent = (x_range[0],x_range[1],y_range[0],y_range[1]), aspect = "auto", vmin = -0.25, vmax = 0.25, cmap = cmap)
        img3 = axes[2].imshow(pred-exact,extent = (x_range[0],x_range[1],y_range[0],y_range[1]), aspect = "auto", vmin = -0.25, vmax = 0.25, cmap = cmap)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[2].set_xticks([])
        axes[2].set_yticks([])

    return img1, img2, img3

def compute_W_cumulatives(w_flow: Wigner, w_flow_params, num_dims, x_range, y_range):
    probs_cross_sections = []
    if num_dims > 2 and not isinstance(w_flow, W_Tensor_Product):
        pred_samples_0 = []
        pred_samples_1 = []
        
        if isinstance(w_flow, QWigner):
            key = PRNGKey(0)
            for i in range(1000):
                print(i)
                key, subkey = split(key)
                pred_samples_0.append(w_flow.flow_sample(key, w_flow_params["Q_params"][0], 5000))
                pred_samples_1.append(w_flow.flow_sample(key, w_flow_params["Q_params"][1], 5000))

            pred_samples_0 = jnp.concatenate(pred_samples_0, axis=0)
            pred_samples_1 = jnp.concatenate(pred_samples_1, axis=0)
            
            for dimension in range(num_dims//2):
                probs_pred_0, _, _ = jnp.histogram2d(
                    pred_samples_0[:, 2*dimension], 
                    pred_samples_0[:, 2*dimension+1], 
                    bins=[
                        jnp.linspace(x_range[0],x_range[1],x_range[2]),
                        jnp.linspace(y_range[0],y_range[1],y_range[2]),
                    ],
                    density=True,
                )

                probs_pred_1, _, _ = jnp.histogram2d(
                    pred_samples_1[:, 2*dimension], 
                    pred_samples_1[:, 2*dimension+1], 
                    bins=[
                        jnp.linspace(x_range[0],x_range[1],x_range[2]),
                        jnp.linspace(y_range[0],y_range[1],y_range[2]),
                    ],
                    density=True,
                )

                scale = w_flow_params["scale"][0]
                w_pred = (scale+1) * probs_pred_0 - scale * probs_pred_1

                probs_cross_sections.append(w_pred.T)

            return probs_cross_sections
    
    elif num_dims == 2:
        x = jnp.arange(x_range[0],x_range[1],x_range[2])
        y = jnp.arange(y_range[0],y_range[1],y_range[2])

        x,y = jnp.meshgrid(x,y)

        xFlat = x.flatten()
        yFlat = y.flatten()

        ins = jnp.concatenate((jnp.expand_dims(xFlat,axis=1),jnp.expand_dims(yFlat,axis=1)),axis=1)

        w_values = w_flow.w(w_flow_params, ins, subkey)

        return w_values
    
    else:
        x = jnp.linspace(x_range[0],x_range[1],x_range[2])
        x = (x[:-1] + x[1:])/2
        y = jnp.linspace(y_range[0],y_range[1],y_range[2])
        y = (y[:-1] + y[1:])/2

        x,y = jnp.meshgrid(x,y)

        xFlat = x.flatten()
        yFlat = y.flatten()

        ins = jnp.concatenate((jnp.expand_dims(xFlat,axis=1),jnp.expand_dims(yFlat,axis=1)),axis=1)

        w_values = []

        key = PRNGKey(0)

        for w_flow_slice, w_flow_params_slice in zip(w_flow.Ws, w_flow_params):
            key, subkey = split(key)
            w_values.append(w_flow_slice.w(w_flow_params_slice, ins, subkey).reshape(x_range[2]-1, y_range[2]-1))

        return w_values
    
def plot(model, model_params, target, target_params, x_range, y_range, filename, input_dim):
    fig, axes = plt.subplots(3, input_dim//2, squeeze=False,
                             constrained_layout=False, figsize=(2*(input_dim//2+1)+0.5, 2.5*3))
    
    fig.subplots_adjust(wspace=0.02, hspace=0.02, right=0.85)

    model_values = compute_W_cumulatives(model, model_params, input_dim, x_range, y_range)
    target_values = compute_W_cumulatives(target, target_params, input_dim, x_range, y_range)

    print(len(model_values), model_values[0].shape)
    print(len(target_values), target_values[0].shape)
    
    for i in range(input_dim//2):
        img1, img2, img3 = plot_data(axes[:,i], model_values[i], target_values[i], x_range, y_range)

    axes[0,0].set_ylabel('Predicted', fontsize=16)
    axes[1,0].set_ylabel('Target', fontsize=16)
    axes[2,0].set_ylabel('Difference', fontsize=16)

    for i in range(input_dim//2):
        axes[0,i].set_title(f'Well {i}\nCumulative', fontsize=16)
    
    # Add two colorbars
    cax1 = fig.add_axes([0.86, 0.11, 0.02, 0.77])
    cbar1 = fig.colorbar(img1, cax=cax1)
    cbar1.ax.tick_params(labelsize=14)
        
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

def get_model_and_params(target, model_param_file, input_dim):
    target_params = target.init(PRNGKey(0),jnp.zeros((1,target.input_dim)))

    model_params = load_params(model_param_file)

    rescale = find_rescale(model_params)

    if rescale is None:
        target_samples = target.sample(target_params,600,PRNGKey(0),sampler = "mcmc")
        rescale = target_samples.std(axis = 0)
    else:
        print("RESCALE FOUND:", rescale)

    flow = get_FFJORD(
        num_layers = 5,
        internal_layer_size = 80,
        scale_layer=True,
        adaptive = True,
        dt = 0.1,
        model = FFJORDNet,
        combine_ty = False,
        activation = jax.nn.gelu,
        num_encodings = 0,
        rescale = rescale,
    )

    model = QWigner(
        flow_init = flow,
        input_dim = input_dim,
    )

    return model, model_params, target, target_params

distribution1 = W_CatState(1,1.5,2)
distribution2 = W_n_particle(2)
distribution3 = W_Num(0)

# distribution4 = W_Rotation(Q_CatState(1,1), rotation = jnp.array([[0,1],[1,0]]))
# distribution5 = W_Num(1)

distribution = W_Tensor_Product([distribution1, distribution2, distribution3])#, distribution4, distribution5])

input_dim = distribution.input_dim

model, model_params, target, target_params = get_model_and_params(
    distribution,
    "flow_params/Wigner/CNF_all3_FFJORD_Adaptive_layer_size=80_network_layers=5_L1_efficient_num_samples=600_epochs=8000_cosine_decay_warmup=0.1.flow",
    input_dim,
)

plot(
    model,
    model_params,
    target,
    target_params,
    x_range = (-5,5,100),
    y_range = (-5,5,100),
    filename = "figure_plotting/figure6.pdf",
    input_dim = input_dim,
)