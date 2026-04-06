import pickle
import random
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from jax.random import PRNGKey,uniform,split, normal

from os.path import exists

from Q_flows.continuous import get_FFJORD
from Q_flows.flow_util import FFJORDNet, sine
from distributions.Q_function import Q, QFlow
from distributions.Q_targets import Q_BEC, Q_GKP, Q_Binomial, Q_CatState, Q_Num, Q_Tensor_Product
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

def plot_data(axes, x, y, pred, exact, x_range, y_range):
    img1 = axes[0].imshow(pred, extent = (x_range[0],x_range[1],y_range[0],y_range[1]), aspect = "auto", vmin = 0, vmax = 0.18)

    if exact != None:
        img2 = axes[1].imshow(exact,extent = (x_range[0],x_range[1],y_range[0],y_range[1]), aspect = "auto", vmin = 0, vmax = 0.18)
        img3 = axes[2].imshow(pred-exact,extent = (x_range[0],x_range[1],y_range[0],y_range[1]), aspect = "auto", vmin = -0.025, vmax = 0.025)

    return img1, img2, img3

def plot_Q_2dim(model, axes, dimension, model_params, x_range, y_range, exact = None, exact_params = None, plot_difference = False, num_dims = 2, mc_sample_num = 50, model_samples = None, exact_samples = None):
    x = jnp.arange(x_range[0],x_range[1],x_range[2])
    y = jnp.arange(y_range[0],y_range[1],y_range[2])

    x,y = jnp.meshgrid(x,y)

    xFlat = x.flatten()
    yFlat = y.flatten()

    key = PRNGKey(0)

    if model_samples == None:
        pred_samples = []
        for i in range(100):
            key, subkey = split(key)
            pred_samples.append(model.sample(model_params, 5000, key))
            #pred_samples.append(normal(key, (5000, 12)))

        pred_samples = jnp.concatenate(pred_samples, axis=0)
    else:
        pred_samples = model_samples

    probs_pred, _, _ = jnp.histogram2d(
        pred_samples[:, 2*dimension], 
        pred_samples[:, 2*dimension+1], 
        bins=[
            jnp.arange(x_range[0],x_range[1],x_range[2]),
            jnp.arange(y_range[0],y_range[1],y_range[2]),
        ],
    )

    probs_pred /= 500000 * x_range[2] * y_range[2]

    if plot_difference:#exact != None and exact_params != None and plot_difference:
        if exact_samples == None:
            exact_samples = []
            for i in range(100):
                key, subkey = split(key)
                exact_samples.append(exact.sample(exact_params, 5000, key))
                #exact_samples.append(normal(key, (5000, 12)))

            exact_samples = jnp.concatenate(exact_samples, axis=0)
        else:
            exact_samples = exact_samples

        probs_exact, _, _ = jnp.histogram2d(
            exact_samples[:, 2*dimension], 
            exact_samples[:, 2*dimension+1], 
            bins=[
                jnp.arange(x_range[0],x_range[1],x_range[2]),
                jnp.arange(y_range[0],y_range[1],y_range[2]),
            ],
        )
        probs_exact /= 500000 * x_range[2] * y_range[2]
    else:
        probs_exact = None

    print(f"Exact total prob: {jnp.sum(probs_exact)*x_range[2]*y_range[2]}")
    print(f"Predicted total prob: {jnp.sum(probs_pred)*x_range[2]*y_range[2]}")

    img1, img2, img3 = plot_data(axes, x, y, probs_pred, probs_exact, x_range, y_range)

    if dimension == 0:
        # Set y ticks
        axes[0].set_yticks([-4,-2,0,2,4])
        axes[1].set_yticks([-4,-2,0,2,4])
        axes[2].set_yticks([-4,-2,0,2,4])
    else:
        axes[0].set_yticks([])
        axes[1].set_yticks([])
        axes[2].set_yticks([])

    # Only axes[2] gets x ticks
    axes[0].set_xticks([])
    axes[1].set_xticks([])
    axes[2].set_xticks([-4,-2,0,2,4])

    return model_samples, exact_samples, img1, img2, img3
        
def plot(model, model_params, target, target_params, x_range, y_range, filename):
    fig, axes = plt.subplots(3, 6, squeeze=False,
                             constrained_layout=False, figsize=(2*6+0.5, 2.5*3))
    
    fig.subplots_adjust(wspace=0.02, hspace=0.02, right=0.85)

    model_samples = None
    exact_samples = None
    
    for i in range(6):
        model_samples, exact_samples, img1, img2, img3 = plot_Q_2dim(
            model, 
            axes[:,i], 
            i, 
            model_params, 
            x_range, 
            y_range, 
            exact = target, 
            exact_params = target_params, 
            plot_difference = True, 
            model_samples = model_samples, 
            exact_samples = exact_samples,
        )

    axes[0,0].set_ylabel('Predicted', fontsize=12)
    axes[1,0].set_ylabel('Target', fontsize=12)
    axes[2,0].set_ylabel('Difference', fontsize=12)

    axes[0,0].set_title('Well 1\nCumulative', fontsize=12)
    axes[0,1].set_title('Well 2\nCumulative', fontsize=12)
    axes[0,2].set_title('Well 3\nCumulative', fontsize=12)
    axes[0,3].set_title('Well 4\nCumulative', fontsize=12)
    axes[0,4].set_title('Well 5\nCumulative', fontsize=12)
    axes[0,5].set_title('Well 6\nCumulative', fontsize=12)
    
    # Add two colorbars
    cax1 = fig.add_axes([0.86, 0.37, 0.02, 0.51])  # [left, bottom, width, height]
    cbar1 = fig.colorbar(img1, cax=cax1)

    cax2 = fig.add_axes([0.86, 0.11, 0.02, 0.25])
    cbar2 = fig.colorbar(img3, cax=cax2)
        
    plt.savefig(filename, bbox_inches='tight')
    print(f"Result file saved to {filename}")

def get_model_and_params(target, model_param_file):
    target_params = target.init(PRNGKey(0),jnp.zeros((1,target.input_dim)))
    target_samples = target.sample(target_params,2000,PRNGKey(0),sampler = "mcmc")
    
    rescale = target_samples.std(axis = 0)

    flow = get_FFJORD(
        num_layers = 5,
        internal_layer_size = 80,
        scale_layer=True,
        adaptive = True,
        dt = 0.1,
        model = FFJORDNet,
        combine_ty = False,
        activation = sine,
        num_encodings = 0,
        rescale = rescale,
    )

    model = QFlow(
        flow_init = flow,
        input_dim = 12,
    )

    model_params = load_params(model_param_file)

    return model, model_params, target, target_params


model, model_params, target, target_params = get_model_and_params(
    Q_Tensor_Product([Q_Num(0)]*8),
    "flow_params/Q/CNF_8_Num_FFJORD_Adaptive_layer_size=80_network_layers=5_KL_model_control_num_samples=600_epochs=4000_cosine_decay_warmup=0.1.flow",
)

plot(
    model,
    model_params,
    target,
    target_params,
    x_range = (-5,5,0.1),
    y_range = (-5,5,0.1),
    filename = "figure_plotting/figure_a1.pdf",
)