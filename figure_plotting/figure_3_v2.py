import pickle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from jax.random import PRNGKey,uniform,split

from os.path import exists

import seaborn as sns

import numpy as np

from Q_flows.continuous import get_FFJORD
from Q_flows.flow_util import FFJORDNet, sine
from distributions.Q_function import Q, QFlow
from distributions.Q_targets import Q_BEC, Q_GKP, Q_Binomial, Q_CatState, Q_Num
from distributions.W_function import QWigner, Wigner
from distributions.W_targets import W_GKP, W_Binomial, W_CatState, W_Num, W_n_particle
from flow_IO import load_params

def get_inputs():
    xvec = np.load("baselines/data/xvector.npy")
    yvec = np.load("baselines/data/yvector.npy")

    X,Y = np.meshgrid(xvec,yvec)
    samples = np.stack([X.flatten(),Y.flatten()], axis = -1)

    w_experimental = np.load("baselines/data/wigner_data.npy")
    w_reconstructed = np.load("baselines/data/wigner_reconstructed.npy")

    return samples, w_experimental, w_reconstructed, X, Y

def plot(model, model_params):
    # Set global font sizes
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11
    })
    
    # Create figure
    fig = plt.figure(figsize=(8, 8))  # Keep square layout
    
    # Create gridspec for 2x2 layout with adjusted spacing
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], 
                         hspace=0.35, wspace=0.2)  # Increased vertical spacing, kept horizontal tight
    
    # Create the subplots
    ax1 = fig.add_subplot(gs[0, 0])  # Loss plot (top-left)
    ax2 = fig.add_subplot(gs[0, 1])  # Our reconstruction (top-right)
    ax3 = fig.add_subplot(gs[1, 0])  # QST-CGAN (bottom-left)
    ax4 = fig.add_subplot(gs[1, 1])  # Experimental (bottom-right)
    
    # Store axes in the same format as before for compatibility
    axes = [[ax1, ax2, ax3, ax4]]
    
    inputs, w_experimental, w_reconstructed, X, Y = get_inputs()
    
    probs_pred = model.w(model_params, inputs, PRNGKey(0))
    
    probs_pred = jnp.reshape(probs_pred, X.shape)

    print(f"Max prob: {jnp.max(probs_pred)}")
    print(f"Min prob: {jnp.min(probs_pred)}")

    # New line plot on the leftmost axis
    l1_loss_cst = np.load("baselines/data/l1_loss.npy")
    l1_loss_cnf = pickle.load(open("loss_evolution_results/Wigner/CNF_FFJORD_Adaptive_layer_size=30_network_layers=5_QST_CGAN_W_Neg_num_samples=1000_epochs=1000_cosine_decay_warmup=0.1.dat", "rb"))
    
    axes[0][0].plot(l1_loss_cnf["L1_sample_uniform"], label = "Q-Flow")
    axes[0][0].plot(l1_loss_cst, label = "QST-CGAN")
    axes[0][0].legend(frameon=False)  # Made legend cleaner without frame
    axes[0][0].set_yscale("log")
    axes[0][0].set_xscale("log")
    axes[0][0].set_title('L1 Loss', pad=10)  # Added some padding to title
    axes[0][0].set_xlabel('Epoch')
    axes[0][0].set_ylabel('Loss')

    c_range = (-0.3, 0.3)

    im1 = axes[0][1].imshow(
        probs_pred,
        extent = (
            X[0,0],
            X[0,-1],
            Y[0,0],
            Y[-1,0],
        ), 
        vmin=c_range[0], 
        vmax=c_range[1], 
        aspect = "auto",
        cmap=sns.color_palette("icefire", as_cmap=True),
    )
    axes[0][1].set_title('Our Reconstruction')
    axes[0][1].set_xlabel('Re(α)')
    axes[0][1].set_ylabel('Im(α)')

    im2 = axes[0][2].imshow(
        w_reconstructed,
        extent = (
            X[0,0],
            X[0,-1],
            Y[0,0],
            Y[-1,0],
        ), 
        aspect = "auto",
        vmin=c_range[0], 
        vmax=c_range[1], 
        cmap=sns.color_palette("icefire", as_cmap=True),
    )
    axes[0][2].set_title('QST-CGAN')
    axes[0][2].set_xlabel('Re(α)')
    axes[0][2].set_yticks([])

    mask = (X**2 + Y**2) <= 2**2

    print("Q_FLOW", jnp.sum(jnp.abs(probs_pred - w_experimental)[mask]) * (X[0,1] - X[0,0]) * (Y[1,0] - Y[0,0]))
    print("CGAN", jnp.sum(jnp.abs(w_reconstructed - w_experimental)[mask]) * (X[0,1] - X[0,0]) * (Y[1,0] - Y[0,0]))

    im3 = axes[0][3].imshow(
        w_experimental,
        extent = (
            X[0,0],
            X[0,-1],
            Y[0,0],
            Y[-1,0],
        ), 
        vmin=c_range[0], 
        vmax=c_range[1], 
        aspect = "auto",
        cmap=sns.color_palette("icefire", as_cmap=True),
    )
    axes[0][3].set_title('Experimental')
    axes[0][3].set_xlabel('Re(α)')
    axes[0][3].set_yticks([])

    # Adjust colorbar position for tighter layout
    fig.subplots_adjust(right=0.9, left=0.1, top=0.9, bottom=0.1)  # Tighter margins
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # Adjusted colorbar position
    cbar = fig.colorbar(im3, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=11)

    plt.savefig("figure_plotting/figure3.pdf", bbox_inches='tight', dpi=300)  # Added higher DPI for better quality

def get_model_and_params(model_param_file):
    rescale = 1

    flow = get_FFJORD(
        num_layers = 5,
        internal_layer_size = 30,
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
        input_dim = 2,
        positive_scale = False,
    )

    model_params = load_params(model_param_file)

    return model, model_params

plot(*get_model_and_params("flow_params/Wigner/CNF_FFJORD_Adaptive_layer_size=30_network_layers=5_QST_CGAN_W_Neg_num_samples=1000_epochs=1000_cosine_decay_warmup=0.1.flow"))