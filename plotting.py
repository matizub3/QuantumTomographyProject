import pickle
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from jax.random import PRNGKey,uniform,split

from os.path import exists

from distributions.W_function import QWigner, Wigner
from distributions.W_targets import W_Tensor_Product

def resolve_range(r):
    lo, hi, third = r

    # integer-like -> interpret as number of grid points / bin edges
    if isinstance(third, int) or (isinstance(third, float) and third.is_integer() and third >= 2):
        n = int(third)
        edges = jnp.linspace(lo, hi, n)
        step = edges[1] - edges[0]
        return {
            "lo": lo,
            "hi": hi,
            "n": n,
            "edges": edges,
            "step": step,
            "mode": "count",
        }

    # non-integer float -> interpret as step size
    step = float(third)
    n = int(round((hi - lo) / step)) + 1
    edges = jnp.linspace(lo, hi, n)
    actual_step = edges[1] - edges[0]

    return {
        "lo": lo,
        "hi": hi,
        "n": n,
        "edges": edges,
        "step": actual_step,
        "mode": "step",
    }

def plot_data(axes, x, y, pred, exact, x_range, y_range):
    img1 = axes[0].imshow(pred, extent = (x_range[0],x_range[1],y_range[0],y_range[1]), aspect = "auto")
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img1, cax=cax)
    axes[0].set_title("Predicted")

    if exact is not None:
        img2 = axes[1].imshow(exact,extent = (x_range[0],x_range[1],y_range[0],y_range[1]), aspect = "auto")
        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(img2, cax=cax)
        axes[1].set_title("Exact")

        img3 = axes[2].imshow(pred-exact,extent = (x_range[0],x_range[1],y_range[0],y_range[1]), aspect = "auto")
        divider = make_axes_locatable(axes[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(img3, cax=cax)
        axes[2].set_title("Difference")

def plot_Q_2dim(model, axes, dimension, model_params, x_range, y_range, exact = None, exact_params = None, plot_difference = False, num_dims = 2, mc_sample_num = 50, model_samples = None, exact_samples = None):
    x_info = resolve_range(x_range)
    y_info = resolve_range(y_range)

    x_edges = x_info["edges"]
    y_edges = y_info["edges"]

    x, y = jnp.meshgrid(x_edges, y_edges)

    xFlat = x.flatten()
    yFlat = y.flatten()

    dx = x_info["step"]
    dy = y_info["step"]

    key = PRNGKey(0)

    if num_dims == 2:
        ins = jnp.concatenate((jnp.expand_dims(xFlat,axis=1),jnp.expand_dims(yFlat,axis=1)),axis=1)

        key, subkey = split(key)
        probs_pred = model.Q(model_params, ins, subkey)

        probs_pred = jnp.reshape(probs_pred,x.shape)

        if exact is not None and exact_params is not None and plot_difference:
            key, subkey = split(key)
            probs_exact = exact.Q(exact_params, ins, subkey)

            probs_exact = jnp.reshape(probs_exact,x.shape)
        else:
            probs_exact = None

    else:
        if model_samples == None:
            pred_samples = []
            for i in range(10):
                key, subkey = split(key)
                pred_samples.append(model.sample(model_params, 5000, key))

            pred_samples = jnp.concatenate(pred_samples, axis=0)
        else:
            pred_samples = model_samples

        probs_pred, _, _ = jnp.histogram2d(
            pred_samples[:, 2*dimension], 
            pred_samples[:, 2*dimension+1], 
            bins=[
                x_edges, y_edges
            ],
        )

        N_pred = pred_samples.shape[0]
        probs_pred /= (N_pred * dx * dy)

        if exact is not None and exact_params is not None and plot_difference:
            if exact_samples == None:
                exact_samples = []
                for i in range(10):
                    key, subkey = split(key)
                    exact_samples.append(exact.sample(exact_params, 5000, key))

                exact_samples = jnp.concatenate(exact_samples, axis=0)
            else:
                exact_samples = exact_samples

            probs_exact, _, _ = jnp.histogram2d(
                exact_samples[:, 2*dimension], 
                exact_samples[:, 2*dimension+1], 
                bins=[
                    x_edges, y_edges
                ],
            )

            N_exact = exact_samples.shape[0]

            probs_exact /= (N_exact * dx * dy)
        else:
            probs_exact = None

    if probs_exact is not None:
        print(f"Exact total prob: {float(jnp.sum(probs_exact) * dx * dy):.6f}")
    
    # print(f"Predicted total prob: {float(jnp.sum(probs_pred) * dx * dy):.6f}")

    plot_data(axes, x, y, probs_pred, probs_exact, x_range, y_range)

    return model_samples, exact_samples
        
def plot_Q(model, model_params, x_range, y_range, exact = None, exact_params = None, plot_difference = False, save = False, filename = None, filetype = None, num_dims = 2):
    fig, axes = plt.subplots( 1 if exact == None else 3, num_dims//2, sharex = True, sharey = True, squeeze = False)
    
    fig.set_size_inches(3.5 * num_dims//2, 6)


    model_samples = None
    exact_samples = None

    for i in range(num_dims//2):
        print(i)
        model_samples, exact_samples = plot_Q_2dim(
            model,
            [axes[j][i] for j in range(len(axes))],
            i,
            model_params,
            x_range,
            y_range,
            exact,
            exact_params,
            plot_difference,
            num_dims,
            model_samples = model_samples,
            exact_samples = exact_samples,
        )
        
    if save:
        plt.tight_layout()
        
        if not exists(filename+filetype):
            plt.savefig(filename+filetype)
            print(f"Result file saved to {filename+filetype}")
        else:
            filenum = 1
            while exists(filename+" "+str(filenum)+filetype):
                filenum += 1
            plt.savefig(filename+" "+str(filenum)+filetype)
            print(f"Result file saved to: {filename} {str(filenum)}{filetype}")
    else:
        plt.tight_layout()
        plt.show()
        plt.close()

def compute_W_cumulatives(w_flow: Wigner, w_flow_params, num_dims, x_range, y_range):
    
    x_info = resolve_range(x_range)
    y_info = resolve_range(y_range)

    x_edges = x_info["edges"]
    y_edges = y_info["edges"]

    dx = x_info["step"]
    dy = y_info["step"]
    
    probs_cross_sections = []
    if num_dims > 2 and not isinstance(w_flow, W_Tensor_Product):
        pred_samples_0 = []
        pred_samples_1 = []
        
        if isinstance(w_flow, QWigner):
            key = PRNGKey(0)
            for i in range(10):
                key, subkey = split(key)
                pred_samples_0.append(w_flow.flow_sample(key, w_flow_params["Q_params"][0], 5000))
                pred_samples_1.append(w_flow.flow_sample(key, w_flow_params["Q_params"][1], 5000))

            pred_samples_0 = jnp.concatenate(pred_samples_0, axis=0)
            pred_samples_1 = jnp.concatenate(pred_samples_1, axis=0)

            N0 = pred_samples_0.shape[0]
            N1 = pred_samples_1.shape[0]

            scale = w_flow_params["scale"][0]
            
            for dimension in range(num_dims//2):
                probs_pred_0, _, _ = jnp.histogram2d(
                    pred_samples_0[:, 2*dimension], 
                    pred_samples_0[:, 2*dimension+1], 
                    bins=[x_edges, y_edges],
                )

                probs_pred_1, _, _ = jnp.histogram2d(
                    pred_samples_1[:, 2*dimension], 
                    pred_samples_1[:, 2*dimension + 1], 
                    bins=[x_edges, y_edges],
                )

                probs_pred_0 = probs_pred_0 / (N0 * dx * dy)
                probs_pred_1 = probs_pred_1 / (N1 * dx * dy)

                # print(probs_pred_0.shape)
                # print(probs_pred_1.shape)

                w_pred = (scale+1) * probs_pred_0 - scale * probs_pred_1
                probs_cross_sections.append(w_pred.T)

                
            return probs_cross_sections
    
    elif num_dims == 2:
        
        # print("x_range =", x_range, "x_range[2] =", x_range[2], type(x_range[2]))
        # print("y_range =", y_range, "y_range[2] =", y_range[2], type(y_range[2]))

        x = x_edges
        y = y_edges

        x,y = jnp.meshgrid(x,y)

        xFlat = x.flatten()
        yFlat = y.flatten()

        ins = jnp.concatenate(
            (jnp.expand_dims(xFlat,axis=1),
             jnp.expand_dims(yFlat,axis=1)),
             axis=1)

        key = PRNGKey(0)
        if isinstance(w_flow, W_Tensor_Product):
            # print("--------------------------------")
            # print("Tensor product!!!!")
            # print("--------------------------------")
            w_values = w_flow.Ws[0].w(w_flow_params[0], ins, key)
        else:
            w_values = w_flow.w(w_flow_params, ins, key)
        # print(w_values.shape)
        # print(w_values.reshape(x.shape).shape)

        return [w_values.reshape(x.shape)]
    
    else:

        # print("x_range =", x_range, "x_range[2] =", x_range[2], type(x_range[2]))
        # print("y_range =", y_range, "y_range[2] =", y_range[2], type(y_range[2]))

        x = (x_edges[:-1] + x_edges[1:]) / 2

        y = (y_edges[:-1] + y_edges[1:])/2

        x,y = jnp.meshgrid(x,y)

        xFlat = x.flatten()
        yFlat = y.flatten()

        ins = jnp.concatenate((jnp.expand_dims(xFlat,axis=1),jnp.expand_dims(yFlat,axis=1)),axis=1)

        w_values = []

        key = PRNGKey(0)

        for w_flow_slice, w_flow_params_slice in zip(w_flow.Ws, w_flow_params):
            key,subkey = split(key)

            # Reshape the output of w to be the same shape as x and y for plotting, use x.shape since x_range[2] was not integer
            w_values.append(w_flow_slice.w(w_flow_params_slice, ins, subkey).reshape(x.shape))

        return w_values


def plot_W_2dim(w_values, axes, x_range, y_range, w_true_values=None, plot_difference=False):
    def draw(ax, values, title):
        img = ax.imshow(
            values,
            extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
            aspect="auto",
            origin="lower",
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(img, cax=cax)
        ax.set_title(title)

    if plot_difference and w_true_values is not None:
        draw(axes[0], w_values, "Predicted")
        draw(axes[1], w_true_values, "Exact")
        draw(axes[2], w_values - w_true_values, "Difference")
    else:
        draw(axes[0], w_values, "Predicted")

        
def plot_W(
    w_flow,
    model_params,
    x_range,
    y_range,
    w_true=None,
    true_params=None,
    plot_difference=False,
    save=False,
    filename=None,
    filetype=None,
    num_dims=2,
):
    w_cumulatives = compute_W_cumulatives(
        w_flow,
        model_params,
        num_dims,
        x_range,
        y_range,
    )

    w_true_cumulatives = (
        compute_W_cumulatives(w_true, true_params, num_dims, x_range, y_range)
        if w_true is not None
        else None
    )

    n_rows = 3 if (w_true is not None and plot_difference) else 1

    fig, axes = plt.subplots(
        n_rows,
        num_dims // 2,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    fig.set_size_inches(4.0 * (num_dims // 2), 3.5 * n_rows)

    for i in range(num_dims // 2):
        plot_W_2dim(
            w_cumulatives[i],
            axes[:, i],
            x_range,
            y_range,
            w_true_values=w_true_cumulatives[i] if w_true_cumulatives is not None else None,
            plot_difference=plot_difference,
        )

    if save:
        plt.tight_layout()

        if not exists(filename + filetype):
            plt.savefig(filename + filetype)
            # print(f"Result file saved to {filename + filetype}")
        else:
            filenum = 1
            while exists(filename + " " + str(filenum) + filetype):
                filenum += 1
            plt.savefig(filename + " " + str(filenum) + filetype)
            # print(f"Result file saved to: {filename} {str(filenum)}{filetype}")
    else:
        plt.tight_layout()
        plt.show()
        plt.close()


def plot_complexity_losses(kwargs, filename, filetype):
    losses = []

    for num_samples in list(jnp.floor(jnp.logspace(1, kwargs["num_samples"], num = 20))):
        new_filename = filename + f"_training_samples={num_samples}"

        with open(f"loss_evolution_results/{new_filename}.dat", "rb") as f:
            loss_dict = pickle.load(f)
        
        for key in loss_dict:
            if "Validation" in key:
                losses.append(loss_dict[key][-1])


    # Plot the last validation loss for each training sample

    plt.clf()
    fig = plt.figure()
    fig.set_size_inches(6.2,4)

    plt.plot(jnp.floor(jnp.logspace(1, kwargs["num_samples"], num = 20)), losses)

    plt.xscale("log")

    plt.xlabel("Training Samples")
    plt.ylabel("Validation Loss")
    plt.tight_layout()
    plt.savefig(f"plots/losses/{filename}_sample_complexity{filetype}")
    plt.close()
