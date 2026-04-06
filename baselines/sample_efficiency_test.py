"""
The below script is used to test the sample efficiency of the QST-CGAN model. It is modified from the code provided by the QST-CGAN paper.
"""

import cmath
import pickle
import numpy as np

from qutip import fock, coherent, coherent_dm, expect, Qobj, fidelity
from qutip.wigner import wigner, qfunc
from qutip.visualization import hinton


import tensorflow as tf

from tqdm.auto import tqdm

from qst_cgan.ops import convert_to_real_ops, batched_expect, tf_fidelity
from qst_cgan.gan import generator_loss, discriminator_loss, Generator, Discriminator


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from dataclasses import dataclass

tf.keras.backend.set_floatx('float64') # Set float64 as the default


import argparse

parser = argparse.ArgumentParser(description='Test the sample efficiency of the QST-CGAN model')

# State type
parser.add_argument('--state_type', type=str, default='coherent', help='Type of state to generate. Options: coherent, squeezed, cat')

hilbert_size=32

def binom(x, y):
  return math.exp(math.lgamma(x + 1) - math.lgamma(y + 1) - math.lgamma(x - y + 1))

def distribution_init(problem):
    if problem == "cat":
        psi = (coherent(hilbert_size, 2) + coherent(hilbert_size, -2)).unit()
    elif problem == "fock":
        psi = fock(hilbert_size, 10)
    elif problem == "num":
        psi = math.sqrt(7 - math.sqrt(17)) * fock(hilbert_size, 0)
        psi += math.sqrt(math.sqrt(17) - 1) * fock(hilbert_size, 3)

        psi = psi.unit()

    elif problem == "binom":
        N=5
        S=2
        mu=0

        psi = fock(hilbert_size, 0)

        for m in range(1, N+2):
            psi += math.sqrt(binom(N+1,m)) * fock(hilbert_size, (S+1)*m)

        psi = psi.unit()

    elif problem == "GKP":
        delta = 0.3
        mu = 0
        cutoff = 20

        phi = 0

        for i in range(-cutoff,cutoff+1):
            for k in range(-cutoff,cutoff+1):
                alpha_re = math.sqrt(math.pi/2) * (2*i+mu)
                alpha_im = math.sqrt(math.pi/2) * k

                alpha = alpha_re + 1j * alpha_im

                normalization = math.exp(-delta**2 * (alpha_re**2 + alpha_im**2))
                normalization *= cmath.exp(- 1j * alpha_re * alpha_im)

                phi += normalization * coherent(hilbert_size, alpha)

        psi = phi.unit()

    else:
        raise ValueError(f"Unknown problem type: {problem}")

    rho = psi * psi.dag()

    return psi, rho

def get_x_y(num_samples, radius):
    """
    Sample randomly from inside a sphere of radius
    """

    grid = num_samples
    x = np.linspace(-5, 5, grid)
    y = np.linspace(-5, 5, grid)

    return x, y

def get_Q(problem, num_samples):
    psi, rho = distribution_init(problem)

    x, y = get_x_y(num_samples, 5)

    X, Y, ops_tf = get_density_matrices(x,y)

    rho_numpy = rho.full().reshape((1, hilbert_size, hilbert_size)) # Conversion to NumPy array and reshaping into (1, N, N) to allow batching
    rho_tf = tf.complex(rho_numpy.real, rho_numpy.imag) # Conversion to TensorFlow tensor

    A = convert_to_real_ops(ops_tf)

    q = batched_expect(ops_tf, rho_tf)

    """cmap = "Blues"
    im = plt.pcolor(x, y, q.numpy().reshape(x.shape[0], x.shape[0]), vmin=0, vmax=np.max(q), cmap=cmap, shading='auto')
    plt.colorbar(im)
    plt.xlabel(r"Re($\beta$)")
    plt.ylabel(r"Im($\beta$)")
    plt.title("Husimi Q function")
    plt.show()
    """

    return X, Y, psi, ops_tf, A, rho_tf, q

def get_density_matrices(x, y):
    X, Y = np.meshgrid(x, y)
    betas = (X + 1j*Y).ravel()
    m_ops = [(1/np.pi)*coherent_dm(hilbert_size, beta) for beta in betas]

    ops_numpy = [op.full() for op in m_ops] # convert the QuTiP Qobj to numpy arrays
    ops_tf = tf.convert_to_tensor([ops_numpy]) # convert the numpy arrays to complex TensorFlow tensors

    return X, Y, ops_tf


import math

def rho_to_Q(rho, xvec, yvec):
    alphas = xvec + 1j*yvec

    alphas = np.expand_dims(alphas, -1)
    
    As = []
    for i in range(rho.shape[-1]):
        As.append(
            (alphas ** i)/math.sqrt(math.factorial(i))
        )

    As = np.concatenate(As, axis=-1)

    As *= np.exp(-np.abs(alphas)**2/2)

    return np.fmax(np.real(np.einsum("ijk,...kl,ijl->...ij",np.conj(As),rho,As)/math.pi), 1e-30)


def L1_loss(Q_true, Q_pred, xvec, yvec):
    return np.sum(np.abs(Q_true - Q_pred)) * (xvec[0,1] - xvec[0,0]) * (yvec[1,0] - yvec[0,0])

def KL_rev_loss(Q_true, Q_pred, xvec, yvec):
    return np.sum(Q_true * np.log(Q_true/Q_pred)) * (xvec[0,1] - xvec[0,0]) * (yvec[1,0] - yvec[0,0])

def KL_loss(Q_true, Q_pred, xvec, yvec):
    return np.sum(Q_pred * np.log(Q_pred/Q_true)) * (xvec[0,1] - xvec[0,0]) * (yvec[1,0] - yvec[0,0])

def Q_fid(Q_true, Q_pred, xvec, yvec):
    return np.sum(np.sqrt(Q_true * Q_pred)) * (xvec[0,1] - xvec[0,0]) * (yvec[1,0] - yvec[0,0])

def add_losses(losses, rho_true, rho_pred, xvec, yvec, i):


    if i % 100 == 0:
        Q_true = rho_to_Q(rho_true, xvec, yvec)
        Q_pred = rho_to_Q(rho_pred, xvec, yvec)

        losses["L1"].append(L1_loss(Q_true, Q_pred, xvec, yvec))
        losses["KL"].append(KL_loss(Q_true, Q_pred, xvec, yvec))
        losses["KL_rev"].append(KL_rev_loss(Q_true, Q_pred, xvec, yvec))
        losses["Q_fid"].append(Q_fid(Q_true, Q_pred, xvec, yvec))

    losses["p_L2"].append(np.sum(np.abs(np.array(rho_true) - np.array(rho_pred)) ** 2))

def train_step(A, x, generator, discriminator, generator_optimizer, discriminator_optimizer, loss, lam):
    """Takes one step of training for the full A matrix representing the
    measurement operators and data x.

    Note that the `generator`, `discriminator`, `generator_optimizer` and the
    `discriminator_optimizer` has to be defined before calling this function.

    Args:
        A (tf.Tensor): A tensor of shape (m, hilbert_size, hilbert_size, n x 2)
                       where m=1 for a single reconstruction, and n represents
                       the number of measured operators. We split the complex
                       operators as real and imaginary in the last axis. The 
                       helper function `convert_to_real_ops` can be used to
                       generate the matrix A with a set of complex operators
                       given by `ops` with shape (1, n, hilbert_size, hilbert_size)
                       by calling `A = convert_to_real_ops(ops)`.

        x (tf.Tensor): A tensor of shape (m, n) with m=1 for a single
                       reconstruction and `n` representing the number of
                       measurements. 
    """
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator([A, x], training=True)

        disc_real_output = discriminator([A, x, x], training=True)
        disc_generated_output = discriminator([A, x, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
            disc_generated_output, gen_output, x, lam=lam
        )
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(
        gen_total_loss, generator.trainable_variables
    )
    discriminator_gradients = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables)
    )

    loss.generator.append(gen_gan_loss)
    loss.l1.append(gen_l1_loss)
    loss.discriminator.append(disc_loss)

def train(A, x, psi, rho_tf, X, Y, num_measurements):
    losses = {"L1": [], "KL": [], "KL_rev": [], "p_L2": [], "Q_fid": []}
    fidelities = []

    density_layer_idx = None

    generator = Generator(hilbert_size, num_measurements, noise=0.) # Specify the number of measurement settings and Gaussian noise
    discriminator = Discriminator(hilbert_size, num_measurements)

    print(generator.summary())
    print(discriminator.summary())

    for i, layer in enumerate(generator.layers):
        if "density_matrix" in layer._name:
            density_layer_idx = i
            break

    print(density_layer_idx)
    model_dm = tf.keras.Model(inputs=generator.input, outputs=generator.layers[density_layer_idx].output)

    @dataclass
    class LossHistory:
        """Class for keeping track of loss"""
        generator: list
        discriminator: list
        l1: list

    loss = LossHistory([], [], [])
    fidelities = []

    initial_learning_rate = 0.0002

    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate,
                                                                decay_steps=10000,
                                                                decay_rate=.96,
                                                                staircase=False)

    lam = 100.

    generator_optimizer = tf.keras.optimizers.legacy.Adam(lr_schedule, 0.5, 0.5)
    discriminator_optimizer = tf.keras.optimizers.legacy.Adam(lr_schedule, 0.5, 0.5)

    test_X = np.arange(-5, 5, 0.02)
    test_Y = np.arange(-5, 5, 0.02)
    test_X, test_Y = np.meshgrid(test_X, test_Y)

    max_iterations = 500

    pbar = tqdm(range(max_iterations))
    for i in pbar:
        train_step(
            A, 
            x,
            generator,
            discriminator,
            generator_optimizer,
            discriminator_optimizer,
            loss,
            lam,
        )
        density_matrix = model_dm([A, x])
        f = tf_fidelity(density_matrix, rho_tf)[-1]
        fidelities.append(f)
        add_losses(losses, rho_tf, density_matrix, test_X, test_Y, i)
        pbar.set_description(f"L1: {losses['L1'][-1]:.3f}, KL: {losses['KL'][-1]:.3f}, KL_rev: {losses['KL_rev'][-1]:.3f}, p_L2: {losses['p_L2'][-1]:.3f}, Q_fid: {losses['Q_fid'][-1]:.3f}")

    
    """iterations = np.arange(len(losses["L1"]))
    plt.plot(iterations + 1, fidelities, color="red", label="QST-CGAN")
    plt.plot(iterations+1, losses["L1"], color="blue", label="L1")
    plt.plot(iterations+1, losses["KL"], color="green", label="KL")
    plt.plot(iterations+1, losses["KL_rev"], color="orange", label="KL_rev")
    plt.xlabel("Iterations")
    plt.ylabel("Fidelity")
    #plt.ylim(0, 1.02)
    plt.grid(which='minor', alpha=0.2)
    plt.grid(which='major', alpha=0.2)
    #plt.xscale('log')
    plt.yscale('log')
    plt.show()


    xvec = np.linspace(-5, 5, 100)
    yvec = np.linspace(-5, 5, 100)

    density_matrix = model_dm([A, x])
    rho_predicted = Qobj(density_matrix.numpy().reshape((hilbert_size, hilbert_size)))
    data = qfunc(psi, xvec, yvec, g= 2)
    data_predicted = qfunc(rho_predicted, xvec, yvec, g= 2)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)

    im = ax[0].pcolor(xvec, yvec, data.reshape((100, 100)), cmap="Blues")
    ax[1].pcolor(xvec, yvec, data_predicted.reshape((100, 100)), cmap="Blues")

    ax[0].set_xlabel(r"Re($\beta$)")
    ax[1].set_xlabel(r"Re($\beta$)")
    ax[0].set_ylabel(r"Im($\beta$)")
    ax[0].set_title("Noisy data")
    ax[1].set_title("Reconstructed\nFidelity {}".format(fidelity(psi * psi.dag(), rho_predicted)))
    plt.colorbar(im, ax=[axis for axis in ax])
    plt.show()"""

    return {
        #"Fidelity" : fidelities[-1],
        "L1" : losses["L1"][-1],
        "KL" : losses["KL"][-1],
        "KL_rev" : losses["KL_rev"][-1],
        #"p_L2" : losses["p_L2"][-1],
        "Q_fid" : losses["Q_fid"][-1],
    }


if __name__ == "__main__":
    args = parser.parse_args()

    results_mean = []
    results_std = []

    size_range = range(5, 15, 1)

    num_repeats = 3

    for grid_size in size_range:
        repeat_results = []
        for _ in range(num_repeats):
            X, Y, psi, ops_tf, A, rho_tf, x = get_Q(args.state_type, grid_size)

            repeat_results.append(
                train(A, x, psi, rho_tf, X, Y, num_measurements = grid_size ** 2)
            )

        repeat_results_mean = {
            key: np.mean([result[key] for result in repeat_results]) for key in repeat_results[0]
        }

        repeat_results_std = {
            key: np.std([result[key] for result in repeat_results], ddof=1) for key in repeat_results[0]
        }

        results_mean.append(repeat_results_mean)
        results_std.append(repeat_results_std)

    pickle.dump((list(size_range), results_mean, results_std), open(f"sample_efficiency/qst_{args.state_type}.pkl", "wb"))


    for key in results_mean[0]:
        plt.fill_between(np.array(size_range)**2, [result[key] - results_std[i][key] for i, result in enumerate(results_mean)], [result[key] + results_std[i][key] for i, result in enumerate(results_mean)], alpha=0.3)
        plt.plot(np.array(size_range)**2, [result[key] for result in results_mean], label=key)

    plt.xlabel("Number of measurements")
    plt.ylabel("Fidelity")

    plt.legend()

    plt.show()



