from jax.example_libraries import stax
import jax.numpy as jnp
import jax.nn as nn
import flows

from .flow_util import Loop,Diagonal

"""MADE Normalizing Flow"""


# This function is a modified version of the MADE function from the 
# jax-flows library.
def MADE(transform):
    """An implementation of `MADE: Masked Autoencoder for Distribution Estimation`
    (https://arxiv.org/abs/1502.03509).
    Args:
        transform: maps inputs of dimension ``num_inputs`` to ``2 * num_inputs``
    Returns:
        An ``init_fun`` mapping ``(rng, input_dim)`` to a ``(params, direct_fun, inverse_fun)`` triplet.
    """

    def init_fun(rng, input_dim, **kwargs):
        params, apply_fun = transform(rng, input_dim)

        def direct_fun(params, inputs, **kwargs):
            log_weight, bias = apply_fun(params, inputs).split(2, axis=1)
            outputs = (inputs - bias) * jnp.exp(-log_weight)
            log_det_jacobian = -log_weight.sum(-1)
            return outputs, log_det_jacobian

        def inverse_fun(params, inputs, **kwargs):
            outputs = jnp.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                log_weight, bias = apply_fun(params, outputs).split(2, axis=1)
                outputs = outputs.at[:,i_col].set(inputs[:, i_col] * jnp.exp(log_weight[:, i_col]) + bias[:, i_col])
            log_det_jacobian = -log_weight.sum(-1)
            return outputs, log_det_jacobian

        return params, direct_fun, inverse_fun

    return init_fun

def get_masks(input_dim, hidden_dim=64, num_hidden=1):
        masks = []
        input_degrees = jnp.arange(input_dim)
        degrees = [input_degrees]

        for n_h in range(num_hidden + 1):
            degrees += [jnp.arange(hidden_dim) % (input_dim - 1)]
        degrees += [input_degrees % input_dim - 1]

        for (d0, d1) in zip(degrees[:-1], degrees[1:]):
            masks += [jnp.transpose(jnp.expand_dims(d1, -1) >= jnp.expand_dims(d0, 0)).astype(jnp.float32)]
        return masks

def masked_transform(rng, input_dim):
    masks = get_masks(input_dim, hidden_dim=64, num_hidden=1)
    
    act = stax.Gelu
    init_fun, apply_fun = stax.serial(
        flows.MaskedDense(masks[0]),
        act,
        flows.MaskedDense(masks[1]),
        act,
        flows.MaskedDense(masks[2].tile(2)),
    )
    _, params = init_fun(rng, (input_dim,))
    #print(params)
    return params, apply_fun

MADE_init = flows.Flow(
    flows.Serial(
        Loop(
            flows.Serial(
                MADE(masked_transform), 
                flows.Reverse()
            ),5),
        Diagonal()
    ),
    flows.Normal(),
)