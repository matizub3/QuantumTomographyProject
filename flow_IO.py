import jax.numpy as jnp
from jax.tree_util import tree_map

import pickle

import numpy as np

def save_params(params,filename):
    with open(filename, 'wb') as f:
        pickle.dump(
            tree_map(
                np.array,
                params,
            ),
            f,
        )

def load_params(filename):
    with open(filename, 'rb') as f:
        params = pickle.load(f)
        params = tree_map(
            jnp.array,
            params,
        )
    return tree_map(
        jnp.copy,
        params,
    )