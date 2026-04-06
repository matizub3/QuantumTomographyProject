from functools import partial
import jax.numpy as jnp
import jax.nn as nn
import haiku as hk

from jax import random

import flows
from .flow_util import AffineCouplingSplit, Flow, Loop,Diagonal, RandomShift,ResNet, Shift

"""RealNVP and GLoW"""

def transform_init(num_layers, network=ResNet, internal_layer_size = None, activation = nn.gelu, random_init = False, divide = 10000):
    print(f"divide={divide}")

    def transform(rng, cutoff, masked):
        if internal_layer_size == None:
            init,apply = hk.without_apply_rng(
                            hk.transform(
                                lambda x: network([masked]*num_layers, activation = activation, random_init = random_init, divide = divide)(x)
                            )
                        )
        else:
            init,apply = hk.without_apply_rng(
                            hk.transform(
                                lambda x: network([internal_layer_size]*(num_layers-1)+[masked], activation = activation, random_init = random_init, divide = divide)(x)
                            )
                        )
            
        params = init(rng=rng,x=jnp.zeros((1,cutoff)))

        return params, apply
    
    return transform


def get_rqs_network(num_layers, network=ResNet, activation = nn.gelu, random_init = False):
    def rqs_network_init(num_out, num_hidden):
        init,apply = hk.without_apply_rng(
                hk.transform(
                    lambda x: network([num_hidden]*(num_layers-1)+[num_out], activation = activation, random_init = random_init)(x)
                    )
                )

        return init, apply
    
    return rqs_network_init

def get_realNVP_slow(num_layers=3, num_transform_layers=5, internal_layer_size = None, prior=flows.Normal(), scaling_layer = True):
    layers = []
    for i in range(num_layers):
        layers.append(AffineCouplingSplit(transform_init(num_transform_layers, internal_layer_size=internal_layer_size), transform_init(num_transform_layers, internal_layer_size=internal_layer_size)))
        layers.append(flows.Reverse())

    if scaling_layer:
        layers.append(Diagonal())

    #module = flows.Serial(flows.AffineCouplingSplit(transform_init(1), transform_init(1)),flows.Shuffle())
    return flows.Flow(flows.Serial(*layers),prior)

def get_realNVP(
        num_layers=6, 
        num_transform_layers=5, 
        internal_layer_size = None, 
        network = ResNet, 
        prior=flows.Normal(), 
        scaling_layer = True, 
        shift = None, 
        random_shift = False,
        random_init = False,
        coupling_layer = "affine",
        knots = 5,
        B = 3,
        permutation_layer = "reverse",
        divide = 10000):
    
    print(f"divide={divide}")
    
    if permutation_layer == "reverse":
        permute = flows.Reverse()
    elif permutation_layer == "linear":
        permute = flows.FixedInvertibleLinear()

    if coupling_layer == "affine":
        coupling = flows.AffineCouplingSplit(
                        transform_init(num_transform_layers, network = network, internal_layer_size = internal_layer_size, random_init = random_init, divide = divide,),
                        transform_init(num_transform_layers, network = network, internal_layer_size = internal_layer_size, random_init = random_init, divide = divide,)
        )
    elif coupling_layer == "RQS":
        coupling = flows.NeuralSplineCoupling(
                    K = knots,
                    B = B,
                    hidden_dim = internal_layer_size,
        )
        
    module = flows.Serial(
        coupling,
        permute,
    )
    
    if scaling_layer:
        if shift != None:
            return Flow(
                flows.Serial(
                    Loop(
                        module,
                        num_layers,
                    ),
                    RandomShift() if random_shift else Shift(shift),
                    Diagonal(),
                ),
                prior,
            )
        
        return Flow(
                flows.Serial(
                    Loop(
                        module,
                        num_layers,
                    ),
                    Diagonal(),
                ),
                prior,
            )
    if shift != None:
        return Flow(
            flows.Serial(
                Loop(
                    module,
                    num_layers,
                ),
                RandomShift() if random_shift else Shift(shift),
            ),
            prior,
        )
    
    return Flow(
        Loop(
            module,
            num_layers,
        ),
        prior,
    )