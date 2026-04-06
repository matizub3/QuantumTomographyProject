import jax
from jax.random import PRNGKey,normal,split
from jax import jit,vmap
from jax.example_libraries import stax
import jax.numpy as jnp
import jax.nn as nn
import haiku as hk
import math

from random import randint

from functools import partial

import flows
import diffrax
from diffrax import RecursiveCheckpointAdjoint,BacksolveAdjoint
from .flow_util import Flow, ResNet,Diagonal

# Most of the following functions are slight modifications of functions
# taken from the excellent Jupyter notebook https://docs.kidger.site/diffrax/examples/continuous_normalising_flow/
def exact_logp_wrapper(func):
    def logp(t, input, params):
        y, _ = input
        fn = lambda y: func(t, y, params)
        f, vjp_fn = jax.vjp(fn, y)
        (size,) = y.shape  # this implementation only works for 1D input
        eye = jnp.eye(size)
        (dfdy,) = jax.vmap(vjp_fn)(eye)
        logp = jnp.trace(dfdy)
        return f, logp
    return logp


def ContinuousFlow(model, internal_layer_size, num_layers, activation, 
                    approximate_adjoint = False, adaptive = True, t0=0, t1=0.5, dt0=0.05, combine_ty = True,
                    num_encodings = 0):
    
    print(adaptive)
    print(dt0)

    def init_fun(rng, input_dim, **kwargs):
        if combine_ty:
            field_init, apply = hk.without_apply_rng(
                hk.transform(
                    lambda x: model(
                        [internal_layer_size]*(num_layers-1)+[input_dim],
                        activation = activation,
                        random_init = True,
                        positional_embeddings = num_encodings,
                    )(x)
                )
            )

            params = field_init(rng, jnp.zeros((1,input_dim+1),float))
        else:
            field_init, apply = hk.without_apply_rng(
                hk.transform(
                    lambda t, x: model(
                        [internal_layer_size]*(num_layers-1)+[input_dim],
                        activation = activation,
                        random_init = True,
                        positional_embeddings = num_encodings,
                    )(t, x)
                )
            )
            
            params = field_init(rng, jnp.zeros((1,1),float), jnp.zeros((1,input_dim),float))

        if approximate_adjoint:
            adjoint = BacksolveAdjoint()
        else:
            adjoint = RecursiveCheckpointAdjoint()

        if combine_ty:
            # Only works for a single input. Vmap to parallelize
            def vector_field(t, y, params):
                inputs = jnp.insert(y,0,t)
                return apply(params,inputs)
        else:
            def vector_field(t, y, params):
                return apply(params,jnp.expand_dims(t,axis=0),y)

        vector_field_prob = exact_logp_wrapper(vector_field)

        solver = diffrax.Tsit5()

        print(f"adaptive = {adaptive}")
        if adaptive:
            step_controller = diffrax.PIDController(rtol = 1e-4, atol = 1e-4)
        else:
            try:
                step_controller = diffrax.ConstantStepSize(compile_steps = True)
            except:
                print("THERE WAS AN ERROR, NOT USING COMPILE STEPS")
                step_controller = diffrax.ConstantStepSize()
        
        @partial(vmap, in_axes=(None,0))
        def direct_fun(params, y, **kwargs):
            term = diffrax.ODETerm(vector_field_prob)
        
            sol = diffrax.diffeqsolve(
                term, 
                solver, 
                t1, t0, 
                -dt0, 
                y0 = (y,0.0), 
                args = params, 
                adjoint = adjoint,
                stepsize_controller = step_controller,
            )
            (y,), (log_prob,) = sol.ys
            #id_print(sol.stats)
            return y,log_prob

        # Runs forward-in-time to draw samples from the CNF.
        @partial(vmap, in_axes=(None,0))
        def inverse_fun(params, y, **kwargs):
            term = diffrax.ODETerm(vector_field)

            sol = diffrax.diffeqsolve(
                term, solver, 
                t0, t1, dt0, 
                y0=y, args = params, 
                adjoint = adjoint,
                stepsize_controller = step_controller,
            )
            (y,) = sol.ys
            #id_print(sol.stats)
            return y,0.0

        return params,direct_fun,inverse_fun

    return init_fun

def get_FFJORD(model = ResNet, num_layers = 5, internal_layer_size = 5, 
                activation = nn.gelu, scale_layer = True, adaptive = True, dt = 0.05, prior = flows.Normal(), combine_ty = True,
                num_encodings = 0, rescale = 1):
    if scale_layer or rescale != 1:
        return Flow(
                    flows.Serial(
                        Diagonal(rescale = rescale),
                        ContinuousFlow(model, internal_layer_size = internal_layer_size, 
                                        num_layers = num_layers, activation = activation, 
                                        approximate_adjoint = False, 
                                        adaptive = adaptive,
                                        t0=0, t1=0.5, dt0=dt, combine_ty = combine_ty,
                                        num_encodings = num_encodings),
                        #Diagonal(),
                    ),
                    prior,
                )
    return Flow(ContinuousFlow(model, internal_layer_size = internal_layer_size, 
                                        num_layers = num_layers, activation = activation, 
                                        approximate_adjoint = False, 
                                        adaptive = adaptive,
                                        t0=0, t1=0.5, dt0=dt, combine_ty = combine_ty,
                                        num_encodings = num_encodings),prior)