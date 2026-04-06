import jax.numpy as jnp
import jax.nn as nn
import haiku as hk

from jax import random

import flows
from .flow_util import AffineCouplingSplit, Flow, Loop, Diagonal, ResNet, Shift
import distrax

class RationalQuadraticSplineCoupling(distrax.Bijector):

  def __init__(self, num_layers, hidden_layer_size, num_splines):
    super().__init__()

    self.num_layers = num_layers
    self.hidden_layer_size = hidden_layer_size
    self.num_splines = num_splines

  def forward_and_log_det(self, x):
    y = x
    logdet = ...
    return y, logdet

  def inverse_and_log_det(self, y):
    x = ...
    logdet = ...
    return x, logdet



def forward_rqs(x, spline_parameters, ):
    rqs = distrax.RationalQuadraticSpline(
        bin_widths = ,
        bin_heights = ,
        knot_slopes = ,
        range_min = ,
        validate_args = ,
    )



"""
Rational Quadratic Spline Transformation
"""
def RQST(network):

    def init_fun(rng, input_dim, **kwargs):
        
        def pass_forward(y):
            params = network()(y)

            rqs = distrax.RationalQuadraticSpline(
                bin_widths = params[...,],
                bin_heights = ,
                knot_slopes = ,
                range_min = ,
                validate_args = ,
            )
            
            return y,log_prob

        # Runs forward-in-time to draw samples from the CNF.
        @partial(vmap, in_axes=(None,0))
        def inverse_fun(params, y):
            term = diffrax.ODETerm(vector_field)
            solver = diffrax.Dopri5(scan_stages=True)
            sol = diffrax.diffeqsolve(
                term, solver, 
                t0, t1, dt0, 
                y0=y, args = params, 
                adjoint = adjoint,
                stepsize_controller = diffrax.ConstantStepSize(compile_steps = True),
            )
            (y,) = sol.ys
            return y,0.0

        return params,direct_fun,inverse_fun

    return init_fun