import jax
from jax.random import PRNGKey,normal,uniform,split
from jax import jit,grad,vmap,jacfwd,jacrev
from jax.example_libraries import stax
from jax.scipy.optimize import minimize
import jax.numpy as jnp
import jax.nn as nn
import haiku as hk

from jaxopt import LBFGS

import math

from random import randint

from functools import partial

import flows
import diffrax
from diffrax import RecursiveCheckpointAdjoint,BacksolveAdjoint

from Q_flows.affine import transform_init
from .flow_util import Loop, ResNet,Diagonal,Flow, Shift

from jax.experimental.ode import odeint

class Linear(hk.Module):
    """Linear module. Modified from the Haiku source code."""
    def __init__(
        self,
        output_size: int,
        with_bias: bool = True,
        name: str | None = None,
        small_init: bool = False,
    ):
        super().__init__(name=name)
        self.input_size = None
        self.output_size = output_size
        self.with_bias = with_bias

        if small_init:
            self.w_init = jnp.zeros#lambda shape, dtype: normal(PRNGKey(0),shape,dtype)/10000
            self.b_init = lambda num_in, shape, dtype: normal(PRNGKey(0),shape,dtype)/10000
        else:
            self.w_init = lambda shape, dtype: uniform(PRNGKey(0),shape,dtype,-jnp.sqrt(1/shape[0]),jnp.sqrt(1/shape[0]))
            self.b_init = lambda num_in, shape, dtype: uniform(PRNGKey(0),shape,dtype,-jnp.sqrt(1/num_in),jnp.sqrt(1/num_in))
    
    def __call__(
        self,
        inputs: jnp.ndarray,
    ) -> jnp.ndarray:
    
        input_size = self.input_size = inputs.shape[-1]
        output_size = self.output_size
        dtype = inputs.dtype
        
        w = hk.get_parameter("w", [input_size, output_size], dtype, init=self.w_init)
        
        out = jnp.dot(inputs, w)
        
        if self.with_bias:
            b = hk.get_parameter("b", [self.output_size], dtype, init=partial(self.b_init,input_size))
            b = jnp.broadcast_to(b, out.shape)
            out = out + b
        
        return out

class PositiveLinear(hk.Module):
    """Linear module. Modified from the Haiku source code."""
    
    def __init__(
        self,
        output_size: int,
        with_bias: bool = True,
        name: str | None = None,
        small_init: bool = False,
    ):
        super().__init__(name=name)
        self.input_size = None
        self.output_size = output_size
        self.with_bias = with_bias

        if small_init:
            self.w_init = jnp.zeros#lambda shape, dtype: normal(PRNGKey(0),shape,dtype)/10000
            self.b_init = lambda num_in, shape, dtype: normal(PRNGKey(0),shape,dtype)/10000
        else:
            self.w_init = lambda shape, dtype: uniform(PRNGKey(0),shape,dtype,-jnp.sqrt(1/shape[0]),jnp.sqrt(1/shape[0]))
            self.b_init = lambda num_in, shape, dtype: uniform(PRNGKey(0),shape,dtype,-jnp.sqrt(1/num_in),jnp.sqrt(1/num_in))
    
    def __call__(
        self,
        inputs: jnp.ndarray,
    ) -> jnp.ndarray:
        
        input_size = self.input_size = inputs.shape[-1]
        output_size = self.output_size
        dtype = inputs.dtype
        
        w = hk.get_parameter("w", [input_size, output_size], dtype, init=self.w_init)
        
        out = jnp.dot(inputs, nn.softplus(w))
        
        if self.with_bias:
            b = hk.get_parameter("b", [self.output_size], dtype, init=partial(self.b_init,input_size))
            b = jnp.broadcast_to(b, out.shape)
            out = out + b
            
        return out/input_size

class ActNorm(hk.Module):
    """ActNorm Module"""
    
    def __init__(
        self,
        input_dim: int,
        input: jnp.ndarray | None = None,
        name: str | None = None,
    ):
        super().__init__(name=name)

        if input != None:
            mean = jnp.mean(input, axis = 0)
            log_std = jnp.log(jnp.std(input, axis = 0)+1e-6)
        else:
            mean = jnp.zeros(input_dim)
            log_std = jnp.zeros(input_dim)

        self.mean_init = lambda a,b: mean
        self.log_std_init = lambda a,b: log_std
        
    def __call__(
        self,
        input: jnp.ndarray,
    ) -> jnp.ndarray:

        mean = hk.get_parameter(f"mean", shape=[input.shape[-1]], dtype=input.dtype, init=self.mean_init)
        log_std = hk.get_parameter(f"log_std", shape=[input.shape[-1]], dtype=input.dtype, init=self.log_std_init)
        
        return (input-mean)/jnp.exp(log_std)


def softplus2(x):
    return 0.5*(nn.softplus(x-1)+nn.softplus(x+1))

class ICNN(hk.Module):
  def __init__(
        self, 
        num_layers, 
        hidden_size, 
        augmented_size, 
        activation = nn.softplus, 
        init = jnp.zeros, 
        name=None, 
        a1 = jnp.log(jnp.exp(1)-1),
        data_dependent = True,
        small_init: bool = False,):
    super().__init__(name=name)
    self.num_layers = num_layers
    self.hidden_size = hidden_size
    self.augmented_size = augmented_size
    self.activation = activation
    self.data_dependent = data_dependent

    self.small_init = small_init

    self.init_alpha = lambda a,b: jnp.array([jnp.log(jnp.exp(1)-1),a1])

  def __call__(self, x):
    in_shape = x.shape[-1]

    linear_init = Linear(self.hidden_size, name = "linear_init")
    h = self.activation(linear_init(x))

    for i in range(1,self.num_layers-1):
        linear_aug = Linear(self.augmented_size, name = f"linear_aug{i}")
        linear = Linear(self.hidden_size-self.augmented_size, name = f"linear_input{i}")
        positive_linear = PositiveLinear(self.hidden_size-self.augmented_size, name = f"positive_linear{i}")
        
        h_aug = self.activation(linear_aug(x))
        h_tilde = positive_linear(h)+linear(x)
        actnorm = ActNorm(h_tilde.shape[-1], name = f"ActNorm{i}", input = h_tilde if self.data_dependent else None)
        h_tilde = self.activation(actnorm(h_tilde))

        h = jnp.concatenate((h_tilde, h_aug),-1)

    i=self.num_layers-1

    positive_linear = PositiveLinear(1, with_bias = False, name = f"positive_linear{i}", small_init = self.small_init)
    linear_input = Linear(1, with_bias = False, name = f"linear_input{i}", small_init = self.small_init)

    icnn_out = positive_linear(h) + linear_input(x)
    actnorm = ActNorm(icnn_out.shape[-1], name = f"ActNorm{i}", input = icnn_out if self.data_dependent else None)
    icnn_out = actnorm(icnn_out)

    alpha = hk.get_parameter(f"alpha", shape=[2,], dtype=x.dtype, init=self.init_alpha)
    alpha = nn.softplus(alpha)

    return alpha[0]*icnn_out[...,0] + alpha[1]*jnp.sum(x**2, axis=-1)/2

class ICNN2(hk.Module):
  def __init__(self, num_layers, hidden_size, augmented_size, activation = nn.softplus, init = jnp.zeros, name=None):
    super().__init__(name=name)
    self.num_layers = num_layers
    self.hidden_size = hidden_size
    self.augmented_size = augmented_size
    self.activation = activation

    def init_positive(shape,dtype):
        return jnp.full(shape,-4,dtype)

    def init_quadratic(shape,dtype):
        return jnp.full(shape,-10,dtype)


    self.init_positive = init_positive
    self.init_linear = lambda shape, dtype: (normal(PRNGKey(0), shape, dtype)/100)
    self.init_quadratic = init_quadratic

  def __call__(self, x):
    in_shape = x.shape[-1]

    w_init = hk.get_parameter(f"w_init", shape=[in_shape, self.hidden_size], dtype=x.dtype, init=self.init_linear)
    b_init = hk.get_parameter(f"b_init", shape=[self.hidden_size], dtype=x.dtype, init=self.init_linear)

    h_prev = self.activation(jnp.dot(x,w_init)+b_init)

    hidden_shape = self.hidden_size

    for i in range(1,self.num_layers-1):
        w_aug = hk.get_parameter(f"w_aug{i}", shape=[in_shape, self.augmented_size], dtype=x.dtype, init=self.init_linear)
        x_aug = jnp.dot(x, w_aug)

        input = jnp.concatenate((h_prev,x_aug),axis = -1)

        w_plus = hk.get_parameter(f"w+{i}", shape=[hidden_shape+self.augmented_size, self.hidden_size], dtype=x.dtype, init=self.init_positive)
        w_plus = jnp.exp(w_plus)
        b_plus = hk.get_parameter(f"b+{i}", shape=[self.hidden_size], dtype=x.dtype, init=self.init_linear)

        h = self.activation(jnp.dot(input,w_plus)+b_plus)

        h_prev = jnp.concatenate((h,h_prev), axis = -1)

        hidden_shape += self.hidden_size

        print(i)

    w_aug = hk.get_parameter(f"w_aug_final", shape=[in_shape, self.augmented_size], dtype=x.dtype, init=self.init_linear)
    x_aug = jnp.dot(x, w_aug)

    input = jnp.concatenate((h_prev,x_aug),axis = -1)

    w_plus = hk.get_parameter(f"w+_final", shape=[hidden_shape+self.augmented_size, 1], dtype=x.dtype, init=self.init_positive)
    w_plus = jnp.exp(w_plus)
    icnn_out = jnp.dot(input,w_plus)

    alpha = hk.get_parameter(f"alpha", shape=[1,], dtype=x.dtype, init=self.init_quadratic)
    alpha = jnp.exp(alpha)

    return icnn_out[...,0] + alpha[0]*jnp.sum(x**2, axis=-1)/2

def ConvexPotentialFlow(model, num_layers, hidden_layer_size, augmented_layer_size, activation, alpha1 = 0, small_init = False):

    def init_fun(rng, input_dim, **kwargs):
        init_inputs = kwargs.pop("init_inputs", None)

        if init_inputs is None:
            init_inputs = jnp.zeros((1,input_dim),float)
            data_dependent = False
        else:
            data_dependent = True

        potential_init, potential = hk.without_apply_rng(
            hk.transform(
                lambda x: model(
                    num_layers = num_layers,
                    hidden_size = hidden_layer_size,
                    augmented_size = augmented_layer_size,
                    activation = activation,
                    data_dependent = data_dependent,
                    small_init = small_init,
                )(x)
            )
        )

        params = potential_init(rng, init_inputs, )

        def potential_to_minimize(input,args):
            params, x = args
            return jnp.sum(potential(params,input))-jnp.sum(x*input)

        solver = LBFGS(potential_to_minimize)
        
        @partial(vmap, in_axes=(None,0))
        def direct_fun(params, x):
            gradient = grad(potential, argnums = 1)(params,x)
            hessian = jacfwd(jacrev(potential,argnums=1),argnums=1,)(params,x)

            _,logdet = jnp.linalg.slogdet(hessian)

            return gradient,logdet

        #@partial(vmap, in_axes=(None,0))
        def inverse_fun(params, x):
            out = solver.run(x, (params, x))
            return out.params,0.0

        return params,direct_fun,inverse_fun

    return init_fun

def get_CPF(model = ICNN, num_layers = 5, hidden_layer_size = 10, augmented_layer_size = 5, num_repeats = 1,
                activation = nn.softplus, scale_layer = True, shift = None, prior = flows.Normal(), small_init = False):

    flow = ConvexPotentialFlow(
        model = model,
        num_layers = num_layers,
        hidden_layer_size = hidden_layer_size,
        augmented_layer_size = augmented_layer_size,
        activation = activation,
        small_init = small_init,
    )

    if scale_layer:
        if shift != None:
            return Flow(
                        flows.Serial(
                            Loop(
                                flows.Serial(
                                    flows.ActNorm(),
                                    flow,
                                ),
                                num_repeats,
                            ),
                            Shift(shift),
                            Diagonal(),
                        ),
                        prior,
                    )
        return Flow(
                    flows.Serial(
                        Loop(
                            flows.Serial(
                                flows.ActNorm(),
                                flow,
                            ),
                            num_repeats,
                        ),
                        Diagonal(),
                    ),
                    prior,
                )
    if shift != None:
        return Flow(flows.Serial(Loop(flow, num_repeats),Shift(shift)),prior)
    return Flow(Loop(flow, num_repeats),prior)


def get_RCPF(model = ICNN2, num_layers_CPF = 5, hidden_layer_size = 10, augmented_layer_size = 5,
                activation = nn.softplus, scale_layer = True, num_layers_RealNVP=3, num_transform_layers=3, 
                internal_layer_size = 5, network = ResNet, prior = flows.Normal()):

    module = flows.Serial(
                    flows.AffineCouplingSplit(
                        transform_init(num_transform_layers, network = network, internal_layer_size = internal_layer_size),
                        transform_init(num_transform_layers, network = network, internal_layer_size = internal_layer_size)
                    ),
                    flows.Reverse()
             )
    
    flow1 = Loop(module,num_layers_RealNVP)

    flow2 = ConvexPotentialFlow(
        model = model,
        num_layers = num_layers_CPF,
        hidden_layer_size = hidden_layer_size,
        augmented_layer_size = augmented_layer_size,
        activation = activation,
    )

    if scale_layer:
        return flows.Flow(
                    flows.Serial(
                        flow1,
                        flow2,
                        flows.Diagonal(),
                        Shift()
                    ),
                    prior,
                )
    return flows.Flow(flows.Serial(flow1,flow2),prior)