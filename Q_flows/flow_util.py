import math
import jax
from jax.random import split,normal,categorical,uniform,PRNGKey
from jax import nn, vmap
import jax.numpy as jnp
from jax.lax import scan
import haiku as hk
from jax.flatten_util import ravel_pytree
from jax.nn import softplus

from jax.scipy.special import logsumexp

import flows
from functools import partial

class MLPLayer(hk.Module):
    def __init__(self, out_size, first_layer, random_init = True, divide = 10000, activation = nn.gelu, name = "MLPLayer"):
        super().__init__(name=name)
        self.out_size = out_size

        self.first_layer = first_layer

        self.activation = activation

        if activation == sine:
            if first_layer:
                self.weight_init = lambda shape,dtype : uniform(PRNGKey(0),shape, minval = -math.sqrt(6/shape[0])/30, maxval = math.sqrt(6/shape[0])/30)
            else:
                self.weight_init = lambda shape,dtype : uniform(PRNGKey(0),shape, minval = -1/shape[0], maxval = 1/shape[0])

        else:
            if random_init:
                self.weight_init = (lambda shape,dtype : normal(PRNGKey(0),shape)/divide)
            else:
                self.weight_init = jnp.zeros

    def __call__(self, x):
        in_shape = x.shape[-1]
        w = hk.get_parameter("w", shape=[in_shape, self.out_size], dtype=x.dtype, init=self.weight_init)
        b = hk.get_parameter("b", shape=[self.out_size], dtype=x.dtype, init=jnp.zeros)
        
        if self.activation == None:
            return jnp.dot(x, w) + b
        
        if self.first_layer:
            return self.activation(jnp.dot(x, w) + b)
        return self.activation(jnp.dot(x, w) + b) + x

class ResNet(hk.Module):
    def __init__(self, layer_size, activation, random_init = True, divide = 10000, name=None, end_activation = False, end_activation_type = nn.softplus, positional_embeddings = 0,):
        super().__init__(name=name)
        
        self.layer_size = layer_size
        self.activation = activation

        self.random_init = random_init
        self.divide = divide
        
        self.end_activation = end_activation
        
        self.end_activation_type = end_activation_type
        self.positional_embeddings = positional_embeddings
        
    def __call__(self, x):
        x = PositionalEncoding(self.positional_embeddings)(x)

        in_shape = x.shape[-1]
        for i in range(len(self.layer_size)-1):
            layer = MLPLayer(self.layer_size[i], first_layer = (i == 0), random_init = self.random_init, divide = self.divide, activation = self.activation)
            
            x = layer(x)
        
        i=len(self.layer_size)-1

        layer = MLPLayer(self.layer_size[i], first_layer = (i == 0), random_init = self.random_init, divide = self.divide, activation = self.end_activation_type if self.end_activation else None)
        
        return layer(x)

class ConcatSquash(hk.Module):
    def __init__(self, out_size, name = "ConcatSquash", first_layer = True, sine_activation = False):
        super().__init__(name=name)

        self.out_size = out_size

        if sine_activation:
            if first_layer:
                self.weight_init = lambda shape,dtype : uniform(PRNGKey(0),shape, minval = -math.sqrt(6/shape[0])/30, maxval = math.sqrt(6/shape[0])/30)
            else:
                self.weight_init = lambda shape,dtype : uniform(PRNGKey(0),shape, minval = -1/shape[0], maxval = 1/shape[0])

        else:
            self.weight_init = None


    def __call__(self, t, x):
        lin1 = hk.Linear(self.out_size, w_init = self.weight_init)
        lin2 = hk.Linear(self.out_size)
        lin3 = hk.Linear(self.out_size, with_bias = False)

        return lin1(x) * jax.nn.sigmoid(lin2(t)) + lin3(t)

def sine(x):
    return jnp.sin(30*x)

class FFJORDNet(hk.Module):
  def __init__(self, layer_size, name = "FFJORDNet", activation = jax.nn.gelu, positional_embeddings = 0, **kwargs):
    super().__init__(name=name)

    self.layer_size = layer_size
    self.activation = activation

    if self.activation == sine:
        self.sine_activation = True
    else:
        self.sine_activation = False

    self.positional_embeddings = positional_embeddings

  def __call__(self, t, x):
    x = PositionalEncoding(self.positional_embeddings)(x)

    x = self.activation(ConcatSquash(self.layer_size[0], sine_activation=self.sine_activation)(t, x))

    for layer_size in self.layer_size[1:-1]:
        x = self.activation(ConcatSquash(layer_size, first_layer=False, sine_activation=self.sine_activation)(t, x)) + x

    return ConcatSquash(self.layer_size[-1], first_layer=False, sine_activation=self.sine_activation)(t, x)
  
class PositionalEncoding(hk.Module):
    """
    Takes encoding_size and returns a function that takes in a batch of inputs and returns the positional encoding for each input.
    The encodings are of the form sin(Wx+b) where W and b are learned parameters.
    """
    def __init__(self, encoding_size, name = "PositionalEncoding"):
        super().__init__(name=name)
        self.encoding_size = encoding_size

    def __call__(self, x):
        encoding = jnp.sin(
            hk.Linear(
                self.encoding_size, 
                w_init = lambda shape,dtype : uniform(
                    PRNGKey(0),
                    shape, 
                    minval = -10, 
                    maxval = 10,
                ),
            )(x),
        )
        return jnp.concatenate((x,encoding),axis=-1)

def _tree_transpose(list_of_trees):
    list_of_trees = jax.tree_map(lambda x: jnp.expand_dims(x,axis=0),list_of_trees)
    return jax.tree_map(lambda *xs: jnp.concatenate(xs,axis=0), *list_of_trees)

def Diagonal(rescale = 1):
    print(f"RESCALE = {rescale}")
    def init_fun(rng, input_dim, **kwargs):
        params = {"scale":jnp.zeros((input_dim,),float), "rescale": rescale}

        def direct_fun(params, input, *args, **kwargs):
            positive_params = jnp.exp(params["scale"])/rescale
            return (input * positive_params), jnp.sum(params["scale"] - jnp.log(rescale))

        def inverse_fun(params, input, **kwargs):
            positive_params = jnp.exp(params["scale"])/rescale
            return (input / positive_params), -jnp.sum(params["scale"] + jnp.log(rescale))

        return params, direct_fun, inverse_fun

    return init_fun

def Shift(amount):
    def init_fun(rng, input_dim, **kwargs):
        params = {"shift":amount}

        def direct_fun(params, input, *args, **kwargs):
            return (input + params["shift"]), jnp.zeros(input.shape[0])

        def inverse_fun(params, input, **kwargs):
            print(input.shape)
            return (input - params["shift"]), jnp.zeros(input.shape[0])

        return params, direct_fun, inverse_fun

    return init_fun

def RandomShift():
    def init_fun(rng, input_dim, **kwargs):
        params = {"shift": uniform(rng, (input_dim,), minval=-2, maxval=2)}

        def direct_fun(params, input, *args, **kwargs):
            return (input + params["shift"]), jnp.zeros(input.shape[0])

        def inverse_fun(params, input, **kwargs):
            return (input - params["shift"]), jnp.zeros(input.shape[0])

        return params, direct_fun, inverse_fun

    return init_fun

def Loop(subunit,num_interations):
    def init_fun(rng, input_dim, **kwargs):
        init_inputs = kwargs.pop("init_inputs", None)

        all_params = []
        for i in range(num_interations):
            rng, layer_rng = split(rng)
            param, apply_fun, reverse_fun = subunit(layer_rng, input_dim, init_inputs=init_inputs)

            all_params.append(param)

            if not (init_inputs is None):
                init_inputs = apply_fun(param, init_inputs)[0]

        all_params = _tree_transpose(all_params)

        def forward(input,params):
            inputs,log_prob = input
            outputs,log_det_jacobian = apply_fun(params,inputs)
            return (outputs,log_prob+log_det_jacobian),0

        def feed_forward(params, inputs):
            log_det_jacobians = jnp.zeros(inputs.shape[:1])
            return scan(forward,(inputs,log_det_jacobians),params)[0]

        def direct_fun(params, inputs, *args, **kwargs):
            return feed_forward(params, inputs)

        def reverse(input,params):
            inputs,log_prob = input
            outputs,log_det_jacobian = reverse_fun(params,inputs)
            return (outputs,log_prob+log_det_jacobian),0

        def feed_backward(params, inputs):
            log_det_jacobians = jnp.zeros(inputs.shape[:1])
            return scan(reverse,(inputs,log_det_jacobians),params,reverse=True)[0]

        def inverse_fun(params, inputs, *args, **kwargs):
            return feed_backward(params, inputs)

        return all_params, direct_fun, inverse_fun

    return init_fun



def get_gaussian(prior=flows.Normal(), scaling_layer = True):
    if scaling_layer:
        return Flow(Shift(),prior)#flows.Serial(Shift(),Diagonal()),prior)
    else:
        return prior


def flow_to_prior(flow_params,log_pdf,sample):
    def init_fun(rng,input_dim):
        def new_log_pdf(params,inputs):
            return log_pdf(flow_params,inputs)

        def new_sample(rng,params,num_samples=1):
            return sample(rng,flow_params,num_samples)

        return None,new_log_pdf,new_sample
    return init_fun


def flow_and_exact_to_prior(flow_params,exact_log_pdf,sample):
    def init_fun(rng,input_dim):
        def new_sample(rng,params,num_samples=1):
            return sample(rng,flow_params,num_samples)

        return None,exact_log_pdf,new_sample
    return init_fun


def vectorize_flow(params, log_pdf, sample):
    vector_params, vector_to_params = ravel_pytree(params)

    def log_pdf_vector(params,inputs):
        return log_pdf(vector_to_params(params),inputs)
    
    def sample_vector(rng,params,inputs):
        return sample(rng,vector_to_params(params),inputs)

    return (vector_params,log_pdf_vector,sample_vector),vector_to_params


def Flow_with_transform(transformation, prior=flows.Normal()):
    """
    Args:
        transformation: a function mapping ``(rng, input_dim)`` to a
            ``(params, direct_fun, inverse_fun)`` triplet
        prior: a function mapping ``(rng, input_dim)`` to a
            ``(params, log_pdf, sample)`` triplet
    Returns:
        A function mapping ``(rng, input_dim)`` to a ``(params, log_pdf, sample, x_to_z, z_to_x)`` tuple.
    Examples:
        >>> import flows
        >>> input_dim, rng = 3, random.PRNGKey(0)
        >>> transformation = flows.Serial(
        ...     flows.Reverse(),
        ...     flows.Reverse()
        ... )
        >>> init_fun = flows.Flow(transformation, Normal())
        >>> params, log_pdf, sample = init_fun(rng, input_dim)
    """

    def init_fun(rng, input_dim):
        transformation_rng, prior_rng = split(rng)

        params, direct_fun, inverse_fun = transformation(transformation_rng, input_dim)
        prior_params, prior_log_pdf, prior_sample = prior(prior_rng, input_dim)

        def log_pdf(params, inputs):
            u, log_det = direct_fun(params, inputs)
            log_probs = prior_log_pdf(prior_params, u)
            return log_probs + log_det

        def sample(rng, params, num_samples=1):
            prior_samples = prior_sample(rng, prior_params, num_samples)
            return inverse_fun(params, prior_samples)[0]

        return params, log_pdf, sample, direct_fun, inverse_fun

    return init_fun


def AffineCouplingSplit(scale, translate):
    """An implementation of a coupling layer from `Density Estimation Using RealNVP`
    (https://arxiv.org/abs/1605.08803).
    Args:
        scale: An ``(params, apply_fun)`` pair characterizing a trainable scaling function
        translate: An ``(params, apply_fun)`` pair characterizing a trainable translation function
    Returns:
        An ``init_fun`` mapping ``(rng, input_dim)`` to a ``(params, direct_fun, inverse_fun)`` triplet.
    """

    def init_fun(rng, input_dim, **kwargs):
        cutoff = input_dim // 2

        scale_rng, rng = split(rng)
        scale_params, scale_apply_fun = scale(scale_rng, cutoff, input_dim - cutoff)

        translate_rng, rng = split(rng)
        translate_params, translate_apply_fun = translate(translate_rng, cutoff, input_dim - cutoff)

        def direct_fun(params, inputs, **kwargs):
            scale_params, translate_params = params
            lower, upper = inputs[:, :cutoff], inputs[:, cutoff:]

            log_weight = scale_apply_fun(scale_params, lower)
            bias = translate_apply_fun(translate_params, lower)
            id_print(jnp.max(jnp.exp(log_weight)))
            id_print(jnp.min(jnp.exp(log_weight)))
            upper = upper * jnp.exp(log_weight) + bias

            outputs = jnp.concatenate([lower, upper], axis=1)
            log_det_jacobian = log_weight.sum(-1)
            return outputs, log_det_jacobian

        def inverse_fun(params, inputs, **kwargs):
            scale_params, translate_params = params
            lower, upper = inputs[:, :cutoff], inputs[:, cutoff:]

            log_weight = scale_apply_fun(scale_params, lower)
            bias = translate_apply_fun(translate_params, lower)
            id_print(jnp.max(jnp.exp(log_weight)))
            id_print(jnp.min(jnp.exp(log_weight)))
            upper = (upper - bias) * jnp.exp(-log_weight)

            outputs = jnp.concatenate([lower, upper], axis=1)
            log_det_jacobian = log_weight.sum(-1)
            return outputs, log_det_jacobian

        return (scale_params, translate_params), direct_fun, inverse_fun

    return init_fun



def Flow(transformation, prior=flows.Normal()):
    """
    Taken from jax-flows library.

    Args:
        transformation: a function mapping ``(rng, input_dim)`` to a
            ``(params, direct_fun, inverse_fun)`` triplet
        prior: a function mapping ``(rng, input_dim)`` to a
            ``(params, log_pdf, sample)`` triplet
    Returns:
        A function mapping ``(rng, input_dim)`` to a ``(params, log_pdf, sample)`` triplet.
    Examples:
        >>> import flows
        >>> input_dim, rng = 3, random.PRNGKey(0)
        >>> transformation = flows.Serial(
        ...     flows.Reverse(),
        ...     flows.Reverse()
        ... )
        >>> init_fun = flows.Flow(transformation, Normal())
        >>> params, log_pdf, sample = init_fun(rng, input_dim)
    """

    def init_fun(rng, input_dim, **kwargs):
        transformation_rng, prior_rng = jax.random.split(rng)

        params, direct_fun, inverse_fun = transformation(transformation_rng, input_dim, **kwargs)
        prior_params, prior_log_pdf, prior_sample = prior(prior_rng, input_dim)

        def log_pdf(params, inputs):
            u, log_det = direct_fun(params, inputs)
            log_probs = prior_log_pdf(prior_params, u)
            return log_probs + log_det

        def sample(rng, params, num_samples=1):
            prior_samples = prior_sample(rng, prior_params, num_samples)
            return inverse_fun(params, prior_samples)[0]

        return params, log_pdf, sample

    return init_fun


def ShiftedGaussianInitialDistribution():

    def init_fun(rng, input_dim):
        def log_pdf(params, inputs):
            return -(inputs.shape[-1]//2)*jnp.log(math.pi)-jnp.sum((inputs+1)**2,axis=-1)

        def sample(rng, params, num_samples=1):
            return (1/jnp.sqrt(2))*normal(rng, (num_samples, input_dim))-1

        return (), log_pdf, sample

    return init_fun

def ShiftedGaussianInitialDistribution2():

    def init_fun(rng, input_dim):
        def log_pdf(params, inputs):
            return -(inputs.shape[-1]//2)*jnp.log(2*math.pi)-jnp.sum(((inputs-jnp.array([1,0]))**2)/2,axis=-1)

        def sample(rng, params, num_samples=1):
            return normal(rng, (num_samples, input_dim))+jnp.array([1,0])

        return (), log_pdf, sample

    return init_fun


def flow_mixture(flow_init, num_flows):
    def init_fun(rng, input_dim, **kwargs):
        _, logprob_single, sample_single = flow_init(PRNGKey(0), input_dim)

        logprob_all = vmap(logprob_single, in_axes=(0, None))

        new_params = []

        keys = split(rng, num_flows)

        for i in range(num_flows):
            new_params.append(flow_init(keys[i],input_dim)[0])

        def concatenate(*xs):
            return jnp.concatenate([jnp.expand_dims(x, axis = 0) for x in xs])

        new_params = jax.tree_map(concatenate, *new_params)

        full_params = {"params": new_params, "scale_logits": jnp.zeros((num_flows,), dtype = float)}

        def log_pdf(params, inputs):
            scales = jax.nn.log_softmax(params["scale_logits"])
            scaled_logprobs = logprob_all(params["params"], inputs) + scales[:, jnp.newaxis, jnp.newaxis]

            return logsumexp(scaled_logprobs, axis = 0)
        
        @partial(vmap, in_axes = (0, 0, None))
        def sample_i(i, rng, params):
            return sample_single(rng, jax.tree_map(lambda x: x[i], params))[0]

        def sample(rng, params, num_samples=1):
            key1,key2 = split(rng)
            keys = split(key2, num_flows)

            flow_choice = categorical(key1, params["scale_logits"], shape = (num_flows,))
            
            return sample_i(flow_choice, keys, params["params"])

        print(full_params)
        return full_params, log_pdf, sample

    return init_fun