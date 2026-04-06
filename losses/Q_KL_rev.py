import jax.numpy as jnp
from jax.lax import stop_gradient

from jax.random import split, PRNGKey

from losses.base_loss import Loss
from training_sampler import SampleFromDistribution, SampleShufflerFromDistribution

from distributions.Q_function import JaxQ
from distributions.W_function import JaxWigner

import equinox as eqx

class KL_rev_sample_target(Loss):
    def __init__(self, 
                 rng,
                 model, 
                 target, 
                 target_params,
                 num_samples = 10000, 
                 batch_size = 1000, 
                 resample = True,
                 has_validation = False,
                 grad_loss = False,
                 name = "KL",
                 **kwargs):
        
        self.model = model
        self.target = target

        self.grad_loss = grad_loss

        if resample:
            self.sampler = SampleFromDistribution(target.sample, num_samples, batch_size, has_validation)
        else:
            self.sampler = SampleShufflerFromDistribution(rng, target.sample, target_params, num_samples, batch_size, has_validation)

        super().__init__(has_validation = has_validation, name = name, **kwargs)

    @eqx.filter_jit
    def loss(self, model_params, inputs, target_logprobs, rng):
        model_logprobs = self.model.log_Q(model_params, inputs)
            
        log_ratio = model_logprobs - target_logprobs
            
        return (jnp.exp(log_ratio)*log_ratio).mean()


    def __call__(self, rng, model_params, target_params, sampler_state = None, validation = False):
        key1, key2, key3 = split(rng, 3)
        inputs, sampler_state = self.sampler.sample(key1, target_params, sampler_state, validation = validation)

        target_logprobs = self.target.log_Q(target_params, inputs, key2)
        
        if self.grad_loss and not validation:
            return eqx.filter_jit(eqx.filter_value_and_grad(self.loss))(model_params, inputs, target_logprobs, key3), sampler_state
        return self.loss(model_params, inputs, target_logprobs, key3), sampler_state
    

class KL_rev_sample_model_reparam(Loss):
    def __init__(self, 
                 rng,
                 model, 
                 target, 
                 num_samples = 10000, 
                 batch_size = 1000,
                 has_validation = False,
                 grad_loss = False,
                 name = "KL",
                 **kwargs):
        
        self.model = model
        self.target = target

        if not (isinstance(self.target, JaxQ) or isinstance(self.target, JaxWigner)):
            raise ValueError("Target distribution must be a Jax-compatible Q or W function (it must extend JaxQ or JaxWigner)")

        self.grad_loss = grad_loss

        self.sampler = SampleFromDistribution(model.sample, num_samples, batch_size, has_validation)

        super().__init__(has_validation = has_validation, name = name, **kwargs)

    @eqx.filter_jit
    def loss(self, model_params, target_params, rng, sampler_state, validation = False):
        key1, key2, key3 = split(rng, 3)
        inputs, sampler_state = self.sampler.sample(key1, model_params, sampler_state, validation = validation)

        target_logprobs = self.target.log_Q(target_params, inputs, key2)
        model_logprobs = self.model.log_Q(model_params, inputs, key3)
            
        log_ratio = model_logprobs - target_logprobs
            
        return log_ratio.mean(), sampler_state

    def __call__(self, rng, model_params, target_params, sampler_state = None, validation = False):
        if self.grad_loss and not validation:
            (value,sampler_state),grad = eqx.filter_jit(eqx.filter_value_and_grad(self.loss, has_aux = True))(model_params, target_params, rng, sampler_state, validation)
            return (value,grad), sampler_state
        
        return self.loss(model_params, target_params, rng, sampler_state, validation)
    
class KL_rev_sample_model_control_variance(Loss):
    def __init__(self, 
                 rng,
                 model, 
                 target, 
                 num_samples = 10000, 
                 batch_size = 1000,
                 has_validation = False,
                 grad_loss = False,
                 name = "KL",
                 **kwargs):
        
        self.model = model
        self.target = target

        if not (isinstance(self.target, JaxQ) or isinstance(self.target, JaxWigner)):
            raise ValueError("Target distribution must be a Jax-compatible Q or W function (it must extend JaxQ or JaxWigner)")

        self.grad_loss = grad_loss

        self.sampler = SampleFromDistribution(model.sample, num_samples, batch_size, has_validation)

        super().__init__(has_validation = has_validation, name = name, **kwargs)

    def loss(self, model_params, inputs, target_logprobs, rng):
        model_logprobs = self.model.log_Q(model_params, inputs, rng)
            
        log_ratio = model_logprobs - target_logprobs
            
        return log_ratio.mean()
    
    def loss_grad(self, model_params, inputs, target_logprobs, rng):
        model_logprobs = self.model.log_Q(model_params, inputs, rng)
            
        log_ratio = model_logprobs - target_logprobs

        return (stop_gradient(log_ratio - log_ratio.mean()) * model_logprobs).mean()

    def __call__(self, rng, model_params, target_params, sampler_state = None, validation = False):
        key1, key2, key3, key4 = split(rng, 4)

        inputs, sampler_state = self.sampler.sample(key1, model_params, sampler_state, validation = validation)
        target_logprobs = self.target.log_Q(target_params, inputs, key2)

        L1 = self.loss(model_params, inputs, target_logprobs, key3)

        if self.grad_loss and not validation:
            L1_grad = eqx.filter_jit(eqx.filter_grad(self.loss_grad))(model_params, inputs, target_logprobs, key4)
            return (L1,L1_grad), sampler_state
        
        return L1, sampler_state
    

class KL_rev_sample_uniform(Loss):
    samples: jnp.ndarray
    target_log_Q: jnp.ndarray
    scale: float

    def __init__(self,
                 model, 
                 target, 
                 target_params,
                 x_range,
                 y_range,
                 has_validation = False,
                 grad_loss = False,
                 name = "KL_rev_sample_uniform",
                 **kwargs):
        
        self.model = model
        self.target = target

        self.grad_loss = grad_loss

        self.scale = ((x_range[1] - x_range[0])/x_range[2]) * ((y_range[1] - y_range[0])/y_range[2])

        print(self.scale)

        xs = jnp.linspace(x_range[0], x_range[1], x_range[2])
        ys = jnp.linspace(y_range[0], y_range[1], y_range[2])
        X,Y = jnp.meshgrid(xs,ys)
        self.samples = jnp.stack([X,Y], axis = -1).reshape(-1,2)

        self.sampler = SampleFromDistribution(model.sample, self.samples.shape[0], self.samples.shape[0], has_validation)

        self.target_log_Q = self.target.log_Q(target_params, self.samples, PRNGKey(0))

        super().__init__(has_validation = has_validation, name = name, **kwargs)

    @eqx.filter_jit
    def loss(self, model_params, rng):
        model_log_Q = self.model.log_Q(model_params, self.samples, rng)

        id_print(jnp.max(self.target_log_Q))
        id_print(jnp.max(model_log_Q))

        log_ratio = model_log_Q - self.target_log_Q
            
        return (jnp.exp(model_log_Q) * log_ratio).sum() * self.scale

    def __call__(self, rng, model_params, target_params, sampler_state = None, validation = False):
        if self.grad_loss and not validation:
            return eqx.filter_jit(eqx.filter_value_and_grad(self.loss))(model_params, rng), sampler_state
        return self.loss(model_params), sampler_state
