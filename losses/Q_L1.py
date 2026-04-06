import jax
import jax.numpy as jnp
from jax.lax import stop_gradient

from losses.base_loss import Loss
from training_sampler import SampleFromDistribution, SampleShufflerFromDistribution

from distributions.Q_function import JaxQ
from distributions.W_function import JaxWigner

import equinox as eqx

class L1_sample_target(Loss):
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
                 name = "L1",
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
    def loss(self, model_params, inputs, target_logprobs):
        model_logprobs = self.model.log_Q(model_params, inputs)
            
        log_ratio = model_logprobs - target_logprobs

        ratio = jnp.exp(log_ratio)
            
        return (jnp.abs(1-ratio)).mean()


    def __call__(self, rng, model_params, target_params, sampler_state = None, validation = False):
        inputs, sampler_state = self.sampler.sample(rng, target_params, sampler_state, validation = validation)

        target_logprobs = self.target.log_Q(target_params, inputs)
        
        if self.grad_loss and not validation:
            return eqx.filter_jit(eqx.filter_value_and_grad(self.loss))(model_params, inputs, target_logprobs), sampler_state
        return self.loss(model_params, inputs, target_logprobs), sampler_state
    


class L1_sample_model_reparam(Loss):
    def __init__(self, 
                 rng,
                 model, 
                 target, 
                 num_samples = 10000, 
                 batch_size = 1000,
                 has_validation = False,
                 grad_loss = False,
                 name = "L1",
                 **kwargs):
        
        self.model = model
        self.target = target

        if not (isinstance(self.target, JaxQ) or isinstance(self.target, JaxWigner)):
            raise ValueError("Target distribution must be a Jax-compatible Q or W function (it must extend JaxQ or JaxWigner)")

        self.grad_loss = grad_loss

        self.sampler = SampleFromDistribution(model.sample, num_samples, batch_size, has_validation)

        super().__init__(has_validation = has_validation, name = name, **kwargs)

    def loss(self, model_params, target_params, rng, sampler_state, validation = False):
        inputs, sampler_state = self.sampler.sample(rng, model_params, sampler_state, validation = validation)

        target_logprobs = self.target.log_Q(target_params, inputs)
        model_logprobs = self.model.log_Q(model_params, inputs)
            
        log_ratio = target_logprobs - model_logprobs

        ratio = jnp.exp(log_ratio)
            
        return (jnp.abs(1-ratio)).mean(), sampler_state

    def __call__(self, rng, model_params, target_params, sampler_state = None, validation = False):
        if self.grad_loss and not validation:
            (value,sampler_state),grad = eqx.filter_jit(eqx.filter_value_and_grad(self.loss, has_aux = True))(model_params, target_params, rng, sampler_state, validation)
            return (value,grad), sampler_state
        
        return eqx.filter_jit(self.loss)(model_params, target_params, rng, sampler_state, validation)



class L1_sample_model_control_variance(Loss):
    control_variance: bool

    def __init__(self, 
                 rng,
                 model, 
                 target, 
                 num_samples = 10000, 
                 batch_size = 1000,
                 has_validation = False,
                 grad_loss = False,
                 name = "L1",
                 control_variance = True,
                 **kwargs):
        
        self.model = model
        self.target = target

        self.grad_loss = grad_loss

        self.control_variance = control_variance

        self.sampler = SampleFromDistribution(model.sample, num_samples, batch_size, has_validation)

        super().__init__(has_validation = has_validation, name = name, **kwargs)

    def loss(self, model_params, inputs, target_logprobs):
        model_logprobs = self.model.log_Q(model_params, inputs)
            
        log_ratio = target_logprobs - model_logprobs

        ratio = jnp.exp(log_ratio)
            
        return (jnp.abs(1-ratio)).mean()
    
    def loss_grad(self, model_params, inputs, target_logprobs):
        model_logprobs = self.model.log_Q(model_params, inputs)
            
        sign = jnp.sign(model_logprobs - target_logprobs)

        if self.control_variance:
            return (stop_gradient(sign - sign.mean()) * model_logprobs).mean()
        else:
            return (stop_gradient(sign) * model_logprobs).mean()

    def __call__(self, rng, model_params, target_params, sampler_state = None, validation = False):
        inputs, sampler_state = self.sampler.sample(rng, model_params, sampler_state, validation = validation)
        target_logprobs = self.target.log_Q(target_params, inputs)

        L1 = self.loss(model_params, inputs, target_logprobs)

        if self.grad_loss and not validation:
            L1_grad = eqx.filter_jit(eqx.filter_grad(self.loss_grad))(model_params, inputs, target_logprobs)
            return (L1,L1_grad), sampler_state
        
        return L1, sampler_state

class L1_sample_uniform(Loss):
    samples: jnp.ndarray
    validation_samples: jnp.ndarray
    target_Q: jnp.ndarray
    scale: float

    def __init__(self,
                 rng,
                 model, 
                 target, 
                 target_params,
                 x_range,
                 y_range,
                 has_validation = False,
                 train_samples = None,
                 grad_loss = False,
                 name = "L1_sample_uniform",
                 **kwargs):
        
        self.model = model
        self.target = target

        self.grad_loss = grad_loss

        self.scale = ((x_range[1] - x_range[0])/x_range[2]) * ((y_range[1] - y_range[0])/y_range[2])

        xs = jnp.linspace(x_range[0], x_range[1], x_range[2])
        ys = jnp.linspace(y_range[0], y_range[1], y_range[2])
        X,Y = jnp.meshgrid(xs,ys)
        samples = jnp.stack([X,Y], axis = -1).reshape(-1,2)

        if has_validation:
            if train_samples is None:
                raise ValueError("Must provide train_samples if validation is True")
            
            self.validation_samples = samples

            # Sample train_sample number of points from samples
            self.samples = samples[jax.random.choice(rng, samples.shape[0], (train_samples,), replace = False)]

        else:
            self.samples = samples
            self.validation_samples = None

        self.sampler = SampleFromDistribution(model.sample, self.samples.shape[0], self.samples.shape[0], has_validation)

        self.target_Q = self.target.Q(target_params, self.samples)

        super().__init__(has_validation = has_validation, name = name, **kwargs)

    @eqx.filter_jit
    def loss(self, model_params, validation = False):
        if validation:
            model_Q = self.model.Q(model_params, self.validation_samples)
        else:
            model_Q = self.model.Q(model_params, self.samples)
            
        return (jnp.abs(model_Q - self.target_Q)).sum() * self.scale

    def __call__(self, rng, model_params, target_params, sampler_state = None, validation = False):
        if self.grad_loss and not validation:
            return eqx.filter_jit(eqx.filter_value_and_grad(self.loss))(model_params), sampler_state
        return self.loss(model_params, validation = validation), sampler_state