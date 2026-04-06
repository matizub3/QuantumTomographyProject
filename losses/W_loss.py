from functools import partial
import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
from jax.debug import print as id_print

from losses.base_loss import Loss
from training_sampler import SampleEfficientSampler, SampleFromDistribution, SampleShufflerFromDistribution

from distributions.Q_function import JaxQ
from distributions.W_function import JaxWigner

import equinox as eqx

class W_L1_sample_target(Loss):
    Z: float

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
                 scale_by_Z = True,
                 Z_samples = None,
                 **kwargs):
        
        self.model = model
        self.target = target

        if scale_by_Z:
            if Z_samples is None:
                Z_samples = num_samples

            #self.Z = self.target.z(target_params, num_samples, rng)
            self.Z = 1
        else:
            self.Z = 1

        self.grad_loss = grad_loss

        if resample:
            self.sampler = SampleFromDistribution(target.sample, num_samples, batch_size, has_validation)
        else:
            self.sampler = SampleShufflerFromDistribution(rng, target.sample, target_params, num_samples, batch_size, has_validation)

        super().__init__(has_validation = has_validation, name = name, **kwargs)

    @eqx.filter_jit
    def loss(self, model_params, inputs, target_W, key):
        model_W = self.model.w(model_params, inputs, key)
            
        ratio = model_W/target_W
        id_print("mean: {x}, std: {y}", x = ratio.mean(), y = ratio.std())

        return self.Z*(jnp.abs(1-ratio)).mean()

    def __call__(self, rng, model_params, target_params, sampler_state = None, validation = False):
        key1,key2,key3,key4 = jax.random.split(rng, 4)
        inputs, sampler_state = self.sampler.sample(key1, target_params, sampler_state, validation = validation)

        target_W = self.target.w(target_params, inputs, key2)
        
        if self.grad_loss and not validation:
            return eqx.filter_jit(eqx.filter_value_and_grad(self.loss))(model_params, inputs, target_W, key3), sampler_state
        return self.loss(model_params, inputs, target_W, key4), sampler_state
    

class W_L1_sample_model(Loss):
    sample_method: str

    def __init__(self, 
                 model,
                 target,
                 num_samples = 10000, 
                 batch_size = 1000,
                 has_validation = False,
                 grad_loss = False,
                 name = "L1_sample_model",
                 sample_method = "mcmc",
                 **kwargs):
        
        self.model = model
        self.target = target

        self.grad_loss = grad_loss

        if sample_method not in ["mcmc", "average", "not_mcmc"]:
            raise ValueError("sample_method must be one of 'mcmc', 'average', or 'not_mcmc'")
        else:
            self.sample_method = sample_method

        self.sampler = SampleFromDistribution(
            partial(model.sample, sampler = sample_method),
            num_samples, 
            batch_size, 
            has_validation,
        )

        super().__init__(has_validation = has_validation, name = name, **kwargs)

    @eqx.filter_jit
    def loss(self, model_params, inputs, target_W, sample_probs, key):
        model_W = self.model.w(model_params, inputs, key)
            
        ratio = jnp.abs(model_W-target_W)/sample_probs

        return ratio.mean()
    
    def loss_grad(self, model_params, inputs, target_W, sample_probs, key):
        model_W = self.model.w(model_params, inputs, key)
            
        ratio = stop_gradient((jnp.sign(model_W-target_W) - jnp.sign(model_W-target_W).mean())/sample_probs)

        return (ratio * model_W).mean()

    def __call__(self, rng, model_params, target_params, sampler_state = None, validation = False):
        key1,key2,key3,key4,key5 = jax.random.split(rng, 5)
        
        inputs, sampler_state = eqx.filter_jit(self.sampler.sample)(
            key1,
            model_params,
            sampler_state,
            validation = validation,
        )

        sample_probs = eqx.filter_jit(self.model.sample_probs)(
            model_params, 
            inputs,
            key2,
            sampler = self.sample_method,
        )

        target_W = self.target.w(target_params, inputs, key3)

        loss_value = self.loss(model_params, inputs, target_W, sample_probs, key4)
        
        if self.grad_loss and not validation:
            loss_grad = eqx.filter_jit(eqx.filter_grad(self.loss_grad))(model_params, inputs, target_W, sample_probs, key5)
            return (loss_value, loss_grad), sampler_state
        return loss_value, sampler_state
    

class W_L1_sample_uniform(Loss):
    samples: jnp.ndarray
    target_W: jnp.ndarray
    scale: float

    def __init__(self,
                 model, 
                 target, 
                 target_params,
                 x_range = None,
                 y_range = None,
                 samples = None, 
                 target_W = None,
                 has_validation = False,
                 grad_loss = False,
                 name = "L1_sample_uniform",
                 **kwargs):
        
        self.model = model
        self.target = target

        self.grad_loss = grad_loss

        if x_range is not None and y_range is not None:
            self.scale = ((x_range[1] - x_range[0])/x_range[2]) * ((y_range[1] - y_range[0])/y_range[2])

            xs = jnp.linspace(x_range[0], x_range[1], x_range[2])
            ys = jnp.linspace(y_range[0], y_range[1], y_range[2])
            X,Y = jnp.meshgrid(xs,ys)
            self.samples = jnp.stack([X,Y], axis = -1).reshape(-1,2)
        elif samples is not None:
            self.samples = samples
            
            #the smallest distance between sampled points
            self.scale = jnp.min(jnp.linalg.norm(self.samples[:-1,:] - self.samples[1:,:], axis = -1)) ** 2

            if target_W is not None:
                print("Using QST_CGAN_W_Neg PROVIDED TARGET W")
                self.target_W = target_W
            else:
                self.target_W = self.target.w(target_params, self.samples)

        self.sampler = SampleFromDistribution(model.sample, self.samples.shape[0], self.samples.shape[0], has_validation)

        super().__init__(has_validation = has_validation, name = name, **kwargs)

    @eqx.filter_jit
    def loss(self, model_params, rng):
        model_W = self.model.w(model_params, self.samples, rng)
            
        return jnp.abs(model_W - self.target_W).sum() * self.scale

    def __call__(self, rng, model_params, target_params, sampler_state = None, validation = False):
        if self.grad_loss and not validation:
            return eqx.filter_jit(eqx.filter_value_and_grad(self.loss))(model_params, rng), sampler_state
        return self.loss(model_params, rng), sampler_state
    



class W_L1_sample_efficient(Loss):
    def __init__(self, 
                 model, 
                 target,
                 samples_per_resample = 100,
                 resample_every = 10,
                 batch_size = 1000,
                 has_validation = False,
                 grad_loss = False,
                 name = "L1_sample_efficient",
                 sample_method = "mcmc",
                 **kwargs):

        if sample_method not in ["mcmc", "average", "not_mcmc"]:
            raise ValueError("sample_method must be one of 'mcmc', 'average', or 'not_mcmc'")
        
        self.model = model
        self.target = target

        self.grad_loss = grad_loss

        self.sampler = SampleEfficientSampler(
            partial(model.sample, sampler = sample_method), 
            partial(self.model.sample_probs, sampler = sample_method), 
            target.w,
            batch_size, 
            samples_per_resample, 
            resample_every,
            logprobs = False,
        )

        super().__init__(has_validation = has_validation, name = name, **kwargs)
    
    @eqx.filter_jit
    def loss(self, model_params, inputs, target_W, sample_probs, rng):
        model_W = self.model.w(model_params, inputs, rng)
            
        ratio = jnp.abs(model_W-target_W)/sample_probs

        return ratio.mean()
    
    def loss_grad(self, model_params, inputs, target_W, sample_probs, rng):
        model_W = self.model.w(model_params, inputs, rng)
            
        ratio = stop_gradient((jnp.sign(model_W-target_W) - jnp.sign(model_W-target_W).mean())/sample_probs)

        return (ratio * model_W).mean()

    def __call__(self, rng, model_params, target_params, sampler_state = None, validation = False):
        key1, key2, key3 = jax.random.split(rng, 3)
        samples, sample_probs, target_W, sampler_state = self.sampler.sample(key1, model_params, target_params, sampler_state, validation = validation)

        loss_value = eqx.filter_jit(self.loss)(model_params, samples, target_W, sample_probs, key2)
        
        if self.grad_loss and not validation:
            loss_grad = eqx.filter_jit(eqx.filter_grad(self.loss_grad))(model_params, samples, target_W, sample_probs, key3)
            return (loss_value, loss_grad), sampler_state
        return loss_value, sampler_state
    

class W_trace_sample_target(Loss):
    Z: float

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
                 name = "Trace",
                 scale_by_Z = True,
                 Z_samples = None,
                 **kwargs):
        
        self.model = model
        self.target = target

        if scale_by_Z:
            if Z_samples is None:
                Z_samples = num_samples
                
            self.Z = self.target.z(target_params, num_samples, rng)
        else:
            self.Z = 1

        self.grad_loss = grad_loss

        if resample:
            self.sampler = SampleFromDistribution(target.sample, num_samples, batch_size, has_validation)
        else:
            self.sampler = SampleShufflerFromDistribution(rng, target.sample, target_params, num_samples, batch_size, has_validation)

        super().__init__(has_validation = has_validation, name = name, **kwargs)

    @eqx.filter_jit
    def loss(self, model_params, inputs, target_W):
        model_W = self.model.w(model_params, inputs)
            
        signed_W = jnp.sign(target_W) * model_W

        return 1 - 2 * jnp.pi * self.Z * jnp.mean(signed_W)

    def __call__(self, rng, model_params, target_params, sampler_state = None, validation = False):
        inputs, sampler_state = self.sampler.sample(rng, target_params, sampler_state, validation = validation)

        target_W = self.target.w(target_params, inputs)
        
        if self.grad_loss and not validation:
            return eqx.filter_jit(eqx.filter_value_and_grad(self.loss))(model_params, inputs, target_W), sampler_state
        return self.loss(model_params, inputs, target_W), sampler_state
    

