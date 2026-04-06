from typing import Callable
from jax.random import split, PRNGKey
from jax import nn
import jax.numpy as jnp
from distributions.distribution import Distribution

from pyparsing import Any

from Q_flows.mcmc import mcmc_sample

import equinox as eqx

import abc

class Wigner(Distribution):

    input_dim: int

    @abc.abstractmethod
    def w(self, params, inputs, rng):
        ...
        
    @eqx.filter_jit
    @abc.abstractmethod
    def sample_probs(self, params, samples, rng, sampler = "mcmc"):
        ...

    def abs_w(self, params, inputs, rng):
        return jnp.abs(self.w(params, inputs, rng))
    
    def log_abs_w(self, params, inputs, rng):
        return jnp.log(self.abs_w(params, inputs, rng))
    
    def z(self, params, num_samples, key, sampler = "not_mcmc"):
        if sampler == "mcmc":
            raise Exception("MCMC sampling not allowed for computing Z!")
        
        rng1, rng2 = split(key)

        samples = self.sample(params, num_samples, rng1, sampler = sampler)
        sample_abs_ws = self.abs_w(params, samples, rng2)
        sample_probabilities = self.sample_probs(params, samples, sampler = sampler)

        return jnp.mean(sample_abs_ws/sample_probabilities)
    
    def w_tilde(self, params, samples, key, sampler = "not_mcmc"):
        rng1, rng2 = split(key)
        return self.w(self, params, samples, rng1)/self.z(params, samples.shape[0], rng2, sampler = sampler)
    
    def abs_w_tilde(self, params, samples, key, sampler = "not_mcmc"):
        rng1, rng2 = split(key)
        return self.abs_w(params, samples, rng1)/self.z(params, samples.shape[0], rng2, sampler = sampler)
    
    def log_abs_w_tilde(self, params, samples, key, sampler = "not_mcmc"):
        return jnp.log(self.abs_w_tilde(params, samples, key, sampler = sampler))
    
    def plot(self, params, ranges, true_distribution = None, true_params = None, plot_difference = False, save = False, filename = None, filetype = None):
        from plotting import plot_W
        
        plot_W(
            self,
            params,
            x_range=ranges[0],
            y_range=ranges[1],
            w_true = true_distribution,
            true_params = true_params,
            plot_difference = plot_difference,
            save = save,
            filename = filename,
            filetype = filetype,
            num_dims = self.input_dim,
        )
    

class JaxWigner(Wigner):

    input_dim: int

    @eqx.filter_jit
    @abc.abstractmethod
    def init(self, rng, inputs=None, real_data=False):
        ...

    @eqx.filter_jit
    @abc.abstractmethod
    def w(self, params, inputs, rng):
        ...
        
    @eqx.filter_jit
    @abc.abstractmethod
    def sample_probs(self, params, samples, rng, sampler = "mcmc"):
        if sampler == "mcmc":
            return self.abs_w_tilde(params, samples, rng)

    @eqx.filter_jit
    def abs_w(self, params, inputs, rng):
        return jnp.abs(self.w(params, inputs, rng))
    
    @eqx.filter_jit
    def log_abs_w(self, params, inputs, rng):
        return jnp.log(self.abs_w(params, inputs, rng))
    
    @eqx.filter_jit
    def z(self, params, num_samples, key, sampler = "not_mcmc"):
        if sampler == "mcmc":
            raise Exception("MCMC sampling not allowed for computing Z!")
        
        rng1, rng2, rng3 = split(key, 3)

        samples = self.sample(params, num_samples, rng1, sampler = sampler)
        sample_abs_ws = self.abs_w(params, samples, rng2)
        sample_probabilities = self.sample_probs(params, samples, rng3, sampler = sampler)

        return jnp.mean(sample_abs_ws/sample_probabilities)
    
    @eqx.filter_jit
    def w_tilde(self, params, samples, key, sampler = "not_mcmc"):
        rng1, rng2 = split(key)
        return self.w(self, params, samples, rng1)/self.z(params, samples.shape[0], rng2, sampler = sampler)
    
    @eqx.filter_jit
    def abs_w_tilde(self, params, samples, key, sampler = "not_mcmc"):
        rng1, rng2 = split(key)
        return self.abs_w(params, samples, rng1)/self.z(params, samples.shape[0], rng2, sampler = sampler)
    
    @eqx.filter_jit
    def log_abs_w_tilde(self, params, samples, key, sampler = "not_mcmc"):
        return jnp.log(self.abs_w_tilde(params, samples, key, sampler = sampler))
    
    @eqx.filter_jit
    def sample(self, params, num_samples, key, sampler = "flow", **kwargs):
        if sampler == "mcmc":
            return self.mcmc_sample(params, num_samples, key)

    @eqx.filter_jit
    def mcmc_sample(self, params, num_samples, key):
        return mcmc_sample(
            self.log_abs_w,
            self.input_dim,
            params,
            num_samples,
            key,
        )


class QWigner(JaxWigner):

    positive_scale: bool
    kwargs: Any

    flow_init: Callable
    log_pdf: Callable
    flow_sample: Callable

    def __init__(self, flow_init, input_dim, positive_scale = False, **kwargs):
        self.input_dim = input_dim
        
        _, log_pdf, flow_sample = flow_init(
            PRNGKey(0), 
            input_dim=input_dim, 
            input_inits=None,
        )

        self.flow_init = flow_init
        self.log_pdf = log_pdf
        self.flow_sample = flow_sample

        self.positive_scale = positive_scale

        self.kwargs = kwargs

    def init(self, rng, inputs=None, real_data=False):
        key1,key2 = split(rng)
        params1, _, _ = self.flow_init(
            key1, input_dim=self.input_dim, input_inits=inputs if real_data else None)
        params2, _, _ = self.flow_init(
            key2, input_dim=self.input_dim, input_inits=inputs if real_data else None)
        return {"Q_params": (params1, params2), "scale": jnp.array([1.])}

    def w(self, params, inputs, rng):
        if self.positive_scale:
            scale = nn.softplus(params["scale"][0]) + 1
        else:
            scale = params["scale"][0]
        
        return (scale+1)*jnp.exp(self.log_pdf(params["Q_params"][0], inputs)) - scale*jnp.exp(self.log_pdf(params["Q_params"][1], inputs))

    def sample(self, params, num_samples, key, sampler = "not_mcmc"):
        if sampler == "average" or sampler == "not_mcmc":
            rng1,rng2 = split(key)

            samples1 = self.flow_sample(rng1, params["Q_params"][0], num_samples)
            samples2 = self.flow_sample(rng2, params["Q_params"][1], num_samples)
                
            return jnp.concatenate((samples1, samples2), axis=0)
            
        if sampler == "mcmc":
            return self.mcmc_sample(params, num_samples, key)
        
    def sample_probs(self, params, samples, rng, sampler = "not_mcmc"):
        """
        Only works if samples are taken using the sample function *for this class*
        """

        if sampler == "average" or sampler == "not_mcmc":
            prob1 = jnp.exp(self.log_pdf(params["Q_params"][0], samples))
            prob2 = jnp.exp(self.log_pdf(params["Q_params"][1], samples))

            return (prob1+prob2)/2
            
        if sampler == "mcmc":
            return self.abs_w_tilde(params, samples, rng)