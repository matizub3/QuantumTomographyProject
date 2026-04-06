import abc
from typing import Callable
from jax.random import PRNGKey
import jax.numpy as jnp
from distributions.distribution import Distribution

import equinox as eqx

from Q_flows.mcmc import mcmc_sample

class Q(Distribution):
    input_dim: int

    @abc.abstractmethod
    def Q(self, params, inputs, rng):
        ...

    @abc.abstractmethod
    def log_Q(self, params, inputs, rng):
        ...

    def plot(self, params, ranges, true_distribution = None, true_params = None, plot_difference = False, save = False, filename = None, filetype = None):
        from plotting import plot_Q
        
        plot_Q(
            self,
            params,
            x_range=ranges[0],
            y_range=ranges[1],
            exact=true_distribution,
            exact_params=true_params,
            plot_difference = plot_difference,
            save = save,
            filename = filename,
            filetype = filetype,
            num_dims = self.input_dim,
        )


class JaxQ(Q):
    input_dim: int

    @eqx.filter_jit
    @abc.abstractmethod
    def init(self, rng, inputs=None, real_data=False):
        ...

    @eqx.filter_jit
    def Q(self, params, inputs, rng):
        return jnp.exp(self.log_Q(params, inputs, rng))

    @eqx.filter_jit
    @abc.abstractmethod
    def log_Q(self, params, inputs, rng):
        ...

    @eqx.filter_jit
    def sample(self, params, num_samples, key, **kwargs):
        return self.mcmc_sample(params, num_samples, key)

    @eqx.filter_jit
    def mcmc_sample(self, params, num_samples, key):
        return mcmc_sample(
            self.log_Q,
            self.input_dim,
            params,
            num_samples,
            key,
        )

class QFlow(JaxQ):
    input_dim: int

    flow_init: Callable
    log_pdf: Callable
    flow_sample: Callable

    def __init__(self, flow_init, input_dim, random_init=False):
        self.input_dim = input_dim

        _, log_pdf, flow_sample = flow_init(
            PRNGKey(0), 
            input_dim=input_dim, 
            input_inits=None,
        )

        self.flow_init = flow_init
        self.log_pdf = log_pdf
        self.flow_sample = flow_sample

    def init(self, rng, inputs=None, real_data=False):
        params, _, _ = self.flow_init(
            rng, 
            input_dim=self.input_dim, 
            input_inits=inputs if real_data else None,
        )

        return params

    def log_Q(self, params, inputs, rng):
        return self.log_pdf(params, inputs)

    def sample(self, params, num_samples, key, sampler = "flow"):
        if sampler == "flow":
            return self.flow_sample(key, params, num_samples)
        return self.mcmc_sample(params, num_samples, key)