import math
import jax
from jax import jit,vmap
from jax.random import split,normal,uniform,PRNGKey
import jax.numpy as jnp
from jax.lax import fori_loop,scan,cond
import haiku as hk
from jax.flatten_util import ravel_pytree

from functools import partial

def mcmc_sample(
            true_pdf, 
            input_dim, 
            params, 
            num_samples, 
            rng_key, 
            sigma = 1.0, 
            n_chains = 16, 
            n_sweeps = 10, 
            n_discard_per_chain = 100,
            **kwargs):

    def mcmc_substep(i, data):
        data["rng"],key,subkey = split(data["rng"], 3)
        new_samples = sigma * normal(key, (n_chains, input_dim)) + data["samples"]

        new_log_prob = true_pdf(params, new_samples, None)
        log_prob_ratio = new_log_prob - data["prev_log_prob"]
        prob_ratio = jnp.exp(log_prob_ratio)

        uniform_sample = uniform(subkey, (n_chains,))

        accept = (uniform_sample < prob_ratio)

        data["accepted"] += accept.sum()

        data["prev_log_prob"] = jnp.where(
            accept,
            new_log_prob,
            data["prev_log_prob"],
            )

        data["samples"] = jnp.where(
            accept.reshape(-1, 1),
            new_samples,
            data["samples"],
        )

        return data

    def mcmc_step(data, samples):
        data = fori_loop(0, n_sweeps, mcmc_substep, data)
        return data, data["samples"]

    def mcmc(rng, chain_length):
        key, subkey = split(rng)

        first_samples = sigma * normal(subkey, (n_chains, input_dim))

        data = {
            "rng" : key,
            "prev_log_prob" : true_pdf(params, first_samples, None),
            "samples" : first_samples,
            "accepted" : 0,
        }

        data,samples = scan(
            f = mcmc_step,
            init = data,
            xs = None,
            length = chain_length,
        )

        return samples

    def sample():
        keep_chain_length = math.ceil(num_samples/n_chains)

        full_chain_length = keep_chain_length+n_discard_per_chain

        samples = mcmc(rng_key, full_chain_length)[n_discard_per_chain:,:,:]

        out = jnp.reshape(samples, (-1, samples.shape[-1]))

        return out[:num_samples,:]
    
    return sample()


def get_mcmc_sampled_flow(ground_truth, sigma = 1.0, n_chains = 16, n_sweeps = 10, n_discard_per_chain = 100):
    
    def init_fun(rng, input_dim):
        _,true_pdf,_ = ground_truth(rng, input_dim)

        def mcmc_substep(i, data):
            data["rng"],key,subkey = split(data["rng"], 3)
            new_samples = sigma * normal(key, (n_chains, input_dim)) + data["samples"]

            new_log_prob = true_pdf(None,new_samples)
            log_prob_ratio = new_log_prob - data["prev_log_prob"]
            prob_ratio = jnp.exp(log_prob_ratio)

            uniform_sample = uniform(subkey, (n_chains,))

            accept = (uniform_sample < prob_ratio)

            data["accepted"] += accept.sum()

            data["prev_log_prob"] = jnp.where(
                accept,
                new_log_prob,
                data["prev_log_prob"],
            )

            data["samples"] = jnp.where(
                accept.reshape(-1, 1),
                new_samples,
                data["samples"],
            )

            return data

        def mcmc_step(data, samples):
            data = fori_loop(0, n_sweeps, mcmc_substep, data)
            return data, data["samples"]

        def mcmc(rng, chain_length):
            key, subkey = split(rng)

            first_samples = sigma * normal(subkey, (n_chains, input_dim))

            data = {
                "rng" : key,
                "prev_log_prob" : true_pdf(None, first_samples),
                "samples" : first_samples,
                "accepted" : 0,
            }

            data,samples = scan(
                f = mcmc_step,
                init = data,
                xs = None,
                length = chain_length,
            )

            #id_print(data["accepted"])

            return samples

        def sample(rng, params, num_samples = 1):
            keep_chain_length = math.ceil(num_samples/n_chains)

            full_chain_length = keep_chain_length+n_discard_per_chain

            samples = mcmc(rng, full_chain_length)[n_discard_per_chain:,:,:]

            out = jnp.reshape(samples, (-1, samples.shape[-1]))

            return out[:num_samples,:]

        return (), true_pdf, sample
    
    return init_fun
