import math
from typing import Callable
import equinox as eqx
import jax.numpy as jnp
from jax.random import split, permutation, choice
from abc import ABC, abstractmethod
from jax.scipy.special import logsumexp

class Sampler(eqx.Module):
    has_validation: bool
    num_batches: int
    num_samples: int
    batch_size: int

    @abstractmethod
    def sample(self, rng, sampler_params, sampler_state, validation = False):
        ...


class SampleShuffler(Sampler):
    samples: jnp.ndarray
    validation_samples: jnp.ndarray | None

    def __init__(self, samples, batch_size, validation_samples = None):
        self.samples = samples
        self.batch_size = batch_size
        self.num_batches = samples.shape[0] // batch_size
        self.validation_samples = validation_samples
        self.num_samples = samples.shape[0]

        self.has_validation = (validation_samples is not None)

        if self.samples.shape != self.validation_samples.shape:
            raise Exception("Validation samples must have the same shape as training samples")

    def sample(self, rng, sampler_params, sampler_state, validation = False):
        if validation and self.has_validation:
            return self.validation_samples, ()
        else:
            if sampler_state == ():
                permuted_samples = permutation(rng, self.samples)

                return permuted_samples[:self.batch_size], (1, permuted_samples)
            else:
                samples = sampler_state[1][sampler_state[0]*self.batch_size:(sampler_state[0]+1)*self.batch_size]
                
                return samples, (sampler_state[0]+1, sampler_state[1])
                

class SampleShufflerFromDistribution(SampleShuffler):
    def __init__(self, rng, sample_function, sampler_params, num_samples, batch_size, has_validation = False):
        key1,key2 = split(rng)
        samples = sample_function(sampler_params, num_samples, key1)
        validation_samples = sample_function(sampler_params, num_samples, key2) if has_validation else None
        super().__init__(samples, batch_size, validation_samples)

class SampleFromDistribution(Sampler):
    sample_function: Callable

    def __init__(self, sample_function, num_samples, batch_size, has_validation = False):
        num_batches = num_samples // batch_size
        
        self.sample_function = sample_function
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.has_validation = has_validation
        self.num_samples = batch_size * num_batches

    def sample(self, rng, sampler_params, sampler_state, validation = False):
        if validation:
            return self.sample_function(sampler_params, self.num_samples, rng),()
        return self.sample_function(sampler_params, self.batch_size, rng),()
    


class SampleEfficientSampler:
    sample_function: Callable
    probability_function: Callable
    target_function: Callable

    samples_per_resample: int
    resample_every: int

    batch_size: int
    has_validation: bool = False
    num_batches: int = 1

    logprobs: bool

    samples: jnp.ndarray | None
    sample_probs: jnp.ndarray | None
    target_values: jnp.ndarray | None

    sample_parameters: list | None

    def __init__(self, 
                 sample_function: Callable,
                 probability_function: Callable,
                 target_function: Callable,
                 batch_size: int,
                 samples_per_resample: int,
                 resample_every: int,
                 logprobs: bool = True):
        
        self.batch_size = batch_size
        
        self.sample_function = sample_function
        self.probability_function = probability_function
        self.target_function = target_function

        self.samples_per_resample = samples_per_resample
        self.resample_every = resample_every

        print(f"Batch size: {self.batch_size}")
        print(f"Samples per resample: {self.samples_per_resample}")
        print(f"Resample every: {self.resample_every}")

        self.samples = None
        self.sample_probs = None
        self.target_values = None
        self.sample_parameters = None
        self.logprobs = logprobs

    def new_samples(self, rng, model_params, target_params):
        key1, key2, key3 = split(rng, 3)

        new_samples = self.sample_function(model_params, self.samples_per_resample, key1)
        new_target_values = self.target_function(target_params, new_samples, key2)

        print("Sampling again")
        print(f"Shape of new samples: {new_samples.shape}")

        if self.samples is None:
            self.samples = new_samples
            self.sample_probs = self.probability_function(model_params, new_samples, key3)
            self.target_values = new_target_values
            #self.sample_parameters = [model_params]
        else:
            self.samples = jnp.concatenate([self.samples, new_samples])
            #self.sample_parameters.append(model_params)
            self.sample_probs = jnp.concatenate([self.sample_probs, self.probability_function(model_params, new_samples, rng)])
            self.target_values = jnp.concatenate([self.target_values, new_target_values])
            #self.calculate_probs()

    def calculate_probs(self):
        sample_probs = []

        for i in range(self.samples.shape[0]//self.batch_size):
            sample_probs.append(
                self.average_probs(
                    self.samples[i*self.batch_size:(i+1)*self.batch_size]
                )
            )

        loc = (self.samples.shape[0]//self.batch_size)*self.batch_size
        
        if loc != self.samples.shape[0]:
            sample_probs.append(
                self.average_probs(
                    self.samples[loc:]
                )
            )

        self.sample_probs = jnp.concatenate(sample_probs)

    def average_probs(self, samples):
        batch_sample_probs = jnp.zeros(samples.shape[0], dtype=float)

        print(f"Shape of samples: {samples.shape}")

        for model_params in self.sample_parameters:
            print(f"Shape of batch_sample_probs: {batch_sample_probs.shape}")

            if self.logprobs:
                batch_sample_probs = logsumexp(
                    jnp.stack(
                        [
                            batch_sample_probs, 
                            self.probability_function(
                                   model_params, 
                                   samples,
                            ),
                        ], 
                        axis=0,
                    ),
                    axis = 0,
                )
            else:
                batch_sample_probs += self.probability_function(
                    model_params, 
                    samples,
                )
        
        if self.logprobs:
            return batch_sample_probs - math.log(len(self.sample_parameters))
        
        return batch_sample_probs / len(self.sample_parameters)

    def sample_shuffle(self, rng):
        sample_indices = choice(rng, self.samples.shape[0], (self.batch_size,))

        if self.logprobs:
            return self.samples[sample_indices], self.sample_probs[sample_indices], self.target_values[sample_indices]
        
        return self.samples[sample_indices], self.sample_probs[sample_indices], self.target_values[sample_indices]

    def sample(self, rng, model_params, target_params, sampler_state, validation = False):
        if validation:
            raise Exception("Validation not supported for SampleEfficientSampler")
        
        if self.samples is None:
            print("Initializing samples")
        else:
            print(f"Num_samples: {self.samples.shape[0]}")

        if sampler_state == ():
            sampler_state = 0

        if sampler_state % self.resample_every == 0:
            rng, rng2 = split(rng)
            self.new_samples(rng2, model_params, target_params)

        samples, sample_probs, sample_target_values = self.sample_shuffle(rng)

        return samples, sample_probs, sample_target_values, sampler_state+1