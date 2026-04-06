import equinox as eqx
from jax.random import split, permutation
from abc import ABC, abstractmethod

import optax

from losses.base_loss import Loss

class MultipleLoss(eqx.Module):
    losses: list[Loss]
    num_batches: int
    has_validation: bool

    def __init__(self, losses):
        num_batches = losses[0].sampler.num_batches
        has_validation = losses[0].sampler.has_validation

        for i, loss in enumerate(losses):
            if loss.sampler.num_batches != num_batches:
                raise Exception("All losses must have the same number of batches")
            
            if i != 0 and loss.grad_loss:
                raise Exception("Only the first loss can have a gradient")
            
            if i == 0 and not loss.grad_loss:
                raise Exception("The first loss must have a gradient")
            
            if loss.sampler.has_validation != has_validation:
                raise Exception("All losses must have the same validation status")

        self.losses = losses
        self.num_batches = num_batches
        self.has_validation = has_validation

    def train_losses(self, rng, model_params, target_params, optimizer, opt_state, sampler_states = None):
        if sampler_states is None:
            sampler_states = [() for _ in self.losses]

        total_losses = [0 for _ in self.losses]

        for _ in range(self.num_batches):
            rng, key = split(rng)
            keys = split(key, num = len(self.losses))

            for i, loss in enumerate(self.losses):
                if loss.grad_loss:
                    (loss_value,loss_grad), sampler_state = loss(keys[i], model_params, target_params, sampler_states[i], validation = False)
                    
                    total_losses[i] += loss_value
                    sampler_states[i] = sampler_state

                else:
                    loss_value, sampler_state = loss(keys[i], model_params, target_params, sampler_states[i], validation = False)

                    total_losses[i] += loss_value
                    sampler_states[i] = sampler_state

            updates, opt_state = optimizer.update(loss_grad, opt_state, model_params)
            model_params = eqx.apply_updates(model_params, updates)

        return model_params, opt_state, [loss / self.num_batches for loss in total_losses], rng, sampler_states

    def get_validation_losses(self, rng, model_params, target_params):
        losses = [0 for _ in self.losses]

        rng,key = split(rng)
        keys = split(key, num = len(self.losses))

        for i, loss in enumerate(self.losses):
            losses[i], _ = loss(keys[i], model_params, target_params, (), validation = True)

        return losses, rng