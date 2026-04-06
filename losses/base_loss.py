import abc

import equinox as eqx

from distributions.distribution import Distribution
from training_sampler import Sampler

class Loss(eqx.Module, abc.ABC):
    sampler: Sampler
    model: Distribution
    target: Distribution
    
    grad_loss: bool
    has_validation: bool

    name: str
    color: str

    def __init__(self, has_validation = False, name = "loss", color = "tab:blue"):
        self.has_validation = has_validation
        self.name = name
        self.color = color

    @abc.abstractmethod
    def __call__(self, rng, model_params, target_params, sampler_state = None, validation = False):
        ...