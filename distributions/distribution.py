import equinox as eqx
import abc

class Distribution(eqx.Module, abc.ABC):
    input_dim: int

    @abc.abstractmethod
    def init(self, rng, inputs=None, real_data=False):
        ...

    @abc.abstractmethod
    def sample(self, params, num_samples, key, **kwargs):
        ...

    @abc.abstractmethod
    def plot(self, params, ranges, true_distribution = None, true_params = None, plot_difference = False, save = False, filename = None, filetype = None):
        ...