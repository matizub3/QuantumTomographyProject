import math
from typing import Any
from jax import numpy as jnp
from jax import jit
import jax
import cmath

from jax import random
import numpy as np
from Q_flows.mcmc import mcmc_sample
from distributions.Q_targets import get_coherent_state

from distributions.W_function import JaxWigner, Wigner

def sample_signed(key, wigner, params, num_samples, sign):
    def log_signed_w(params, inputs):
        out = wigner.w(params, inputs)

        signed = jnp.fmax(out*sign,1e-20)

        return jnp.log(signed)

    return mcmc_sample(
        log_signed_w,
        wigner.input_dim,
        params,
        num_samples,
        key,
    )


"""
The two functions are jax reimplementations of functions from https://qutip.org/docs/4.4/modules/qutip/wigner.html
"""
@jit
def _wigner_clenshaw(rho, x, y, g=math.sqrt(2)):
    """
    Using Clenshaw summation - numerically stable and efficient
    iterative algorithm to evaluate polynomial series.
    
    The Wigner function is calculated as
    :math:`W = e^(-0.5*x^2)/pi * \sum_{L} c_L (2x)^L / sqrt(L!)` where 
    :math:`c_L = \sum_n \\rho_{n,L+n} LL_n^L` where
    :math:`LL_n^L = (-1)^n sqrt(L!n!/(L+n)!) LaguerreL[n,L,x]`
    
    """

    M = rho.shape[0]
    #A = 0.5 * g * (X + 1.0j * Y)
    A2 = g * (x+ 1.0j * y) #this is A2 = 2*A

    B = jnp.abs(A2)
    B *= B
    w0 = (2*rho[0,-1])*jnp.ones_like(A2)
    L = M-1

    #calculation of \sum_{L} c_L (2x)^L / sqrt(L!)
    #using Horner's method
    rho = rho * (2*jnp.ones((M,M)) - jnp.diag(jnp.ones(M)))
    while L > 0:
        L -= 1
        #here c_L = _wig_laguerre_val(L, B, np.diag(rho, L))
        w0 = _wig_laguerre_val(L, B, jnp.diag(rho, L)) + w0 * A2 * (L+1)**-0.5

    return w0.real * jnp.exp(-B*0.5) * (g*g*0.5 / math.pi)


def _wig_laguerre_val(L, x, c):
    """
    this is evaluation of polynomial series inspired by hermval from numpy.    
    Returns polynomial series
    \sum_n b_n LL_n^L,
    where
    LL_n^L = (-1)^n sqrt(L!n!/(L+n)!) LaguerreL[n,L,x]    
    The evaluation uses Clenshaw recursion
    """

    if len(c) == 1:
        y0 = c[0]
        y1 = 0
    elif len(c) == 2:
        y0 = c[0]
        y1 = c[1]
    else:
        k = len(c)
        y0 = c[-2]
        y1 = c[-1]
        for i in range(3, len(c) + 1):
            k -= 1
            y0,    y1 = c[-i] - y1 * (float((k - 1)*(L + k - 1))/((L+k)*k))**0.5, \
            y0 - y1 * ((L + 2*k -1) - x) * ((L+k)*k)**-0.5

    return y0 - y1 * ((L + 1) - x) * (L + 1)**-0.5


class W_density(JaxWigner):
    rho: jnp.ndarray
    kwargs: Any

    def __init__(self, rho, **kwargs):
        self.input_dim = 2
        self.rho = rho
        self.kwargs = kwargs

    def init(self, rng, inputs, real_data=False):
        return None
    
    def w(self, params, inputs, rng):
        return _wigner_clenshaw(self.rho, inputs[:,0], inputs[:,1])
    
    def sample(self, params, num_samples, key, sampler = "mcmc"):
        if sampler == "mcmc":
            return self.mcmc_sample(params, num_samples, key)
        
        elif sampler == "not_mcmc":
            samples = self.rho.shape[0]*random.normal(key, (num_samples,self.input_dim), float)
            return samples
        
    def sample_probs(self, params, samples, rng, sampler = "mcmc"):
        """
        Only works if samples are taken using the sample function *for this class*
        """

        if sampler == "mcmc":
            return self.abs_w_tilde(params, samples, rng)
        
        if sampler == "not_mcmc":
            samples /= self.rho.shape[0]
            return (1/(jnp.sqrt(2*jnp.pi)))*(jnp.exp(-jnp.sum(samples**2, axis=1)/2))


class W_n_particle(W_density):
    def __init__(self, num_particles):
        rho = jnp.zeros((num_particles+1, num_particles+1), dtype=float)
        rho = rho.at[num_particles, num_particles].set(1)
        super().__init__(rho)

def binom(x, y):
  return jnp.exp(jax.scipy.special.gammaln(x + 1) - jax.scipy.special.gammaln(y + 1) - jax.scipy.special.gammaln(x - y + 1))

class W_Binomial(W_density):
    def __init__(self, N, S, mu = 0):
        if not mu in [0,1]:
            raise ValueError("mu must be 0 or 1")

        psi = jnp.zeros(((S+1)*(N+1),), dtype=complex)

        for m in range(N+2):
            value = (-1)**(mu*m)
            value *= jnp.sqrt(binom(N+1, m))

            psi = psi.at[(S+1)*m].set(value)

        psi /= 2 ** ((N+1)/2)

        rho = jnp.outer(psi, jnp.conj(psi))
        super().__init__(rho)


num_states = []
num_states.append([
    [
        (np.sqrt(7 - np.sqrt(17))) / np.sqrt(6),
        0,
        0,
        (np.sqrt(np.sqrt(17) - 1) / np.sqrt(6)),
        0,
    ],
    [
        0,
        (np.sqrt(9 - np.sqrt(17)) / np.sqrt(6)),
        0,
        0,
        (np.sqrt(np.sqrt(17) - 3) / np.sqrt(6)),
    ],
])

num_states.append([
    [
        0.5458351325482939,
        -3.7726009161224436e-9,
        4.849511177634774e-8,
        -0.7114411727633639,
        -7.48481181758003e-8,
        -1.3146003192319789e-8,
        0.44172510726665587,
        1.1545802803733896e-8,
        1.0609402576342428e-8,
        -0.028182506843720707,
        -6.0233214626778965e-9,
        -6.392041552216322e-9,
        0.00037641909140801935,
        -6.9186916801058116e-9,
    ],
    [
        2.48926815257019e-9,
        -0.7446851186077535,
        -8.040831059521339e-9,
        6.01942995399906e-8,
        -0.5706020908811399,
        -3.151900508005823e-8,
        -7.384935824733578e-10,
        -0.3460030551087218,
        -8.485651303145757e-9,
        -1.2114327561832047e-8,
        0.011798401879159238,
        -4.660460771433317e-9,
        -5.090374160706911e-9,
        -0.00010758601713550998,
    ],
])

class W_Num(W_density):
    def __init__(self, num = 0, mu = 0):
        psi = jnp.array(num_states[num][mu], dtype=complex)

        rho = jnp.outer(psi, jnp.conj(psi))
        super().__init__(rho)

class W_GKP(W_density):
    def __init__(self, delta, mu, cutoff, resolution):
        phi = 0
        for i in range(-cutoff,cutoff+1):
            for k in range(-cutoff,cutoff+1):
                alpha_re = math.sqrt(math.pi/2) * (2*i+mu)
                alpha_im = math.sqrt(math.pi/2) * k

                alpha = alpha_re + 1j * alpha_im

                normalization = math.exp(-delta**2 * (alpha_re**2 + alpha_im**2))
                normalization *= cmath.exp(- 1j * alpha_re * alpha_im)

                phi += normalization * get_coherent_state(alpha, resolution)

        phi /= jnp.sqrt(jnp.vdot(phi,phi))

        rho = jnp.outer(phi, jnp.conj(phi))
    
        super().__init__(rho)

class W_GKP(W_density):
    def __init__(self, delta, mu, cutoff, resolution):
        phi = 0
        for i in range(-cutoff,cutoff+1):
            for k in range(-cutoff,cutoff+1):
                alpha_re = math.sqrt(math.pi/2) * (2*i+mu)
                alpha_im = math.sqrt(math.pi/2) * k

                alpha = alpha_re + 1j * alpha_im

                normalization = math.exp(-delta**2 * (alpha_re**2 + alpha_im**2))
                normalization *= cmath.exp(- 1j * alpha_re * alpha_im)

                phi += normalization * get_coherent_state(alpha, resolution)

        phi /= jnp.sqrt(jnp.vdot(phi,phi))

        rho = jnp.outer(phi, jnp.conj(phi))
    
        super().__init__(rho)

class W_CatState(JaxWigner):

    sign: float
    beta: float
    kwargs: Any

    def __init__(self, sign, beta, input_dim, **kwargs):
        self.input_dim = input_dim
        self.sign = sign
        self.beta = beta
        self.kwargs = kwargs

    def init(self, rng, inputs, real_data=False):
        return None

    def w(self, params, inputs, rng):
        N = jnp.pi*(1+self.sign*jnp.exp(-2*self.beta**2))

        term1 = jnp.exp(
            -2 * ( (inputs[:,0] - self.beta)**2 + inputs[:,1]**2 )
        )

        term2 = jnp.exp(
            -2 * ( (inputs[:,0] + self.beta)**2 + inputs[:,1]**2)
        )

        term3 = 2*jnp.cos(4*self.beta*inputs[:,1])*jnp.exp(-2*jnp.sum(inputs**2, axis=1))

        return (term1+term2+self.sign*term3)/N

    def sample(self, params, num_samples, key, sampler = "mcmc"):
        if sampler == "mcmc":
            return self.mcmc_sample(params, num_samples, key)
        
        elif sampler == "not_mcmc":
            samples = random.normal(key, (num_samples,self.input_dim), float)
            samples2 = jnp.concatenate(
                (samples[:,:1]*self.beta, samples[:,1:]),
                axis = 1,
            )

            return samples2
        
    def sample_probs(self, params, samples, rng, sampler = "mcmc"):
        """
        Only works if samples are taken using the sample function *for this class*
        """

        if sampler == "mcmc":
            return self.abs_w_tilde(params, samples, rng)
        
        if sampler == "not_mcmc":
            samples = jnp.concatenate(
                (samples[:,:1]/self.beta, samples[:,1:]),
                axis = 1,
            )

            return (1/(jnp.sqrt(2*jnp.pi)))*(jnp.exp(-jnp.sum(samples**2, axis=1)/2))
        

class W_Tensor_Product(JaxWigner):
    Ws: list[JaxWigner]

    def __init__(self, Ws: list[JaxWigner]):
        self.Ws = Ws

        self.input_dim = sum([W.input_dim for W in Ws])

    def init(self, rng, inputs=None, real_data=False):
        params = []

        rngs = jax.random.split(rng, len(self.Ws))
        for i in range(len(self.Ws)):
            params.append(self.Ws[i].init(rngs[i], inputs, real_data))

        return params

    def w(self, params, inputs, rng):
        w = 1
        input_start = 0
        if rng is not None:
            rngs = jax.random.split(rng, len(self.Ws))
        else:
            rngs = [None for i in range(len(self.Ws))]
            
        for W,param,rng in zip(self.Ws, params, rngs):
            w *= W.w(param, inputs[:,input_start:input_start+W.input_dim], rng)
            input_start += W.input_dim
        
        return w

    def sample(self, params, num_samples, key, sampler = "mcmc"):
        assert sampler == "mcmc"

        samples = []
        for i, W in enumerate(self.Ws):
            samples.append(W.mcmc_sample(params[i], num_samples, key))

        #print(samples)

        return jnp.concatenate(samples, axis=1)
    
    def sample_probs(self, params, samples, rng, sampler = "mcmc"):
        assert sampler == "mcmc"

        start_index = 0
        probs = jnp.ones(samples.shape[0])
        for i, W in enumerate(self.Ws):
            end_index = start_index + W.input_dim
            probs *= W.abs_w(params[i], samples[:,start_index:end_index], rng)
            start_index = end_index

        return probs
    

class W_Mixture(JaxWigner):
    Ws: list[JaxWigner]
    weights: jnp.ndarray

    def __init__(self, Ws: list[JaxWigner], weights: list[float] | jnp.ndarray | None = None):
        self.Ws = Ws

        if weights is None:
            weights = jnp.array([1/len(Ws) for W in Ws])
        else:
            if weights is not jnp.ndarray:
                weights = jnp.array(weights)

            weights = weights / jnp.sum(weights)
            self.weights = weights

        self.input_dim = Ws[0].input_dim
        for W in Ws:
            if W.input_dim != self.input_dim:
                raise ValueError("All Ws must have the same input_dim")
            
    def init(self, rng, inputs=None, real_data=False):
        params = []

        rngs = jax.random.split(rng, len(self.Ws))
        for i in range(len(self.Ws)):
            params.append(self.Ws[i].init(rngs[i], inputs, real_data))

        return params
    
    def w(self, params, inputs, rng):
        w = 0
        
        rngs = jax.random.split(rng, len(self.Ws))
        for i, (W,param, rng) in enumerate(zip(self.Ws, params, rngs)):
            w += self.weights[i] * W.w(param, inputs, rng)
        
        return w
    

_default_rotation = jnp.array(
    [
        [1,0,-1,0],
        [0,1,0,-1],
        [1,0,1,0],
        [0,1,0,1],
    ]
)

class W_Rotation(JaxWigner):
    unrotated_W: JaxWigner
    rotation: jnp.ndarray

    def __init__(self, unrotated_W: JaxWigner, rotation: jnp.ndarray = _default_rotation):
        self.unrotated_W = unrotated_W
        self.rotation = rotation

        self.input_dim = unrotated_W.input_dim

        if rotation.shape != (self.input_dim, self.input_dim):
            raise ValueError("Rotation matrix must have shape (input_dim, input_dim)")
        
        if not jnp.allclose(rotation@rotation.T, jnp.eye(self.input_dim)):
            raise ValueError("Rotation matrix must be orthogonal")
        
    def init(self, rng, inputs=None, real_data=False):
        return self.unrotated_W.init(rng, inputs, real_data)
    
    def w(self, params, inputs, rng):
        rotated_inputs = inputs @ self.rotation

        return self.unrotated_W.w(params, rotated_inputs, rng)
    
    def sample(self, params, num_samples, key, sampler = "mcmc"):
        print(sampler)
        return self.unrotated_W.sample(params, num_samples, key, sampler = sampler) @ self.rotation.T
    
    def sample_probs(self, params, samples, rng, sampler = "mcmc"):
        return self.unrotated_W.sample_probs(params, samples @ self.rotation, rng, sampler = sampler)
    
class NoisyW(JaxWigner):
    W: JaxWigner
    noise_std: float

    def __init__(self, W: JaxWigner, noise_std: float):
        self.W = W
        self.noise_std = noise_std

        self.input_dim = W.input_dim

    def init(self, rng, inputs=None, real_data=False):
        return self.W.init(rng, inputs, real_data)
    
    def w(self, params, inputs, rng):
        return self.W.w(params, inputs) + random.normal(rng, inputs.shape, float) * self.noise_std
    
    def sample_probs(self, params, samples, rng, sampler = "mcmc"):
        return self.W.sample_probs(params, samples, rng, sampler = sampler)
    
    def sample(self, params, num_samples, key, sampler = "mcmc"):
        return self.W.sample(params, num_samples, key, sampler = sampler)
    
