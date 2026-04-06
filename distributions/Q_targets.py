import math
from jax import numpy as jnp
import jax
import cmath

import numpy as np

from distributions.Q_function import JaxQ

class Q_BEC(JaxQ):
    phi: jnp.ndarray
    normalization: float
    n: int

    def __init__(self,phi,n,num_wells,normalized=True):
        self.input_dim = num_wells*2
        self.phi = phi
        self.n = n

        if normalized:
            normalization = -num_wells*jnp.log(math.pi)
            normalization -= jnp.sum(jnp.log(jnp.arange(1,n+1)))
        else:
            normalization = 0

        self.normalization = normalization

    def init(self, rng, inputs=None, real_data=False):
        return ()

    def log_Q(self,params,inputs, rng):
        log_exp = jnp.sum(-inputs**2,axis=-1)

        q = inputs[...,:self.input_dim//2]
        p = inputs[...,self.input_dim//2:]
        q_term = jnp.sum(self.phi*q,axis=-1)**2
        p_term = jnp.sum(self.phi*p,axis=-1)**2
        polyterm = self.n*jnp.log(q_term+p_term)

        return self.normalization+log_exp+polyterm

class Q_CatState(JaxQ):
    beta: float
    N: float
    normalization: float

    def __init__(self, sign, beta):
        N = 2*(1+sign*jnp.exp(-2*beta**2))
        normalization = -jnp.log(math.pi * N)

        self.beta = beta
        self.N = N
        self.normalization = normalization

        self.input_dim = 2

    def init(self, rng, inputs=None, real_data=False):
        return ()

    def log_Q(self, params, inputs, rng):
        term1 = jnp.exp(
            - (inputs[:,0] - self.beta)**2 - inputs[:,1]**2
        )
        
        term2 = jnp.exp(
            - (inputs[:,0] + self.beta)**2 - inputs[:,1]**2
        )

        term3 = 2*jnp.cos(2*self.beta*inputs[:,1])*jnp.exp(-jnp.sum(inputs**2)-self.beta**2)

        return self.normalization + jnp.log(term1+term2+term3)

class Q_ShiftedGaussian(JaxQ):
    def __init__(self):
        self.input_dim = 2

    def init(self, rng, inputs=None, real_data=False):
        return ()

    def log_Q(self, params,inputs,rng):
        log_exp = jnp.sum(-(inputs-jnp.array([1,0]))**2,axis=-1)/2

        normalization = math.log(1/(2*math.pi))

        return normalization+log_exp
    
class Q_Coherent(JaxQ):
    shift: jnp.ndarray
    
    def __init__(self, shift):
        self.shift = shift
        self.input_dim = self.shift.shape[0]

    def init(self, rng, inputs=None, real_data=False):
        return ()

    def log_Q(self,params,inputs,rng):
        log_exp = jnp.sum(-(inputs-self.shift)**2,axis=-1)

        normalization = math.log(1/(math.pi))

        return normalization+log_exp


"""
Retuns a coherent state in the particle number basis, up to a given resolution
Works for multiple different alpha in parallel
"""
def get_coherent_state(alpha, resolution):
    norm = jnp.exp(
        -jnp.abs(alpha)**2/2
    )

    ascending = jnp.arange(resolution)

    sqrt_log_factorial = 0.5*jax.scipy.special.gammaln(ascending + 1)

    log_numerator = ascending * jnp.log(alpha)

    return norm * jnp.where(
        alpha == 0,
        jnp.array([1]+[0 for i in range(resolution-1)]),
        jnp.exp(log_numerator - sqrt_log_factorial),
    )

class Q_PureState(JaxQ):
    psi: jnp.ndarray
   # n: int

    def __init__(self, psi):#, n):
        self.psi = psi
        self.input_dim = 2
        #self.n = n

    def init(self, rng, inputs=None, real_data=False):
        return ()

    def log_Q(self,params,inputs,rng):
        alphas = get_coherent_state(inputs[:,:1] + 1j*inputs[:,1:], self.psi.shape[0])

        dot = jnp.sum(jnp.conjugate(self.psi)*alphas,axis=-1)

        return jnp.log(jnp.abs(dot)**2) - jnp.log(math.pi)

class Q_GKP(Q_PureState):
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

        super().__init__(phi)


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


class Q_Num(Q_PureState):
    def __init__(self, num = 0, mu = 0):
        if not num in [0,1,2,3,4]:
            raise ValueError("num must be between 0 and 4")
        if not mu in [0,1]:
            raise ValueError("mu must be 0 or 1")
        
        psi = jnp.array(num_states[num][mu], dtype=complex)

        super().__init__(psi)

def binom(x, y):
  return jnp.exp(jax.scipy.special.gammaln(x + 1) - jax.scipy.special.gammaln(y + 1) - jax.scipy.special.gammaln(x - y + 1))

class Q_Binomial(Q_PureState):
    def __init__(self, N, S, mu = 0):
        if not mu in [0,1]:
            raise ValueError("mu must be 0 or 1")

        psi = jnp.zeros(((S+1)*(N+1),), dtype=complex)

        for m in range(N+2):
            value = (-1)**(mu*m)
            value *= jnp.sqrt(binom(N+1, m))

            psi = psi.at[(S+1)*m].set(value)

        psi /= 2 ** ((N+1)/2)

        super().__init__(psi)

class Q_WState(JaxQ):
    def __init__(self, N):
        self.input_dim = 2*N

    def init(self, rng, inputs=None, real_data=False):
        return ()

    def log_Q(self, params, inputs, rng):
        log_sum = jnp.log(
                jnp.sum(
                (
                    jnp.reshape(
                        inputs, 
                        (-1, self.input_dim//2, 2),
                    ).sum(axis=1)
                )**2, 
                axis=1,
            ),
        )

        return - (self.input_dim//2)*jnp.log(math.pi) - jnp.log(self.input_dim//2) - jnp.sum(inputs**2, axis = 1) + log_sum

class Q_Tensor_Product(JaxQ):
    Qs: list[JaxQ]

    def __init__(self, Qs: list[JaxQ]):
        self.Qs = Qs

        self.input_dim = sum([Q.input_dim for Q in Qs])

    def init(self, rng, inputs=None, real_data=False):
        params = []

        rngs = jax.random.split(rng, len(self.Qs))
        for i in range(len(self.Qs)):
            params.append(self.Qs[i].init(rngs[i], inputs, real_data))

        return params

    def log_Q(self, params, inputs, rng):
        log_Q = 0
        input_start = 0
        for Q,param in zip(self.Qs, params):
            log_Q += Q.log_Q(param, inputs[:,input_start:input_start+Q.input_dim], rng)
            input_start += Q.input_dim
        
        return log_Q

class Q_Mixture(JaxQ):
    Qs: list[JaxQ]
    weights: jnp.ndarray

    def __init__(self, Qs: list[JaxQ], weights: list[float] | jnp.ndarray | None = None):
        self.Qs = Qs

        if weights is None:
            weights = jnp.array([1/len(Qs) for Q in Qs])
        else:
            if weights is not jnp.ndarray:
                weights = jnp.array(weights)

            weights = weights / jnp.sum(weights)
            self.weights = weights

        self.input_dim = Qs[0].input_dim
        for Q in Qs:
            if Q.input_dim != self.input_dim:
                raise ValueError("All Qs must have the same input_dim")
            
    def init(self, rng, inputs=None, real_data=False):
        params = []

        rngs = jax.random.split(rng, len(self.Qs))
        for i in range(len(self.Qs)):
            params.append(self.Qs[i].init(rngs[i], inputs, real_data))

        return params
    
    def log_Q(self, params, inputs, rng):
        Q = 0
        for i, (Q,param) in enumerate(zip(self.Qs, params)):
            Q += self.weights[i] * Q.Q(param, inputs, rng)
        
        return jnp.log(Q)
    

_default_rotation = jnp.array(
    [
        [1,0,-1,0],
        [0,1,0,-1],
        [1,0,1,0],
        [0,1,0,1],
    ]
)

class Q_Rotation(JaxQ):
    unrotated_Q: JaxQ
    rotation: jnp.ndarray

    def __init__(self, unrotated_Q: JaxQ, rotation: jnp.ndarray = _default_rotation):
        self.unrotated_Q = unrotated_Q
        self.rotation = rotation

        self.input_dim = unrotated_Q.input_dim

        if rotation.shape != (self.input_dim, self.input_dim):
            raise ValueError("Rotation must be a square matrix with the same shape as the input_dim of unrotated_Q")
        
        if not jnp.allclose(rotation@rotation.T, jnp.eye(self.input_dim)):
            raise ValueError("Rotation matrix must be orthogonal")

    def init(self, rng, inputs=None, real_data=False):
        return self.unrotated_Q.init(rng, inputs, real_data)
    
    def log_Q(self, params, inputs, rng):
        rotated_inputs = inputs @ self.rotation

        return self.unrotated_Q.log_Q(params, rotated_inputs, rng)
    

class NoisyQ(JaxQ):
    Q: JaxQ
    noise_std: float

    def __init__(self, Q: JaxQ, noise_std: float):
        self.Q = Q
        self.noise_std = noise_std

        self.input_dim = Q.input_dim

    def init(self, rng, inputs=None, real_data=False):
        return self.Q.init(rng, inputs, real_data)
    
    def Q(self, params, inputs, rng):
        return self.Q.Q(params, inputs) + jax.random.normal(rng, inputs.shape) * self.noise_std
    
    def log_Q(self, params, inputs, rng):
        return jnp.log(self.Q(params, inputs, rng))
    
    def sample(self, params, num_samples, key, **kwargs):
        return self.Q.sample(params, num_samples, key, **kwargs)