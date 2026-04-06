import jax.numpy as jnp
from jax.random import split
from jax import jit
from jax.experimental.host_callback import id_print

from functools import partial

#@partial(jit,static_argnums=(1,))
def expect_n_i(samples,i):
    #print(samples[:,i]**2+samples[:,samples.shape[1]//2+i]**2-1)
    return jnp.mean(samples[:,i]**2+samples[:,samples.shape[1]//2+i]**2)-1

@jit
def expect_n_tot(samples):
    return jnp.mean(jnp.sum(samples**2,axis=1))-(samples.shape[1]//2)

@partial(jit,static_argnums=(1,2))
def expect_n_ij(samples,i,j):
    value = (samples[:,i]**2+samples[:,samples.shape[1]//2+i]**2-1)*(samples[:,j]**2+samples[:,samples.shape[1]//2+j]**2-1)
    if i==j:
        return jnp.mean(value - samples[:,i]**2 - samples[:,samples.shape[1]//2+i]**2)
    return jnp.mean(value)

@partial(jit,static_argnums=(1,2))
def expect_a_a_bar(samples,i,j):
    out = jnp.mean((samples[:,i]-1j*samples[:,samples.shape[1]//2+i])*(samples[:,j]+1j*samples[:,samples.shape[1]//2+j]))
    if i==j:
        return jnp.abs(out-1)
    return jnp.abs(out)

@partial(jit,static_argnums=(1,2))
def get_g_1(samples,i,j):
    return expect_a_a_bar(samples,i,j)/jnp.sqrt(expect_n_i(samples,i)*expect_n_i(samples,j))

@partial(jit,static_argnums=(1,2))
def get_g_2(samples,i,j):
    return expect_n_ij(samples,i,j)/(expect_n_i(samples,i)*expect_n_i(samples,j))

def evaluate_with_std(sampler, distribution, params, num_samples, functions, num_repeats, flow_sampler, rng):
    rngs = split(rng, num_repeats)
    outs = [[0 for _ in range(len(functions))] for _ in range(num_repeats)]
    for i in range(num_repeats):
        if flow_sampler:
            samples = distribution.sample(params,num_samples,rngs[i])
        else:
            samples = sampler.sample(distribution,params,chain_length=num_samples//sampler.n_chains)[0]
            samples = jnp.reshape(samples,(-1,samples.shape[-1]))
            #print(samples)

        for j in range(len(functions)):
            outs[i][j] = functions[j](samples)
    outs = jnp.array(outs)

    return jnp.mean(outs,axis=0),jnp.std(outs,ddof=1,axis=0)/jnp.sqrt(num_repeats)
