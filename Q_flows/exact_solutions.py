from jax.random import normal
import jax.numpy as jnp
import math
from jax.experimental.host_callback import id_print

"""Exact solutions to test that quantum systems are implemented correctly"""
def damped_harmonic_solution(key,input_dim):
    def log_pdf(n_bar,inputs):
        return jnp.log(1/(math.pi*(n_bar[0]+1)))-jnp.sum(inputs**2,axis=-1)/(n_bar[0]+1)
    def sample(key,n_bar,num_samples):
        return normal(key,shape=(num_samples,2))*((n_bar[0]+1)**0.5)
    return [jnp.array([0.1])[0]], log_pdf, sample


def damped_harmonic_solution_multiparticle2(key,input_dim):
    num_particles = input_dim//2
    def log_pdf(n_bar,inputs):
        return num_particles*jnp.log(1/(math.pi*(n_bar[0]+1)))-jnp.sum(inputs**2,axis=-1)/(n_bar[0]+1)
    def sample(key,n_bar,num_samples):
        return normal(key,shape=(num_samples,input_dim))*((n_bar[0]+1)**0.5)
    return [jnp.array([0.1])[0]], log_pdf, sample

def damped_harmonic_solution_multiparticle(key,input_dim):
    if input_dim%2 != 0:
        raise Exception("Need an even imput_dim!!")
    num_particles = input_dim//2

    def log_pdf(n_bar,inputs):
        logp = 0
        for i in range(num_particles):
            logp += jnp.log(1/(math.pi*(n_bar[0][i]+1)))-jnp.sum(inputs[:,2*i:2*(i+1)]**2,axis=-1)/(n_bar[0][i]+1)
        return logp

    def sample(key,n_bar,num_samples):
        return normal(key,shape=(num_samples,input_dim))*((jnp.repeat(n_bar[0],2)+1)**0.5)

    return [jnp.array([3.1]*num_particles)], log_pdf, sample

def gaussian(key,input_dim):
    params = [jnp.array([0.01])[0]]
    def log_pdf(params,inputs):
        return -jnp.log((math.pi*(params[0]+1)))-jnp.sum(inputs**2,axis=-1)/(params[0]+1)
    def sample(key,params,num_samples):
        return normal(key,shape=(num_samples,2))*(((params[0]+1)/2)**0.5)
    return params, log_pdf, sample

def fokker_plank_exact(initial, n, gamma, omega):
    def init(key, input_dim):
        def log_pdf(t,inputs):
            alpha = initial[::2] + 1j*initial[1::2]
            alpha *= jnp.exp(-gamma*t/2)*jnp.exp(-1j*omega*t)
            alpha = jnp.expand_dims(alpha, axis=1)

            new_center = jnp.concatenate((jnp.real(alpha),jnp.imag(alpha)),axis=1)
            new_center = jnp.reshape(new_center, (-1,))

            std = jnp.sqrt((1+n*(1-jnp.exp(-gamma*t)))/2)
            std = jnp.expand_dims(std, -1)
            std = jnp.concatenate((std,std), -1)
            std = jnp.reshape(std, (-1,))
            
            return -jnp.sum(jnp.log(math.pi*2*(std**2)))/2-jnp.sum((inputs-new_center)**2/(2*(std**2)),axis=-1)

        def sample(key,t,num_samples):
            alpha = initial[::2] + 1j*initial[1::2]
            alpha *= jnp.exp(-gamma*t/2)*jnp.exp(-1j*omega*t)
            alpha = jnp.expand_dims(alpha, axis=1)

            new_center = jnp.concatenate((jnp.real(alpha),jnp.imag(alpha)),axis=1)
            new_center = jnp.reshape(new_center, (-1,))

            std = jnp.sqrt((1+n*(1-jnp.exp(-gamma*t)))/2)
            std = jnp.expand_dims(std, -1)
            std = jnp.concatenate((std,std), -1)
            std = jnp.reshape(std, (-1,))
            
            num_wells = new_center.shape[0]//2
            return std*normal(key,shape=(num_samples,2*num_wells))+new_center
        return 0, log_pdf, sample

    return init