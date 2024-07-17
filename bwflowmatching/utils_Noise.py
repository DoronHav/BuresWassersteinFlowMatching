from jax import random   # type: ignore
import jax.numpy as jnp  # type: ignore 

def gaussian(size, dimention, key = random.key(0)):

    key_means, key_covs = random.split(key)
    
    # Generate k random means in d dimensions
    means = random.normal(key_means, shape=(size, dimention))
    
    L = random.normal(key_covs, shape=(size, dimention, dimention))
    covariances = jnp.matmul(L, jnp.transpose(L, axes=(0, 2, 1)))/dimention
    
    return means, covariances

