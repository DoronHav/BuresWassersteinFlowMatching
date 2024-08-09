from jax import random   # type: ignore
import jax.numpy as jnp  # type: ignore 

def gaussian(size, dimention, mean_scale = 1, cov_scale = 1/32, degrees_of_freedom_scale = 2.0, key = random.key(0)):

    key_means, key_covs = random.split(key)
    
    # Generate k random means in d dimensions
    means = mean_scale * random.normal(key_means, shape=(size, dimention))
    
    L = random.normal(key_covs, shape=(size, dimention, int(dimention  * degrees_of_freedom_scale)))
    covariances = cov_scale * jnp.matmul(L, jnp.transpose(L, axes=(0, 2, 1)))/int(dimention  * degrees_of_freedom_scale)
    
    return means, covariances
