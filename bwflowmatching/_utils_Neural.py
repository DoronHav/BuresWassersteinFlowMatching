import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
import jax.random as random  # type: ignore
from flax import linen as nn  # type: ignore

from wassersteinflowmatching.DefaultConfig import DefaultConfig

class FeedForward(nn.Module):
    """Transformer MLP / feed-forward block.

    Attributes:
    config: DefaultConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
    """
    config: DefaultConfig

    @nn.compact
    def __call__(self, inputs):
        config = self.config
        mlp_hidden_dim = config.mlp_hidden_dim

        x = nn.Dense(features = mlp_hidden_dim)(inputs)
        x = nn.relu(x)
        output = nn.Dense(inputs.shape[-1])(x) + inputs
        return output


class BuresWassersteinNN(nn.Module):

    config: DefaultConfig

    @nn.compact
    def __call__(self, means, covariances, t,  labels = None, deterministic = True, dropout_rng=random.key(0)):
        
        config = self.config

        embedding_dim = config.embedding_dim
        num_layers = config.num_layers

        space_dim = means.shape[-1]


        means_emb = nn.Dense(features = embedding_dim)(means)
        covariances_emb = nn.Dense(features = embedding_dim)(jnp.tril(covariances))
        t_emb = nn.Dense(features = embedding_dim)(t[:, None, None])

        x = means_emb + covariances_emb + t_emb

        if(labels is not None):
            l_emb = nn.Dense(features = embedding_dim)(jax.nn.one_hot(labels, config.label_dim)[:, None, :])
            x = x + l_emb

        for _ in range(num_layers):
            x = FeedForward(config)(inputs = x, deterministic = deterministic, dropout_rng = dropout_rng)

        mean = nn.Dense(space_dim)(x)
        
        tril_vec = nn.Dense(space_dim * (space_dim + 1) // 2)(x)

        idx = jnp.tril_indices(space_dim)
        covariance_chol = jnp.zeros((space_dim, space_dim), dtype=tril_vec.dtype).at[idx].set(tril_vec)
        covariance_chol = covariance_chol + covariance_chol.T - jnp.diag(jnp.diag(covariance_chol))
        
        covariance = covariance_chol @ covariance_chol.T

        return mean, covariance





    
