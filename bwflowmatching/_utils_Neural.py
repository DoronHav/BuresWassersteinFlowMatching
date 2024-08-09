import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
import jax.random as random  # type: ignore
from flax import linen as nn  # type: ignore
import tensorflow_probability.substrates.jax.math as jax_prob # type: ignore

from bwflowmatching.DefaultConfig import DefaultConfig


class FeedForward(nn.Module):
    """Transformer MLP / feed-forward block.

    Attributes:
    config: DefaultConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
    """
    config: DefaultConfig

    @nn.compact
    def __call__(self, inputs, deterministic = True, dropout_rng=random.key(0)):
        config = self.config
        
        mlp_hidden_dim = config.mlp_hidden_dim
        # dropout_rate = config.dropout_rate

        x = nn.Dense(features = mlp_hidden_dim)(inputs)
        x = nn.relu(x)
        # x = nn.Dropout(
        #     rate = dropout_rate,
        #     deterministic = deterministic,
        #     rng = dropout_rng,
        # )(x)
        x = nn.Dense(inputs.shape[-1])(x) + inputs
        output = nn.LayerNorm()(x)
        return output

class InputMeanCovarianceNN(nn.Module):

    config: DefaultConfig

    @nn.compact
    def __call__(self, means, covariances, t,  labels = None, deterministic = True, dropout_rng=random.key(0)):
        config = self.config

        embedding_dim = config.embedding_dim
        num_layers = config.num_layers

        freqs = jnp.arange(embedding_dim//2) 

        cov_tril = jax_prob.fill_triangular_inverse(covariances)

        means_emb = nn.Dense(features = embedding_dim)(means)
        covariances_emb = nn.Dense(features = embedding_dim)(cov_tril)

        t_freq = freqs[None, :] * t[:, None]
        t_four = jnp.concatenate([jnp.cos(t_freq), jnp.sin(t_freq)], axis = -1)

        t_emb = nn.Dense(features = embedding_dim)(t_four)

        x = jnp.concat([means_emb, covariances_emb, t_emb], axis = -1)

        #means_emb + covariances_emb + t_emb
        
        if(labels is not None):
            l_emb = nn.Dense(features = embedding_dim)(jax.nn.one_hot(labels, config.label_dim))
            x = jnp.concatenate([x, l_emb], axis = -1)

        for _ in range(num_layers):
            x = FeedForward(config)(inputs = x, deterministic = deterministic, dropout_rng = dropout_rng)

        return(x)

class BuresWassersteinNN(nn.Module):

    config: DefaultConfig

    @nn.compact
    def __call__(self, means, covariances, t,  labels = None, deterministic = True, dropout_rng=random.key(0)):
        
        config = self.config
        architecture = config.architecture

        space_dim = means.shape[-1]


        if(architecture == 'separate'):
              
            mean_dot_emb = InputMeanCovarianceNN(config)(means, covariances, t, labels, deterministic, dropout_rng)  
            sigma_dot_emb = InputMeanCovarianceNN(config)(means, covariances, t, labels, deterministic, dropout_rng)  

            mean_dot = nn.Dense(space_dim)(mean_dot_emb)
            tril_vec = nn.Dense(space_dim * (space_dim + 1) // 2)(sigma_dot_emb)

        else:
            dot_emb = InputMeanCovarianceNN(config)(means, covariances, t, labels, deterministic, dropout_rng)

            mean_dot = nn.Dense(space_dim)(dot_emb)
            tril_vec = nn.Dense(space_dim * (space_dim + 1) // 2)(dot_emb)

        lower_triangular = jax_prob.fill_triangular(tril_vec)
        covariance_dot = lower_triangular + jnp.triu(lower_triangular.transpose([0,2,1]), k=1)

        return mean_dot, covariance_dot


