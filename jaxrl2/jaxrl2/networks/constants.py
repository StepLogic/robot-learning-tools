import flax.linen as nn
import jax.numpy as jnp


def default_init(scale: float = jnp.sqrt(2)):
    return nn.initializers.xavier_uniform()

def default_orthogonal_init(scale: float = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale=scale)

def default_bias_init(scale: float = 0.0):
    return nn.initializers.constant(0.0)
