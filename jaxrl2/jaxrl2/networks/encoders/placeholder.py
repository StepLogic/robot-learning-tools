from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from jaxrl2.networks.constants import default_init


class PlaceholderEncoder(nn.Module):
    @nn.compact
    def __call__(self, observations: jnp.ndarray,train=True) -> jnp.ndarray:
        return observations
