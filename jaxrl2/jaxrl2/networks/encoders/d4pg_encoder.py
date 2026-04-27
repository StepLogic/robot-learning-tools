from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from jaxrl2.networks.constants import default_init


class D4PGEncoder(nn.Module):
    features: Sequence[int] = (32, 32, 32, 32)
    filters: Sequence[int] = (2, 1, 1, 1)
    strides: Sequence[int] = (2, 1, 1, 1)
    padding: str = "VALID"
    kernel_init: Optional[Callable] = None
    @nn.compact
    def __call__(self, observations: jnp.ndarray,train=True) -> jnp.ndarray:
        assert len(self.features) == len(self.strides)
        # breakpoint()
        observations=observations.astype(jnp.float32)
        x = jax.lax.cond(
                    jnp.max(observations) > 1.0,
                    lambda observations: (observations).astype(jnp.float32)/ 255.0,  # true_fn: normalize
                    lambda observations: observations.astype(jnp.float32),               # false_fn: keep as is
                    observations
                )
        # x=(observations - jnp.mean(observations)) / jnp.std(observations)  # Normalize using mean and std :scale agnostic
     

        x = jnp.reshape(x, (*x.shape[:-2], -1))

        for features, filter_, stride in zip(self.features, self.filters, self.strides):
            kernel_init= self.kernel_init or default_init
            x = nn.Conv(
                features,
                kernel_size=(filter_, filter_),
                strides=(stride, stride),
                kernel_init=kernel_init(),
                padding=self.padding,
            )(x)
            x = nn.relu(x)

        return x.reshape((*x.shape[:-3], -1))
