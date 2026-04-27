from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp

from jaxrl2.networks.mlp import MLP
from jaxrl2.networks.plain_mlp import PlainMLP


class StateValue(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh
    kernel_init: Optional[Callable] = None
    bias_init: Optional[Callable] = None
    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, training: bool = False
    ) -> jnp.ndarray:
        critic = PlainMLP((*self.hidden_dims, 1), activations=self.activations,kernel_init=self.kernel_init,bias_init=self.bias_init)(
            observations, training=training
        )
        return jnp.squeeze(critic, -1)
