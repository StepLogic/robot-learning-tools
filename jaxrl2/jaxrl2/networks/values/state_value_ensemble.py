from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp

from jaxrl2.networks.values.state_action_value import StateActionValue
from jaxrl2.networks.values.state_value import StateValue


class StateValueEnsemble(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    num_qs: int = 2
    kernel_init: Optional[Callable] = None
    bias_init: Optional[Callable] = None
    @nn.compact
    def __call__(self, states, training: bool = False):
        VmapCritic = nn.vmap(
            StateValue,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_qs,
        )
        qs = VmapCritic(self.hidden_dims, activations=self.activations)(
            states,training
        )
        return qs
