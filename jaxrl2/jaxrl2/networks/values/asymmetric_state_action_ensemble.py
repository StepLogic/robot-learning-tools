from typing import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp

from jaxrl2.networks.values.state_action_value import StateActionValue


class AsymmetricStateActionEnsemble(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    num_qs: int = 2

    @nn.compact
    def __call__(self, pixels,states, actions, training: bool = False):
        PixelsVmapCritic = nn.vmap(
            StateActionValue,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_qs,
        )
        StatesVmapCritic = nn.vmap(
            StateActionValue,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_qs,
        )
        qs1 = PixelsVmapCritic(self.hidden_dims, activations=self.activations)(
            states, actions, training
        )
        qs2=StatesVmapCritic(self.hidden_dims, activations=self.activations)(
            states, actions, training
        )
        return jnp.concatenate([qs1,qs2],axis=-1)
