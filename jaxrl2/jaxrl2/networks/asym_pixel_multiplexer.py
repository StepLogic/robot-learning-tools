from typing import Dict, Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from jaxrl2.networks.constants import default_init
from jaxrl2.utils.misc import is_image_space, process_observation

# from typing import Dict, Optional, Union
# import flax.linen as nn
# import jax
# import jax.numpy as jnp
# from flax.core.frozen_dict import FrozenDict
# from jaxrl2.networks.constants import default_init

# class PixelMultiplexer(nn.Module):
#     encoder: nn.Module
#     network: nn.Module
#     latent_dim: int
#     stop_gradient: bool = False

#     @nn.compact
#     def __call__(
#         self,
#         observations: Union[FrozenDict, Dict],
#         actions: Optional[jnp.ndarray] = None,
#         training: bool = False,
#     ) -> jnp.ndarray:
#         observations = FrozenDict(observations)
#         assert (
#             len(observations.keys()) <= 2
#         ), "Can include only pixels and states fields."

#         x = self.encoder(observations["pixels"])

#         if self.stop_gradient:
#             # We do not update conv layers with policy gradients.
#             x = jax.lax.stop_gradient(x)

#         x = nn.Dense(self.latent_dim, kernel_init=default_init())(x)
#         x = nn.LayerNorm()(x)
#         x = nn.tanh(x)

#         if "states" in observations:
#             y = nn.Dense(self.latent_dim, kernel_init=default_init())(
#                 observations["states"]
#             )
#             y = nn.LayerNorm()(y)
#             y = nn.tanh(y)

#             x = jnp.concatenate([x, y], axis=-1)

#         if actions is None:
#             return self.network(x, training=training)
#         else:
#             return self.network(x, actions, training=training)


class AsymmetricPixelMultiplexer(nn.Module):
    encoder: nn.Module
    network: nn.Module
    latent_dim: int
    stop_gradient: bool = False
    exclude_states:bool=False
    # siamese:bool=False
    # def setup(self):
    #     # Submodule names are derived by the attributes you assign to. In this
    #     # case, "dense1" and "dense2". This follows the logic in PyTorch.
    #     # self.encoder_dict={
    #     #     "encoder_1":self.encoder
    #     # }
    #     if self.siamese:

    #     self.dense1 = nn.Dense(32)
    #     self.dense2 = nn.Dense(32)
    @nn.compact
    def __call__(
        self,
        observations: Union[jnp.ndarray, Dict, FrozenDict],
        actions: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        # Handle both array and dict inputs
        observations = process_observation(observations)
        # Convert to FrozenDict if needed
        if not isinstance(observations, FrozenDict):
            observations = FrozenDict(observations)
        # processed_features = []
        pixels=None
        states=None
        for key, value in observations.items():
            if is_image_space(value):
                pixels = self.encoder(name=f"encoder_{key}")(value)
                if self.stop_gradient:
                    pixels = jax.lax.stop_gradient(pixels)
                pixels = nn.Dense(self.latent_dim, kernel_init=default_init())(pixels)
                pixels = nn.LayerNorm()(pixels)
                pixels = nn.tanh(pixels)
            else:
                if self.exclude_states:
                    continue
                # Handle continuous observations
                states = nn.Dense(self.latent_dim, kernel_init=default_init())(value)
                states = nn.LayerNorm()(states)
                states = nn.tanh(states)
        
        # Pass through the network
        if self.exclude_states:
            if actions is None:
                return self.network(pixels, training=training)
            else:
                return self.network(pixels, actions, training=training)
        else:
            if actions is None:
                return self.network(pixels,states,training=training)
            else:
                return self.network(pixels,states,actions, training=training)