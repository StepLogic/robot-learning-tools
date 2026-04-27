from typing import Callable, Dict, Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from jaxrl2.networks.constants import default_init,default_bias_init
from jaxrl2.utils.misc import augment_batch, is_image_space, process_observation

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


class PixelMultiplexer(nn.Module):
    encoder: nn.Module
    network: nn.Module
    latent_dim: int
    stop_gradient: bool = False
    kernel_init: Optional[Callable] = None
    bias_init: Optional[Callable] = None
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
        # train_encoder:bool=True
    ) -> jnp.ndarray:
        
        # Handle both array and dict inputs
        observations = process_observation(observations)
        # Convert to FrozenDict if needed
        if not isinstance(observations, FrozenDict):
            observations = FrozenDict(observations)
        processed_features = []
        kernel_init= self.kernel_init or default_init
        bias_init = self.bias_init or default_bias_init
        for key, value in observations.items():
            # value=jax.numpy.nan_to_num(value)
            if is_image_space(value):
                value=value.astype(jnp.float32)
                
                # value = jax.lax.cond(
                #     jnp.max(value) > 1.0,
                #     lambda x: x.astype(jnp.float32)/255.0,  # true_fn: normalize
                #     lambda x: x.astype(jnp.float32), # false_fn: keep as is
                #     value
                # )

                x = self.encoder(name=f"encoder_{key}")(value)
                # breakpoint()
                self.sow('intermediates', 'features', x)
                x = nn.Dense(self.latent_dim, kernel_init=kernel_init(),bias_init=bias_init(),name=f"encoder_pre_latent")(x)
                x = nn.LayerNorm(name="encoder_layer_norm_enc")(x)
            else:
                # Handle continuous observations
                # jax.debug.print("value {value}",value=value)
                # breakpoint()
                x = nn.Dense(self.latent_dim, kernel_init=kernel_init(),name=f"encoder_{key}")(value)
                x = nn.LayerNorm(name=f"encoder_{key}_layer_norm")(x)
            x = nn.tanh(x)
            if self.stop_gradient:
                x = jax.lax.stop_gradient(x)

            processed_features.append(x)
        # breakpoint()
        # Combine all processed features
        if len(processed_features) > 1:
            # breakpoint()
            x = jnp.concatenate(processed_features, axis=-1)
        else:
            x = processed_features[0]

        # Pass through the network
        if actions is None:
            return self.network(x, training=training)
        else:
            return self.network(x, actions, training=training)