from typing import Callable, Optional, Sequence, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from jaxrl2.networks.constants import default_bias_init, default_init
def _flatten_dict(x: Union[FrozenDict, jnp.ndarray]) -> jnp.ndarray:
    if hasattr(x, "values"):
        return jnp.concatenate([_flatten_dict(v) for k, v in sorted(x.items())], -1)
    else:
        return x


from typing import Callable, Optional, Sequence
import flax.linen as nn
import jax.numpy as jnp

default_init = nn.initializers.orthogonal


class PlainMLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh
    activate_final: bool = False
    use_layer_norm: bool = False
    scale_final: Optional[float] = None
    dropout_rate: Optional[float] = None
    # default_init = nn.initializers.xavier_uniform
    kernel_init: Optional[Callable] = None
    bias_init: Optional[Callable] = None
    # @nn.compact
    # def __call__(self, x: jnp.ndarray,training: bool = False) -> jnp.ndarray:
    #     # for i, size in enumerate(self.hidden_dims):
    #     #     x = nn.Dense(size)(x)
    #     #     x = self.activations(x)
    #     # x = nn.Dense(1)(x)
    #     # print(x)
    #     x = nn.Dense(256,kernel_init=self.kernel_init(),bias_init=self.bias_init())(x)
    #     x = nn.tanh(x)
    #     x = nn.Dense(256,kernel_init=self.kernel_init(),bias_init=self.bias_init())(x)
    #     x = nn.tanh(x)
    #     x = nn.Dense(1,kernel_init=nn.initializers.orthogonal(1.0),bias_init=nn.initializers.constant(0.1))(x)
    #     # jax.debug.print("{v}",v=x)
    #     return x
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = _flatten_dict(x)
        kernel_init= self.kernel_init or default_init
        bias_init = self.bias_init or default_bias_init
        for i, size in enumerate(self.hidden_dims):
            if i + 1 == len(self.hidden_dims) and self.scale_final is not None:
                x = nn.Dense(size, kernel_init=kernel_init(self.scale_final),bias_init=bias_init())(x)
            else:
                x = nn.Dense(size, kernel_init=kernel_init(),bias_init=bias_init())(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training
                    )
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
        return x
