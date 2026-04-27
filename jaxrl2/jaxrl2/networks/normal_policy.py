from typing import Callable, Optional, Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import constant
from jaxrl2.networks import MLP
from jaxrl2.networks.constants import default_bias_init, default_init,default_orthogonal_init
from jaxrl2.networks.plain_mlp import PlainMLP


class UnitStdNormalPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None
    apply_tanh: bool = True
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, training: bool = False
    ) -> distrax.Distribution:
        outputs = MLP(
            self.hidden_dims,
            activate_final=True,
            dropout_rate=self.dropout_rate,
            activations=self.activations,
        )(observations, training=training)

        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)

        if self.apply_tanh:
            means = nn.tanh(means)

        return distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.ones_like(means)
        )

class VariableStdNormalPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None
    apply_tanh: bool = True
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    low: Optional[jnp.ndarray] = None
    high: Optional[jnp.ndarray] = None
    log_std_init: float = 1.0
    kernel_init: Optional[Callable] = None
    bias_init: Optional[Callable] = None
    use_layer_norm: bool = False

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, training: bool = False
    ) -> distrax.Distribution:
        kernel_init= self.kernel_init or default_orthogonal_init
        bias_init = self.bias_init or default_bias_init
        outputs = PlainMLP(
            self.hidden_dims, activate_final=False, dropout_rate=self.dropout_rate,kernel_init=kernel_init,bias_init=bias_init,use_layer_norm=self.use_layer_norm
        )(observations, training=training)

        action_logits = nn.Dense(self.action_dim,kernel_init=kernel_init(),bias_init=nn.initializers.constant(0.1))(outputs)
        # log_stds = self.param("log_std", constant(self.log_std_init), (self.action_dim,))
        log_stds=nn.Dense(self.action_dim,kernel_init=kernel_init(),bias_init=nn.initializers.constant(self.log_std_init))(outputs)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        # jax.debug.print("log std {log_std}",log_std=log_std)
        return distrax.MultivariateNormalDiag(loc=action_logits, scale_diag=jnp.exp(log_stds))
