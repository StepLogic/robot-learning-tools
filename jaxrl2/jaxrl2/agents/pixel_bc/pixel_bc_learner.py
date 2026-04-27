"""Implementations of algorithms for continuous control."""

from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jaxrl2.networks.encoders.pretrained_resnet import PretrainedResNet
from jaxrl2.networks.normal_tanh_policy import NormalTanhPolicy
from jaxrl2.utils.misc import augment_batch
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from jaxrl2.agents.agent import Agent
from jaxrl2.agents.bc.actor_updater import log_prob_update
from jaxrl2.utils.augmentations import batched_random_crop, batched_random_cutout
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.networks.encoders import D4PGEncoder, ResNetV2Encoder,PlaceholderEncoder
from jaxrl2.networks.normal_policy import UnitStdNormalPolicy, VariableStdNormalPolicy
from jaxrl2.networks.pixel_multiplexer import PixelMultiplexer
from jaxrl2.types import Params, PRNGKey
import flaxmodels as fm
@jax.jit
def _update_jit(
    rng: PRNGKey, actor: TrainState, batch: TrainState
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str, float]]:
    rng, key = jax.random.split(rng)
    # if augument:
    rng, batch = augment_batch(key, batch,batched_random_crop)
    # rng, key = jax.random.split(rng)
    # rng, batch = augment_batch(key, batch,batched_random_cutout)
    rng, new_actor, actor_info = log_prob_update(rng, actor, batch)

    return rng, new_actor, actor_info


class PixelBCLearner(Agent):
    def __init__(
        self,
        seed: int,
        observations: Union[jnp.ndarray, DatasetDict],
        actions: jnp.ndarray,
        actor_lr: float = 3e-4,
        decay_steps: Optional[int] = None,
        hidden_dims: Sequence[int] = (256, 256),
        cnn_features: Sequence[int] = (32, 32, 32, 32),
        cnn_filters: Sequence[int] = (3, 3, 3, 3),
        cnn_strides: Sequence[int] = (2, 1, 1, 1),
        cnn_padding: str = "VALID",
        latent_dim: int = 50,
        dropout_rate: Optional[float] = None,
        encoder: str = "d4pg",
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        action_dim = actions.shape[-1]

        rng = jax.random.PRNGKey(seed)
        rng, actor_key = jax.random.split(rng)

        if encoder == "d4pg":
            encoder_def = partial(
                D4PGEncoder,
                features=cnn_features,
                filters=cnn_filters,
                strides=cnn_strides,
                padding=cnn_padding,
            )
        elif encoder == "resnet":
            encoder_def = partial(ResNetV2Encoder, stage_sizes=(2, 2, 2, 2))
        else:
            encoder_def = partial(PlaceholderEncoder)

        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)
        policy_def = NormalTanhPolicy(
            hidden_dims, action_dim, dropout_rate=dropout_rate
        )
        actor_def = PixelMultiplexer(
            encoder=encoder_def, network=policy_def, latent_dim=latent_dim
        )
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )

        self._rng = rng
        self._actor = actor

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        new_rng, new_actor, info = _update_jit(self._rng, self._actor, batch)

        self._rng = new_rng
        self._actor = new_actor

        return info
