"""Implementations of algorithms for continuous control."""

import copy
import functools
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import distrax
import jax
import jax.numpy as jnp
from jaxrl2.networks.encoders.pretrained_resnet import PretrainedResNet
from jaxrl2.networks.normal_tanh_policy import NormalTanhPolicy
from jaxrl2.utils.misc import augment_batch
import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
from jaxrl2.agents.agent import Agent
from jaxrl2.utils.augmentations import batched_random_crop
# from jaxrl2.agents.drq.drq_learner import _share_encoder, _unpack
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.networks.encoders import D4PGEncoder,ResNetV2Encoder,PlaceholderEncoder
from jaxrl2.networks.normal_policy import UnitStdNormalPolicy
from jaxrl2.networks.pixel_multiplexer import PixelMultiplexer
from jaxrl2.networks.values import StateActionEnsemble, StateValue
from jaxrl2.types import Params, PRNGKey
from jaxrl2.utils.target_update import soft_target_update
from functools import partial
from  flax.training import train_state
from flax.core.frozen_dict import FrozenDict,freeze,unfreeze
class TrainState(train_state.TrainState):
  batch_stats: Any

@partial(jax.jit, static_argnames="actor_apply_fn")
def sample_actions_jit(
    rng: PRNGKey,
    actor_apply_fn: Callable[..., distrax.Distribution],
    actor_params: Params,
     batch_stats:Any,
    observations: np.ndarray,
) -> Tuple[PRNGKey, jnp.ndarray]:
    dist = actor_apply_fn({"params": actor_params,'batch_stats': batch_stats}, observations)
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)
@partial(jax.jit, static_argnames="actor_apply_fn")
def extract_feature(
    actor_apply_fn: Callable[..., distrax.Distribution],
    actor_params: Params,
    batch_stats:Any,
    observations: np.ndarray,
) -> jnp.ndarray:
    (_,outputs) = actor_apply_fn({"params": actor_params,'batch_stats': batch_stats}, observations,mutable='intermediates')
    features = outputs['intermediates']['features']
    return features

@partial(jax.jit, static_argnames="actor_apply_fn")
def action_dist_jit(
    actor_apply_fn: Callable[..., distrax.Distribution],
    actor_params: Params,
    batch_stats:Any,
    observations: np.ndarray,
) -> jnp.ndarray:
    dist = actor_apply_fn({"params": actor_params,'batch_stats': batch_stats},observations)
    return dist

@partial(jax.jit, static_argnames="actor_apply_fn")
def eval_actions_jit(
    actor_apply_fn: Callable[..., distrax.Distribution],
    actor_params: Params,
    batch_stats:Any,
    observations: np.ndarray,
) -> jnp.ndarray:
    dist = actor_apply_fn({"params": actor_params,'batch_stats': batch_stats},observations)
    return dist.mode()



def _share_encoder(source, target):
    replacers = {}
    for k, v in source.params.items():
        if "encoder" in k:
            replacers[k] = v
    new_params = unfreeze(FrozenDict(target.params).copy(add_or_replace=replacers))
    target=target.replace(params=new_params)

    replacers = {}
    new_batch_stats = unfreeze(FrozenDict(target.batch_stats).copy(add_or_replace=replacers))
    for k, v in source.batch_stats.items():
        replacers[k] = v
    target=target.replace(batch_stats=new_batch_stats)
    return target



def update_actor(
    key: PRNGKey,
    actor: TrainState,
    target_critic: TrainState,
    value: TrainState,
    batch: FrozenDict,
    A_scaling: float,
    critic_reduction: str,
) -> Tuple[TrainState, Dict[str, float]]:
    v,_ = value.apply_fn({"params": value.params,"batch_stats":value.batch_stats}, batch["observations"],
                           mutable=["batch_stats"])
    qs,_ = target_critic.apply_fn(
        {"params": target_critic.params,"batch_stats":target_critic.batch_stats},
        batch["observations"],
        batch["actions"],
        mutable=["batch_stats"]
    )
    if critic_reduction == "min":
        q = qs.min(axis=0)
    elif critic_reduction == "mean":
        q = qs.mean(axis=0)
    else:
        raise NotImplemented()
    exp_a = jnp.exp((q - v) * A_scaling)
    exp_a = jnp.minimum(exp_a, 100.0)

    def actor_loss_fn(actor_params: Params,batch_stats:any) -> Tuple[jnp.ndarray, Dict[str, float]]:
        dist,_ = actor.apply_fn(
            {"params": actor_params,"batch_stats":batch_stats},
            batch["observations"],
            training=True,
            rngs={"dropout": key},
            mutable=["batch_stats"]
        )
        log_probs = dist.log_prob(batch["actions"])
        actor_loss = -(exp_a * log_probs).mean()

        return actor_loss, {"actor_loss": actor_loss, "adv": jnp.mean(q - v)}

    grads, info = jax.grad(actor_loss_fn, has_aux=True)(actor.params,actor.batch_stats)
    new_actor = actor.apply_gradients(grads=grads)

    return new_actor, info


def loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


def update_v(
    target_critic: TrainState,
    value: TrainState,
    batch: FrozenDict,
    expectile: float,
    critic_reduction: str,
) -> Tuple[TrainState, Dict[str, float]]:
    qs,_ = target_critic.apply_fn(
        {"params": target_critic.params,"batch_stats":value.batch_stats},
        batch["observations"],
        batch["actions"],
        mutable=["batch_stats"]
    )

    if critic_reduction == "min":
        q = qs.min(axis=0)
    elif critic_reduction == "mean":
        q = qs.mean(axis=0)
    else:
        raise NotImplemented()

    def value_loss_fn(value_params: Params,batch_stats:Any) -> Tuple[jnp.ndarray, Dict[str, float]]:
        v,_ = value.apply_fn({"params": value_params,"batch_stats":batch_stats}, batch["observations"],
                           mutable=["batch_stats"])
        value_loss = loss(q - v, expectile).mean()
        return value_loss, {"value_loss": value_loss, "v": v.mean()}

    grads, info = jax.grad(value_loss_fn, has_aux=True)(value.params,target_critic.batch_stats)
    new_value = value.apply_gradients(grads=grads)

    return new_value, info


def update_q(
    critic: TrainState, value: TrainState, batch: FrozenDict, discount: float
) -> Tuple[TrainState, Dict[str, float]]:
    next_v,_ = value.apply_fn({"params":value.params ,"batch_stats":value.batch_stats},batch["next_observations"],mutable=["batch_stats"])

    target_q = batch["rewards"] + discount * batch["masks"] * next_v

    def critic_loss_fn(critic_params: Params,batch_stats:Any) -> Tuple[jnp.ndarray, Dict[str, float]]:
        qs,updates = critic.apply_fn(
            {"params": critic_params ,"batch_stats":batch_stats}, batch["observations"], batch["actions"],
            training=True,
            mutable=['batch_stats']
        )
        critic_loss = ((qs - target_q) ** 2).mean()
        return critic_loss, ({"critic_loss": critic_loss, "q": qs.mean()},updates)

    grads, (info,updates) = jax.grad(critic_loss_fn, has_aux=True)(critic.params,critic.batch_stats)
    new_critic = critic.apply_gradients(grads=grads)
    new_critic = new_critic.replace(batch_stats=updates['batch_stats'])
    return new_critic, info


@functools.partial(jax.jit, static_argnames=("critic_reduction", "share_encoder"))
def _update_jit(
    rng: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    target_critic_params: Params,
    value: TrainState,
    batch: TrainState,
    discount: float,
    tau: float,
    expectile: float,
    A_scaling: float,
    critic_reduction: str,
    share_encoder: bool,
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str, float]]:

    if share_encoder:
        actor = _share_encoder(source=critic, target=actor)
        value = _share_encoder(source=critic, target=value)

    rng, key = jax.random.split(rng)
    # breakpoint()
    rng, batch = augment_batch(key, batch,batched_random_crop)

    target_critic = critic.replace(params=target_critic_params)
    # breakpoint()
    new_value, value_info = update_v(
        target_critic, value, batch, expectile, critic_reduction
    )
    key, rng = jax.random.split(rng)
    new_actor, actor_info = update_actor(
        key, actor, target_critic, new_value, batch, A_scaling, critic_reduction
    )

    new_critic, critic_info = update_q(critic, new_value, batch, discount)

    new_target_critic_params = soft_target_update(
        new_critic.params, target_critic_params, tau
    )

    return (
        rng,
        new_actor,
        new_critic,
        new_target_critic_params,
        new_value,
        {**critic_info, **value_info, **actor_info},
    )


class PixelResNetIQLLearner(Agent):
    def __init__(
        self,
        seed: int,
        observations: Union[jnp.ndarray, DatasetDict],
        actions: jnp.ndarray,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        decay_steps: Optional[int] = None,
        hidden_dims: Sequence[int] = (256, 256),
        cnn_features: Sequence[int] = (32, 32, 32, 32),
        cnn_filters: Sequence[int] = (3, 3, 3, 3),
        cnn_strides: Sequence[int] = (2, 1, 1, 1),
        cnn_padding: str = "VALID",
        latent_dim: int = 50,
        discount: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.9,
        A_scaling: float = 10.0,
        critic_reduction: str = "min",
        dropout_rate: Optional[float] = None,
        share_encoder: bool = False,
        freeze_encoders:bool=False,
        cosine_decay:bool=False,
        encoder: str = "d4pg",
        num_qs=2,

    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        action_dim = actions.shape[-1]

        self.expectile = expectile
        self.tau = tau
        self.discount = discount
        self.critic_reduction = critic_reduction
        self.A_scaling = A_scaling
        self.share_encoder = share_encoder

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        # if encoder == "d4pg":
        #     encoder_def = partial(
        #         D4PGEncoder,
        #         features=cnn_features,
        #         filters=cnn_filters,
        #         strides=cnn_strides,
        #         padding=cnn_padding,
        #     )
        # elif encoder == "resnet":
        #     encoder_def = partial(ResNetV2Encoder, stage_sizes=(2, 2, 2, 2))
        # elif encoder == "embeddings":
        #     encoder_def = partial(PlaceholderEncoder)
        encoder_def = partial(PretrainedResNet)

        # actor_def = PixelMultiplexer(
        #     encoder=encoder_def, network=policy_def, latent_dim=latent_dim)
        
        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)
        policy_def = NormalTanhPolicy(
            hidden_dims, action_dim, dropout_rate=dropout_rate
            # , apply_tanh=False
        )
        actor_def = PixelMultiplexer(
            encoder=encoder_def,
            network=policy_def,
            latent_dim=latent_dim,
            stop_gradient=share_encoder or freeze_encoders,
        )
        params = actor_def.init(actor_key, observations)
        
        actor_params=params["params"]
        batch_stats=params["batch_stats"]

        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            batch_stats=batch_stats,
            tx=optax.adam(learning_rate=actor_lr),
        )

        critic_def = StateActionEnsemble(hidden_dims, num_qs=max(2,num_qs))
        critic_def = PixelMultiplexer(
            encoder=encoder_def, network=critic_def, latent_dim=latent_dim,
                      stop_gradient=freeze_encoders,
        )
        params = critic_def.init(critic_key, observations, actions)
        critic_params = params["params"]
        batch_stats=params["batch_stats"]
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            batch_stats=batch_stats,
            tx=optax.adam(learning_rate=critic_lr),
        )
        target_critic_params = copy.deepcopy(critic_params)

        value_def = StateValue(hidden_dims)
        value_def = PixelMultiplexer(
            encoder=encoder_def,
            network=value_def,
            latent_dim=latent_dim,
            stop_gradient=share_encoder or freeze_encoders,
        )
        params = value_def.init(value_key, observations)
        
        value_params = params["params"]
        batch_stats=params["batch_stats"]

        value = TrainState.create(
            apply_fn=value_def.apply,
            params=value_params,
            batch_stats=batch_stats,
            tx=optax.adam(learning_rate=value_lr),
        )

        self._rng = rng
        self._actor = actor
        self._critic = critic
        self._target_critic_params = target_critic_params
        self._value = value
    
    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        rng, actions = sample_actions_jit(
            self._rng, self._actor.apply_fn, self._actor.params,self._actor.batch_stats, observations
        )
        self._rng = rng
        return np.asarray(actions)
    def extract_features(self, batch:DatasetDict):
        return np.asarray(extract_feature(self._actor.apply_fn, self._actor.params,self._actor.batch_stats,batch))

    def action_dist(self, observations: np.ndarray):
        return action_dist_jit(self._actor.apply_fn, self._actor.params,self._actor.batch_stats, observations)

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        actions = eval_actions_jit(
                    self._actor.apply_fn, self._actor.params,self._actor.batch_stats,observations
                )
        return np.asarray(actions)
    

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        (
            new_rng,
            new_actor,
            new_critic,
            new_target_critic,
            new_value,
            info,
        ) = _update_jit(
            self._rng,
            self._actor,
            self._critic,
            self._target_critic_params,
            self._value,
            batch,
            self.discount,
            self.tau,
            self.expectile,
            self.A_scaling,
            self.critic_reduction,
            self.share_encoder,
        )

        self._rng = new_rng
        self._actor = new_actor
        self._critic = new_critic
        self._target_critic_params = new_target_critic
        self._value = new_value

        return info
