"""Implementations of algorithms for continuous control."""

import copy
import functools
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import distrax
import jax
import jax.numpy as jnp
from jaxrl2.networks.encoders.placeholder import PlaceholderEncoder
from jaxrl2.networks.encoders.pretrained_resnet import PretrainedResNet
from jaxrl2.utils.augmentations import batched_random_crop, batched_random_cutout
from jaxrl2.utils.misc import augment_batch
import optax
from flax.core.frozen_dict import FrozenDict,freeze,unfreeze
# from flax.training.train_state import TrainState
import numpy as np
from jaxrl2.agents.agent import Agent


from jaxrl2.agents.sac.temperature import Temperature
from jaxrl2.agents.sac.temperature_updater import update_temperature
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.networks.encoders import D4PGEncoder, ResNetV2Encoder
from jaxrl2.networks.normal_tanh_policy import NormalTanhPolicy
from jaxrl2.networks.gsde_policy import GSDEPolicy
from jaxrl2.networks.pixel_multiplexer import PixelMultiplexer
from jaxrl2.networks.values import StateActionEnsemble
from jaxrl2.types import Params, PRNGKey
from jaxrl2.utils.target_update import soft_target_update
from functools import partial
from  flax.training import train_state



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
    try:
     dist = actor_apply_fn({"params": actor_params,'batch_stats': batch_stats}, observations,training=False)
    except:
        breakpoint()
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)
@partial(jax.jit, static_argnames="actor_apply_fn")
def extract_feature(
    actor_apply_fn: Callable[..., distrax.Distribution],
    actor_params: Params,
    batch_stats:Any,
    observations: np.ndarray,
) -> jnp.ndarray:
    (_,outputs) = actor_apply_fn({"params": actor_params,'batch_stats': batch_stats}, observations,training=False,mutable='intermediates')
    features = outputs['intermediates']['features']
    return features

@partial(jax.jit, static_argnames="actor_apply_fn")
def action_dist_jit(
    actor_apply_fn: Callable[..., distrax.Distribution],
    actor_params: Params,
    batch_stats:Any,
    observations: np.ndarray,
) -> jnp.ndarray:
    dist = actor_apply_fn({"params": actor_params,'batch_stats': batch_stats},observations,training=False)
    return dist

@partial(jax.jit, static_argnames="actor_apply_fn")
def eval_actions_jit(
    actor_apply_fn: Callable[..., distrax.Distribution],
    actor_params: Params,
    batch_stats:Any,
    observations: np.ndarray,
) -> jnp.ndarray:
    dist = actor_apply_fn({"params": actor_params,'batch_stats': batch_stats},observations,training=False)
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


def update_critic(
    key: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    target_critic: TrainState,
    temp: TrainState,
    batch: DatasetDict,
    discount: float,
    backup_entropy: bool,
    critic_reduction: str,
) -> Tuple[TrainState, Dict[str, float]]:
    # breakpoint()
    dist,_ = actor.apply_fn({"params": actor.params,"batch_stats":actor.batch_stats}, batch["next_observations"],training=False,mutable=["batch_stats"])
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)
    next_qs,_ = target_critic.apply_fn(
        {"params": target_critic.params,"batch_stats":target_critic.batch_stats}, batch["next_observations"], next_actions,training=False,mutable=["batch_stats"]
    )
    if critic_reduction == "min":
        next_q = next_qs.min(axis=0)
    elif critic_reduction == "mean":
        next_q = next_qs.mean(axis=0)
    else:
        raise NotImplemented()
    # value=jax.numpy.nan_to_num(            value=jax.numpy.nan_to_num(value))
    target_q = jax.numpy.nan_to_num(batch["rewards"]) + discount * batch["masks"] * next_q

    if backup_entropy:
        target_q -= (
            discount
            * batch["masks"]
            * temp.apply_fn({"params": temp.params})
            * next_log_probs
        )

    def critic_loss_fn(critic_params: Params,batch_stats:Any) -> Tuple[jnp.ndarray, Dict[str, float]]:
        qs,updates = critic.apply_fn(
            {"params": critic_params,"batch_stats":batch_stats}, batch["observations"], batch["actions"],training=True,mutable=["batch_stats"]
        )
        critic_loss = ((qs - target_q) ** 2).mean()
        return critic_loss, ({
            "critic_loss": critic_loss,
            "q": qs.mean(),
            "target_actor_entropy": -next_log_probs.mean(),
        },updates)

    grads, (info,updates) = jax.grad(critic_loss_fn, has_aux=True)(critic.params,critic.batch_stats)
    new_critic = critic.apply_gradients(grads=grads)
    new_critic = new_critic.replace(batch_stats=updates['batch_stats'])
    return new_critic, info

def update_actor(
    key: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    temp: TrainState,
    batch: DatasetDict,
) -> Tuple[TrainState, Dict[str, float]]:
    def actor_loss_fn(actor_params: Params,batch_stats:Any) -> Tuple[jnp.ndarray, Dict[str, float]]:
        # breakpoint()
        dist,_ = actor.apply_fn({"params": actor_params,"batch_stats":batch_stats}, batch["observations"],mutable=["batch_stats"])
        actions, log_probs = dist.sample_and_log_prob(seed=key)
        qs,_ = critic.apply_fn({"params": critic.params,"batch_stats":critic.batch_stats}, batch["observations"], actions,mutable=["batch_stats"])
        q = qs.mean(axis=0)
        actor_loss = (log_probs * temp.apply_fn({"params": temp.params}) - q).mean()
        return actor_loss, {"actor_loss": actor_loss, "entropy": -log_probs.mean()}

    grads, info = jax.grad(actor_loss_fn, has_aux=True)(actor.params,actor.batch_stats)
    if not isinstance(grads, dict):
            grads = unfreeze(grads)
    
    assert grads.keys() == actor.params.keys(), "Gradients and parameters keys do not match!"
    # print(type(grads.keys()))
    new_actor = actor.apply_gradients(grads=grads)
    return new_actor, info



@functools.partial(jax.jit, static_argnames=("backup_entropy", "critic_reduction","augument"))
def _update_critic_jit(
    rng: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    target_critic_params: Params,
    temp: TrainState,
    batch: FrozenDict,
    discount: float,
    tau: float,
    backup_entropy: bool,
    critic_reduction: str,
    augument:bool=True
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str, float]]:
    rng, key = jax.random.split(rng)
    if augument:
        rng, batch = augment_batch(key, batch,batched_random_crop)
    
    target_critic = critic.replace(params=target_critic_params)
  
    new_critic, critic_info = update_critic(
                    key,
                    actor,
                    critic,
                    target_critic,
                    temp,
                    batch,
                    discount,
                    backup_entropy=backup_entropy,
                    critic_reduction=critic_reduction,
                )
    
    new_target_critic_params = soft_target_update(
        new_critic.params, target_critic_params, tau
    )
    return (
        rng,
        new_critic,
        new_target_critic_params,
        {**critic_info},
    )


@functools.partial(jax.jit, static_argnames=("backup_entropy", "critic_reduction","utd_ratio","enable_update_temperature","augument"))
def _update_jit(
    rng: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    target_critic_params: Params,
    temp: TrainState,
    batch: FrozenDict,
    discount: float,
    tau: float,
    target_entropy: float,
    backup_entropy: bool,
    critic_reduction: str,
    utd_ratio=None,
    enable_update_temperature=True,
    augument:bool=True
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str, float]]:
    # batch = _unpack(batch)
    actor = _share_encoder(source=critic, target=actor)

    rng, key = jax.random.split(rng)
    # aug_pixels = batched_random_crop(key, batch["observations"]["pixels"])
    # observations = batch["observations"].copy(add_or_replace={"pixels": aug_pixels})
    # batch = batch.copy(add_or_replace={"observations": observations})
    # rng, key = jax.random.split(rng)
    # aug_next_pixels = batched_random_crop(key, batch["next_observations"]["pixels"])
    # next_observations = batch["next_observations"].copy(
    #     add_or_replace={"pixels": aug_next_pixels}
    # )
    # batch = batch.copy(add_or_replace={"next_observations": next_observations})
    if augument:
        rng, batch = augment_batch(key, batch,batched_random_crop)
        # rng, key = jax.random.split(rng)
        # rng, batch = augment_batch(key, batch,batched_random_cutout)
        
    target_critic = critic.replace(params=target_critic_params)
  
    new_critic, critic_info = update_critic(
                    key,
                    actor,
                    critic,
                    target_critic,
                    temp,
                    batch,
                    discount,
                    backup_entropy=backup_entropy,
                    critic_reduction=critic_reduction,
                )
    
    new_target_critic_params = soft_target_update(
        new_critic.params, target_critic_params, tau
    )

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, critic, temp, batch)
    new_temp=None
    alpha_info={}
    # print(enable_update_temperature)
    if enable_update_temperature:
        new_temp, alpha_info = update_temperature(
            temp, actor_info["entropy"], target_entropy
        )
    return (
        rng,
        new_actor,
        new_critic,
        new_target_critic_params,
        new_temp,
        {**critic_info, **actor_info, **alpha_info},
    )


class PixelResNetDrQLearner(Agent):
    def __init__(
        self,
        seed: int,
        observations: Union[jnp.ndarray, DatasetDict],
        actions: jnp.ndarray,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        cnn_features: Sequence[int] = (32, 32, 32, 32),
        cnn_filters: Sequence[int] = (3, 3, 3, 3),
        cnn_strides: Sequence[int] = (2, 1, 1, 1),
        cnn_padding: str = "VALID",
        latent_dim: int = 50,
        discount: float = 0.99,
        tau: float = 0.005,
        target_entropy: Optional[float] = None,
        backup_entropy: bool = True,
        critic_reduction: str = "min",
        init_temperature: float = 1.0,
        encoder: str = "d4pg",
        num_qs=2
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        action_dim = actions.shape[-1]
        # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/sac/sac.py
        if target_entropy is None:
            self.target_entropy=-action_dim/2 
        else:
            self.target_entropy = target_entropy

        self.backup_entropy = backup_entropy
        self.critic_reduction = critic_reduction

        self.tau = tau
        self.discount = discount

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

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
        # else:
        #     encoder_def = partial(PlaceholderEncoder)
        encoder_def = partial(PretrainedResNet)

        self.augument=not encoder is None
        policy_def = NormalTanhPolicy(hidden_dims, action_dim)
        actor_def = PixelMultiplexer(
            encoder=encoder_def,
            network=policy_def,
            latent_dim=latent_dim,
            stop_gradient=True,
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

        critic_def = StateActionEnsemble(hidden_dims, num_qs=max(num_qs,2))
        critic_def = PixelMultiplexer(
            encoder=encoder_def, network=critic_def, latent_dim=latent_dim
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

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = train_state.TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=temp_lr),
        )

        self._actor = actor
        self._critic = critic
        self._target_critic_params = target_critic_params
        self._temp = temp
        self._rng = rng


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
    


    def update(self, batch: FrozenDict,utd_ratio: int=None, enable_update_temperature: bool = True, output_range: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None) -> Dict[str, float]:



        (
            new_rng,
            new_actor,
            new_critic,
            new_target_critic_params,
            new_temp,
            info,
        ) = _update_jit(
            self._rng,
            self._actor,
            self._critic,
            self._target_critic_params,
            self._temp,
            batch,
            self.discount,
            self.tau,
            self.target_entropy,
            self.backup_entropy,
            self.critic_reduction,
            utd_ratio=utd_ratio,
            enable_update_temperature=enable_update_temperature,
            augument=self.augument
        )


        self._rng = new_rng
        self._actor = new_actor
        self._critic = new_critic
        self._target_critic_params = new_target_critic_params
        if not new_temp is None:
            self._temp = new_temp
        return info
    

    def update_expert(self, batch: FrozenDict) -> Dict[str, float]:
        (
        new_rng,
        new_critic,
        new_target_critic_params,
        info
        ) = _update_critic_jit(
            self._rng,
            self._actor,
            self._critic,
            self._target_critic_params,
            self._temp,
            batch,
            self.discount,
            self.tau,
            # self.target_entropy,
            self.backup_entropy,
            self.critic_reduction,
            # utd_ratio=utd_ratio,
            # enable_update_temperature=enable_update_temperature,
            augument=self.augument
        )
        # rng: PRNGKey,
        # actor: TrainState,
        # critic: TrainState,
        # target_critic_params: Params,
        # temp: TrainState,
        # batch: FrozenDict,
        # discount: float,
        # tau: float,
        # backup_entropy: bool,
        # critic_reduction: str,
        # augument:bool=True

        self._rng = new_rng
        # self._actor = new_actor
        self._critic = new_critic
        self._target_critic_params = new_target_critic_params
        # if not new_temp is None:
        #     self._temp = new_temp
        return info
