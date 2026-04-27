"""Implementations of algorithms for continuous control."""

from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import distrax
import jax
import jax.numpy as jnp
from jaxrl2.networks.encoders.pretrained_resnet import PretrainedResNet
from jaxrl2.networks.normal_tanh_policy import NormalTanhPolicy
from jaxrl2.utils.misc import augment_batch, augment_state_batch
import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict
from  flax.training import train_state
import flax.linen as nn
from jaxrl2.agents.agent import Agent
from jaxrl2.utils.augmentations import  augment_bc_random_shift_batch, batch_bc_augmentation, batched_add_noise, batched_random_crop, batched_random_cutout
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.networks.encoders import D4PGEncoder, ResNetV2Encoder,PlaceholderEncoder
from jaxrl2.networks.normal_policy import UnitStdNormalPolicy, VariableStdNormalPolicy
from jaxrl2.networks.pixel_multiplexer import PixelMultiplexer
from jaxrl2.types import Params, PRNGKey
import flaxmodels as fm

class TrainState(train_state.TrainState):
  batch_stats: Any


#     def action_dist_jit(
#     actor_apply_fn: Callable[..., distrax.Distribution],
#     actor_params: Params,
#     observations: np.ndarray,
# ) -> jnp.ndarray:
#     dist = actor_apply_fn({"params": actor_params}, observations)
#     return dist

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

@partial(jax.jit, static_argnames=("train_encoder"))
def _update_jit(
    rng: PRNGKey, actor: TrainState, batch: TrainState,train_encoder:bool=True,
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str, float]]:
    rng, key = jax.random.split(rng)
    # if augument:
    rng, batch = augment_batch(key, batch,batched_random_crop)
    rng, key = jax.random.split(rng)
    # rng, batch = augumen_(key, batch,batched_random_cutout)
    rng,batch=augment_state_batch(key,batch,batched_add_noise)
    # rng,batch=augment_batch(key,batch,batch_bc_augmentation)

    # rng,batch=augment_bc_random_shift_batch(rng,batch)
    
    # rng, key = jax.random.split(rng)

    def loss_fn(actor_params: Params,batch_stats:Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        dist,updates = actor.apply_fn(
            {"params": actor_params,"batch_stats":batch_stats},
            batch["observations"],
            training=True,
            rngs={"dropout": key},
            mutable=['batch_stats']
        )
        log_probs = dist.log_prob(batch["actions"]) 
        
        # log_probs = log_probs.mean() 
        # clipped_log_probs=jnp.clip(log_probs,-2.9,0)
        # actor_loss = jnp.mean(jnp.square(dist.mode() - batch["actions"]))
        # actor_loss = -log_probs.mean() 
        # actor_loss = jnp.mean(jnp.square(dist.mode() - batch["actions"]) / ( 2*jnp.square(dist.stddev()) )) + log_probs.mean()
        # + jnp.mean(jnp.square(dist.mean() - batch["actions"]))
        # + 0.1*jnp.mean(jnp.square(dist.mode()-batch["actions"]))
        # standard_deviation=jnp.mean(dist.stddev(),axis=0)
        # + jnp.mean(jnp.square(dist.mean() - batch["actions"]))
        # + 0.1*jnp.mean(jnp.square(dist.mode()-batch["actions"]))
        # entropy = dist.entropy().mean()
        # actor_loss = jnp.mean(jnp.square(dist.mode()-batch["actions"])) - entropy
        actor_loss=-log_probs.mean()
        return actor_loss,({"bc_loss": actor_loss},updates)

    grads, (info,updates) = jax.grad(loss_fn, has_aux=True)(actor.params,actor.batch_stats)
    new_actor = actor.apply_gradients(grads=grads)
    new_actor = new_actor.replace(batch_stats=updates['batch_stats'])
    return rng, new_actor, info

@partial(jax.jit, static_argnames=("train_encoder"))
def _eval_jit(
    rng: PRNGKey, actor: TrainState, batch: TrainState,train_encoder:bool=True,
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str, float]]:
    rng, key = jax.random.split(rng)
    # if augument:
    # rng, batch = augment_batch(key, batch,batched_random_crop)
    # rng, key = jax.random.split(rng)
    # rng, batch = augment_batch(key, batch,batched_random_cutout)
    # rng, key = jax.random.split(rng)

    def loss_fn(actor_params: Params,batch_stats:Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        dist,updates = actor.apply_fn(
            {"params": actor_params,"batch_stats":batch_stats},
            batch["observations"],
            training=False,
            train_encoder=train_encoder,
            rngs={"dropout": key},
            mutable=['batch_stats']
        )
        log_probs = dist.log_prob(batch["actions"]) 
        
        # log_probs = log_probs.mean() 
        # clipped_log_probs=jnp.clip(log_probs,-2.9,0)
        # actor_loss = jnp.mean(jnp.square(dist.mean() - batch["actions"]))
        actor_loss = log_probs.mean() 
        # standard_deviation=jnp.mean(dist.stddev(),axis=0)
        # + jnp.mean(jnp.square(dist.mean() - batch["actions"]))
        # + 0.1*jnp.mean(jnp.square(dist.mode()-batch["actions"]))
        return actor_loss,({"bc_loss": actor_loss},updates)

    loss,(info,_)=loss_fn(actor.params,actor.batch_stats)
    # new_actor = actor.apply_gradients(grads=grads)
    # new_actor = new_actor.replace(batch_stats=updates['batch_stats'])
    return rng,info 


class PixelResNetBCLearner(Agent):
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


        encoder_def = partial(PretrainedResNet)

        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)
        policy_def = VariableStdNormalPolicy(
            hidden_dims, action_dim, dropout_rate=dropout_rate ,use_layer_norm=False,activations=nn.relu
        )
        actor_def = PixelMultiplexer(
            encoder=encoder_def, network=policy_def, latent_dim=latent_dim)
        
        params = actor_def.init(actor_key, observations)
        actor_params=params["params"]
        batch_stats=params["batch_stats"]
        # warmup_steps = 2*int(4e5)  # Adjust based on your dataset size
        # total_steps = 10*int(4e5)  # Total training steps planned

        # lr_schedule = optax.warmup_cosine_decay_schedule(
        #     init_value=0.0,
        #     peak_value=actor_lr,
        #     warmup_steps=warmup_steps,
        #     decay_steps=total_steps - warmup_steps,
        #     end_value=actor_lr * 0.1  # Final learning rate will be 10% of peak
        # )
  
        # Create TrainState with the scheduler
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            batch_stats=batch_stats,
            tx=optax.adam(learning_rate=actor_lr),
        )

        # Create TrainState with the scheduler
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            batch_stats=batch_stats,
            tx=optax.adam(learning_rate=actor_lr),
        )
        self._rng = rng
        self._actor = actor

    def update(self, batch: FrozenDict,train_encoder:bool=True) -> Dict[str, float]:
        new_rng, new_actor, info = _update_jit(self._rng, self._actor, batch,train_encoder=train_encoder)

        self._rng = new_rng
        self._actor = new_actor

        return info
    
    def eval(self, batch: FrozenDict,train_encoder:bool=True) -> Dict[str, float]:
        rng,info =_eval_jit(self._rng, self._actor, batch,train_encoder=train_encoder)
        self._rng = rng
        return info
    
    def extract_features(self, batch:DatasetDict):
        return np.asarray(extract_feature(self._actor.apply_fn, self._actor.params,self._actor.batch_stats,batch))

    def action_dist(self, observations: np.ndarray):
        return action_dist_jit(self._actor.apply_fn, self._actor.params,self._actor.batch_stats, observations)

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        actions = eval_actions_jit(
                    self._actor.apply_fn, self._actor.params,self._actor.batch_stats,observations
                )
        return np.asarray(actions)
    

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        rng, actions = sample_actions_jit(
            self._rng, self._actor.apply_fn, self._actor.params,self._actor.batch_stats, observations
        )
        self._rng = rng
        return np.asarray(actions)