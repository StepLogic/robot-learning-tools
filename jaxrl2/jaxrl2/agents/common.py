from functools import partial
from gc import unfreeze
from typing import Callable, Tuple

import distrax
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict,freeze,unfreeze
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey


@partial(jax.jit, static_argnames="actor_apply_fn")
def eval_log_prob_jit(
    actor_apply_fn: Callable[..., distrax.Distribution],
    actor_params: Params,
    batch: DatasetDict,
) -> float:
    dist = actor_apply_fn({"params": actor_params}, batch["observations"])
    log_probs = dist.log_prob(batch["actions"])
    return log_probs.mean()


@partial(jax.jit, static_argnames="actor_apply_fn")
def eval_actions_jit(
    actor_apply_fn: Callable[..., distrax.Distribution],
    actor_params: Params,
    observations: np.ndarray,
) -> jnp.ndarray:
    dist= actor_apply_fn({"params": actor_params},observations)
    return dist.mode()


@partial(jax.jit, static_argnames="actor_apply_fn")
def action_dist_jit(
    actor_apply_fn: Callable[..., distrax.Distribution],
    actor_params: Params,
    observations: np.ndarray,
) -> jnp.ndarray:
    dist = actor_apply_fn({"params": actor_params}, observations)
    return dist

@partial(jax.jit, static_argnames="actor_apply_fn")
def extract_feature(
    actor_apply_fn: Callable[..., distrax.Distribution],
    actor_params: Params,
    observations: np.ndarray,
) -> jnp.ndarray:
    (_,outputs) = actor_apply_fn({"params": actor_params}, observations,mutable='intermediates')
    # breakpoint()
    features = outputs['intermediates']['features']
    return features

@partial(jax.jit, static_argnames="actor_apply_fn")
def sample_actions_jit(
    rng: PRNGKey,
    actor_apply_fn: Callable[..., distrax.Distribution],
    actor_params: Params,
    observations: np.ndarray,
) -> Tuple[PRNGKey, jnp.ndarray]:
    dist = actor_apply_fn({"params": actor_params}, observations)
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)

@partial(jax.jit, static_argnames=("actor_apply_fn","critic_apply_fn"))
def sample_actions_log_probs_values_jit(
    rng: PRNGKey,
    actor_apply_fn: Callable[..., distrax.Distribution],
    critic_apply_fn: Callable[...,np.ndarray],
    actor_params: Params,
    critic_params:Params,
    observations: np.ndarray,
) -> Tuple[PRNGKey, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    
    dist = actor_apply_fn({"params": actor_params}, observations)
    rng, key = jax.random.split(rng)
    actions = dist.sample(seed=key)
    log_probs = dist.log_prob(actions)
    values = critic_apply_fn({"params": critic_params}, observations)
    # values=jnp.mean(values,axis=0)
    return rng,actions,log_probs,values 


def _share_encoder(source, target):
    replacers = {}
    for k, v in source.params.items():
        if "encoder" in k:
            replacers[k] = v
            # print(k)
    # Use critic conv layers in actor:
    new_params = unfreeze(FrozenDict(target.params).copy(add_or_replace=replacers))
    return target.replace(params=new_params)
