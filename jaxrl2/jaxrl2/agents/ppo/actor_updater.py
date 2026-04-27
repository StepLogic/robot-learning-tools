from typing import Dict, Tuple

import distrax
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax.core.frozen_dict import freeze,unfreeze
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey
import numpy as np


# def compute_advantage(
#     critic:TrainState,
#     batch: DatasetDict,
#     discount:float,
#     gae_lambda:float
# ) -> Tuple[jnp.ndarray, jnp.ndarray]:
#     values=critic.apply_fn(
#             {"params": critic.params}, batch["observations"]
#         )
#     values=jnp.mean(values,axis=0)
#     mask=batch['masks']
#     rewards=batch["rewards"]
#     last_gae_lam = 0
#     batch_size=jnp.size(mask)
#     advantages=jnp.array([])
#     for step in reversed(range(batch_size)):
#         if step == batch_size - 1:
#             next_non_terminal = mask[step]
#             next_values = jnp.where(mask[step] == 0, 0.0, values[step])
#         else:
#             next_non_terminal = mask[step + 1]
#             next_values = values[step + 1]
#         # breakpoint()
#         delta = rewards[step] + discount * next_values * next_non_terminal - values[step]
#         last_gae_lam = delta + discount * gae_lambda * next_non_terminal * last_gae_lam
#         advantages = jnp.append(advantages,last_gae_lam)
#     # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
#     # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
#     # breakpoint()
#     returns = advantages + values
#     return returns,advantages


def update_actor(
    actor: TrainState,
    batch: DatasetDict,
    clip_ratio:float,
    target_kl:float,
    ent_coeff:float,
    utd_ratio:int=1,
    normalize_advantage:bool=True
) -> Tuple[TrainState, Dict[str, float]]:
   
    def actor_loss_fn(actor_params)-> Tuple[jnp.ndarray, Dict[str, float]]:
        actions ,advantages,logp_old =  batch['actions'],batch["advantages"],batch["logps"]
        
        # jax.debug.print("{advantages}",advantages=advantages)
        if normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        dist:distrax.Distribution = actor.apply_fn({"params": actor_params}, batch["observations"])
        logp = dist.log_prob(actions)
        # breakpoint()
        # ratio between old and new policy, should be one at the first iteration
        ratio = jnp.exp(logp - logp_old)
        # jax.debug.print("ratios {ratio} logps_old {logp_old} logps {logps}",logp_old=logp_old,logps=logp,ratio=ratio)
     
        clip_adv = jnp.clip(ratio, 1-clip_ratio, 1+clip_ratio) * advantages
        policy_loss= -jnp.minimum(ratio * advantages, clip_adv).mean()
        # entropy_loss = -jnp.mean(entropy)
        # # if  hasattr(dist,"entropy"):
        entropy_loss= -jnp.mean(dist.entropy())
        # # else:
        # entropy_loss = -jnp.mean(-logp)
        actor_loss = policy_loss + ent_coeff * entropy_loss

        # Useful extra info
        clipped = jnp.logical_or(
            ratio > (1 + clip_ratio),
            ratio < (1 - clip_ratio)
        )
        clipfrac =jnp.mean(clipped)
        log_ratio = logp - logp_old
        # approx_kl = jnp.mean((jnp.exp(log_ratio) - 1) - log_ratio)
        approx_kl = jnp.mean(logp_old - logp)
       
        return actor_loss,  {"actor_loss": actor_loss,"mean_advantage":advantages.mean(),"entropy_loss": entropy_loss,"kl":approx_kl,"cf":clipfrac}

    # def loop_body(i, carry):
    #         actor,info = carry
    #         grads, _info = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    #         kl = _info["kl"]
    #         # Early stopping condition
    #         should_continue = kl <= 1.5 * target_kl
    #         # Update actor if continuing
    #         grads = jax.tree_util.unfreeze(grads) if not isinstance(grads, dict) else grads
    #         actor = jax.lax.cond(
    #             should_continue,
    #             lambda: actor.apply_gradients(grads=grads),
    #             lambda: actor
    #         )
    #         info.update(_info)
    #         return (actor,info)
    # info=dict(actor_loss=0.0,entropy=0.0,kl=0.0,cf=0.0)
    # actor, info=jax.lax.fori_loop(0, utd_ratio, loop_body, (actor,info))
        #         kl = _info["kl"]
    #         # Early stopping condition

    grads, info = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    # kl = info["kl"]
    # should_continue = kl <= 1.5 * target_kl
    # new_actor = jax.lax.cond(
    #             should_continue,
    #             lambda: actor.apply_gradients(grads=grads),
    #             lambda: actor
    #         )
    new_actor=actor.apply_gradients(grads=grads)
    return new_actor, info
