from typing import Dict, Tuple

import distrax
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey

# import jax
# import jax.numpy as jnp
# import distrax
# from typing import Tuple, Dict
# from flax.training.train_state import TrainState
# from chex import PRNGKey, Array, DatasetDict

def update_actor_critic(
    key: PRNGKey,
    actor_critic: TrainState,
    batch: DatasetDict,
    clip_ratio: float,
    target_kl: float,
    ent_coeff: float,
    vf_coeff: float,
    utd_ratio: int = 1,
) -> Tuple[TrainState, TrainState, Dict[str, float]]:
    

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        values = actor_critic.critic_fn(
            {"params": critic_params}, batch["observations"]
        )
        returns=batch["returns"]
        critic_loss = ((values - returns) ** 2).mean()
        # v_loss_unclipped = (newvalue - ret) ** 2
        critic_loss_clipped = batch["values"] + jnp.clip(
            values - batch["values"],
            -0.2,
            0.2,
        )

        critic_loss = jnp.maximum(critic_loss, critic_loss_clipped)
        critic_loss = 0.5 * critic_loss.mean()
        return critic_loss, {
            "critic_loss": critic_loss,
            "value": values.mean(),
            "returns":returns.mean()
        }


    def actor_loss_fn(actor_params)-> Tuple[jnp.ndarray, Dict[str, float]]:
        actions ,advantages,logp_old =  batch['actions'],batch["advantages"],batch["logps"]
        
        # jax.debug.print("{advantages}",advantages=advantages)
        # if normalize_advantage and len(advantages) > 1:
        #     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        dist:distrax.Distribution = actor_critic.actor_fn({"params": actor_params}, batch["observations"])
        logp = dist.log_prob(actions)
        # breakpoint()
        # ratio between old and new policy, should be one at the first iteration
        ratio = jnp.exp(logp - logp_old)
        
        clip_adv = jnp.clip(ratio, 1-clip_ratio, 1+clip_ratio) * advantages
        policy_loss= -jnp.minimum(ratio * advantages, clip_adv).mean()
        # entropy_loss = -jnp.mean(entropy)
        if  hasattr(dist,"entropy"):
            entropy_loss= -jnp.mean(dist.entropy())
        else:
            entropy_loss = -jnp.mean(-logp)
        actor_loss = policy_loss + ent_coeff * entropy_loss

        # Useful extra info
        clipped = jnp.logical_or(
            ratio > (1 + clip_ratio),
            ratio < (1 - clip_ratio)
        )
        clipfrac =jnp.mean(clipped)
        log_ratio = logp - logp_old
        approx_kl = jnp.mean((jnp.exp(log_ratio) - 1) - log_ratio)
        return actor_loss,  {"actor_loss": actor_loss,"entropy_loss":entropy_loss,"mean_advantage":advantages.mean(),"entropy": -logp.mean(),"kl":approx_kl,"cf":clipfrac}
    def loss_fn(params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        actor_loss, actor_info = actor_loss_fn(params["params"]["actor"])
        critic_loss, critic_info = critic_loss_fn(params["params"]["critic"])
        loss = actor_loss + critic_loss
        info = {**actor_info, **critic_info,"loss":loss}
        return loss, info
    
    grads, info = jax.grad(loss_fn, has_aux=True)(actor_critic.params)
    # kl = info["kl"]
    # should_continue = kl <= 1.5 * target_kl
    # new_actor = jax.lax.cond(
    #             should_continue,
    #             lambda: actor.apply_gradients(grads=grads),
    #             lambda: actor
    #         )
    new_actor_critic=actor_critic.apply_gradients(grads=grads)
    return new_actor_critic,info