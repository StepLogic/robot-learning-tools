from typing import Dict, Tuple

import distrax
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey




def update_critic(
    critic: TrainState,
    batch: DatasetDict,
    vf_coeff:float
) -> Tuple[TrainState, Dict[str, float]]:
    # dist:distrax.Distribution = actor.apply_fn({"params": actor.params}, batch["observations"])
    # dist.sample_and_log_prob(seed=key)
    # breakpoint()
    # logp = dist.log_prob(batch["actions"])
    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        values = critic.apply_fn(
            {"params": critic_params}, batch["observations"]
        )
        returns=batch["returns"]
        values = batch["values"] + jnp.clip(
            values - batch["values"],
            -0.2,
            0.2,
        )
        critic_loss = vf_coeff*((values - returns) ** 2).mean()
        # v_loss_unclipped = (newvalue - ret) ** 2
 

        # critic_loss = jnp.maximum(critic_loss, critic_loss_clipped)
        critic_loss = critic_loss.mean()
        return critic_loss, {
            "critic_loss": critic_loss,
            "value": values.mean(),
            "returns":returns.mean()
        }

    grads, info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    new_critic = critic.apply_gradients(grads=grads)
    return new_critic, info
