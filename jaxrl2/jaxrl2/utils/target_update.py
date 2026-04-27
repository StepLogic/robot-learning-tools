import jax
from  jax import tree_util
from jaxrl2.types import Params
import optax

def soft_target_update(
    critic_params: Params, target_critic_params: Params, tau: float
) -> Params:
    # new_target_params = tree_util.tree_map(
    #     lambda p, tp: p * tau + tp * (1 - tau), critic_params, target_critic_params
    # )
    target_critic_params = optax.incremental_update(
        critic_params, target_critic_params, tau
    )
    return target_critic_params
