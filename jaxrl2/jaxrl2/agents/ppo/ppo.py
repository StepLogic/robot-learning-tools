"""Implementations of algorithms for continuous control."""

import copy
import functools
from typing import Callable, Dict, Optional, Sequence, Tuple, Union
import jax
import jax.numpy as jnp
from jaxrl2.agents.common import _share_encoder, sample_actions_jit, sample_actions_log_probs_values_jit
from jaxrl2.agents.ppo.actor_critic_updater import update_actor_critic
from jaxrl2.agents.ppo.actor_updater import update_actor
from jaxrl2.agents.ppo.critic_updater import update_critic
from jaxrl2.networks.encoders.placeholder import PlaceholderEncoder
from jaxrl2.networks.normal_policy import VariableStdNormalPolicy
from jaxrl2.networks.values.state_value import StateValue
from jaxrl2.utils.misc import augment_batch
import optax
from flax.core.frozen_dict import FrozenDict,unfreeze,freeze
from flax.training.train_state import TrainState
import numpy as np
from jaxrl2.agents.agent import Agent
from jaxrl2.utils.augmentations import batched_random_crop
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.networks.encoders import D4PGEncoder, ResNetV2Encoder
from jaxrl2.networks.normal_tanh_policy import NormalTanhPolicy
from jaxrl2.networks.gsde_policy import GSDEPolicy
from jaxrl2.networks.pixel_multiplexer import PixelMultiplexer
from jaxrl2.networks.values import StateValueEnsemble
from jaxrl2.types import Params, PRNGKey
from functools import partial
import flax.linen as nn



@functools.partial(jax.jit, static_argnames=("utd_ratio","target_kl","vf_coeff","ent_coeff","gae_lambda","clip_ratio","augument"))
def _update_jit(
    rng: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    batch: FrozenDict,
    discount: float,
    clip_ratio:float,
    target_kl:float,
    vf_coeff:float,
    ent_coeff:float,
    gae_lambda:float,
    utd_ratio=1,
    augument=True
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str, float]]:

    rng, key = jax.random.split(rng)    
    # rng, batch = augment_batch(key, batch,batched_random_crop)
    # rng, key = jax.random.split(rng)
    # batch=unfreeze(batch)
    # batch["returns"],batch["advantages"]=compute_advantage(
    #     critic,batch,discount,gae_lambda
    # )
    # batch=freeze(batch)
    # rng, key = jax.random.split(rng)
    # new_actor, actor_info = update_actor(key,actor,batch,clip_ratio,target_kl,utd_ratio)
    # rng, key = jax.random.split(rng)
    # new_actor,new_critic, info=update_actor_critic(key,actor,critic,batch,clip_ratio,target_kl,ent_coeff,vf_coeff,utd_ratio)
    # if augument:
    #     rng, batch = augment_batch(key, batch,batched_random_crop)
    new_info={}
    new_critic, critic_info = update_critic(critic, batch, vf_coeff)
    new_actor, actor_info = update_actor(actor, batch, clip_ratio, target_kl, ent_coeff, utd_ratio)
    # new_info = info.copy()
    new_info.update(actor_info)
    new_info.update(critic_info)
    
    # def loop_body(i, carry: Tuple[TrainState, TrainState, Dict]) -> Tuple[TrainState, TrainState, Dict]:
    #     actor, critic, info = carry
        
    #     def do_update():
    #         new_actor, actor_info = update_actor(actor, batch, clip_ratio, target_kl, ent_coeff, utd_ratio)
    #         new_critic, critic_info = update_critic(critic, batch, vf_coeff)
            
    #         new_info = info.copy()
    #         new_info.update(actor_info)
    #         new_info.update(critic_info)
            
    #         return (new_actor, new_critic, new_info)
        
    #     def skip_update():
    #         return carry
        
    #     # Check KL threshold only after first iteration
    #     should_continue = jnp.logical_or(
    #         i == 0,
    #         info.get("kl", 0.0) <= 1.5 * target_kl
    #     )
        
    #     return jax.lax.cond(
    #         should_continue,
    #         do_update,
    #         skip_update
    #     )
    # new_info = dict(actor_loss=0.0, entropy=0.0, kl=0.0, cf=0.0,q=0.0,critic_loss=0.0)
    # new_actor, new_critic, new_info = jax.lax.fori_loop(0, utd_ratio, loop_body, (actor, critic, new_info))
    new_actor = _share_encoder(source=new_critic, target=new_actor)
    return (
        rng,
        new_actor,
        new_critic,
        # {**actor_info,**critic_info},
        {**new_info}
    )


class PPOLearner(Agent):
    def __init__(
        self,
        observations: Union[jnp.ndarray, DatasetDict],
        actions: jnp.ndarray,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        cnn_features: Sequence[int] = (32, 32, 32, 32),
        cnn_filters: Sequence[int] = (3, 3, 3, 3),
        cnn_strides: Sequence[int] = (2, 1, 1, 1),
        cnn_padding: str = "VALID",
        latent_dim: int = 50,
        discount: float = 0.99,
        gae_lambda:float=0.95,
        target_kl=0.015,
        ent_coeff:float=0.01,
        vf_coeff:float=0.5,
        clip_ratio: float = 0.2,
        critic_reduction: str = "min",
        encoder: str = "d4pg",

        max_grad_norm=0.5,
        num_qs=2,
        seed: int=0 
    ):
        """
        An implementation of the version of Proximal Policy Optimization described in https://arxiv.org/abs/1707.06347
        See here #https://costa.sh/blog-the-32-implementation-details-of-ppo.html
        
        """
        action_dim = actions.shape[-1]
        self.critic_reduction = critic_reduction
        self.clip_ratio = clip_ratio
        self.target_kl=target_kl
        self.discount = discount
        self.vf_coeff=vf_coeff
        self.ent_coeff=ent_coeff
        self.gae_lambda=gae_lambda
        rng = jax.random.PRNGKey(0)
        rng, actor_key, critic_key = jax.random.split(rng, 3)

        if encoder == "d4pg":
            encoder_def = partial(
                D4PGEncoder,
                features=cnn_features,
                filters=cnn_filters,
                strides=cnn_strides,
                padding=cnn_padding,
                kernel_init=lambda:nn.initializers.orthogonal(jnp.sqrt(2)),
            )
        elif encoder == "resnet":
            encoder_def = partial(ResNetV2Encoder, stage_sizes=(2, 2, 2, 2))
        else:
            encoder_def = partial(PlaceholderEncoder)
        self.augument=not encoder is None
        policy_def = VariableStdNormalPolicy(hidden_dims, action_dim,
                                             activations=nn.tanh,
                                            kernel_init=lambda:nn.initializers.orthogonal(jnp.sqrt(2)),
                                            bias_init=lambda:nn.initializers.constant(0.0)
                                             )
        # policy_def = NormalTanhPolicy(hidden_dims, action_dim,activations=nn.tanh,
        #                                     # kernel_init=lambda:nn.initializers.orthogonal(jnp.sqrt(2)),
        #                                     # bias_init=lambda:nn.initializers.constant(0.0)
        #                                      )
        actor_def = PixelMultiplexer(
            encoder=encoder_def,
            network=policy_def,
            latent_dim=latent_dim,
            stop_gradient=True,
            # kernel_init=lambda:nn.initializers.orthogonal(jnp.sqrt(2)),
            # bias_init=lambda:nn.initializers.constant(0.0)
        )
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx= optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
            learning_rate=actor_lr,
            eps=1e-5
        )
    )
        )

        # critic_def = StateValueEnsemble(hidden_dims, num_qs=max(num_qs,2))
        critic_def=StateValue(hidden_dims,activations=nn.relu,
                                kernel_init=lambda:nn.initializers.orthogonal(jnp.sqrt(2)),
                                bias_init=lambda:nn.initializers.constant(0.0),
                                # num_qs=max(num_qs,2)
                              )
        critic_def = PixelMultiplexer(
            encoder=encoder_def, 
            network=critic_def, 
            latent_dim=latent_dim,
            # kernel_init=lambda:nn.initializers.orthogonal(jnp.sqrt(2)),
            # bias_init=lambda:nn.initializers.constant(0.0)
        )
        critic_params = critic_def.init(critic_key, observations)["params"]
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx= optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
            learning_rate=critic_lr,
            eps=1e-5
        ),
        ))

        # critic = Critic(n_units=64, activation_fn=nn.tanh)

        # critic = TrainState.create(
        #     apply_fn=critic.apply,
        #     params=critic.init({"params": critic_key}, observations)["params"],
        #     tx=optax.chain(
        #     optax.clip_by_global_norm(max_grad_norm),
        #     optax.inject_hyperparams(optax.adamw)(
        #     learning_rate=critic_lr,
        #     eps=1e-5
        # ),
        # ))
        # actor = _share_encoder(source=critic, target=actor)  #very important!!!!
        self._actor = actor
        self._critic = critic
        self._rng = rng
    
    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        # breakpoint()
        rng, actions,log_probs,values = sample_actions_log_probs_values_jit(
            self._rng, 
            self._actor.apply_fn,
            self._critic.apply_fn,
            self._actor.params,
            self._critic.params,
            observations
        )
        self._rng = rng
        return np.asarray(actions),np.array(log_probs),np.asarray(values)

    def update(self, batch: FrozenDict,
                utd_ratio: int=1,
                output_range: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None) -> Dict[str, float]:

        (
            new_rng,
            new_actor,
            new_critic,
            info,
        ) = _update_jit(
            self._rng,
            self._actor,
            self._critic,
            batch,
            self.discount,
            self.clip_ratio,
            self.target_kl,
            self.vf_coeff,
            self.ent_coeff,
            self.gae_lambda,
            utd_ratio=utd_ratio,
            augument=self.augument
        )
      

        self._rng = new_rng
        self._actor = new_actor
        self._critic = new_critic
        return info
