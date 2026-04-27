from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
from torch.utils.tensorboard import SummaryWriter
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
# import dm
from jaxrl2.networks.constants import default_init
import numpy as np

def is_image_space(observation: jnp.ndarray) -> bool:
    """Check if an observation is image-like (has 3+ dimensions)."""
    try:
        return len(observation.shape) >= 3
    except:
        breakpoint()

def process_observation(observation: Union[jnp.ndarray, Dict, FrozenDict]) -> Dict:
    """Convert observation to consistent dict format."""
    if isinstance(observation, (dict, FrozenDict)):
        return observation
    else:
        # If single array passed, treat as primary observation
        return {'obs': observation}



# def augment_observations(
#     rng: jnp.ndarray,
#     observations: Union[np.ndarray, jnp.ndarray, Dict],
#     aug_func: None
# ) -> Tuple[jnp.ndarray, Union[jnp.ndarray, Dict]]:

#     # Handle direct array input
#     if isinstance(observations, (np.ndarray, jnp.ndarray)):
#         if is_image_space(observations):
#             rng, split_rng = jax.random.split(rng)
#             return rng, aug_func(split_rng, observations)
#         return rng, observations
#     # Process dictionary observations
#     new_observations = observations.copy()

#     # Iterate through observations and augment image-like ones
#     for key, value in observations.items():
#         if is_image_space(value):
#             rng, split_rng = jax.random.split(rng)
#             aug_value = aug_func(split_rng, value)
#             new_observations = new_observations.copy(add_or_replace={key: aug_value})
    
#     return rng, new_observations



def augment_observations(
    rng: jnp.ndarray,
    observations: Union[np.ndarray, jnp.ndarray, Dict],
    aug_func: None
) -> Tuple[jnp.ndarray, Union[jnp.ndarray, Dict]]:

    # Handle direct array input
    if isinstance(observations, (np.ndarray, jnp.ndarray)):
        if is_image_space(observations):
            rng, split_rng = jax.random.split(rng)
            
            return rng, aug_func(split_rng, observations)
        return rng, observations
    
    # Process dictionary observations
    new_observations = observations.copy()

    # Iterate through observations and augment image-like ones
    for key, value in observations.items():
        if is_image_space(value):
            rng, split_rng = jax.random.split(rng)
            aug_value = aug_func(split_rng, value)
            new_observations = new_observations.copy(add_or_replace={key: aug_value})
    return rng, new_observations


def augment_state(
    rng: jnp.ndarray,
    observations: Union[np.ndarray, jnp.ndarray, Dict],
    aug_func: None
) -> Tuple[jnp.ndarray, Union[jnp.ndarray, Dict]]:

    # Handle direct array input
    if isinstance(observations, (np.ndarray, jnp.ndarray)):
        if not is_image_space(observations):
            rng, split_rng = jax.random.split(rng)
            
            return rng, aug_func(split_rng, observations)
        return rng, observations
    
    # Process dictionary observations
    new_observations = observations.copy()

    # Iterate through observations and augment image-like ones
    for key, value in observations.items():
        if  not is_image_space(value):
            rng, split_rng = jax.random.split(rng)
            aug_value = aug_func(split_rng, value)
            new_observations = new_observations.copy(add_or_replace={key: aug_value})
    return rng, new_observations


def augment_state_batch(
    rng: jnp.ndarray,
    batch: Dict,
    aug_func: None,
) -> Tuple[jnp.ndarray, Dict]:
    # Get observations and next_observations
    observations = batch["observations"]
    if "next_observations" in batch.keys():
        next_observations = batch["next_observations"]

    
    # Handle observations
    rng, aug_observations = augment_state(rng, observations, aug_func)
    new_batch = batch.copy(add_or_replace={"observations": aug_observations})
    
    # Handle next_observations
    # if "next_observations" in batch.keys():
    #     rng, aug_next_observations = augment_state(rng, next_observations, aug_func)
    #     new_batch = new_batch.copy(add_or_replace={"next_observations": aug_next_observations})
    return rng, new_batch



def augment_batch(
    rng: jnp.ndarray,
    batch: Dict,
    aug_func: None,
) -> Tuple[jnp.ndarray, Dict]:
    # Get observations and next_observations
    observations = batch["observations"]
    if "next_observations" in batch.keys():
        next_observations = batch["next_observations"]

    
    # Handle observations
    rng, aug_observations = augment_observations(rng, observations, aug_func)
    new_batch = batch.copy(add_or_replace={"observations": aug_observations})
    
    # Handle next_observations
    if "next_observations" in batch.keys():
        rng, aug_next_observations = augment_observations(rng, next_observations, aug_func)
        new_batch = new_batch.copy(add_or_replace={"next_observations": aug_next_observations})
    return rng, new_batch




    # th.autograd.set_detect_anomaly(True)
# import torch as th
import ssl
import os

from PIL import Image
# import torch
# from torch.utils.data import Dataset

import numpy as np
from jax.tree_util import tree_map
# from torch.utils import data
import os
from flax.core.frozen_dict import unfreeze
from flax.training import checkpoints
import numpy as np

def load_pretrained(checkpoint_dir,agent):
    loaded_params = checkpoints.restore_checkpoint(checkpoint_dir, target=None)
    actor_params = unfreeze(agent.actor.params)
    for k, v in loaded_params["critic"]["params"].items():
        if 'encoder' in k: #load only encoder
            actor_params[k] = v
    critic_params = unfreeze(agent.critic.params)

    return agent.replace(
        actor=agent.actor.replace(
            params=unfreeze(actor_params),
        ),
        critic=agent.critic.replace(
            params=unfreeze(critic_params),
        ),
        target_critic=agent.target_critic.replace(
            params=unfreeze(critic_params),
        ),
    )

def load_checkpoints(checkpoint_dir,agent):
    loaded_params = checkpoints.restore_checkpoint(checkpoint_dir, target=agent)
    return loaded_params




class Logger:
    def __init__(self, log_dir: str,prefix:str=""):
        # Create timestamped log directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, prefix+timestamp)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Store metrics for console printing
        self.train_metrics = {}
        self.eval_metrics = {}
        self.episode_metrics = {}
        
        print(f"\nLogging to: {self.log_dir}\n")
    
    def log_training(self, metrics: Dict[str, Any], step: int,prefix=""):
        """Log training metrics to both tensorboard and console."""
        for k, v in metrics.items():
            self.writer.add_scalar(f"training{prefix}/{k}", np.array(v), step)
            self.train_metrics[f"{k}{prefix}"] = np.array(v)
    
    def log_eval(self, metrics: Dict[str, Any], step: int):
        """Log evaluation metrics to both tensorboard and console."""
        for k, v in metrics.items():
            self.writer.add_scalar(f"evaluation/{k}",np.array(v), step)
            self.eval_metrics[k] = np.array(v)
    
    def log_episode(self, metrics: Dict[str, Any], step: int):
        """Log episode metrics to both tensorboard and console."""
        for k, v in metrics.items():
            self.writer.add_scalar(f"episode/{k}", np.array(v), step)
            self.episode_metrics[k] = np.array(v)
    
    def print_status(self, step: int, total_steps: int):
        """Print current status in a nicely formatted way."""
        print("\n" + "="*80)
        print(f"Step: {step}/{total_steps} ({step/total_steps*100:.1f}%)")
        
        if self.train_metrics:
            print("\nTraining Metrics:")
            for k, v in self.train_metrics.items():
                print(f"  {k:<20} {np.array(v):>10.4f}")
        
        if self.episode_metrics:
            print("\nLatest Episode:")
            for k, v in self.episode_metrics.items():
                print(f"  {k:<20} {np.array(v):>10.4f}")
        
        if self.eval_metrics:
            print("\nLatest Evaluation:")
            for k, v in self.eval_metrics.items():
                print(f"  {k:<20} {np.array(v):>10.4f}")
        
        print("="*80 + "\n")