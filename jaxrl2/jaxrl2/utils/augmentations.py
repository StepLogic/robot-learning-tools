from typing import Dict, Tuple
import cv2
import dm_pix
import jax
import jax.numpy as jnp
from jaxrl2.utils.misc import is_image_space
import numpy as np
# import dm_p
import random
def random_crop(key, img, padding):
    crop_from = jax.random.randint(key, (2,), 0, 2 * padding + 1)
    crop_from = jnp.concatenate([crop_from, jnp.zeros((2,), dtype=jnp.int32)])
    padded_img = jnp.pad(
        img, ((padding, padding), (padding, padding), (0, 0), (0, 0)), mode="edge"
    )
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


def batched_random_crop(key, imgs, padding=4):
    keys = jax.random.split(key, imgs.shape[0])
    return jax.vmap(random_crop, (0, 0, None))(keys, imgs, padding)


def add_noise_to_states(key,states, noise_scale=0.09):
    noise = jax.random.normal(key, shape=states.shape) * noise_scale
    return states + noise

def batched_add_noise(key,states, noise_scale=0.09):
    keys = jax.random.split(key, states.shape[0])
    return jax.vmap(add_noise_to_states, (0, 0, None))(keys, states,noise_scale)


def random_cutout(key, img, max_size=16):
    """Apply random cutout to a single image.
    
    Args:
        key: JAX random key
        img: Input image array of shape (H, W, C)
        max_size: Maximum size of cutout square
    
    Returns:
        Image with random cutout applied
    """
    h, w = img.shape[:2]
    keys = jax.random.split(key, 2)
    
    # Sample cutout location
    start_x = jax.random.randint(keys[0], (), 0, w - max_size + 1)
    start_y = jax.random.randint(keys[1], (), 0, h - max_size + 1)
    
    # Create mask directly using dynamic_update_slice
    mask = jnp.ones_like(img)
    cutout = jnp.zeros((max_size, max_size) + img.shape[2:])
    
    # Place the cutout in the mask
    mask = jax.lax.dynamic_update_slice(
        mask, 
        cutout,
        (start_y, start_x) + (0,) * len(img.shape[2:])
    )
    # cv2.imwrite("sample.jpg",(np.array(img*mask)*255).astype(np.uint8))
    return img * mask

def batched_random_cutout(key, imgs, max_size=16):
    return jax.vmap(lambda k, x: random_cutout(k, x, max_size))(
        jax.random.split(key, imgs.shape[0]), 
        imgs
    )


def process_image(key, img):
    keys = jax.random.split(key, 3)
    img = dm_pix.random_brightness(keys[0], img, max_delta=0.2)
    img = dm_pix.random_contrast(keys[1], img, lower=0.9, upper=1.1)
    return img

def batch_bc_augmentation(key, imgs):
    keys = jax.random.split(key, imgs.shape[0])
    return jax.vmap(process_image)(keys, imgs)


def horizontal_flip_augmentation(key, observations, actions, flip_probability=0.5):
    batch_size = observations.shape[0]

    # Generate random values to determine which samples to flip
    flip_key = jax.random.split(key, 1)[0]  # Split key for flipping
    flip_mask = jax.random.uniform(flip_key, shape=(batch_size,)) < flip_probability

    # Expand dimensions for broadcasting with observations
    flip_mask_expanded = flip_mask[:, None, None, None,None]

    # Flip observations horizontally (along width dimension)
    flipped_observations = jnp.flip(observations, axis=2)
    augmented_observations = jnp.where(flip_mask_expanded, flipped_observations, observations)

    # Flip just the steering component of actions (negate it)
    # Create a mask to only flip steering (first component), not throttle
    action_flip_mask = jnp.tile(flip_mask[:, None], (1, 2)) * jnp.array([1.0, 0.0])

    # Where mask is 1, multiply by -1 (steering), where 0 multiply by 1 (throttle)
    flip_multiplier = 1.0 - 2.0 * action_flip_mask
    augmented_actions = actions * flip_multiplier

    return augmented_observations, augmented_actions

def bc_augment_random_shift(rng, observations, actions):
    
    # Apply horizontal flipping augmentation
    # flip_key, rng = jax.random.split(key)
    # observations, actions = horizontal_flip_augmentation(
    #     flip_key, observations, actions, flip_probability=0.5
    # )

    # Add noise to observations for better generalization
    # noise = jax.random.normal(noise_key, shape=observations.shape) * 0.01
    # observations = observations + noise
    # Handle direct array input
    if isinstance(observations, (np.ndarray, jnp.ndarray)):
        if is_image_space(observations):
            rng, split_rng = jax.random.split(rng)
            return rng, horizontal_flip_augmentation(
        split_rng, observations, actions, flip_probability=0.5
    )
        return rng, observations,actions
    
    # Process dictionary observations
    new_observations = observations.copy()

    # Iterate through observations and augment image-like ones
    for key, value in observations.items():
        if  is_image_space(value):
            rng, split_rng = jax.random.split(rng)
            aug_value,actions = horizontal_flip_augmentation(
                    split_rng, value, actions, flip_probability=0.5
                )
            new_observations = new_observations.copy(add_or_replace={key: aug_value})
    return rng, new_observations,actions



def augment_bc_random_shift_batch(
    rng: jnp.ndarray,
    batch: Dict,
) -> Tuple[jnp.ndarray, Dict]:
    # Get observations and next_observations
    observations = batch["observations"]
    actions=batch["actions"]
    # if "next_observations" in batch.keys():
    #     next_observations = batch["next_observations"]
        

    
    # Handle observations
    rng, aug_observations,actions = bc_augment_random_shift(rng, observations, actions)
    new_batch = batch.copy(add_or_replace={"observations": aug_observations,"actions":actions})
    
    # Handle next_observations
    # if "next_observations" in batch.keys():
    #     rng, aug_next_observations = bc_augment_random_shift(rng, next_observations, aug_func)
    #     new_batch = new_batch.copy(add_or_replace={"next_observations": aug_next_observations})
    return rng, new_batch



# def adjust_brightness(image: jnp.Array, delta:float):
#   return image + jnp.asarray(delta, image.dtype)

# def random_brightness(
#     key: jax.PRNGKey,
#     image: jnp.Array,
#     max_delta: float=0.1,
# ) -> jnp.Array:
#   """`adjust_brightness(...)` with random delta in `[-max_delta, max_delta)`."""
#   # DO NOT REMOVE - Logging usage.
#   delta = jax.random.uniform(key, (), minval=-max_delta, maxval=max_delta)
#   return adjust_brightness(image, delta)

# def batched_random_cutout(key, imgs):
#     keys = jax.random.split(key, imgs.shape[0])
#     rng = jax.random.uniform(key)
   
#     return jax.lax.cond(
#         rng < 0.1,
#         lambda x: jax.vmap(random_cutout, (0,0))(keys, imgs),
#         lambda x: x,
#         imgs)

