#!/usr/bin/env python
"""
DrQ Training with Pre-trained Agent Initialization
- Load pre-trained checkpoint
- Collect initial data using trained agent
- Continue training from loaded checkpoint
"""
import random
from collections import deque
import os
from datetime import datetime
import tqdm
from typing import Dict, Any
import uuid
import pickle

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import random as jax_random
import numpy as np
from absl import app, flags
from flax.training import checkpoints
from ml_collections import config_flags
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from gymnasium import spaces

# PyTorch imports for MobileNetV3
import torch
import torch.nn as torch_nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import cv2

# JAX RL imports
from jaxrl2.agents import DrQLearner
from jaxrl2.data import ReplayBuffer
from jaxrl2.noise import OrnsteinUhlenbeckActionNoise
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from jaxrl2.wrappers.timelimit import TimeLimit

# Donkey Car env classes
from gym_donkeycar.envs.donkey_env import (
    OfficeEnv, WaveshareEnv, MiniMonacoEnv, WarehouseEnv
)

# Import wrappers
from wrappers import (
    EnvCompatibility, 
    Logger, 
    MobileNetFeatureWrapper, 
    StackingWrapper, 
    save_checkpoint,
    load_checkpoint
)

flax.config.update("flax_use_orbax_checkpointing", True)

FLAGS = flags.FLAGS

# Configuration flags
flags.DEFINE_string("env_name", "donkey-warehouse-v0", "Environment name.")
flags.DEFINE_string(
    "pretrained_checkpoint", 
    "/home/kojogyaase/Projects/Research/recovery-from-failure/checkpoints/drq_mobilenet_20260214_080226/step_45000/checkpoint_45000",
    "Path to pre-trained checkpoint to initialize from."
)
flags.DEFINE_boolean("load_pretrained", True, "Whether to load pre-trained checkpoint.")
flags.DEFINE_integer("data_collection_steps", 3000, "Steps to collect data with pre-trained agent before training.")
flags.DEFINE_integer("port", 9091, "Port to use for tcp.")
flags.DEFINE_string("save_dir", "./logs/", "Tensorboard logging dir.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", int(50000), "Eval interval.")
flags.DEFINE_integer("checkpoint_interval", 5000, "Checkpoint saving interval.")
flags.DEFINE_integer("batch_size", 16, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps (after data collection).")
flags.DEFINE_integer("start_training", int(1e3), "Additional steps before starting training updates.")
flags.DEFINE_integer("replay_buffer_size", int(1e5), "Replay buffer size.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_integer("frame_stack", 4, "Number of frames to stack for temporal modeling.")
flags.DEFINE_integer("mobilenet_blocks", 4, "Number of MobileNetV3 blocks to use.")
flags.DEFINE_integer("mobilenet_input_size", 84, "Input size for MobileNetV3.")

config_flags.DEFINE_config_file(
    "config",
    "./configs/drq_default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


# ============================================================================
# Data Collection with Pre-trained Agent
# ============================================================================

def collect_data_with_agent(
    agent,
    env,
    replay_buffer,
    num_steps: int,
    deterministic: bool = False,
    noise_scale: float = 0.1
):
    """
    Collect data using a pre-trained agent.
    
    Args:
        agent: Pre-trained DrQLearner agent
        env: Environment
        replay_buffer: Replay buffer to store transitions
        num_steps: Number of steps to collect
        deterministic: Whether to use deterministic policy
        noise_scale: Scale of exploration noise (if not deterministic)
    
    Returns:
        Number of episodes completed
    """
    print("\n" + "="*70)
    print(f"Collecting {num_steps:,} steps using pre-trained agent")
    print(f"Deterministic: {deterministic}, Noise scale: {noise_scale}")
    print("="*70 + "\n")
    
    observation, info = env.reset()
    episode_count = 0
    episode_return = 0.0
    episode_length = 0
    
    # Add noise for exploration during data collection
    if not deterministic:
        action_dim = env.action_space.shape[0]
        noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(action_dim),
            sigma=noise_scale * np.ones(action_dim)
        )
    
    for step in tqdm.tqdm(range(num_steps), desc="Data Collection"):
        # Select action using pre-trained agent
        if deterministic:
            action = agent.eval_actions(observation)
        else:
            action = agent.sample_actions(observation)
            action = action + noise()
            action = np.clip(action, env.action_space.low, env.action_space.high)
        
        # Execute action
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store transition
        mask = 0.0 if done else 1.0
        replay_buffer.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward,
                masks=mask,
                dones=done,
                next_observations=next_observation,
            )
        )
        
        episode_return += reward
        episode_length += 1
        observation = next_observation
        
        # Handle episode end
        if done:
            episode_count += 1
            if not deterministic:
                noise.reset()
            
            # Log episode
            if "episode" in info:
                ep_return = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                print(f"\rEpisode {episode_count}: Return={ep_return:.2f}, Length={ep_length}", 
                      end='', flush=True)
            
            observation, info = env.reset()
            episode_return = 0.0
            episode_length = 0
    
    print(f"\n\n[Data Collection] Completed!")
    print(f"  Steps collected: {num_steps:,}")
    print(f"  Episodes: {episode_count}")
    print(f"  Buffer size: {replay_buffer._size}")
    print("="*70 + "\n")
    
    return episode_count


# ============================================================================
# Main Training Loop
# ============================================================================

def main(_):
    print("\n" + "="*70)
    print("DrQ Training with Pre-trained Initialization")
    print("="*70 + "\n")
    
    # Environment configuration
    conf = {
        "host": "127.0.0.1",
        "port": FLAGS.port,
        "body_style": "f1",
        "body_rgb": (128, 128, 128),
        "car_name": "",
        "font_size": 100,
        "racer_name": "",
        "country": "USA",
        "bio": "Learning to drive with DrQ + MobileNetV3",
        "guid": str(uuid.uuid4()),
        "max_cte": 10,
        "frame_skip": 3,
    }

    # Set random seeds
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(FLAGS.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create environment with wrapper chain
    print("\nCreating environment...")
    old_gym_env = WaveshareEnv(conf=conf)
    env = EnvCompatibility(old_gym_env)
    
    # Stack RGB frames
    env = StackingWrapper(env, num_stack=FLAGS.frame_stack)
    
    # Extract MobileNetV3 features from stacked frames
    env = MobileNetFeatureWrapper(
        env,
        device=device,
        num_blocks=FLAGS.mobilenet_blocks,
        input_size=FLAGS.mobilenet_input_size
    )
    
    env = RecordEpisodeStatistics(env)
    env = TimeLimit(env, max_episode_steps=1000)
    
    print(f"\n{'='*60}")
    print("Environment Setup Complete")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"{'='*60}\n")
    
    # RL Training Setup
    action_dim = env.action_space.shape[0]
    mean = np.zeros(action_dim)
    sigma = 0.2 * np.ones(action_dim)
    noise = OrnsteinUhlenbeckActionNoise(mean=mean, sigma=sigma)

    logger = Logger(log_dir=FLAGS.save_dir)
    policy_folder = os.path.join(
        "checkpoints", 
        f"drq_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(policy_folder, exist_ok=True)

    # Initialize DrQ agent
    print("Initializing DrQ agent...")
    kwargs = dict(FLAGS.config) if FLAGS.config else {}
    
    sample_obs, _ = env.reset()
    sample_action = env.action_space.sample()
    
    print(f"Sample observation shapes:")
    print(f"  Pixels (features): {sample_obs['pixels'].shape}")
    print(f"  Actions: {sample_obs['actions'].shape}")
    print(f"Sample action shape: {sample_action.shape}")
    
    agent = DrQLearner(
        FLAGS.seed, 
        sample_obs, 
        sample_action, 
        **kwargs
    )

    # Load pre-trained checkpoint if specified
    if FLAGS.load_pretrained and FLAGS.pretrained_checkpoint:
        print(f"\nLoading pre-trained checkpoint from: {FLAGS.pretrained_checkpoint}")
        agent = load_checkpoint(agent, FLAGS.pretrained_checkpoint)
        print("[Checkpoint] ✓ Pre-trained agent loaded successfully\n")
    else:
        print("\n[Info] Starting training from scratch (no pre-trained checkpoint)\n")

    # Replay buffer
    replay_buffer = ReplayBuffer(
        env.observation_space, 
        env.action_space, 
        FLAGS.replay_buffer_size
    )
    replay_buffer.seed(FLAGS.seed)
    replay_buffer_iterator = replay_buffer.get_iterator(
        sample_args={"batch_size": FLAGS.batch_size}
    )

    # ========================================================================
    # PHASE 1: Data Collection with Pre-trained Agent
    # ========================================================================
    if FLAGS.load_pretrained and FLAGS.data_collection_steps > 0:
        collect_data_with_agent(
            agent=agent,
            env=env,
            replay_buffer=replay_buffer,
            num_steps=FLAGS.data_collection_steps,
            deterministic=False,  # Use some exploration
            noise_scale=0.1  # Small noise for exploration
        )
    
    # ========================================================================
    # PHASE 2: Training Loop
    # ========================================================================
    print("\n" + "="*70)
    print("Starting training phase")
    print(f"Initial buffer size: {replay_buffer._size:,}")
    print(f"Training steps: {FLAGS.max_steps:,}")
    print(f"Will start training updates after: {FLAGS.start_training:,} additional steps")
    print(f"Checkpoints every: {FLAGS.checkpoint_interval:,} steps")
    print("="*70 + "\n")
    
    observation, info = env.reset()
    done = False
    episode_count = 0
    best_return = -float('inf')
    
    # Adjust start_training to account for pre-collected data
    effective_start_training = FLAGS.data_collection_steps + FLAGS.start_training
    
    for step in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1),
        smoothing=0.1,
        disable=not FLAGS.tqdm,
    ):
        # Sample action
        total_step = FLAGS.data_collection_steps + step
        
        if total_step < effective_start_training:
            # Still in exploration phase
            action = env.action_space.sample()
        else:
            # Use policy with noise
            action = agent.sample_actions(observation)
            action = action + noise()
            action = np.clip(action, env.action_space.low, env.action_space.high)
        
        # Execute action
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Compute mask for bootstrapping
        mask = 0.0 if (terminated or truncated) else 1.0

        # Store transition
        replay_buffer.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward,
                masks=mask,
                dones=done,
                next_observations=next_observation,
            )
        )
        observation = next_observation

        # Handle episode end
        if done:
            episode_count += 1
            observation, info = env.reset()
            noise.reset()
            
            if "episode" in info:
                ep_return = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                
                episode_info = {
                    "return": ep_return,
                    "length": ep_length,
                    "distance": info.get("distance", 0),
                }
                logger.log_episode(episode_info, total_step)
                
                # Track best return
                if ep_return > best_return:
                    best_return = ep_return

        # Training updates (only after collecting enough data)
        if total_step >= effective_start_training:
            batch = next(replay_buffer_iterator)
            update_info = agent.update(batch)

            if step % FLAGS.log_interval == 0:
                logger.log_training(update_info, total_step)
                logger.print_status(total_step, FLAGS.data_collection_steps + FLAGS.max_steps)

        # Checkpoint saving
        if step % FLAGS.checkpoint_interval == 0 and total_step >= effective_start_training:
            save_checkpoint(
                agent, 
                replay_buffer,
                os.path.join(policy_folder, f"step_{total_step}"), 
                total_step
            )

        # Evaluation
        if step % FLAGS.eval_interval == 0 and total_step >= effective_start_training:
            # You can add evaluation code here if needed
            pass

    # Final save
    final_step = FLAGS.data_collection_steps + FLAGS.max_steps
    save_checkpoint(
        agent, 
        replay_buffer,
        os.path.join(policy_folder, "final"), 
        final_step
    )
    
    print("\n" + "="*70)
    print("Training completed!")
    print(f"  Total steps: {final_step:,}")
    print(f"    - Data collection: {FLAGS.data_collection_steps:,}")
    print(f"    - Training: {FLAGS.max_steps:,}")
    print(f"  Episodes: {episode_count}")
    print(f"  Best return: {best_return:.2f}")
    print(f"  Final buffer size: {replay_buffer._size:,}")
    print(f"  Checkpoints saved to: {policy_folder}")
    print("="*70 + "\n")
    
    logger.close()
    env.close()


if __name__ == "__main__":
    app.run(main)