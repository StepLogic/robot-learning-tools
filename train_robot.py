#!/usr/bin/env python
"""
DrQ Training with JAX, MobileNetV3 encoder, and Temporal Modeling
Adapted for Donkey Car simulator with Real Robot Environment Architecture
- MobileNetV3 visual feature extraction (PyTorch)
- Frame stacking with feature extraction
- Action+IMU history tracking
- Enhanced reward structure for forward motion and collision avoidance
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

from racer_imu_env import RacerEnv,StackingWrapper,RewardWrapper
from wrappers import EnvCompatibility, Logger, MobileNetFeatureWrapper, load_checkpoint, save_checkpoint

flax.config.update("flax_use_orbax_checkpointing", True)

FLAGS = flags.FLAGS

# Configuration flags
flags.DEFINE_string("env_name", "donkey-warehouse-v0", "Environment name.")
flags.DEFINE_string("sim", "sim_path", "Path to unity simulator.")
flags.DEFINE_integer("port", 9091, "Port to use for tcp.")
flags.DEFINE_string("save_dir", "./logs/", "Tensorboard logging dir.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 5, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", int(50000), "Eval interval.")
flags.DEFINE_integer("checkpoint_interval", 5000, "Checkpoint saving interval.")
flags.DEFINE_integer("batch_size", 16, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer("start_training", int(1e3), "Number of training steps to start training.")
flags.DEFINE_integer("replay_buffer_size", int(1e4), "Replay buffer size.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("domain_randomization", True, "Enable domain randomization.")
flags.DEFINE_boolean("env_randomization", True, "Enable environment randomization.")
flags.DEFINE_integer("switch_env_every", 10, "Switch environment every N episodes.")
flags.DEFINE_integer("frame_stack", 4, "Number of frames to stack for temporal modeling.")
flags.DEFINE_integer("mobilenet_blocks", 4, "Number of MobileNetV3 blocks to use.")
flags.DEFINE_integer("mobilenet_input_size", 84, "Input size for MobileNetV3.")

config_flags.DEFINE_config_file(
    "config",
    "./configs/drq_default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)







def load_replay_buffer(replay_buffer, replay_buffer_path):
    """Load replay buffer from checkpoint."""
    try:

        if os.path.exists(replay_buffer_path):
            with open(replay_buffer_path, 'rb') as f:
                buffer_data = pickle.load(f)
            
            replay_buffer.dataset_dict = buffer_data['data']
            replay_buffer._insert_index = buffer_data['insert_index']
            replay_buffer._size = buffer_data['size']
            
            print(f"[Checkpoint] Loaded replay buffer with {replay_buffer._size} transitions")
            return True
        else:
            print(f"[Checkpoint] Replay buffer file not found at {replay_buffer_path}")
            return False
    except Exception as e:
        print(f"[ERROR] Failed to load replay buffer: {e}")
        return False


# ============================================================================
# Main Training Loop
# ============================================================================

def main(_):
    print("\n" + "="*70)
    print("DrQ Training with MobileNetV3 Feature Extraction")
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
        "throttle_min":-1.0
    }

    # Set random seeds
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(FLAGS.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create the environment in human render mode
    env = RacerEnv(render_mode="human")
    env= EnvCompatibility(env)
    # Wrap with stacking wrapper
    env = StackingWrapper(env, num_stack=4)



    env = RewardWrapper(env)

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
        f"drq_mobilenet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        env.observation_space.sample(), 
        env.action_space.sample(), 
        **kwargs
    )
    agent = load_checkpoint(agent,"/home/kojogyaase/Projects/Research/recovery-from-failure/pretrained_policy/step_520000")
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

        # Replay buffer
    expert_replay_buffer = ReplayBuffer(
        env.observation_space, 
        env.action_space, 
        FLAGS.replay_buffer_size
    )
    expert_replay_buffer =load_replay_buffer(expert_replay_buffer, "/home/kojogyaase/Projects/Research/recovery-from-failure/teleop_buffer.pkl")
    expert_replay_buffer.seed(FLAGS.seed)
    expert_replay_buffer_iterator = expert_replay_buffer.get_iterator(
        sample_args={"batch_size": FLAGS.batch_size}
    )

    # Main Training Loop
    print("\n" + "="*70)
    print("Starting training with MobileNetV3 features")
    print(f"Checkpoints will be saved every {FLAGS.checkpoint_interval:,} steps")
    print("="*70 + "\n")
    
    observation, info = env.reset()
    done = False
    episode_count = 0
    best_return = -float('inf')
    
    for step in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1),
        smoothing=0.1,
        disable=not FLAGS.tqdm,
    ):
        # Sample action
        if step < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
            action = action + noise()
            action = np.clip(action, env.action_space.low, env.action_space.high)
        
        # Execute action
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Compute mask for bootstrapping
        mask = 0.0 if (terminated or truncated) else 1.0

        # Store transition
        # breakpoint()
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
                logger.log_episode(episode_info, step)

        if expert_replay_buffer_iterator is not None:
            batch = next(expert_replay_buffer_iterator)
            update_info = agent.update(batch,enable_update_temperature=False)

        # Training updates
        if step >= FLAGS.start_training:
            batch = next(replay_buffer_iterator)
            update_info = agent.update(batch)

            if step % FLAGS.log_interval == 0:
                logger.log_training(update_info, step)
                logger.print_status(step, FLAGS.max_steps)

        # Checkpoint saving every 5000 steps
        if step % FLAGS.checkpoint_interval == 0 and step >= FLAGS.start_training:
            save_checkpoint(
                agent, 
                replay_buffer,
                os.path.join(policy_folder, f"step_{step}"), 
                step
            )

        # Evaluation (kept separate from checkpointing)
        if step % FLAGS.eval_interval == 0 and step >= FLAGS.start_training:
            # You can add evaluation code here if needed
            pass

    # Final save
    save_checkpoint(
        agent, 
        replay_buffer,
        os.path.join(policy_folder, "final"), 
        FLAGS.max_steps
    )
    
    print("\n" + "="*70)
    print("Training completed!")
    print(f"  Steps: {FLAGS.max_steps:,}")
    print(f"  Episodes: {episode_count}")
    print(f"  Best return: {best_return:.2f}")
    print("="*70 + "\n")
    
    logger.close()
    env.close()


if __name__ == "__main__":
    app.run(main)