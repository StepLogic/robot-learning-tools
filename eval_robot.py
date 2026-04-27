#!/usr/bin/env python
"""
DrQ Agent Evaluation Script for Donkey Car Simulator
- Load trained checkpoints from simulation training
- Evaluate agent performance in simulator
- Record episode statistics and trajectories
- Support multiple environments
- Optional video recording
"""
import random
from collections import deque
import os
from datetime import datetime
import tqdm
from typing import Dict, Any, Optional
import uuid
import pickle
import json

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import random as jax_random
import numpy as np
from absl import app, flags
from flax.training import checkpoints
from ml_collections import config_flags
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
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from jaxrl2.wrappers.timelimit import TimeLimit

# Donkey Car env classes
from gym_donkeycar.envs.donkey_env import (
    OfficeEnv, WaveshareEnv, MiniMonacoEnv, WarehouseEnv
)

from racer_imu_env import RewardWrapper,StackingWrapper,RacerEnv
from wrappers import EnvCompatibility, FrameSkipWrapper, MobileNetFeatureWrapper, load_checkpoint

flax.config.update("flax_use_orbax_checkpointing", True)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "checkpoint_path", 
    "/home/kojogyaase/Projects/Research/recovery-from-failure/pretrained_policy/step_520000", 
    "Path to checkpoint directory (required)."
)
flags.DEFINE_integer("checkpoint_step", -1, "Checkpoint step to load (-1 for latest).")
flags.DEFINE_string("env_name", "donkey-warehouse-v0", "Environment name.")
flags.DEFINE_integer("port", 9091, "Port to use for tcp.")
flags.DEFINE_string("save_dir", "./eval_results/", "Directory to save evaluation results.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("num_episodes", 10, "Number of episodes for evaluation.")
flags.DEFINE_integer("max_episode_steps", 1000, "Maximum steps per episode.")
flags.DEFINE_boolean("deterministic", True, "Use deterministic policy (no exploration noise).")
flags.DEFINE_integer("frame_stack", 3, "Number of frames to stack (must match training).")
flags.DEFINE_integer("mobilenet_blocks", 4, "Number of MobileNetV3 blocks (must match training).")
flags.DEFINE_integer("mobilenet_input_size", 84, "Input size for MobileNetV3 (must match training).")
flags.DEFINE_boolean("verbose", True, "Print detailed episode information.")
flags.DEFINE_boolean("record_video", False, "Record videos of episodes.")

config_flags.DEFINE_config_file(
    "config",
    "./configs/drq_default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

# ============================================================================
# Checkpoint Loading Functions
# ============================================================================
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def find_latest_checkpoint(checkpoint_dir: str) -> Optional[int]:
    """Find the latest checkpoint step in a directory."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Look for checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_")]
    
    if not checkpoint_files:
        return None
    
    # Extract step numbers
    steps = []
    for f in checkpoint_files:
        try:
            step = int(f.split("_")[1])
            steps.append(step)
        except (IndexError, ValueError):
            continue
    
    return max(steps) if steps else None



# ============================================================================
# Evaluation Function
# ============================================================================

def evaluate_agent(
    agent,
    env,
    num_episodes: int,
    deterministic: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate agent on environment.
    
    Args:
        agent: Trained DrQLearner agent
        env: Evaluation environment
        num_episodes: Number of episodes to evaluate
        deterministic: Use deterministic policy (no noise)
        verbose: Print episode statistics
    
    Returns:
        Dictionary with evaluation statistics
    """
    episode_returns = []
    episode_lengths = []
    episode_distances = []
    episode_info_list = []
    
    print(f"\n{'='*70}")
    print(f"Evaluating agent for {num_episodes} episodes")
    print(f"Deterministic policy: {deterministic}")
    print(f"{'='*70}\n")
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        episode_return = 0.0
        episode_length = 0
        done = False
        
        episode_actions = []
        episode_distance = 0.0
        
        while not done:
            # Select action
            if deterministic:
                action = agent.eval_actions(observation)
            else:
                action = agent.sample_actions(observation)
            action = np.array([-action[0],np.clip(sigmoid(action[1]),0,0.160)])
            # print((action))
            action = np.clip(
                    action,
                    env.action_space.low,
                    env.action_space.high,
                )
            episode_actions.append(action.copy())
            
            # Execute action
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_return += reward
            episode_length += 1
            
            # Track distance if available
            if "distance" in info:
                episode_distance = info["distance"]
        
        # Store episode statistics
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        episode_distances.append(episode_distance)
        
        # Store detailed info
        episode_data = {
            "episode": episode,
            "return": episode_return,
            "length": episode_length,
            "distance": episode_distance,
            "actions": episode_actions,
        }
        
        # Add any additional info from the environment
        if "episode" in info:
            episode_data.update(info["episode"])
        
        episode_info_list.append(episode_data)
        
        # Print episode summary
        if verbose:
            print(f"Episode {episode + 1}/{num_episodes}:")
            print(f"  Return:   {episode_return:>10.2f}")
            print(f"  Length:   {episode_length:>10d} steps")
            print(f"  Distance: {episode_distance:>10.1f}")
            
            # Compute action statistics
            actions_array = np.array(episode_actions)
            print(f"  Avg Steering: {np.mean(actions_array[:, 0]):>+7.3f} "
                  f"(std: {np.std(actions_array[:, 0]):.3f})")
            print(f"  Avg Throttle: {np.mean(actions_array[:, 1]):>+7.3f} "
                  f"(std: {np.std(actions_array[:, 1]):.3f})")
            print()
    
    # Compute aggregate statistics
    stats = {
        "num_episodes": num_episodes,
        "mean_return": np.mean(episode_returns),
        "std_return": np.std(episode_returns),
        "min_return": np.min(episode_returns),
        "max_return": np.max(episode_returns),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "min_length": np.min(episode_lengths),
        "max_length": np.max(episode_lengths),
        "mean_distance": np.mean(episode_distances),
        "std_distance": np.std(episode_distances),
        "episodes": episode_info_list,
    }
    
    # Print summary
    print(f"\n{'='*70}")
    print("Evaluation Summary")
    print(f"{'='*70}")
    print(f"Episodes:      {stats['num_episodes']}")
    print(f"Mean Return:   {stats['mean_return']:>10.2f} ± {stats['std_return']:.2f}")
    print(f"Return Range:  [{stats['min_return']:.2f}, {stats['max_return']:.2f}]")
    print(f"Mean Length:   {stats['mean_length']:>10.1f} ± {stats['std_length']:.1f}")
    print(f"Length Range:  [{stats['min_length']}, {stats['max_length']}]")
    print(f"Mean Distance: {stats['mean_distance']:>10.1f} ± {stats['std_distance']:.1f}")
    print(f"{'='*70}\n")
    
    return stats


# ============================================================================
# Main Evaluation
# ============================================================================

def main(_):
    
    print("\n" + "="*70)
    print("DrQ Simulator Evaluation")
    print("="*70 + "\n")
    
    # Set random seeds
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(FLAGS.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Checkpoint: {FLAGS.checkpoint_path}")
    print(f"Environment: {FLAGS.env_name}")
    
    # Environment configuration
    conf = {
        "host": "127.0.0.1",
        "port": FLAGS.port,
        "body_style": "f1",
        "body_rgb": (128, 128, 128),
        "car_name": "EvalAgent",
        "font_size": 100,
        "racer_name": "DrQ Eval",
        "country": "USA",
        "bio": "Evaluating DrQ + MobileNetV3",
        "guid": str(uuid.uuid4()),
        "max_cte": 10,
        "frame_skip": 3,
    }
    
    # Create environment with wrapper chain
    print("\nCreating evaluation environment...")
    # old_gym_env = WaveshareEnv(conf=conf)
    env = RacerEnv()
    env = FrameSkipWrapper(env,skip=3)
    env = RewardWrapper(env)
    # Stack RGB frames
    env = StackingWrapper(env, num_stack=FLAGS.frame_stack)
    
    # Extract MobileNetV3 features from stacked frames
    env = MobileNetFeatureWrapper(
        env,
        device=device,
        num_blocks=FLAGS.mobilenet_blocks,
        input_size=FLAGS.mobilenet_input_size
    )
    
    # Add episode statistics and time limit
    env = RecordEpisodeStatistics(env)
    env = TimeLimit(env, max_episode_steps=FLAGS.max_episode_steps)
    
    
    print(f"\n{'='*60}")
    print("Environment Setup Complete")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"{'='*60}\n")
    
    # Initialize DrQ agent
    print("Initializing agent...")
    kwargs = dict(FLAGS.config) if FLAGS.config else {}
    
    # sample_obs, _ = env.reset()
    # sample_action = env.action_space.sample()
    
    # print(f"Sample observation shapes:")
    # print(f"  Pixels (features): {sample_obs['pixels'].shape}")
    # print(f"  Actions: {sample_obs['actions'].shape}")
    # print(f"Sample action shape: {sample_action.shape}")
    
    agent = DrQLearner(
        FLAGS.seed, 
        env.observation_space.sample(), 
        env.action_space.sample(), 
        **kwargs
    )
    
    # Load checkpoint
    print("\nLoading checkpoint...")
    agent = load_checkpoint(agent, FLAGS.checkpoint_path)
    
    # if loaded_step is None:
    #     print("[ERROR] Failed to load checkpoint. Exiting.")
    #     env.close()
    #     return
    
    # Evaluate agent
    print("\nStarting evaluation...")
    stats = evaluate_agent(
        agent,
        env,
        num_episodes=FLAGS.num_episodes,
        deterministic=FLAGS.deterministic,
        verbose=FLAGS.verbose
    )
    
    # Save results
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    results_file = os.path.join(
        FLAGS.save_dir,
        f"eval_sim_step_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    # Add metadata
    stats["metadata"] = {
        "checkpoint_path": FLAGS.checkpoint_path,
        # "checkpoint_step": loaded_step,
        "env_name": FLAGS.env_name,
        "port": FLAGS.port,
        "deterministic": FLAGS.deterministic,
        "num_episodes": FLAGS.num_episodes,
        "max_episode_steps": FLAGS.max_episode_steps,
        "frame_stack": FLAGS.frame_stack,
        "mobilenet_blocks": FLAGS.mobilenet_blocks,
        "mobilenet_input_size": FLAGS.mobilenet_input_size,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    stats = convert_to_serializable(stats)
    
    with open(results_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    
    # Close environment
    env.close()
    
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    app.run(main)