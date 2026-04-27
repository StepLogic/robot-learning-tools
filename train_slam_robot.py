#!/usr/bin/env python
"""
SLAM-Based Training for Real Robot
===================================

Replaces HER with ORB_SLAM3-based position estimation and distance rewards.
Uses accurate metric position instead of pseudo-odometry.
"""

import random
import os
import re
import pickle
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional, List, Dict, Any
import tqdm
import uuid

import cv2
import flax
from flax.core import frozen_dict
import jax
import numpy as np
from absl import app, flags
from ml_collections import config_flags

import torch
import pygame

import gymnasium as gym
from gymnasium import spaces
from jaxrl2.agents import DrQLearner
from jaxrl2.data import ReplayBuffer
from jaxrl2.noise import OrnsteinUhlenbeckActionNoise
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from jaxrl2.wrappers.timelimit import TimeLimit

from racer_imu_env import RacerEnv, StackingWrapper
from wrappers import (
    EnvCompatibility,
    Logger,
    MobileNetFeatureWrapper,
    load_checkpoint,
    save_checkpoint,
)
from slam_reward_wrapper import SLAMRewardWrapper

flax.config.update("flax_use_orbax_checkpointing", True)
FLAGS = flags.FLAGS

# ── Flags ─────────────────────────────────────────────────────────────────────
flags.DEFINE_string("env_name",        "donkey-warehouse-v0", "Environment name.")
flags.DEFINE_string("sim",             "sim_path",            "Path to unity simulator.")
flags.DEFINE_integer("port",           9091,                  "Port for TCP.")
flags.DEFINE_string("save_dir",        "./logs/",             "Tensorboard log dir.")
flags.DEFINE_integer("seed",           42,                    "Random seed.")
flags.DEFINE_integer("log_interval",   1000,                  "Logging interval.")
flags.DEFINE_integer("eval_interval",  int(50000),            "Eval interval.")
flags.DEFINE_integer("checkpoint_interval", 1000,             "Checkpoint interval.")
flags.DEFINE_integer("batch_size",     64,                    "Batch size.")
flags.DEFINE_integer("max_steps",      int(1e6),              "Total training steps.")
flags.DEFINE_integer("start_training", int(1e3),              "Steps before updates begin.")
flags.DEFINE_integer("replay_buffer_size", int(1e4),          "Online buffer capacity.")
flags.DEFINE_boolean("tqdm",           True,                  "Show tqdm bar.")
flags.DEFINE_integer("frame_stack",    3,                     "Frame stack depth.")
flags.DEFINE_integer("mobilenet_blocks",     4,               "MobileNetV3 blocks.")
flags.DEFINE_integer("mobilenet_input_size", 84,              "MobileNetV3 input size.")

# Checkpoint/teleop flags
flags.DEFINE_string(
    "pretrained_checkpoint",
    "/home/kojogyaase/Projects/Research/recovery-from-failure/pretrained_policy/step_520000",
    "Path to pre-trained checkpoint.",
)
flags.DEFINE_string(
    "teleop_buffer_path",
    "/home/kojogyaase/Projects/Research/recovery-from-failure/teleop_buffer.pkl",
    "Path to save/resume teleop (human) transitions.",
)
flags.DEFINE_float("expert_sample_ratio", 0.25, "Fraction of each mixed batch drawn from the expert teleop buffer.")
flags.DEFINE_float("steer_step",          0.05, "Steering increment per keypress.")
flags.DEFINE_float("throttle_step",       0.02, "Throttle increment per keypress.")
flags.DEFINE_integer("teleop_save_every", 500,  "Auto-save teleop buffer every N human steps.")

# SLAM flags (replaces HER flags)
flags.DEFINE_string("slam_vocab", "/path/to/ORBvoc.txt", "Path to ORB vocabulary file.")
flags.DEFINE_string("slam_settings", "calibration/donkeycar_fisheye_imu.yaml", "Path to ORB_SLAM3 settings YAML.")
flags.DEFINE_float("k_dist", 5.0, "Weight for distance delta reward.")
flags.DEFINE_float("k_goal", 50.0, "Weight for goal completion reward.")
flags.DEFINE_float("k_step", 0.1, "Weight for per-step time penalty.")
flags.DEFINE_float("goal_threshold", 0.5, "Distance threshold to consider goal reached (meters).")

# ── Environment Setup ────────────────────────────────────────────────────────
def make_env() -> gym.Env:
    """Create the environment with SLAM wrapper stack."""
    env = RacerEnv()

    # Wrapper stack (replaces HER-based stack)
    env = EnvCompatibility(env)
    env = StackingWrapper(env, num_stack=FLAGS.frame_stack)
    env = MobileNetFeatureWrapper(env, num_blocks=FLAGS.mobilenet_blocks, input_size=FLAGS.mobilenet_input_size)
    env = RecordEpisodeStatistics(env)
    env = TimeLimit(env, max_episode_steps=1000)

    # SLAM reward wrapper (replaces GoalRelObservationWrapper and RewardWrapper)
    env = SLAMRewardWrapper(
        env,
        vocab_path=FLAGS.slam_vocab,
        settings_path=FLAGS.slam_settings,
        goal_threshold=FLAGS.goal_threshold,
        k_dist=FLAGS.k_dist,
        k_goal=FLAGS.k_goal,
        k_step=FLAGS.k_step
    )

    return env

# ── Training Setup ───────────────────────────────────────────────────────────
def setup_training():
    """Setup training components."""
    # Environment
    env = make_env()

    # Action noise for exploration
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(env.action_space.shape[0]),
        sigma=np.array([0.1, 0.05])  # steering, throttle
    )

    # Replay buffer (standard, not HER)
    replay_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        FLAGS.replay_buffer_size
    )

    # Load teleop buffer if it exists
    teleop_buffer = []
    if os.path.exists(FLAGS.teleop_buffer_path):
        with open(FLAGS.teleop_buffer_path, 'rb') as f:
            teleop_buffer = pickle.load(f)
        print(f"Loaded {len(teleop_buffer)} teleop transitions")

    # Agent
    agent = DrQLearner(
        env.observation_space,
        env.action_space,
        actor_lr=3e-4,
        critic_lr=3e-4,
        hidden_dims=[256, 256],
        discount=0.99,
        tau=0.005,
        target_update_period=2,
        use_tb=True
    )

    # Load pretrained checkpoint if specified
    if FLAGS.pretrained_checkpoint and os.path.exists(FLAGS.pretrained_checkpoint):
        agent = load_checkpoint(agent, FLAGS.pretrained_checkpoint)
        print(f"Loaded pretrained checkpoint from {FLAGS.pretrained_checkpoint}")

    return env, agent, replay_buffer, teleop_buffer, action_noise

# ── Training Loop ───────────────────────────────────────────────────────────
def train():
    """Main training loop."""
    # Setup
    env, agent, replay_buffer, teleop_buffer, action_noise = setup_training()

    # Training state
    obs, info = env.reset()
    episode_reward = 0
    episode_length = 0
    step_count = 0

    # Teleop state
    human_control = False
    human_steps = 0

    # Main loop
    for step in tqdm.tqdm(range(1, FLAGS.max_steps + 1), disable=not FLAGS.tqdm):
        # Check for human override (HitL)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a or event.key == pygame.K_d:
                    human_control = True
                    print("🎮 Human control activated")
                elif event.key == pygame.K_SPACE:
                    human_control = False
                    print("🤖 AI control resumed")

        # Get action
        if human_control:
            # Human teleoperation
            keys = pygame.key.get_pressed()
            steering = 0.0
            throttle = 0.15  # Default forward throttle

            if keys[pygame.K_a]:
                steering = -0.5
            elif keys[pygame.K_d]:
                steering = 0.5

            if keys[pygame.K_w]:
                throttle = 0.2
            elif keys[pygame.K_s]:
                throttle = -0.1

            action = np.array([steering, throttle], dtype=np.float32)
            human_steps += 1

            # Save teleop transition
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            teleop_buffer.append((obs, action, reward, next_obs, terminated))

            # Auto-save teleop buffer periodically
            if human_steps % FLAGS.teleop_save_every == 0:
                with open(FLAGS.teleop_buffer_path, 'wb') as f:
                    pickle.dump(teleop_buffer, f)
                print(f"Saved {len(teleop_buffer)} teleop transitions")
        else:
            # AI policy
            if step < FLAGS.start_training:
                action = env.action_space.sample()  # Random exploration
            else:
                action = agent.sample_actions(obs, noise_scale=0.1)
                action = action_noise(action)

            next_obs, reward, terminated, truncated, next_info = env.step(action)

        # Store transition in replay buffer
        if step >= FLAGS.start_training:
            replay_buffer.add(obs, action, reward, next_obs, terminated)

        # Training step
        if step >= FLAGS.start_training and step % 1 == 0:
            # Mix teleop and online data
            batch_size = FLAGS.batch_size
            teleop_size = int(FLAGS.expert_sample_ratio * batch_size)
            online_size = batch_size - teleop_size

            # Sample from buffers
            online_batch = replay_buffer.sample(online_size) if online_size > 0 else None
            teleop_batch = random.sample(teleop_buffer, min(teleop_size, len(teleop_buffer))) if teleop_size > 0 else []

            # Combine batches
            if online_batch is not None and teleop_batch:
                batch = online_batch + teleop_batch
            elif online_batch is not None:
                batch = online_batch
            elif teleop_batch:
                batch = teleop_batch
            else:
                continue

            # Update agent
            agent.update(batch)

        # Update state
        obs = next_obs
        episode_reward += reward
        episode_length += 1
        step_count += 1

        # Logging
        if step % FLAGS.log_interval == 0:
            slam_stats = env.get_slam_stats() if hasattr(env, 'get_slam_stats') else {}
            print(f"\nStep {step}:")
            print(f"  Episode reward: {episode_reward:.1f}")
            print(f"  Episode length: {episode_length}")
            print(f"  SLAM OK: {slam_stats.get('slam_ok_count', 0)}")
            print(f"  SLAM Lost: {slam_stats.get('slam_lost_count', 0)}")
            print(f"  Replay buffer: {len(replay_buffer)}/{replay_buffer.capacity}")
            print(f"  Teleop buffer: {len(teleop_buffer)}")

        # Checkpointing
        if step % FLAGS.checkpoint_interval == 0:
            checkpoint_path = os.path.join(FLAGS.save_dir, f"step_{step}")
            save_checkpoint(agent, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        # Episode termination
        if terminated or truncated:
            print(f"Episode finished: reward={episode_reward:.1f}, length={episode_length}")
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0

            # Reset SLAM stats
            if hasattr(env, 'get_slam_stats'):
                stats = env.get_slam_stats()
                print(f"Episode SLAM stats: OK={stats['slam_ok_count']}, Lost={stats['slam_lost_count']}")

        # Evaluation
        if step % FLAGS.eval_interval == 0:
            print("Running evaluation...")
            eval_rewards = []
            for _ in range(5):  # 5 evaluation episodes
                eval_obs, _ = env.reset()
                eval_reward = 0
                eval_done = False

                while not eval_done:
                    eval_action = agent.sample_actions(eval_obs, noise_scale=0.0)  # No exploration
                    eval_obs, eval_reward_step, eval_terminated, eval_truncated, _ = env.step(eval_action)
                    eval_reward += eval_reward_step
                    eval_done = eval_terminated or eval_truncated

                eval_rewards.append(eval_reward)

            avg_eval_reward = np.mean(eval_rewards)
            print(f"Evaluation: {avg_eval_reward:.1f} ± {np.std(eval_rewards):.1f}")

    # Cleanup
    env.close()

    # Save final teleop buffer
    if teleop_buffer:
        with open(FLAGS.teleop_buffer_path, 'wb') as f:
            pickle.dump(teleop_buffer, f)
        print(f"Saved final teleop buffer with {len(teleop_buffer)} transitions")

# ── Main ────────────────────────────────────────────────────────────────────
def main(argv):
    """Entry point."""
    # Initialize pygame for HitL
    pygame.init()
    pygame.display.set_mode((100, 100))
    pygame.display.set_caption("HitL Control")

    # Run training
    train()

    print("Training completed!")


if __name__ == "__main__":
    app.run(main)