#!/usr/bin/env python
"""
Curriculum Learning Training Pipeline for ViZDoom Goal Navigation.

Trains a DrQ agent in ViZDoom using a 4-stage curriculum:
  Stage 1 (Easy)    : Open arena, heading goals only
  Stage 2 (Medium)  : Wall obstacles, heading + image goals
  Stage 3 (Hard)    : Narrow corridors, image-goal matching
  Stage 4 (Expert)  : Mixed tasks, adversarial obstacles

Transfer evaluation can be run on OfficeEnv / Donkey Car sim.

Usage:
  # Train from scratch
  python train_vizdoom_curriculum.py --env_name vizdoom --max_steps 1000000

  # Resume from checkpoint
  python train_vizdoom_curriculum.py --checkpoint_dir ./checkpoints/vizdoom_curriculum/step_50000

  # Evaluate on OfficeEnv (transfer)
  python train_vizdoom_curriculum.py --eval_only --eval_env office --checkpoint_dir ./checkpoints/...
"""

import os
import sys
import random
import time
import uuid
import pickle
from collections import deque
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import random as jax_random

import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

import gymnasium as gym
from gymnasium import spaces

from absl import app, flags
from flax.training import checkpoints
from ml_collections import config_flags
from torch.utils.tensorboard import SummaryWriter
import tqdm

# JAX RL imports
from jaxrl2.agents import DrQLearner
from jaxrl2.data import ReplayBuffer
from jaxrl2.noise import OrnsteinUhlenbeckActionNoise
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from jaxrl2.wrappers.timelimit import TimeLimit

# Project imports
from wrappers import (
    DoomStackingWrapper, EnvCompatibility, Logger, MobileNetFeatureWrapper,
    StackingWrapper, FrameSkipWrapper, save_checkpoint,
    MobileNetV3Encoder, sim2real_a, GoalRelObservationWrapper,
    RewardWrapper,
)
from vizdoom_env import make_vizdoom_env, ViZDoomEnv, DummyViZDoomEnv
from curriculum_wrappers import CurriculumWrapper, OfficeEnvCurriculumWrapper, CURRICULUM_STAGES

flax.config.update("flax_use_orbax_checkpointing", True)

FLAGS = flags.FLAGS

# General
flags.DEFINE_string("env_name", "vizdoom", "Environment name: vizdoom or office")
flags.DEFINE_string("save_dir", "./logs/vizdoom_curriculum", "TensorBoard log dir.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of eval episodes.")
flags.DEFINE_integer("log_interval", 1000, "Training log interval.")
flags.DEFINE_integer("eval_interval", int(5000), "Eval interval.")
flags.DEFINE_integer("checkpoint_interval", 50000, "Checkpoint saving interval.")
flags.DEFINE_integer("batch_size", 8, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(2e6), "Total training steps.")
flags.DEFINE_integer("start_training", int(2000), "Steps before training starts.")
flags.DEFINE_integer("replay_buffer_size", int(1e5), "Replay buffer size.")
flags.DEFINE_boolean("tqdm", True, "Show tqdm progress bar.")

# Curriculum
flags.DEFINE_integer("num_stages", 4, "Number of curriculum stages.")
flags.DEFINE_integer("init_stage", 1, "Initial curriculum stage (1-4).")
flags.DEFINE_boolean("eval_mode", False, "Eval mode (no curriculum advancement).")

# Visual encoding
flags.DEFINE_integer("frame_stack", 4, "Number of frames to stack.")
flags.DEFINE_integer("mobilenet_blocks", 4, "MobileNetV3 blocks to use.")
flags.DEFINE_integer("mobilenet_input_size", 84, "MobileNetV3 input size.")
flags.DEFINE_string("device", "cuda" if torch.cuda.is_available() else "cpu", "PyTorch device.")

# ViZDoom config
flags.DEFINE_string("vizdoom_scenario", "easy", "ViZDoom scenario: easy/medium/hard.")
flags.DEFINE_integer("frame_skip", 4, "Frame skip for ViZDoom.")
flags.DEFINE_boolean("vizdoom_visible", True, "Show ViZDoom window.")

# Transfer eval
flags.DEFINE_boolean("eval_only", False, "Only run evaluation (no training).")
flags.DEFINE_string("eval_env", "office", "Eval environment: office/warehouse/barrowshall.")
flags.DEFINE_string("checkpoint_dir", "", "Path to checkpoint for eval/transfer.")

# Donkey Car sim
flags.DEFINE_string("sim_path", "", "Path to donkey car simulator.")
flags.DEFINE_integer("port", 9091, "Simulator TCP port.")
flags.DEFINE_integer("max_episode_steps", 2000, "Max steps per episode.")

config_flags.DEFINE_config_file(
    "config",
    "./configs/drq_default.py",
    "DrQ hyperparameter configuration.",
    lock_config=False,
)


# ============================================================================
# Donkey Car Sim Environments (for OfficeEnv transfer)
# ============================================================================

class OfficeEnvCorner:
    """Placeholder for donkey car OfficeEnv — imported lazily."""
    pass


def make_office_env(eval_env: str = "office", port: int = 9091, seed: int = 42):
    """Create OfficeEnv / Donkey Car simulator environment."""
    from gym_donkeycar.envs.donkey_env import DonkeyEnv

    class OfficeEnv(DonkeyEnv):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    conf = {
        "host": "127.0.0.1",
        # "port": FLAGS.port,
        # "body_style": "f1",
        "body_rgb": (128, 128, 128),
        "car_name": "",
        "font_size": 100,
        "racer_name": "",
        "country": "USA",
        "bio": "Learning to drive with DrQ + MobileNetV3",
        "guid": str(uuid.uuid4()),
        "max_cte": 10,
        "frame_skip": 3,
        # "level":"OfficeBox"
    }

    return OfficeEnv(conf)


# ============================================================================
# Environment Setup
# ============================================================================

def setup_vizdoom_env(seed: int, device: str, frame_stack: int,
                     mobilenet_blocks: int, mobilenet_input_size: int,
                     init_stage: int = 1, frame_skip: int = 4,
                     visible: bool = True, num_stages: int = 4) -> gym.Env:
    """Create and wrap ViZDoom env with full stack."""

    # Stage-specific scenario
    stage_cfg = CURRICULUM_STAGES[init_stage]
    scenario = stage_cfg["scenario"]

    # Create base ViZDoom env (returns dict obs with "image" key)
    env = make_vizdoom_env(
        scenario=scenario,
        frame_skip=frame_skip,
        resolution=(60, 108),
        heading_goals=stage_cfg["heading_goals"],
        image_goals=stage_cfg["image_goals"],
        seed=seed,
        visible=visible,
    )

    env = EnvCompatibility(env)

    # StackingWrapper now handles dict obs internally (extracts "image" key)
    env = DoomStackingWrapper(env, num_stack=frame_stack)
    # o,l=env.reset()
    # breakpoint()
    # Add zero-filled IMU for ViZDoom
    env = _ZeroIMUWrapper(env)

    env = MobileNetFeatureWrapper(
        env,
        device=device,
        num_blocks=mobilenet_blocks,
        input_size=mobilenet_input_size,
    )

    # Curriculum wrapper
    env = CurriculumWrapper(env, num_stages=num_stages, stage=init_stage)

    env = RecordEpisodeStatistics(env)
    env = TimeLimit(env, max_episode_steps=CURRICULUM_STAGES[init_stage]["max_steps"])

    return env


def setup_office_env(seed: int, device: str, frame_stack: int,
                    mobilenet_blocks: int, mobilenet_input_size: int,
                    stage: int = 1, port: int = 9091,
                    max_episode_steps: int = 2000,
                    eval_env_name: str = "office") -> gym.Env:
    """Create and wrap OfficeEnv / Donkey Car sim with curriculum."""

    env = make_office_env(eval_env=eval_env_name, port=port, seed=seed)
    env = EnvCompatibility(env)
    env = StackingWrapper(env, num_stack=frame_stack)
    env = _ZeroIMUWrapper(env)
    env = MobileNetFeatureWrapper(
        env,
        device=device,
        num_blocks=mobilenet_blocks,
        input_size=mobilenet_input_size,
    )
    env = OfficeEnvCurriculumWrapper(env, curriculum_stage=stage)
    env = RecordEpisodeStatistics(env)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    return env


class _ExtractImageWrapper(gym.Wrapper):
    """
    Extracts 'image' key from dict observations and passes it as raw array,
    while preserving the dict for later wrappers.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # StackingWrapper needs raw image space
        if isinstance(env.observation_space, spaces.Dict) and "image" in env.observation_space.spaces:
            h, w, c = env.observation_space["image"].shape
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(h, w, c),
                dtype=np.uint8
            )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Extract image for StackingWrapper
        img = obs["image"]
        self._last_obs = obs  # preserve for later
        return img, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs
        return obs["image"], reward, terminated, truncated, info


class _DictToArrayImageWrapper(gym.Wrapper):
    """
    Wraps ViZDoomEnv (dict obs with "image" key) so it returns raw image array.
    The raw image is then passed to StackingWrapper.
    After StackingWrapper, _ArrayToDictImageWrapper re-wraps as dict.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        if isinstance(env.observation_space, spaces.Dict) and "image" in env.observation_space.spaces:
            h, w, c = env.observation_space["image"].shape
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(h, w, c), dtype=np.uint8
            )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._saved_obs = obs  # keep full obs dict
        return obs["image"], info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._saved_obs = obs
        return obs["image"], reward, terminated, truncated, info


class _ArrayToDictImageWrapper(gym.Wrapper):
    """
    After StackingWrapper produces stacked pixels as a raw array,
    this wraps it back as a dict with 'pixels' key for MobileNetFeatureWrapper.
    """
    def __init__(self, env: gym.Env, frame_stack: int = 4):
        super().__init__(env)
        self.frame_stack = frame_stack
        # env.observation_space is already the stacked Box from StackingWrapper
        raw_obs_space = env.observation_space
        if isinstance(raw_obs_space, spaces.Box):
            self.observation_space = spaces.Dict({
                "pixels": raw_obs_space,
                "actions": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(frame_stack * 2 * 2,),  # placeholder, will be correct
                    dtype=np.float32
                ),
            })

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # StackingWrapper returns {"pixels": array, "actions": ..., "imu": ...}
        # We need to add the "actions" key if not present
        if isinstance(obs, dict) and "actions" not in obs:
            obs = {"pixels": obs.get("pixels", obs), "actions": np.zeros(16, dtype=np.float32)}
        elif not isinstance(obs, dict):
            obs = {"pixels": obs, "actions": np.zeros(16, dtype=np.float32)}
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if isinstance(obs, dict) and "actions" not in obs:
            obs = {"pixels": obs.get("pixels", obs), "actions": np.zeros(16, dtype=np.float32)}
        elif not isinstance(obs, dict):
            obs = {"pixels": obs, "actions": np.zeros(16, dtype=np.float32)}
        return obs, reward, terminated, truncated, info


class _ZeroIMUWrapper(gym.Wrapper):
    """Injects zero-filled IMU vector into observations for envs without IMU."""

    IMU_DIM = 6
    PROP_STACK_MULT = 2  # matches StackingWrapper's 2x factor

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.num_stack_prop = getattr(env, 'num_stack', 4) * self.PROP_STACK_MULT
        imu_stack_shape = (self.num_stack_prop * self.IMU_DIM,)

        if isinstance(env.observation_space, spaces.Dict):
            new_spaces = dict(env.observation_space.spaces)
            new_spaces['imu'] = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=imu_stack_shape,
                dtype=np.float32,
            )
            self.observation_space = spaces.Dict(new_spaces)
        else:
            raise ValueError("_ZeroIMUWrapper requires dict observation space")

    def _zero_imu(self) -> np.ndarray:
        return np.zeros(self.num_stack_prop * self.IMU_DIM, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs['imu'] = self._zero_imu()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs['imu'] = self._zero_imu()
        return obs, reward, terminated, truncated, info


# ============================================================================
# Training Loop
# ============================================================================

def train_curriculum(
    env: gym.Env,
    eval_env: Optional[gym.Env],
    agent,
    replay_buffer,
    logger: Logger,
    policy_folder: str,
    max_steps: int,
    start_training: int,
    batch_size: int,
    checkpoint_interval: int,
    eval_interval: int,
    eval_episodes: int,
    log_interval: int,
    device: str,
):
    """Main curriculum training loop."""

    observation, info = env.reset()

    done = False
    episode_count = 0
    best_return = -float('inf')
    curriculum_metrics = env.unwrapped.get_metrics() if hasattr(env.unwrapped, 'get_metrics') else None

    # Action noise for exploration
    action_dim = env.action_space.shape[0]
    mean = np.zeros(action_dim)
    sigma = np.ones(action_dim)
    sigma[1] = 0.2  # throttle less noisy
    noise = OrnsteinUhlenbeckActionNoise(mean=mean, sigma=sigma)

    replay_buffer_iterator = replay_buffer.get_iterator(
        sample_args={"batch_size": batch_size}
    )

    pbar = tqdm.tqdm(range(1, max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm)

    for step in pbar:
        # Sample action (random until start_training)
        if step < start_training:
            action = env.action_space.sample()
        else:
            action = np.array(agent.sample_actions(observation))
            action = action + noise()
            action = np.clip(action, env.action_space.low, env.action_space.high)

        # Execute
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        mask = 0.0 if terminated else 1.0

        replay_buffer.insert(dict(
            observations=observation,
            actions=action,
            rewards=reward,
            masks=mask,
            dones=terminated,
            next_observations=next_obs,
        ))
        observation = next_obs

        # Episode end
        if done:
            episode_count += 1
            observation, info = env.reset()
            noise.reset()

            if "episode" in info:
                ep_return = info["episode"]["r"]
                ep_length = info["episode"]["l"]

                # Record curriculum metrics
                goal_reached = ep_return > 50.0
                collision = info.get("hit", "none") != "none"

                if curriculum_metrics is not None:
                    curriculum_metrics.record_episode_result(
                        ep_return, ep_length, goal_reached, collision
                    )

                # Log
                episode_info = {
                    "return": ep_return,
                    "length": ep_length,
                    "distance": info.get("distance", 0.0),
                }
                logger.log_episode(episode_info, step)

                if ep_return > best_return:
                    best_return = ep_return

                # Update progress bar description with curriculum info
                if curriculum_metrics is not None:
                    stats = curriculum_metrics.get_stats()
                    pbar.set_description(
                        f"S{stats['stage']}({stats['stage_name']}) "
                        f"SR:{stats['success_rate']:.0%} "
                        f"R:{ep_return:.1f}"
                    )

        # Training update
        if step >= start_training:
            batch = next(replay_buffer_iterator)
            update_info = agent.update(batch)

            if step % log_interval == 0:
                logger.log_training(update_info, step)

                if curriculum_metrics is not None:
                    stats = curriculum_metrics.get_stats()
                    for k, v in stats.items():
                        logger.log_training({f"curriculum/{k}": v}, step)

        # Checkpoint
        if step % checkpoint_interval == 0 and step >= start_training:
            ckpt_path = os.path.join(policy_folder, f"step_{step}")
            save_checkpoint(agent, replay_buffer, ckpt_path, step)

            # Also save curriculum state
            if curriculum_metrics is not None:
                curriculum_state = {
                    "stage": curriculum_metrics.stage,
                    "total_episodes": curriculum_metrics.total_episodes,
                    "stage_history": curriculum_metrics.stage_history,
                }
                with open(os.path.join(ckpt_path, "curriculum_state.pkl"), "wb") as f:
                    pickle.dump(curriculum_state, f)

        # Evaluation
        if step % eval_interval == 0 and step >= start_training and eval_env is not None:
            eval_returns = run_evaluation(eval_env, agent, eval_episodes)
            logger.log_eval({
                "eval/return_mean": np.mean(eval_returns),
                "eval/return_std": np.std(eval_returns),
            }, step)

    # Final save
    save_checkpoint(agent, replay_buffer,
                   os.path.join(policy_folder, "final"), max_steps)
    logger.close()
    env.close()

    print(f"\nTraining completed: {episode_count} episodes, best return: {best_return:.2f}")
    return episode_count


def run_evaluation(eval_env: gym.Env, agent, num_episodes: int) -> list:
    """Run evaluation episodes and return list of returns."""
    returns = []
    for _ in range(num_episodes):
        obs, info = eval_env.reset()
        done = False
        episode_return = 0.0

        while not done:
            action = np.array(agent.sample_actions(obs))
            action = np.clip(action, eval_env.action_space.low, eval_env.action_space.high)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            episode_return += reward

        returns.append(episode_return)
        if "episode" in info:
            print(f"  Eval episode return: {info['episode']['r']:.2f}")

    return returns


# ============================================================================
# Transfer Evaluation on OfficeEnv
# ============================================================================

def evaluate_transfer(
    checkpoint_dir: str,
    eval_env_name: str = "office",
    num_episodes: int = 20,
    stage: int = 4,
    device: str = "cuda",
):
    """Evaluate a ViZDoom-trained checkpoint on OfficeEnv (transfer learning eval)."""

    print(f"\n{'='*70}")
    print(f"TRANSFER EVALUATION: ViZDoom -> {eval_env_name.upper()}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"{'='*70}\n")

    # Setup eval env
    eval_env = setup_office_env(
        seed=42, device=device, frame_stack=FLAGS.frame_stack,
        mobilenet_blocks=FLAGS.mobilenet_blocks,
        mobilenet_input_size=FLAGS.mobilenet_input_size,
        stage=stage, port=FLAGS.port,
        max_episode_steps=FLAGS.max_episode_steps,
        eval_env_name=FLAGS.eval_env,
    )

    # Create a dummy train env for agent initialization
    train_env = setup_vizdoom_env(
        seed=FLAGS.seed, device=device, frame_stack=FLAGS.frame_stack,
        mobilenet_blocks=FLAGS.mobilenet_blocks,
        mobilenet_input_size=FLAGS.mobilenet_input_size,
        init_stage=stage, frame_skip=FLAGS.frame_skip,
        visible=False, num_stages=FLAGS.num_stages,
    )

    # Initialize agent
    kwargs = dict(FLAGS.config) if FLAGS.config else {}
    sample_obs = train_env.observation_space.sample()
    sample_action = train_env.action_space.sample()

    agent = DrQLearner(
        FLAGS.seed, sample_obs, sample_action, **kwargs
    )

    # Load checkpoint
    if checkpoint_dir:
        from wrappers import load_checkpoint
        agent = load_checkpoint(agent, checkpoint_dir)

    # Run eval
    returns = run_evaluation(eval_env, agent, num_episodes)

    print(f"\n{'='*50}")
    print(f"Transfer Results ({eval_env_name}):")
    print(f"  Mean Return: {np.mean(returns):.2f} +/- {np.std(returns):.2f}")
    print(f"  Min Return:  {np.min(returns):.2f}")
    print(f"  Max Return:  {np.max(returns):.2f}")
    print(f"  Success Rate (>0): {np.mean([r > 0 for r in returns]):.2%}")
    print(f"{'='*50}\n")

    eval_env.close()
    return returns


# ============================================================================
# Main
# ============================================================================

def main(_):
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(FLAGS.seed)

    device = FLAGS.device
    print(f"\nDevice: {device}")

    if FLAGS.eval_only:
        assert FLAGS.checkpoint_dir, "--checkpoint_dir required for eval_only"
        evaluate_transfer(
            checkpoint_dir=FLAGS.checkpoint_dir,
            eval_env_name=FLAGS.eval_env,
            num_episodes=FLAGS.eval_episodes,
            stage=FLAGS.init_stage,
            device=device,
        )
        return

    # ===== Training =====

    # Create train env
    print("\n[1/4] Setting up ViZDoom curriculum env...")
    train_env = setup_vizdoom_env(
        seed=FLAGS.seed, device=device, frame_stack=FLAGS.frame_stack,
        mobilenet_blocks=FLAGS.mobilenet_blocks,
        mobilenet_input_size=FLAGS.mobilenet_input_size,
        init_stage=FLAGS.init_stage, frame_skip=FLAGS.frame_skip,
        visible=FLAGS.vizdoom_visible, num_stages=FLAGS.num_stages,
    )

    # Create eval env (OfficeEnv for transfer eval)
    print("[2/4] Setting up OfficeEnv for transfer evaluation...")
    eval_env = setup_office_env(
        seed=FLAGS.seed + 1, device=device, frame_stack=FLAGS.frame_stack,
        mobilenet_blocks=FLAGS.mobilenet_blocks,
        mobilenet_input_size=FLAGS.mobilenet_input_size,
        stage=FLAGS.num_stages,  # evaluate at hardest stage
        port=FLAGS.port + 1,
        max_episode_steps=FLAGS.max_episode_steps,
        eval_env_name=FLAGS.eval_env,
    )

    # Print env info
    print(f"\n  Train env obs space: {train_env.observation_space}")
    print(f"  Train env action space: {train_env.action_space}")
    sample_obs = train_env.observation_space.sample()
    print(f"  Sample obs keys: {sample_obs.keys() if isinstance(sample_obs, dict) else 'not dict'}")

    # Initialize agent
    print("\n[3/4] Initializing DrQ agent...")
    kwargs = dict(FLAGS.config) if FLAGS.config else {}
    agent = DrQLearner(
        FLAGS.seed,
        sample_obs,
        train_env.action_space.sample(),
        **kwargs
    )

    # Replay buffer
    replay_buffer = ReplayBuffer(
        train_env.observation_space,
        train_env.action_space,
        FLAGS.replay_buffer_size,
    )
    replay_buffer.seed(FLAGS.seed)
    # replay_buffer_iterator = replay_buffer.get_iterator(
    #     sample_args={"batch_size": FLAGS.batch_size}
    # )

    # Logger & checkpoint dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(FLAGS.save_dir, f"run_{timestamp}")
    logger = Logger(log_dir=log_dir)

    policy_folder = os.path.join(
        "./checkpoints",
        f"vizdoom_curriculum_{timestamp}"
    )
    os.makedirs(policy_folder, exist_ok=True)

    print(f"\n  Log dir: {log_dir}")
    print(f"  Checkpoint dir: {policy_folder}")

    # Train
    print("\n[4/4] Starting curriculum training...")
    print(f"  Stages: {FLAGS.num_stages}")
    print(f"  Starting stage: {FLAGS.init_stage}")
    print(f"  Max steps: {FLAGS.max_steps:,}\n")

    train_curriculum(
        env=train_env,
        eval_env=eval_env,
        agent=agent,
        replay_buffer=replay_buffer,
        logger=logger,
        policy_folder=policy_folder,
        max_steps=FLAGS.max_steps,
        start_training=FLAGS.start_training,
        batch_size=FLAGS.batch_size,
        checkpoint_interval=FLAGS.checkpoint_interval,
        eval_interval=FLAGS.eval_interval,
        eval_episodes=FLAGS.eval_episodes,
        log_interval=FLAGS.log_interval,
        device=device,
    )

    print("\nDone!")


if __name__ == "__main__":
    app.run(main)
