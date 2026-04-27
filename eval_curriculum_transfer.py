#!/usr/bin/env python
"""
Evaluation script for curriculum-trained goal navigation agents.

Supports two modes:
  1. In-distribution ViZDoom evaluation (--env vizdoom)
  2. Transfer evaluation on OfficeEnv / Donkey Car sim (--env office)

Usage:
  # Evaluate ViZDoom agent
  python eval_curriculum_transfer.py --env vizdoom --checkpoint ./checkpoints/vizdoom_curriculum/run_xxx/step_100000

  # Transfer to OfficeEnv
  python eval_curriculum_transfer.py --env office --checkpoint ./checkpoints/vizdoom_curriculum/run_xxx/step_100000 --num_episodes 20

  # Transfer to BarrowsHall
  python eval_curriculum_transfer.py --env barrowshall --checkpoint ./checkpoints/vizdoom_curriculum/run_xxx/step_100000 --stage 4
"""

import os
import sys
import random
import uuid
from typing import Dict, List, Optional

import numpy as np

import torch
import torchvision.models as models
import torchvision.transforms as T

import gymnasium as gym
from gymnasium import spaces

from absl import app, flags
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import tqdm

from jaxrl2.agents import DrQLearner
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from jaxrl2.wrappers.timelimit import TimeLimit

from wrappers import (
    EnvCompatibility, MobileNetFeatureWrapper, MobileNetV3Encoder,
    StackingWrapper, save_checkpoint, load_checkpoint,
    GoalRelObservationWrapper,
)
from vizdoom_env import make_vizdoom_env
from curriculum_wrappers import CurriculumWrapper, OfficeEnvCurriculumWrapper, CURRICULUM_STAGES

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "vizdoom", "Eval env: vizdoom, office, warehouse, barrowshall.")
flags.DEFINE_string("checkpoint", "", "Path to agent checkpoint.")
flags.DEFINE_string("vizdoom_scenario", "medium", "ViZDoom scenario for eval: easy/medium/hard.")
flags.DEFINE_integer("num_episodes", 20, "Number of evaluation episodes.")
flags.DEFINE_integer("seed", 99, "Random seed for evaluation.")
flags.DEFINE_integer("frame_stack", 4, "Frames to stack.")
flags.DEFINE_integer("mobilenet_blocks", 4, "MobileNetV3 blocks.")
flags.DEFINE_integer("mobilenet_input_size", 84, "MobileNet input size.")
flags.DEFINE_string("device", "cuda" if torch.cuda.is_available() else "cpu", "Device.")
flags.DEFINE_integer("port", 9092, "Sim port for office env.")
flags.DEFINE_integer("stage", 4, "Curriculum stage to evaluate at.")
flags.DEFINE_string("save_dir", "./eval_results", "Dir to save results.")
flags.DEFINE_integer("max_episode_steps", 2000, "Max steps per episode.")
flags.DEFINE_boolean("render", False, "Render the environment.")

flags.DEFINE_config_file("config", "./configs/drq_default.py", "DrQ config.", lock_config=False)


# ============================================================================
# Environment Setup
# ============================================================================

def make_eval_vizdoom_env(scenario: str = "medium", seed: int = 99,
                          frame_skip: int = 4, stage: int = 4) -> gym.Env:
    """Create ViZDoom eval env."""
    from vizdoom_env import DummyViZDoomEnv

    try:
        env = make_vizdoom_env(
            scenario=scenario,
            frame_skip=frame_skip,
            resolution=(60, 108),
            heading_goals=True,
            image_goals=True,
            seed=seed,
            visible=False,
        )
    except Exception:
        print("[Eval] ViZDoom unavailable, using DummyViZDoomEnv")
        env = DummyViZDoomEnv(
            scenario_path="scenarios/maze_medium.cfg",
            frame_skip=frame_skip,
            resolution=(60, 108),
            heading_goals=True,
            image_goals=True,
            seed=seed,
        )

    env = EnvCompatibility(env)
    env = StackingWrapper(env, num_stack=FLAGS.frame_stack)
    env = _ZeroIMUWrapper(env)
    env = MobileNetFeatureWrapper(
        env, device=FLAGS.device,
        num_blocks=FLAGS.mobilenet_blocks,
        input_size=FLAGS.mobilenet_input_size,
    )
    env = CurriculumWrapper(env, num_stages=4, stage=stage, eval_mode=True)
    env = RecordEpisodeStatistics(env)
    cfg = CURRICULUM_STAGES[stage]
    env = TimeLimit(env, max_episode_steps=cfg["max_steps"])
    return env


def make_eval_office_env(eval_env: str = "office", seed: int = 99,
                        stage: int = 4, port: int = 9092) -> gym.Env:
    """Create OfficeEnv / Donkey Car simulator eval env."""
    from gym_donkeycar.envs.donkey_env import DonkeyEnv

    level_map = {
        "office": "OfficeBox",
        "warehouse": "donkey-warehouse-v0",
        "barrowshall": "BarrowsHall",
        "office_corner": "OfficeBox-Corner",
    }
    level = level_map.get(eval_env, "OfficeBox")

    class _Env(DonkeyEnv):
        def __init__(self, conf):
            super().__init__(level=level, **conf)

    conf = {
        "host": "127.0.0.1",
        "port": port,
        "body_rgb": (128, 128, 128),
        "car_name": "",
        "font_size": 100,
        "racer_name": "",
        "country": "USA",
        "bio": "Eval Agent",
        "guid": str(uuid.uuid4()),
        "max_cte": 10,
        "frame_skip": 3,
    }

    env = _Env(conf)
    env = EnvCompatibility(env)
    env = StackingWrapper(env, num_stack=FLAGS.frame_stack)
    env = _ZeroIMUWrapper(env)
    env = MobileNetFeatureWrapper(
        env, device=FLAGS.device,
        num_blocks=FLAGS.mobilenet_blocks,
        input_size=FLAGS.mobilenet_input_size,
    )
    env = OfficeEnvCurriculumWrapper(env, curriculum_stage=stage)
    env = RecordEpisodeStatistics(env)
    env = TimeLimit(env, max_episode_steps=FLAGS.max_episode_steps)
    return env


class _ZeroIMUWrapper(gym.Wrapper):
    """Injects zero-filled IMU for envs without IMU sensor."""

    IMU_DIM = 6
    PROP_STACK_MULT = 2

    def __init__(self, env):
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

    def _zero_imu(self):
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
# Evaluation
# ============================================================================

def evaluate_agent(
    env: gym.Env,
    agent,
    num_episodes: int,
    env_name: str,
    deterministic: bool = True,
) -> Dict[str, float]:
    """
    Run evaluation episodes and compute metrics.

    Returns dict with:
        return_mean, return_std, return_min, return_max,
        length_mean, length_std, success_rate, collision_rate
    """

    returns = []
    lengths = []
    successes = []
    collisions = []

    for ep in tqdm.tqdm(range(num_episodes), desc=f"Eval {env_name}"):
        obs, info = env.reset()
        done = False
        episode_return = 0.0
        episode_length = 0

        while not done:
            action = np.array(agent.sample_actions(obs))
            if not deterministic:
                action = action + np.random.normal(0, 0.1, action.shape)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += reward
            episode_length += 1

            if FLAGS.render:
                try:
                    env.render()
                except Exception:
                    pass

        returns.append(episode_return)
        lengths.append(episode_length)

        # Determine outcome
        hit = info.get("hit", "none")
        goal_reached = episode_return > 50.0
        successes.append(1.0 if goal_reached else 0.0)
        collisions.append(1.0 if hit != "none" else 0.0)

        print(f"  Episode {ep+1}/{num_episodes}: "
              f"return={episode_return:.2f}, length={episode_length}, "
              f"success={goal_reached}, collision={hit != 'none'}")

    returns = np.array(returns)
    lengths = np.array(lengths)
    successes = np.array(successes)
    collisions = np.array(collisions)

    metrics = {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "return_min": float(np.min(returns)),
        "return_max": float(np.max(returns)),
        "length_mean": float(np.mean(lengths)),
        "length_std": float(np.std(lengths)),
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
    }

    return metrics


def save_results(metrics: Dict, env_name: str, checkpoint_name: str,
                save_dir: str):
    """Save evaluation results to file and tensorboard."""
    os.makedirs(save_dir, exist_ok=True)

    # Text summary
    summary_path = os.path.join(save_dir, f"{env_name}_eval_results.txt")
    with open(summary_path, "w") as f:
        f.write(f"Environment: {env_name}\n")
        f.write(f"Checkpoint: {checkpoint_name}\n")
        f.write(f"Num episodes: {FLAGS.num_episodes}\n\n")
        for k, v in metrics.items():
            f.write(f"  {k:25s}: {v:>10.4f}\n")

    # Pickle
    import pickle
    pkl_path = os.path.join(save_dir, f"{env_name}_metrics.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(metrics, f)

    print(f"\nResults saved to {save_dir}/")
    return summary_path


# ============================================================================
# Main
# ============================================================================

def main(_):
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    checkpoint_name = os.path.basename(FLAGS.checkpoint) if FLAGS.checkpoint else "random"

    print(f"\n{'='*70}")
    print(f"CURRICULUM GOAL NAVIGATION EVALUATION")
    print(f"  Env:      {FLAGS.env}")
    print(f"  Stage:    {FLAGS.stage}")
    print(f"  Checkpoint: {checkpoint_name}")
    print(f"  Episodes: {FLAGS.num_episodes}")
    print(f"{'='*70}\n")

    # Create env
    if FLAGS.env == "vizdoom":
        env = make_eval_vizdoom_env(
            scenario=FLAGS.vizdoom_scenario,
            seed=FLAGS.seed,
            stage=FLAGS.stage,
        )
    else:
        env = make_eval_office_env(
            eval_env=FLAGS.env,
            seed=FLAGS.seed,
            stage=FLAGS.stage,
            port=FLAGS.port,
        )

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Build sample obs for agent init
    sample_obs = env.observation_space.sample()
    sample_action = env.action_space.sample()

    # Initialize agent
    kwargs = dict(FLAGS.config) if FLAGS.config else {}
    agent = DrQLearner(FLAGS.seed, sample_obs, sample_action, **kwargs)

    # Load checkpoint
    if FLAGS.checkpoint:
        print(f"\nLoading checkpoint: {FLAGS.checkpoint}")
        agent = load_checkpoint(agent, FLAGS.checkpoint)
        print("Checkpoint loaded successfully.")
    else:
        print("\nNo checkpoint specified — using random agent.")

    # Run evaluation
    metrics = evaluate_agent(
        env=env,
        agent=agent,
        num_episodes=FLAGS.num_episodes,
        env_name=FLAGS.env,
        deterministic=True,
    )

    # Print results
    print(f"\n{'='*50}")
    print(f"RESULTS: {FLAGS.env.upper()}")
    print(f"{'='*50}")
    print(f"  Return mean / std:   {metrics['return_mean']:.2f} +/- {metrics['return_std']:.2f}")
    print(f"  Return min / max:    {metrics['return_min']:.2f} / {metrics['return_max']:.2f}")
    print(f"  Length mean / std:   {metrics['length_mean']:.1f} +/- {metrics['length_std']:.1f}")
    print(f"  Success rate:        {metrics['success_rate']:.2%}")
    print(f"  Collision rate:      {metrics['collision_rate']:.2%}")
    print(f"{'='*50}\n")

    # Save results
    summary_path = save_results(metrics, FLAGS.env, checkpoint_name, FLAGS.save_dir)
    print(f"Results written to {summary_path}")

    env.close()


if __name__ == "__main__":
    app.run(main)
