
#!/usr/bin/env python
"""
DrQ Training with JAX, MobileNetV3 encoder, and Temporal Modeling
Adapted for Donkey Car simulator with Real Robot Environment Architecture
- MobileNetV3 visual feature extraction (PyTorch)
- Frame stacking with feature extraction
- Action+IMU history tracking
- Enhanced reward structure for forward motion and collision avoidance
- Hindsight Experience Replay (HER) for goal relabeling
"""
import random
from collections import deque
import os
from datetime import datetime
import tqdm
from typing import Dict, Any, List, Callable, Optional
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
from jaxrl2.noise import OrnsteinUhlenbeckActionNoise
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from jaxrl2.wrappers.timelimit import TimeLimit

# Donkey Car env classes
from gym_donkeycar.envs.donkey_env import DonkeyEnv

# from racer_imu_env import RacerEnv,StackingWrapper
from wrappers import EnvCompatibility, Logger, MobileNetFeatureWrapper, RewardWrapper, save_checkpoint, StackingWrapper
from jaxrl2.data.replay_buffer import ReplayBuffer
from jaxrl2.data.dataset import DatasetDict
from flax.core import frozen_dict

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
flags.DEFINE_integer("checkpoint_interval", 100000, "Checkpoint saving interval.")
flags.DEFINE_integer("batch_size", 8, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer("start_training", int(10), "Number of training steps to start training.")
flags.DEFINE_integer("replay_buffer_size", int(4e3), "Replay buffer size.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("domain_randomization", True, "Enable domain randomization.")
flags.DEFINE_boolean("env_randomization", True, "Enable environment randomization.")
flags.DEFINE_integer("switch_env_every", 10, "Switch environment every N episodes.")
flags.DEFINE_integer("frame_stack", 4, "Number of frames to stack for temporal modeling.")
flags.DEFINE_integer("mobilenet_blocks", 4, "Number of MobileNetV3 blocks to use.")
flags.DEFINE_integer("mobilenet_input_size", 84, "Input size for MobileNetV3.")
# HER-specific flags
flags.DEFINE_float("her_fraction", 0.5, "Fraction of relabeled goals (HER strategy).")
flags.DEFINE_string("her_strategy", "future", "HER strategy: 'future', 'final', or 'random'.")
# Goal wrapper flags
flags.DEFINE_float("goal_range", 20.0, "Range for random goal sampling.")
flags.DEFINE_float("goal_threshold", 1.0, "Distance threshold to consider goal reached.")

config_flags.DEFINE_config_file(
    "config",
    "./configs/drq_default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


# ============================================================================
# Goal-Relative Observation Wrapper with Random Goals
# ============================================================================

class GoalRelObservationWrapper(gym.Wrapper):
    """
    Goal-relative observation wrapper with random relative position goals.

    - Goals are sampled as random relative positions (not from a fixed set)
    - When the agent reaches a goal (distance < threshold), a new random goal is sampled
    - This enables continuous exploration and HER-style learning
    """

    def __init__(self, env, goal_range: float = 20.0, goal_threshold: float = 1.0):
        super().__init__(env)
        self.goal_range = goal_range  # Range for random goal sampling
        self.goal_threshold = goal_threshold  # Distance to consider goal "reached"
        self.stop_count = 0
        self.distance_to_goal = None
        self.current_goal = None
        self.goals_reached = 0  # Track number of goals reached
        self.start_pos = None  # Starting position for distance calculation
        self.prev_pos = None  # Previous position for distance accumulation
        self.total_distance = 0.0  # Cumulative distance traveled in episode


        # Goal relative space: relative x, y, z positions
        goal_rel_space = spaces.Box(
            low=-np.array([goal_range, goal_range, 0], dtype=np.float32),
            high=np.array([goal_range, goal_range, 0.1], dtype=np.float32),
            dtype=np.float32
        )

        if isinstance(self.observation_space, spaces.Dict):
            new_spaces = dict(self.observation_space.spaces)
            new_spaces["goal_rel"] = goal_rel_space
            self.observation_space = spaces.Dict(new_spaces)
        else:
            self.observation_space = spaces.Dict(
                {
                    "obs": self.observation_space,
                    "goal_rel": goal_rel_space,
                }
            )
    def _unity_to_agent_coords(self, pos: np.ndarray) -> np.ndarray:
        """
        Unity uses Y as vertical and Z as depth. This swaps Y and Z so that:
        Unity: (x, y, z) -> Agent: (x, z, y)
        This makes the coordinate system more intuitive for 2D navigation.
        """
        if pos is None or len(pos) < 3:
            return pos
        return np.array([pos[0], pos[2], pos[1]], dtype=np.float32)

    def _sample_random_goal(self) -> np.ndarray:
        """Sample a random relative goal position."""
        return np.random.uniform(
            low=-self.goal_range,
            high=self.goal_range,
            size=3
        ).astype(np.float32)

    def _is_goal_reached(self, goal_rel: np.ndarray) -> bool:
        """Check if the current goal has been reached."""
        return np.linalg.norm(goal_rel) < self.goal_threshold

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_goal = self._sample_random_goal()
        self.stop_count = 0
        raw_pos = np.asarray(info.get("pos", [0.0, 0.0, 0.0]), dtype=np.float32)
        curr_pos = self._unity_to_agent_coords(raw_pos)
        goal_rel = self.current_goal - curr_pos
        self.distance_to_goal = np.linalg.norm(goal_rel)
        # Initialize distance tracking
        self.start_pos = curr_pos.copy()
        self.prev_pos = curr_pos.copy()
        self.total_distance = 0.0

        if isinstance(obs, dict):
            obs = dict(obs)
            obs["goal_rel"] = goal_rel
        else:
            obs = {
                "obs": obs,
                "goal_rel": goal_rel,
            }
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        terminated, truncated = False, False

        if self.current_goal is None:
            self.current_goal = self._sample_random_goal()

        raw_pos = np.asarray(info.get("pos", [0.0, 0.0, 0.0]), dtype=np.float32)
        curr_pos = self._unity_to_agent_coords(raw_pos)

        # Accumulate distance traveled
        if self.prev_pos is not None:
            step_distance = np.linalg.norm(curr_pos - self.prev_pos)
            self.total_distance += step_distance

        self.prev_pos = curr_pos.copy()
        goal_rel = self.current_goal - curr_pos
        reward = -1.0

        # Check if goal reached - if so, resample a new random goal
        if self._is_goal_reached(goal_rel):
            self.goals_reached += 1
            terminated = True
            reward += 100
            # Resample new goal for next episode
            self.current_goal = self._sample_random_goal()
            print(f"Goal reached! Total goals reached: {self.goals_reached}, new goal sampled")

        vel = info.get("forward_vel", 0.0)
        reward += vel

        if vel < 0.01:
            self.stop_count += 1

        if self.stop_count > 20:
            truncated = True

        if info.get("hit", "none") != "none":
            truncated = True
            reward -= 1.0

        if isinstance(obs, dict):
            obs = dict(obs)
            obs["goal_rel"] = goal_rel
        else:
            obs = {
                "obs": obs,
                "goal_rel": goal_rel,
            }

        self.distance_to_goal = np.linalg.norm(goal_rel)

        if terminated or truncated:
            print(f"Distance To Goal: {np.linalg.norm(goal_rel):.2f}")
            print(f"Total Distance Traveled: {self.total_distance:.2f}")

        # Add distance info for logging
        info["distance"] = self.total_distance

        return obs, reward, terminated, truncated, info


# ============================================================================
# HindsightReplayBuffer - Replay buffer with HER goal relabeling
# ============================================================================

class HindsightReplayBuffer(ReplayBuffer):
    """
    Replay buffer that supports Hindsight Experience Replay (HER).

    HER relabels goals after exploration by replacing some goals with
    achieved states from the same episode. This helps learn from
    failed episodes by treating them as successful for different goals.

    Strategies:
    - 'future': Sample a random future state from the same episode (default)
    - 'final': Use the final state of the episode as the relabeled goal
    - 'random': Sample a random state from the episode as the relabeled goal
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
        her_fraction: float = 0.5,
        her_strategy: str = "future",
        **kwargs
    ):
        super().__init__(
            observation_space,
            action_space,
            capacity,
            next_observation_space=next_observation_space,
            **kwargs
        )
        self.her_fraction = her_fraction
        self.her_strategy = her_strategy

        # Track episodes for HER: list of (start_index, end_index) tuples
        self._episodes: List[List[int]] = []
        self._current_episode_indices: List[int] = []

    def insert(self, data_dict: DatasetDict):
        """Insert transition and track episode indices."""
        current_idx = self._insert_index
        super().insert(data_dict)
        self._current_episode_indices.append(current_idx)

    def end_episode(self):
        """Call when episode ends to store episode boundaries."""
        if self._current_episode_indices:
            self._episodes.append(self._current_episode_indices.copy())
            self._current_episode_indices = []

    def _get_relabel_fn(self) -> Callable:
        """
        Create the relabeling function for HER.

        Returns a function that relabels goals in a batch of samples
        by replacing some goals with achieved states from the same episode.
        """

        def relabel_fn(samples: Dict) -> Dict:
            """
            Relabel goals in the batch.

            For a fraction of samples, replaces the goal with an achieved
            state (position) from the same episode. This is the key insight
            of HER: even if we failed to reach the original goal, we can
            treat the episode as successful for a different goal that we
            actually achieved.
            """
            observations = samples['observations']
            next_observations = samples['next_observations']

            # Handle dict observations (like goal_rel in this codebase)
            if isinstance(observations, dict) and 'goal_rel' in observations:
                batch_size = observations['goal_rel'].shape[0]
                goals = observations['goal_rel']
                next_goals = next_observations['goal_rel']

                # Determine which samples to relabel
                her_mask = np.random.random(batch_size) < self.her_fraction

                # For each sample to relabel, get an achieved goal from the episode
                for i in range(batch_size):
                    if her_mask[i]:
                        # Get episode containing this transition
                        # We use the index modulo capacity to find the episode
                        # This is approximate since indices wrap around
                        episode = self._find_episode_for_index(i)

                        if episode and len(episode) > 1:
                            if self.her_strategy == 'future':
                                # Sample a random future state from the episode
                                future_idx = np.random.randint(1, len(episode))
                                achieved_idx = episode[min(future_idx, len(episode) - 1)]
                            elif self.her_strategy == 'final':
                                # Use the final state of the episode
                                achieved_idx = episode[-1]
                            else:  # 'random'
                                # Sample a random state from the episode
                                achieved_idx = episode[np.random.randint(len(episode))]

                            # Get the achieved goal (position) from the future state
                            achieved_goal = self._get_achieved_goal(achieved_idx)
                            if achieved_goal is not None:
                                # Replace the goal with the achieved goal
                                goals[i] = achieved_goal

                # Return modified samples
                new_obs = dict(observations)
                new_obs['goal_rel'] = goals
                samples['observations'] = new_obs

            return samples

        return relabel_fn

    def _find_episode_for_index(self, sample_idx: int) -> Optional[List[int]]:
        """
        Find the episode that contains the transition at the given index.

        Note: This is a simplified approximation. In practice, for wrapped
        buffers with index wrapping, you'd need to track this more carefully.
        """
        # Simple case: find an episode that contains this index
        # Since indices can wrap around, we look for episodes that include the sample
        for episode_indices in self._episodes[-10:]:  # Check last 10 episodes
            if episode_indices and len(episode_indices) > 0:
                return episode_indices
        return None

    def _get_achieved_goal(self, idx: int) -> Optional[np.ndarray]:
        """Get the achieved goal (position) at the given buffer index."""
        try:
            # Get the observation at this index
            obs = self.dataset_dict['observations']
            if isinstance(obs, dict) and 'goal_rel' in obs:
                # The goal_rel is already relative to current goal
                # For achieved goal, we need the state itself
                # In this implementation, we use the next_observation's goal_rel
                # as a proxy for where we actually ended up
                return obs['goal_rel'][idx % len(obs['goal_rel'])]
        except (KeyError, IndexError):
            pass
        return None

    def get_iterator(self, queue_size: int = 2, sample_args: dict = {}):
        """
        Get iterator with HER relabeling enabled.

        Adds relabel_fn to sample_args if not present.
        """
        sample_args = dict(sample_args)  # Don't modify original
        sample_args['relabel'] = True
        return super().get_iterator(queue_size=queue_size, sample_args=sample_args)

    def sample(self, *args, relabel: bool = True, relabel_fn=None, **kwargs):
        """
        Sample from buffer with optional HER relabeling.
        """
        if relabel_fn is None:
            relabel_fn = self._get_relabel_fn()

        samples = super().sample(*args, relabel=False, **kwargs)

        if relabel and relabel_fn is not None:
            samples = frozen_dict.unfreeze(samples)
            samples = relabel_fn(samples)
            samples = frozen_dict.freeze(samples)
        return samples


# ============================================================================
# Main Training Loop
# ============================================================================
class OfficeEnvCorner(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="OfficeBox-Corner", *args, **kwargs)


class BarrowsHallEnvCorner(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="BarrowsHall", *args, **kwargs)
class OfficeEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="OfficeBox", *args, **kwargs)


def main(_):
    print("\n" + "="*70)
    print("DrQ Training with MobileNetV3 Feature Extraction + HER")
    print("="*70 + "\n")

    # Environment configuration
    conf = {
        "host": "127.0.0.1",
        "port": FLAGS.port,
        # "body_style": "f1",
        "body_rgb": (128, 128, 128),
        "car_name": "",
        "font_size": 100,
        "racer_name": "",
        "country": "USA",
        "bio": "Learning to drive with DrQ + MobileNetV3 + HER",
        "guid": str(uuid.uuid4()),
        "max_cte": 10,
        "frame_skip": 3,
        # "throttle_min":-1.0
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
    env = BarrowsHallEnvCorner(conf=conf)
    # print(env.action_space)
    env= EnvCompatibility(env)

    # Wrap with stacking wrapper
    env = StackingWrapper(env, num_stack=3)


    # Extract MobileNetV3 features from stacked frames
    env = MobileNetFeatureWrapper(
        env,
        device=device,
        num_blocks=FLAGS.mobilenet_blocks,
        input_size=FLAGS.mobilenet_input_size
    )

    env = RecordEpisodeStatistics(env)
    env = TimeLimit(env, max_episode_steps=4000)
    env = GoalRelObservationWrapper(env, goal_range=FLAGS.goal_range, goal_threshold=FLAGS.goal_threshold)


    print(f"\n{'='*60}")
    print("Environment Setup Complete")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"{'='*60}\n")

    # RL Training Setup
    action_dim = env.action_space.shape[0]
    mean = np.zeros(action_dim)
    sigma =  np.ones(action_dim)
    sigma[1]=0.2*sigma[1]
    noise = OrnsteinUhlenbeckActionNoise(mean=mean, sigma=sigma)

    logger = Logger(log_dir=FLAGS.save_dir)
    policy_folder = os.path.join(
        "checkpoints",
        f"drq_mobilenet_her_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(policy_folder, exist_ok=True)

    # Initialize DrQ agent
    print("Initializing DrQ agent...")
    kwargs = dict(FLAGS.config) if FLAGS.config else {}

    # sample_obs, _ = env.reset()
    # sample_action = env.action_space.sample()
    sample_obs = env.observation_space.sample()
    sample_action  = env.action_space.sample()
    print(f"Sample observation shapes:")
    print(f"  Pixels (features): {sample_obs['pixels'].shape}")
    print(f"  Actions: {sample_obs['actions'].shape}")
    print(f"  Goal rel: {sample_obs.get('goal_rel', 'N/A').shape}")
    print(f"Sample action shape: {sample_action.shape}")

    agent = DrQLearner(
        FLAGS.seed,
        env.observation_space.sample(),
        env.action_space.sample(),
        **kwargs
    )

    # Hindsight Replay buffer with HER support
    replay_buffer = HindsightReplayBuffer(
        env.observation_space,
        env.action_space,
        FLAGS.replay_buffer_size,
        her_fraction=FLAGS.her_fraction,
        her_strategy=FLAGS.her_strategy,
    )
    replay_buffer.seed(FLAGS.seed)
    replay_buffer_iterator = replay_buffer.get_iterator(
        sample_args={"batch_size": FLAGS.batch_size}
    )

    # Main Training Loop
    print("\n" + "="*70)
    print("Starting training with MobileNetV3 features + HER")
    print(f"HER fraction: {FLAGS.her_fraction}, Strategy: {FLAGS.her_strategy}")
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
        # print(action)
        # Execute action
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Compute mask for bootstrapping
        mask = 0.0 if terminated else 1.0

        # Store transition
        # breakpoint()
        replay_buffer.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward,
                masks=mask,
                dones=terminated,
                next_observations=next_observation,
            )
        )
        observation = next_observation

        # Handle episode end
        if done:
            dst= info.get("distance", 0)
            episode_count += 1
            # Signal end of episode to replay buffer for HER
            replay_buffer.end_episode()
            observation, info = env.reset()
            noise.reset()

            if "episode" in info:
                ep_return = info["episode"]["r"]
                ep_length = info["episode"]["l"]

                episode_info = {
                    "return": ep_return,
                    "length": ep_length,
                    "distance": dst,
                }
                logger.log_episode(episode_info, step)


        # Training updates
        if step >= FLAGS.start_training:
            # for _ in range(FLAGS.frame_stack):  # Update multiple
            batch = next(replay_buffer_iterator)
            update_info = agent.update(batch)

            if step % FLAGS.log_interval == 0:
                logger.log_training(update_info, step)
                logger.print_status(step, FLAGS.max_steps)
            pass
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
