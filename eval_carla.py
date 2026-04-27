#!/usr/bin/env python
"""
DrQ Agent Evaluation Script for CARLA Simulator
- Load trained checkpoints from Donkey Car simulator (DRQ + HER + goal masking)
- Evaluate agent performance in CARLA
- Record collision count, success rate, episode statistics
- Save results to experiment-named folder
"""
import random
from collections import deque
import os
from datetime import datetime
import uuid
import json

import flax
import jax
import numpy as np
from absl import app, flags
from flax.training import checkpoints
from ml_collections import config_flags
import gymnasium as gym
from gymnasium import spaces

import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

from jaxrl2.agents import DrQLearner
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from jaxrl2.wrappers.timelimit import TimeLimit

from carla_env import CarlaEnv, CarlaConfig, CameraConfig, RewardFunction
from racer_imu_env import RewardWrapper, StackingWrapper
from wrappers import MobileNetFeatureWrapper, load_checkpoint, EnvCompatibility, FrameSkipWrapper

flax.config.update("flax_use_orbax_checkpointing", True)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "checkpoint_path",
    "/home/kojogyaase/Projects/Research/recovery-from-failure/checkpoints/drq_mobilenet_her_masking_20260326_183727/step_300000",
    "Path to checkpoint directory.",
)
flags.DEFINE_integer("checkpoint_step", -1, "Checkpoint step to load (-1 for latest).")
flags.DEFINE_string("save_dir", "./donkey-sim-to-carla-results/", "Directory to save evaluation results.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("num_episodes", 10, "Number of episodes for evaluation.")
flags.DEFINE_integer("max_episode_steps", 1000, "Maximum steps per episode.")
flags.DEFINE_boolean("deterministic", True, "Use deterministic policy (no exploration noise).")
flags.DEFINE_integer("frame_stack", 3, "Number of frames to stack (must match training).")
flags.DEFINE_integer("mobilenet_blocks", 4, "Number of MobileNetV3 blocks (must match training).")
flags.DEFINE_integer("mobilenet_input_size", 84, "Input size for MobileNetV3 (must match training).")
flags.DEFINE_boolean("verbose", True, "Print detailed episode information.")
flags.DEFINE_string("carla_server_path", "/opt/carla-simulator/CarlaUE4.sh", "Path to CARLA server executable.")
flags.DEFINE_string("carla_host", "localhost", "CARLA host.")
flags.DEFINE_integer("carla_port", 2000, "CARLA port.")
flags.DEFINE_string("carla_town", "Town01", "CARLA town.")
flags.DEFINE_integer("carla_fps", 20, "CARLA simulation FPS.")

config_flags.DEFINE_config_file(
    "config",
    "./configs/drq_robot.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

IMU_DIM = 6


# ============================================================================
# CarlaInfoWrapper: bridges CarlaEnv to the RL stack
# ============================================================================

class CarlaInfoWrapper(gym.Wrapper):
    """
    Adds pos, forward_vel, yaw, accel, gyro to info dict on every step/reset,
    and remaps rgb_camera → pixels so StackingWrapper can find it.
    Also adds 'hit' key for RewardWrapper / GoalRelObservationWrapper compatibility.
    """

    def __init__(self, env: CarlaEnv):
        super().__init__(env)
        self._prev_velocity = None

    def _make_info(self, env: CarlaEnv, done: bool) -> dict:
        agent = env._agent
        loc = agent.get_location()
        vel = agent.get_velocity()
        rot = agent.get_transform().rotation

        forward_speed = np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

        imu_data = env._sensor_manager.data.get("imu", {})
        accel = imu_data.get("accelerometer", np.zeros(3))
        gyro = imu_data.get("gyroscope", np.zeros(3))
        # compass: 0=North, positive=East in CARLA
        compass = imu_data.get("compass", 0.0)
        yaw_from_compass = -compass  # convert to math convention (CCW positive)

        info = {
            "pos": np.array([loc.x, loc.y, loc.z], dtype=np.float32),
            "forward_vel": float(forward_speed),
            "yaw": float(yaw_from_compass),
            "accel": np.asarray(accel, dtype=np.float32),
            "gyro": np.asarray(gyro, dtype=np.float32),
            # 'hit' key used by GoalRelObservationWrapper for collision termination
            "hit": "collision" if env._sensor_manager.collision_detected else "none",
        }
        return info

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        info = self._make_info(self.env, done=False)
        # Remap rgb_camera → pixels for StackingWrapper
        pixels = self.env._sensor_manager.data.get("rgb_camera", obs.get("pixels"))
        obs_out = {"pixels": pixels}
        return obs_out, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Inject additional fields
        extra = self._make_info(self.env, done=(terminated or truncated))
        info.update(extra)
        # Remap rgb_camera → pixels
        pixels = self.env._sensor_manager.data.get("rgb_camera", obs.get("pixels"))
        obs_out = {"pixels": pixels}
        return obs_out, reward, terminated, truncated, info


# ============================================================================
# ResizeWrapper: resize camera output to 120x160 to match StackingWrapper
# (StackingWrapper hardcodes 120x160 — this is consistent with training on Donkey)
# ============================================================================

class ResizeWrapper(gym.Wrapper):
    """
    Resize CARLA RGB output from (84, 84, 4) BGRA → (120, 160, 3) RGB
    to match the StackingWrapper shape used during training.
    """

    def __init__(self, env: gym.Env, height: int = 120, width: int = 160):
        super().__init__(env)
        self.height = height
        self.width = width

    def _process_obs(self, obs):
        # obs is {"pixels": array of shape (H, W, 4) BGRA uint8}
        pixels = obs.get("pixels", obs.get("image"))
        if pixels is None:
            return obs
        # BGRA → RGB
        if pixels.shape[-1] == 4:
            import cv2
            rgb = cv2.cvtColor(pixels, cv2.COLOR_BGRA2RGB)
        else:
            rgb = pixels
        # Resize to Donkey sim dimensions
        import cv2
        resized = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        obs = dict(obs)
        obs["pixels"] = resized
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._process_obs(obs), reward, terminated, truncated, info


# ============================================================================
# CarlaStackingWrapper: 84x84-compatible StackingWrapper for CARLA
# (avoids hardcoded 120x160 of the Donkey-trained wrapper)
# ============================================================================

class CarlaStackingWrapper(gym.Wrapper):
    """
    Stacks RGB frames, actions, and IMU readings.
    Uses 84x84 CARLA camera size instead of Donkey's 120x160.

    Observation dict shapes (frame_stack=3):
        'pixels'  : (84, 84, 3 * num_stack)   e.g. (84, 84, 9)
        'actions' : (num_stack * action_dim,)    e.g. (6,)
        'imu'     : (num_stack * IMU_DIM,)       e.g. (18,)
    """

    def __init__(self, env, num_stack: int = 3):
        super().__init__(env)
        self.num_stack = num_stack
        self.num_stack_prop = num_stack  # 1x for CARLA (not 2x like Donkey)

        self.action_history = deque(maxlen=self.num_stack_prop)
        self.rgb_history = deque(maxlen=self.num_stack)
        self.imu_history = deque(maxlen=self.num_stack_prop)

        self.action_dim = env.action_space.shape[0]
        self.H = 84
        self.W = 84

        action_stack_shape = (self.num_stack_prop * self.action_dim,)
        imu_stack_shape = (self.num_stack_prop * IMU_DIM,)

        self.observation_space = spaces.Dict({
            "pixels": spaces.Box(
                low=0, high=255,
                shape=(self.H, self.W, 3 * num_stack),
                dtype=np.uint8,
            ),
            "actions": spaces.Box(
                low=np.tile(env.action_space.low, self.num_stack_prop),
                high=np.tile(env.action_space.high, self.num_stack_prop),
                shape=action_stack_shape,
                dtype=np.float32,
            ),
            "imu": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=imu_stack_shape,
                dtype=np.float32,
            ),
        })

    def _get_stacked_actions(self) -> np.ndarray:
        actions_list = list(self.action_history)
        while len(actions_list) < self.num_stack_prop:
            actions_list.insert(0, np.zeros(self.action_dim, dtype=np.float32))
        return np.concatenate(actions_list).astype(np.float32)

    def _get_stacked_rgb(self) -> np.ndarray:
        return np.concatenate(list(self.rgb_history), axis=-1).astype(np.uint8)

    def _get_stacked_imu(self) -> np.ndarray:
        imu_list = list(self.imu_history)
        while len(imu_list) < self.num_stack_prop:
            imu_list.insert(0, np.zeros(IMU_DIM, dtype=np.float32))
        return np.concatenate(imu_list).astype(np.float32)

    def _build_obs(self) -> dict:
        return {
            "pixels": self._get_stacked_rgb(),
            "actions": self._get_stacked_actions(),
            "imu": self._get_stacked_imu(),
        }

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        import cv2
        # Handle: obs may be {"pixels": ...} or raw array
        if isinstance(obs, dict):
            pixels = obs.get("pixels")
        else:
            pixels = obs

        if pixels is not None and pixels.ndim == 3 and pixels.shape[-1] == 4:
            # BGRA → RGB (from CARLA camera)
            pixels = cv2.cvtColor(pixels, cv2.COLOR_BGRA2RGB)
        elif pixels is not None and pixels.ndim == 2:
            # Grayscale → RGB
            pixels = cv2.cvtColor(pixels, cv2.COLOR_GRAY2RGB)

        self.action_history.append(action.astype(np.float32))
        self.rgb_history.append(pixels)
        self.imu_history.append(extract_imu_vector(info))

        return self._build_obs(), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if isinstance(obs, dict):
            pixels = obs.get("pixels")
        else:
            pixels = obs

        if pixels is not None and pixels.ndim == 3 and pixels.shape[-1] == 4:
            pixels = cv2.cvtColor(pixels, cv2.COLOR_BGRA2RGB)
        elif pixels is not None and pixels.ndim == 2:
            pixels = cv2.cvtColor(pixels, cv2.COLOR_GRAY2RGB)

        self.action_history.clear()
        self.rgb_history.clear()
        self.imu_history.clear()

        zero_imu = np.zeros(IMU_DIM, dtype=np.float32)
        for _ in range(self.num_stack):
            self.rgb_history.append(pixels)
        for _ in range(self.num_stack_prop):
            self.action_history.append(np.zeros(self.action_dim, dtype=np.float32))
            self.imu_history.append(zero_imu)

        return self._build_obs(), info


# ============================================================================
# IMU extraction (compatible with wrappers.py)
# ============================================================================

def extract_imu_vector(info: dict) -> np.ndarray:
    accel = np.asarray(info.get("accel", [0.0, 0.0, 0.0]), dtype=np.float32)
    gyro = np.asarray(info.get("gyro", [0.0, 0.0, 0.0]), dtype=np.float32)
    if accel.shape != (3,):
        accel = np.zeros(3, dtype=np.float32)
    if gyro.shape != (3,):
        gyro = np.zeros(3, dtype=np.float32)
    return np.concatenate([[accel[0], accel[2], accel[1]],
                           [gyro[0], gyro[2], gyro[1]]])


# ============================================================================
# GoalRelObservationWrapper (copied from train_her_robot.py for isolation)
# ============================================================================

class GoalRelObservationWrapper(gym.Wrapper):
    """
    Goal-relative observation wrapper with HER-style polar goal space.
    Tracks pseudo-odometry and computes polar goal-relative state [rel_angle, rel_dist, mask].
    Supports goal masking with stochastic masking of the goal.
    """

    def __init__(
        self,
        env,
        goal_range: float = 20.0,
        goal_threshold: float = 1.0,
        use_goal_masking: bool = True,
        mask_probability: float = 0.3,
    ):
        super().__init__(env)
        self.goal_range = goal_range
        self.goal_threshold = goal_threshold
        self.use_goal_masking = use_goal_masking
        self.mask_probability = mask_probability

        self.agent_x = 0.0
        self.agent_y = 0.0
        self.agent_yaw = 0.0
        self.current_goal_abs = np.zeros(2, dtype=np.float32)

        self.stop_count = 0
        self.goals_reached = 0
        self.total_distance = 0.0

        if use_goal_masking:
            goal_rel_space = spaces.Box(
                low=-np.array([np.pi, goal_range, 0], dtype=np.float32),
                high=np.array([np.pi, goal_range, 1], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            goal_rel_space = spaces.Box(
                low=-np.array([np.pi, goal_range, 0], dtype=np.float32),
                high=np.array([np.pi, goal_range, 0.1], dtype=np.float32),
                dtype=np.float32,
            )

        if isinstance(self.observation_space, spaces.Dict):
            new_spaces = dict(self.observation_space.spaces)
            new_spaces["goal_rel"] = goal_rel_space
            self.observation_space = spaces.Dict(new_spaces)
        else:
            self.observation_space = spaces.Dict(
                {"obs": self.observation_space, "goal_rel": goal_rel_space}
            )

    def _get_obs_dict(self, base_obs):
        dx = self.current_goal_abs[0] - self.agent_x
        dy = self.current_goal_abs[1] - self.agent_y
        dist = np.hypot(dx, dy)

        abs_angle = np.arctan2(dy, dx)
        rel_angle = (abs_angle - self.agent_yaw + np.pi) % (2 * np.pi) - np.pi

        mask = 1.0
        if self.use_goal_masking and np.random.random() < self.mask_probability:
            mask = 0.0

        goal_rel = np.array([rel_angle, dist, mask], dtype=np.float32)

        if isinstance(base_obs, dict) and "goal_rel" not in base_obs:
            out = dict(base_obs)
        elif isinstance(base_obs, dict):
            out = dict(base_obs)
        else:
            out = {"obs": base_obs}

        out["goal_rel"] = goal_rel
        return out

    def _update_info(self, info):
        info["agent_pos"] = np.array([self.agent_x, self.agent_y], dtype=np.float32)
        info["agent_yaw"] = np.array([self.agent_yaw], dtype=np.float32)
        info["distance"] = self.total_distance
        return info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.agent_x = 0.0
        self.agent_y = 0.0
        self.agent_yaw = info.get("yaw", 0.0)
        self.total_distance = 0.0
        self.stop_count = 0

        # Sample initial goal in polar coords relative to current pose
        rel_angle = np.random.uniform(-np.pi, np.pi)
        dist = np.random.uniform(0.0, self.goal_range)
        abs_angle = self.agent_yaw + rel_angle
        self.current_goal_abs = np.array([
            self.agent_x + dist * np.cos(abs_angle),
            self.agent_y + dist * np.sin(abs_angle),
        ], dtype=np.float32)

        return self._get_obs_dict(obs), self._update_info(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        vel = info.get("forward_vel", 0.0)
        self.agent_yaw = info.get("yaw", 0.0)

        if vel > 0.01:
            dist_step = 0.01
            self.total_distance += dist_step
            self.agent_x += dist_step * np.cos(self.agent_yaw)
            self.agent_y += dist_step * np.sin(self.agent_yaw)

        dx = self.current_goal_abs[0] - self.agent_x
        dy = self.current_goal_abs[1] - self.agent_y
        dist = np.hypot(dx, dy)

        reward = -0.1 + vel

        if dist < self.goal_threshold:
            self.goals_reached += 1
            reward += 10.0
            # Resample goal
            rel_angle = np.random.uniform(-np.pi, np.pi)
            new_dist = np.random.uniform(0.0, self.goal_range)
            abs_angle = self.agent_yaw + rel_angle
            self.current_goal_abs = np.array([
                self.agent_x + new_dist * np.cos(abs_angle),
                self.agent_y + new_dist * np.sin(abs_angle),
            ], dtype=np.float32)

        self.stop_count = self.stop_count + 1 if vel < 0.01 else 0
        if self.stop_count > 20:
            terminated = True
            reward -= 100

        if info.get("hit", "none") != "none":
            terminated = True
            reward -= 100

        return self._get_obs_dict(obs), reward, terminated, truncated, self._update_info(info)


# ============================================================================
# Action translation: agent [steer, throttle] → carla.VehicleControl
# ============================================================================

def agent_action_to_control(action: np.ndarray):
    """Convert agent action [steer, throttle] to carla.VehicleControl."""
    steer, throttle = float(action[0]), float(np.clip(action[1], 0, 1))
    import carla
    return carla.VehicleControl(throttle=throttle, steer=steer, brake=0.0)


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_agent(
    env,
    agent,
    num_episodes: int,
    deterministic: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Evaluate agent on CARLA environment.

    Tracks: returns, lengths, distances, collision counts, success rate.
    """
    episode_returns = []
    episode_lengths = []
    episode_distances = []
    episode_collisions = []
    episode_info_list = []

    print(f"\n{'='*70}")
    print(f"Evaluating agent for {num_episodes} episodes")
    print(f"Deterministic policy: {deterministic}")
    print(f"{'='*70}\n")

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_return = 0.0
        episode_length = 0
        done = False

        episode_actions = []
        episode_distance = 0.0
        episode_collision_count = 0

        while not done:
            if deterministic:
                action = agent.eval_actions(obs)
            else:
                action = agent.sample_actions(obs)

            # Agent outputs [steer, throttle] in [-1, 1] x [0, 0.160]
            action = np.array([
                np.clip(action[0], -1.0, 1.0),
                np.clip(action[1], 0.0, 0.160),
            ])
            episode_actions.append(action.copy())

            # Translate to CARLA control and step
            control = agent_action_to_control(action)
            obs, reward, terminated, truncated, info = env.step(control)
            done = terminated or truncated

            episode_return += reward
            episode_length += 1

            if "distance" in info:
                episode_distance = info["distance"]

            # Count collisions (each sensor trigger = one collision event)
            if info.get("hit", "none") != "none":
                episode_collision_count += 1

        # Store episode statistics
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        episode_distances.append(episode_distance)
        episode_collisions.append(episode_collision_count)

        # Success: episode ended without collision
        success = (info.get("hit", "none") == "none") and (episode_length < env._max_episode_steps)

        episode_data = {
            "episode": episode,
            "return": float(episode_return),
            "length": int(episode_length),
            "distance": float(episode_distance),
            "collision_count": int(episode_collision_count),
            "success": bool(success),
            "actions": [a.tolist() for a in episode_actions],
        }
        episode_info_list.append(episode_data)

        if verbose:
            print(f"Episode {episode + 1}/{num_episodes}:")
            print(f"  Return:          {episode_return:>10.2f}")
            print(f"  Length:          {episode_length:>10d} steps")
            print(f"  Distance:        {episode_distance:>10.1f}")
            print(f"  Collisions:      {episode_collision_count:>10d}")
            print(f"  Success:         {success!s:>10s}")

            actions_array = np.array(episode_actions)
            print(f"  Avg Steering:    {np.mean(actions_array[:, 0]):>+7.3f} "
                  f"(std: {np.std(actions_array[:, 0]):.3f})")
            print(f"  Avg Throttle:    {np.mean(actions_array[:, 1]):>+7.3f} "
                  f"(std: {np.std(actions_array[:, 1]):.3f})")
            print()

    # Aggregate statistics
    total_collisions = sum(episode_collisions)
    successes = sum(1 for e in episode_info_list if e["success"])
    stats = {
        "num_episodes": num_episodes,
        "total_collisions": total_collisions,
        "success_rate": successes / num_episodes,
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "min_return": float(np.min(episode_returns)),
        "max_return": float(np.max(episode_returns)),
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
        "mean_distance": float(np.mean(episode_distances)),
        "std_distance": float(np.std(episode_distances)),
        "mean_collisions": float(np.mean(episode_collisions)),
        "std_collisions": float(np.std(episode_collisions)),
        "episodes": episode_info_list,
    }

    print(f"\n{'='*70}")
    print("Evaluation Summary")
    print(f"{'='*70}")
    print(f"Episodes:         {stats['num_episodes']}")
    print(f"Total Collisions: {stats['total_collisions']}")
    print(f"Success Rate:     {stats['success_rate']:.1%}")
    print(f"Mean Return:      {stats['mean_return']:>10.2f} ± {stats['std_return']:.2f}")
    print(f"Return Range:     [{stats['min_return']:.2f}, {stats['max_return']:.2f}]")
    print(f"Mean Length:      {stats['mean_length']:>10.1f} ± {stats['std_length']:.1f}")
    print(f"Mean Distance:    {stats['mean_distance']:>10.1f} ± {stats['std_distance']:.1f}")
    print(f"Mean Collisions:  {stats['mean_collisions']:>10.2f} ± {stats['std_collisions']:.2f}")
    print(f"{'='*70}\n")

    return stats


# ============================================================================
# Main
# ============================================================================

def main(_):
    print("\n" + "="*70)
    print("DrQ + HER + Goal Masking — CARLA Evaluation")
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
    print(f"CARLA: {FLAGS.carla_host}:{FLAGS.carla_port} ({FLAGS.carla_town})")

    # Create CARLA environment
    print("\nCreating CARLA environment...")
    carla_cfg = CarlaConfig(
        server_path="/home/kojogyaase/Apps/CARLA_0.9.16/CarlaUE4.sh",
        host=FLAGS.carla_host,
        port=FLAGS.carla_port,
        town=FLAGS.carla_town,
        fps=FLAGS.carla_fps,
        camera=CameraConfig(width=FLAGS.mobilenet_input_size, height=FLAGS.mobilenet_input_size, fov=90.0),
    )
    carla_env = CarlaEnv(config=carla_cfg, reward_fn=RewardFunction())

    # CarlaEnv exposes carla.VehicleControl actions; define action space for gym wrapper
    carla_env.action_space = spaces.Box(
        low=np.array([-1.0, 0.0], dtype=np.float32),
        high=np.array([1.0, 0.160], dtype=np.float32),
        dtype=np.float32,
    )
    carla_env.observation_space = spaces.Dict({
        "pixels": spaces.Box(low=0, high=255, shape=(84, 84, 4), dtype=np.uint8),
    })

    # Wrap: Carla → EnvCompatibility → CarlaInfo → CarlaStacking(84x84) → MobileNet → GoalRel → RecordStats → TimeLimit
    env = EnvCompatibility(carla_env)
    env = CarlaInfoWrapper(env)
    env = CarlaStackingWrapper(env, num_stack=FLAGS.frame_stack)
    env = MobileNetFeatureWrapper(
        env,
        device=device,
        num_blocks=FLAGS.mobilenet_blocks,
        input_size=FLAGS.mobilenet_input_size,
    )
    env = GoalRelObservationWrapper(
        env,
        goal_range=20.0,
        goal_threshold=1.0,
        use_goal_masking=True,
        mask_probability=0.3,
    )
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

    agent = DrQLearner(
        FLAGS.seed,
        env.observation_space.sample(),
        env.action_space.sample(),
        **kwargs,
    )

    # Load checkpoint
    print("\nLoading checkpoint...")
    agent = load_checkpoint(agent, FLAGS.checkpoint_path)

    # Evaluate
    print("\nStarting evaluation...")
    stats = evaluate_agent(
        env,
        agent,
        num_episodes=FLAGS.num_episodes,
        deterministic=FLAGS.deterministic,
        verbose=FLAGS.verbose,
    )

    # Save results
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    experiment_name = "donkey-sim-to-carla-results"
    results_file = os.path.join(
        FLAGS.save_dir,
        f"{experiment_name}_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )

    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj

    stats = convert_to_serializable(stats)

    stats["metadata"] = {
        "checkpoint_path": FLAGS.checkpoint_path,
        "carla_host": FLAGS.carla_host,
        "carla_port": FLAGS.carla_port,
        "carla_town": FLAGS.carla_town,
        "deterministic": FLAGS.deterministic,
        "num_episodes": FLAGS.num_episodes,
        "max_episode_steps": FLAGS.max_episode_steps,
        "frame_stack": FLAGS.frame_stack,
        "mobilenet_blocks": FLAGS.mobilenet_blocks,
        "mobilenet_input_size": FLAGS.mobilenet_input_size,
        "goal_masking": True,
        "mask_probability": 0.3,
        "goal_range": 20.0,
        "goal_threshold": 1.0,
        "timestamp": datetime.now().isoformat(),
    }

    with open(results_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Results saved to: {results_file}")

    # Cleanup
    carla_env.close()
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    app.run(main)
