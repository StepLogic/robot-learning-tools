
#!/usr/bin/env python
"""
DrQ Training with JAX, MobileNetV3 encoder, and Temporal Modeling
Real Robot Training with Human-in-the-Loop (HitL)
- MobileNetV3 visual feature extraction (PyTorch)
- Frame stacking with feature extraction
- Action+IMU history tracking (RacerEnv)
- Expert/teleop buffer for human demonstrations
- Frozen temperature for online updates
- Human-in-the-Loop override (WASD keys for control)
- Hindsight Experience Replay (HER) for goal relabeling
- Goal masking with state value (0 for masked, 1 for reliable)
- Goals specified as relative angle and relative distance
- Concurrent training thread (asynchronous gradient updates)
- Async checkpoint/teleop saving
"""
import random
import os
import re
import pickle
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import tqdm
import uuid

import flax
import jax
import numpy as np
from absl import app, flags
from ml_collections import config_flags

import torch
import pygame

import gymnasium as gym
from jaxrl2.agents import DrQLearner
from jaxrl2.data import ReplayBuffer
from jaxrl2.noise import OrnsteinUhlenbeckActionNoise
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from jaxrl2.wrappers.timelimit import TimeLimit

from racer_imu_env import RacerEnv, StackingWrapper, RewardWrapper
from wrappers import (
    EnvCompatibility,
    Logger,
    MobileNetFeatureWrapper,
    load_checkpoint,
    save_checkpoint,
)

flax.config.update("flax_use_orbax_checkpointing", True)

FLAGS = flags.FLAGS

# Configuration flags
flags.DEFINE_string("env_name", "donkey-warehouse-v0", "Environment name.")
flags.DEFINE_string("sim", "sim_path", "Path to unity simulator.")
flags.DEFINE_integer("port", 9091, "Port to use for tcp.")
flags.DEFINE_string("save_dir", "./logs/", "Tensorboard logging dir.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", int(50000), "Eval interval.")
flags.DEFINE_integer("checkpoint_interval", 1000, "Checkpoint saving interval.")
flags.DEFINE_integer("batch_size", 64, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer("start_training", int(1e3), "Number of training steps to start training.")
flags.DEFINE_integer("replay_buffer_size", int(1e4), "Replay buffer size.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_integer("frame_stack", 3, "Number of frames to stack for temporal modeling.")
flags.DEFINE_integer("mobilenet_blocks", 4, "Number of MobileNetV3 blocks to use.")
flags.DEFINE_integer("mobilenet_input_size", 84, "Input size for MobileNetV3.")

# Resume flags
flags.DEFINE_string(
    "pretrained_checkpoint",
    "/home/kojogyaase/Projects/Research/recovery-from-failure/pretrained_policy/step_520000",
    "Path to pre-trained checkpoint (used only when robot_policy has no prior runs).",
)
flags.DEFINE_string(
    "teleop_buffer_path",
    "/home/kojogyaase/Projects/Research/recovery-from-failure/teleop_buffer.pkl",
    "Path to save/resume teleop (human) transitions.",
)

# Expert sampling flags
flags.DEFINE_float(
    "expert_sample_ratio", 0.25,
    "Fraction of each mixed batch drawn from the expert teleop buffer.",
)

# HitL flags
flags.DEFINE_float("steer_step", 0.05, "Steering increment per keypress.")
flags.DEFINE_float("throttle_step", 0.02, "Throttle increment per keypress.")
flags.DEFINE_integer("teleop_save_every", 500, "Auto-save teleop buffer every N human steps.")

# HER-specific flags
flags.DEFINE_float("her_fraction", 0.5, "Fraction of relabeled goals (HER strategy).")
flags.DEFINE_string("her_strategy", "future", "HER strategy: 'future', 'final', or 'random'.")
# Goal wrapper flags
flags.DEFINE_float("goal_range", 20.0, "Range for random goal sampling.")
flags.DEFINE_float("goal_threshold", 1.0, "Distance threshold to consider goal reached.")
flags.DEFINE_boolean("use_goal_masking", True, "Enable goal masking with state values.")
flags.DEFINE_float("mask_probability", 0.3, "Probability of masking a goal (setting state value to 0).")
config_flags.DEFINE_config_file(
    "config",
    "./configs/drq_robot.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

# ============================================================================
# Teleop buffer helpers
# ============================================================================

def _load_teleop_buffer(path, obs_space, act_space, seed, capacity):
    buf = ReplayBuffer(obs_space, act_space, capacity)
    buf.seed(seed)
    if not os.path.exists(path):
        print(f"[Teleop] No existing file at {path} - starting fresh.")
        return buf, 0
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, ReplayBuffer):
            buf.dataset_dict  = data.dataset_dict
            buf._insert_index = data._insert_index
            buf._size         = data._size
            prior = data._size
        elif isinstance(data, dict) and "data" in data:
            buf.dataset_dict  = data["data"]
            buf._insert_index = data["insert_index"]
            buf._size         = data["size"]
            prior = data["size"]
        else:
            transitions = data if isinstance(data, list) else [data]
            for t in transitions:
                try:
                    buf.insert(t)
                except Exception:
                    pass
            prior = buf._size
        print(f"[Teleop] Loaded {prior:,} transitions from {path}")
        return buf, prior
    except Exception as e:
        print(f"[Teleop] Load failed ({e}) - starting fresh.")
        return buf, 0


def _save_teleop_buffer(buf: ReplayBuffer, path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({
            "data":         buf.dataset_dict,
            "insert_index": buf._insert_index,
            "size":         buf._size,
        }, f)
    os.replace(tmp, path)   # atomic - never corrupts on crash
    print(f"[Teleop] Saved {buf._size:,} transitions - {path}")


# ============================================================================
# Mixed-batch generator
# ============================================================================

def _mixed_batch_gen(online_iter, teleop_buf, batch_size, expert_ratio):
    n_expert = int(batch_size * expert_ratio) if teleop_buf is not None else 0
    n_online = batch_size - n_expert
    while True:
        online_batch = next(online_iter) if n_online > 0 else None
        expert_batch = (
            teleop_buf.sample(n_expert)
            if n_expert > 0 and teleop_buf is not None and teleop_buf._size >= n_expert
            else None
        )
        if online_batch is None:
            yield expert_batch
        elif expert_batch is None:
            yield online_batch
        else:
            yield jax.tree_util.tree_map(
                lambda o, e: np.concatenate([np.asarray(o), np.asarray(e)], axis=0),
                online_batch, expert_batch,
            )


# ============================================================================
# Freeze temperature
# ============================================================================

def _freeze_temperature(agent):
    try:
        agent._frozen_temp_state = agent._temp
        _orig = agent.update

        def _update_frozen(batch, **kw):
            info = _orig(batch, **kw)
            agent._temp = agent._frozen_temp_state
            return info

        def _update_with_temp(batch, **kw):
            info = _orig(batch, **kw)
            agent._frozen_temp_state = agent._temp
            return info

        agent.update           = _update_frozen
        agent.update_with_temp = _update_with_temp
        print("[Temperature] Frozen for online updates  (agent.update).")
        print("[Temperature] Free   for expert updates  (agent.update_with_temp).")
    except AttributeError as e:
        print(f"[Temperature] Could not patch: {e}")
        agent.update_with_temp = agent.update
    return agent


# ============================================================================
# Human controller (HitL)
# ============================================================================

class HumanController:
    def __init__(self, action_low, action_high):
        self.low      = action_low
        self.high     = action_high
        self.steering  = 0.0
        self.throttle  = 0.130
        self.paused    = False

    def process_events(self):
        quit_req = reset_req = tog_pause = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_req = True
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    quit_req = True
                if event.key == pygame.K_r:
                    reset_req = True
                if event.key == pygame.K_t:
                    tog_pause = True
                if event.key == pygame.K_SPACE:
                    self.throttle = 0.130
        return quit_req, reset_req, tog_pause

    def read(self):
        keys = pygame.key.get_pressed()
        steer_key = throttle_key = False
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self.steering = max(self.low[0],  self.steering - FLAGS.steer_step)
            steer_key = True
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.steering = min(self.high[0], self.steering + FLAGS.steer_step)
            steer_key = True
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self.throttle = min(self.high[1], self.throttle + FLAGS.throttle_step)
            throttle_key = True
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.throttle = max(self.low[1],  self.throttle - FLAGS.throttle_step)
            throttle_key = True
        human_active = steer_key or throttle_key or self.paused
        return np.array([self.steering, self.throttle], dtype=np.float32), human_active

    def reset_controls(self):
        self.steering = 0.0
        self.throttle = 0.130


# ============================================================================
# HUD overlay
# ============================================================================

def _draw_hud(screen, steering, throttle, step, ep, ep_steps,
              ep_reward, human_active, paused, teleop_size, buf_cap):
    if screen is None:
        return
    if not pygame.font.get_init():
        pygame.font.init()
    font = pygame.font.SysFont("monospace", 15, bold=True)

    def txt(msg, x, y, color=(255, 255, 0)):
        screen.blit(font.render(msg, True, color), (x, y))

    bar = pygame.Surface((screen.get_width(), 90), pygame.SRCALPHA)
    bar.fill((0, 0, 0, 170))
    screen.blit(bar, (0, screen.get_height() - 90))
    y0 = screen.get_height() - 88

    if paused:
        lbl, col = "HUMAN [PAUSED]", (255,  80,  80)
    elif human_active:
        lbl, col = "HUMAN OVERRIDE", ( 80, 255,  80)
    else:
        lbl, col = "AGENT",          ( 80, 180, 255)

    txt(f"[ {lbl} ]", screen.get_width() - 220, y0, color=col)
    txt(f"Steer {steering:+.3f}   Throttle {throttle:+.3f}", 10, y0)
    txt(f"Step {step:>7d}   Ep {ep:>4d}   Ep-step {ep_steps:>4d}", 10, y0 + 18)
    txt(f"Ep reward {ep_reward:+.2f}   Teleop {teleop_size:>6d}/{buf_cap}", 10, y0 + 36)
    txt("W/S=throttle  A/D=steer  SPC=coast  T=pause  R=reset  Q=quit",
        10, y0 + 54, color=(190, 190, 190))

    bw, bh = 180, 8
    bx, by = screen.get_width() - bw - 10, y0 + 20
    cx = bx + bw // 2
    pygame.draw.rect(screen, (60, 60, 60), (bx, by, bw, bh))
    fw  = int(abs(steering) * (bw // 2))
    col = (0, 200, 100) if steering >= 0 else (220, 80, 0)
    pygame.draw.rect(screen, col, (cx if steering >= 0 else cx - fw, by, fw, bh))
    pygame.draw.line(screen, (255, 255, 255), (cx, by), (cx, by + bh), 1)
    pygame.display.flip()


# ============================================================================
# Gradient-update thread
# ============================================================================

def _train_thread(
    agent,
    batch_queue:     deque,
    expert_iter,
    episode_event:   threading.Event,
    stop_event:      threading.Event,
    update_info_box: list,
    n_updates:       int,
    utd_ratio:       int,
    teleop_buf:      ReplayBuffer,
):
    """
    Sleeps until main signals episode end, then runs all gradient updates.

    Protocol:
    - Main appends pre-sampled batches to batch_queue every step.
    - At episode end main calls episode_event.set().
    - This thread wakes, runs n_updates gradient steps, writes the last
      update_info into update_info_box[0], then sleeps again.
    - stop_event tells the thread to exit cleanly.
    """
    while not stop_event.is_set():
        triggered = episode_event.wait(timeout=0.5)
        if stop_event.is_set():
            break
        if not triggered:
            continue
        episode_event.clear()

        info = {}
        for _ in range(n_updates):
            # Expert update - temperature FROZEN
            if expert_iter is not None and teleop_buf._size >= FLAGS.batch_size:
                agent.update(next(expert_iter), utd_ratio=utd_ratio)

            # Mixed online+expert update - temperature FREE
            if batch_queue:
                info = agent.update_with_temp(batch_queue.popleft(), utd_ratio=utd_ratio)

        if info:
            update_info_box[0] = info

def _find_latest_checkpoint(folder: str) -> tuple[str | None, int]:
    if not os.path.isdir(folder):
        return None, 0
    best_path, best_step = None, 0
    for name in os.listdir(folder):
        m = re.fullmatch(r"step_(\d+)", name)
        if m:
            n = int(m.group(1))
            if n > best_step:
                best_step = n
                best_path = os.path.join(folder, name)
    return (os.path.abspath(best_path), best_step) if best_path else (None, 0)


# ============================================================================
# Teleop buffer helpers
# ============================================================================

class GoalRelObservationWrapper(gym.Wrapper):
    """
    Goal-relative observation wrapper with random relative position goals.
    Features:
    - Goal masking with state values (0 = masked/random, 1 = reliable)
    - Goals specified as relative angle and relative distance
    - Optional goal image support with MobileNetV3 features
    - Full relabeling after episode completion
    """

    def __init__(
        self,
        env,
        goal_range: float = 20.0,
        goal_threshold: float = 1.0,
        use_goal_masking: bool = True,
        mask_probability: float = 0.3,
        use_goal_image: bool = True,
    ):
        super().__init__(env)
        self.goal_range = goal_range  # Range for random goal sampling
        self.goal_threshold = goal_threshold  # Distance to consider goal "reached"
        self.use_goal_masking = use_goal_masking
        self.mask_probability = mask_probability
        self.use_goal_image = use_goal_image
        self.stop_count = 0
        self.distance_to_goal = None
        self.current_goal = None
        self.goals_reached = 0  # Track number of goals reached
        self.start_pos = None  # Starting position for distance calculation
        self.prev_pos = None  # Previous position for distance accumulation
        self.total_distance = 0.0  # Cumulative distance traveled in episode
        self.goal_state_value = 1.0  # 1 = reliable, 0 = masked
        self.current_goal_image = None  # Store goal image if enabled
        self.encoder = None  # Will be initialized lazily in reset()

        # Goal relative space with masking:
        # [rel_angle, rel_distance, state_value] (if using angle/distance)
        # or [rel_x, rel_y, rel_z, state_value] (if using position)
        if use_goal_masking:
            # Add state value dimension for masking
            goal_rel_space = spaces.Box(
                low=-np.array([np.pi, goal_range, 0], dtype=np.float32),
                high=np.array([np.pi, goal_range, 1], dtype=np.float32),
                dtype=np.float32
            )
        else:
            goal_rel_space = spaces.Box(
                low=-np.array([goal_range, goal_range, 0], dtype=np.float32),
                high=np.array([goal_range, goal_range, 0.1], dtype=np.float32),
                dtype=np.float32
            )

        # Add goal image space if enabled (MobNetV3 features shape)
        if use_goal_image:
            # Note: We don't know the feature dimension yet, will add it in reset()
            self._pending_goal_image = True
        else:
            self._pending_goal_image = False

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

    def _initialize_encoder(self):
        """Initialize the MobileNetV3 encoder lazily."""
        if self.encoder is None and self.use_goal_image:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.encoder = MobileNetV3Encoder(
                device=device,
                num_blocks=FLAGS.mobilenet_blocks,
                input_size=FLAGS.mobilenet_input_size
            )

            # Add goal_image space to observation space
            feature_dim = self.encoder.output_dim
            goal_image_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(feature_dim,),
                dtype=np.float32
            )
            if "goal_image" not in self.observation_space.spaces:
                new_spaces = dict(self.observation_space.spaces)
                new_spaces["goal_image"] = goal_image_space
                self.observation_space = spaces.Dict(new_spaces)

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

    def _convert_to_angle_distance(self, goal_rel: np.ndarray) -> np.ndarray:
        """
        Convert relative position to angle and distance representation.
        Returns: [rel_angle, rel_distance, state_value]
        """
        # Get angle (bearing) to goal
        x, y, z = goal_rel
        rel_angle = np.arctan2(y, x)  # Angle in radians
        rel_distance = np.linalg.norm(goal_rel[:2])  # 2D distance

        return np.array([rel_angle, rel_distance, self.goal_state_value], dtype=np.float32)

    def _convert_to_position(self, angle_dist: np.ndarray) -> np.ndarray:
        """Convert angle/distance back to position for debugging."""
        rel_angle, rel_distance = angle_dist[0], angle_dist[1]
        x = rel_distance * np.cos(rel_angle)
        y = rel_distance * np.sin(rel_angle)
        return np.array([x, y, 0], dtype=np.float32)

    def _is_goal_reached(self, goal_rel: np.ndarray) -> bool:
        """Check if the current goal has been reached."""
        return np.linalg.norm(goal_rel[:2]) < self.goal_threshold

    def _apply_goal_masking(self, goal_angle_dist: np.ndarray) -> np.ndarray:
        """Apply masking to goal - with some probability set state value to 0."""
        if self.use_goal_masking and np.random.random() < self.mask_probability:
            # Mask the goal by setting state value to 0 (random/uncertain)
            return np.array([goal_angle_dist[0], goal_angle_dist[1], 0.0], dtype=np.float32)
        return goal_angle_dist

    def _get_goal_image(self, pos: np.ndarray) -> np.ndarray:
        """
        Get a goal image visualization and extract MobileNetV3 features.
        This returns MobileNetV3 features extracted from a goal visualization image.
        """
        # Create a simple visualization: blank image with a marker at goal position
        img = np.zeros((84, 84, 3), dtype=np.uint8)
        # Scale position to image coordinates
        x = int((pos[0] + self.goal_range) / (2 * self.goal_range) * 84)
        y = int((pos[1] + self.goal_range) / (2 * self.goal_range) * 84)
        x = np.clip(x, 5, 79)
        y = np.clip(y, 5, 79)
        # Draw circle
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)  # Red channel

        # Extract MobileNetV3 features from the goal image
        features = self.encoder.encode(img)
        return features

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Initialize encoder lazily (only once)
        self._initialize_encoder()

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

        # Compute angle/distance representation with masking
        goal_angle_dist = self._convert_to_angle_distance(goal_rel)
        if self.use_goal_masking:
            goal_angle_dist = self._apply_goal_masking(goal_angle_dist)

        if isinstance(obs, dict):
            obs = dict(obs)
            obs["goal_rel"] = goal_angle_dist
            if self.use_goal_image:
                self.current_goal_image = self._get_goal_image(self.current_goal)
                obs["goal_image"] = self.current_goal_image
        else:
            obs = {
                "obs": obs,
                "goal_rel": goal_angle_dist,
            }
            if self.use_goal_image:
                self.current_goal_image = self._get_goal_image(self.current_goal)
                obs["goal_image"] = self.current_goal_image

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
        reward = -0.1

        # Check if goal reached - if so, resample a new random goal
        # Only resample if goal is NOT masked (state_value = 1)
        if self._is_goal_reached(goal_rel) and self.goal_state_value == 1.0:
            self.goals_reached += 1
            # terminated = True
            reward += 100
            # Resample new goal for next episode
            self.current_goal = self._sample_random_goal()
            print(f"Goal reached! Total goals reached: {self.goals_reached}, new goal sampled")

        vel = info.get("forward_vel", 0.0)
        reward += vel

        if vel < 0.01:
            self.stop_count += 1

        if self.stop_count > 20:
            terminated = True
            reward -= 100

        if info.get("hit", "none") != "none":
            terminated = True
            reward -= 100

        # Compute angle/distance representation with masking
        goal_angle_dist = self._convert_to_angle_distance(goal_rel)
        if self.use_goal_masking:
            goal_angle_dist = self._apply_goal_masking(goal_angle_dist)

        if isinstance(obs, dict):
            obs = dict(obs)
            obs["goal_rel"] = goal_angle_dist
            if self.use_goal_image:
                obs["goal_image"] = self.current_goal_image
        else:
            obs = {
                "obs": obs,
                "goal_rel": goal_angle_dist,
            }
            if self.use_goal_image:
                obs["goal_image"] = self.current_goal_image

        self.distance_to_goal = np.linalg.norm(goal_rel[:2])

        if terminated or truncated:
            print(f"Distance To Goal: {np.linalg.norm(goal_rel[:2]):.2f}")
            print(f"Total Distance Traveled: {self.total_distance:.2f}")
            print(f"Goal state value: {goal_angle_dist[2]:.1f} (0=masked, 1=reliable)")

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

        # Store original goals for relabeling
        self._original_goals: Dict[int, np.ndarray] = {}  # idx -> original goal

    def insert(self, data_dict: DatasetDict):
        """Insert transition and track episode indices."""
        current_idx = self._insert_index

        # Store original goal before any potential masking
        observations = data_dict.get('observations', {})
        if isinstance(observations, dict) and 'goal_rel' in observations:
            goal_rel = observations['goal_rel']
            self._original_goals[current_idx] = goal_rel.copy()

        super().insert(data_dict)
        self._current_episode_indices.append(current_idx)

    def end_episode(self):
        """Call when episode ends to store episode boundaries and relabel all goals."""
        if self._current_episode_indices:
            # Store episode boundaries
            self._episodes.append(self._current_episode_indices.copy())

            # Relabel ALL goals in this episode with achieved states from the episode
            self._relabel_all_goals(self._current_episode_indices)

            self._current_episode_indices = []

    def _relabel_all_goals(self, episode_indices: List[int]):
        """
        Relabel all goals in an episode using HER strategy.

        For each transition in the episode, replace the goal with an achieved
        state from the same episode. This enables learning from failed episodes
        by treating them as successful for different goals.
        """
        if len(episode_indices) < 2:
            return

        # Use 'final' strategy: use the final state as the relabeled goal
        # This is a simple but effective approach
        final_idx = episode_indices[-1]

        # Get the achieved goal from the final state
        final_achieved_goal = self._get_achieved_goal(final_idx)

        # Relabel all goals in the episode with the final achieved goal
        for idx in episode_indices:
            if final_achieved_goal is not None:
                # Store the relabeled goal
                self._original_goals[idx] = final_achieved_goal.copy()

    def _get_achieved_goal_from_buffer(self, idx: int) -> Optional[np.ndarray]:
        """Get the achieved goal at a specific buffer index."""
        try:
            # Get the observation at this index
            obs = self.dataset_dict.get('observations', {})
            if isinstance(obs, dict) and 'goal_rel' in obs:
                # Get the relative goal from the observation
                goal_rel = obs['goal_rel'][idx % len(obs['goal_rel'])]
                return goal_rel.copy()
        except (KeyError, IndexError, TypeError):
            pass
        return None

    def _get_achieved_goal(self, idx: int) -> Optional[np.ndarray]:
        """Get the achieved goal at the given buffer index."""
        try:
            # Get the observation at this index
            obs = self.dataset_dict['observations']
            if isinstance(obs, dict) and 'goal_rel' in obs:
                # For achieved goal, we use the stored original goals
                if idx in self._original_goals:
                    return self._original_goals[idx].copy()
                # Fallback: use the goal from the buffer
                return obs['goal_rel'][idx % len(obs['goal_rel'])].copy()
        except (KeyError, IndexError):
            pass
        return None

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
            state from the same episode.
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

                            # Get the achieved goal from the future state
                            achieved_goal = self._get_achieved_goal(achieved_idx)
                            if achieved_goal is not None:
                                # Replace the goal with the achieved goal
                                goals[i] = achieved_goal.copy()
                                # Update state value to 1 (reliable) for relabeled goals
                                if goals[i].shape[0] >= 3:
                                    goals[i][2] = 1.0  # Set state_value to 1

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
    print("DrQ Training with MobileNetV3 Feature Extraction + HER + Goal Masking")
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
        "bio": "Learning to drive with DrQ + MobileNetV3 + HER + Goal Masking",
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
    env = EnvCompatibility(env)

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
    env = GoalRelObservationWrapper(
        env,
        goal_range=FLAGS.goal_range,
        goal_threshold=FLAGS.goal_threshold,
        use_goal_masking=FLAGS.use_goal_masking,
        mask_probability=FLAGS.mask_probability,
        use_goal_image=FLAGS.use_goal_image,
    )

    print(f"\n{'='*60}")
    print("Environment Setup Complete")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Goal masking enabled: {FLAGS.use_goal_masking}")
    print(f"Mask probability: {FLAGS.mask_probability}")
    print(f"Goal image enabled: {FLAGS.use_goal_image}")
    print(f"{'='*60}\n")

    # RL Training Setup
    action_dim = env.action_space.shape[0]
    mean = np.zeros(action_dim)
    sigma = np.ones(action_dim)
    sigma[1] = 0.2 * sigma[1]
    noise = OrnsteinUhlenbeckActionNoise(mean=mean, sigma=sigma)

    logger = Logger(log_dir=FLAGS.save_dir)
    policy_folder = os.path.join(
        "checkpoints",
        f"drq_mobilenet_her_masking_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(policy_folder, exist_ok=True)

    # Initialize DrQ agent
    print("Initializing DrQ agent...")
    kwargs = dict(FLAGS.config) if FLAGS.config else {}

    # sample_obs, _ = env.reset()
    # sample_action = env.action_space.sample()
    sample_obs = env.observation_space.sample()
    sample_action = env.action_space.sample()
    print(f"Sample observation shapes:")
    if isinstance(sample_obs, dict):
        for key, val in sample_obs.items():
            if isinstance(val, np.ndarray):
                print(f"  {key}: {val.shape}")
            else:
                print(f"  {key}: {val}")
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
    print("Starting training with MobileNetV3 features + HER + Goal Masking")
    print(f"HER fraction: {FLAGS.her_fraction}, Strategy: {FLAGS.her_strategy}")
    print(f"Goal masking: {FLAGS.use_goal_masking}, Mask prob: {FLAGS.mask_probability}")
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
            dst = info.get("distance", 0)
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
