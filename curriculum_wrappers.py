#!/usr/bin/env python
"""
Curriculum Learning Wrapper for Goal Navigation RL.

Implements a 4-stage progressive curriculum:
  Stage 1 (Easy)    : Open arena, close heading goals, sparse obstacles
  Stage 2 (Medium)  : Wall obstacles, medium-distance goals, heading + image cues
  Stage 3 (Hard)    : Narrow corridors, distant goals, image-goal matching
  Stage 4 (Expert)  : Mixed tasks, adversarial obstacles, full image goals

The wrapper automatically manages stage transitions based on success rate
and provides a unified observation dict compatible with the existing
StackingWrapper -> MobileNetFeatureWrapper -> GoalRelObservationWrapper stack.

Observation dict keys:
    pixels  : visual features from MobileNet (or raw stacked RGB before encoder)
    actions : stacked action history
    imu     : stacked IMU vector (zeros for ViZDoom)
    goal_rel: relative position to current goal (x, y, z)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, List
from collections import deque
import pickle
import os


# =============================================================================
# Curriculum Stage Definitions
# =============================================================================

CURRICULUM_STAGES = {
    1: {
        "name": "easy",
        "scenario": "easy",
        "heading_goals": True,
        "image_goals": False,
        "goal_radius_min": 3.0,
        "goal_radius_max": 5.0,
        "max_steps": 1000,
        "step_penalty": -0.01,
        "goal_reward": 100.0,
        "collision_penalty": -10.0,
        "success_threshold": 0.70,  # 70% success rate to advance
        "min_episodes": 50,         # minimum episodes before evaluation
        "obstacle_density": 0.0,
        "nav_type": "heading",      # pure heading-based navigation
    },
    2: {
        "name": "medium",
        "scenario": "medium",
        "heading_goals": True,
        "image_goals": True,
        "goal_radius_min": 4.0,
        "goal_radius_max": 7.0,
        "max_steps": 1200,
        "step_penalty": -0.01,
        "goal_reward": 100.0,
        "collision_penalty": -50.0,
        "success_threshold": 0.60,
        "min_episodes": 60,
        "obstacle_density": 0.3,
        "nav_type": "heading+image",
    },
    3: {
        "name": "hard",
        "scenario": "hard",
        "heading_goals": True,
        "image_goals": True,
        "goal_radius_min": 5.0,
        "goal_radius_max": 9.0,
        "max_steps": 1500,
        "step_penalty": -0.02,
        "goal_reward": 150.0,
        "collision_penalty": -100.0,
        "success_threshold": 0.50,
        "min_episodes": 75,
        "obstacle_density": 0.6,
        "nav_type": "image-goal",
    },
    4: {
        "name": "expert",
        "scenario": "mixed",
        "heading_goals": True,
        "image_goals": True,
        "goal_radius_min": 6.0,
        "goal_radius_max": 12.0,
        "max_steps": 2000,
        "step_penalty": -0.02,
        "goal_reward": 200.0,
        "collision_penalty": -150.0,
        "success_threshold": 0.40,
        "min_episodes": 100,
        "obstacle_density": 0.8,
        "nav_type": "image-goal",
    },
}

# Stage transition evaluation interval
EVAL_INTERVAL_EPISODES = 20


# =============================================================================
# Curriculum Metrics Tracker
# =============================================================================

class CurriculumMetrics:
    """Tracks success rates and handles stage transitions."""

    def __init__(self, num_stages: int = 4):
        self.num_stages = num_stages
        self.stage = 1
        self.episode_returns = {s: [] for s in range(1, num_stages + 1)}
        self.episode_lengths = {s: [] for s in range(1, num_stages + 1)}
        self.episode_successes = {s: [] for s in range(1, num_stages + 1)}
        self.stage_episode_count = 0
        self.total_episodes = 0
        self.stage_history = []

    def add_episode(self, episode_return: float, episode_length: int,
                   goal_reached: bool, collision: bool):
        """Record an episode result."""
        self.episode_returns[self.stage].append(episode_return)
        self.episode_lengths[self.stage].append(episode_length)
        self.episode_successes[self.stage].append(1.0 if goal_reached else 0.0)
        self.stage_episode_count += 1
        self.total_episodes += 1

    def get_success_rate(self) -> float:
        """Current stage success rate over recent episodes."""
        if len(self.episode_successes[self.stage]) == 0:
            return 0.0
        return float(np.mean(self.episode_successes[self.stage][-EVAL_INTERVAL_EPISODES:]))

    def get_mean_return(self) -> float:
        """Mean return for current stage."""
        if len(self.episode_returns[self.stage]) == 0:
            return 0.0
        return float(np.mean(self.episode_returns[self.stage][-EVAL_INTERVAL_EPISODES:]))

    def get_mean_length(self) -> float:
        """Mean episode length for current stage."""
        if len(self.episode_lengths[self.stage]) == 0:
            return 0.0
        return float(np.mean(self.episode_lengths[self.stage][-EVAL_INTERVAL_EPISODES:]))

    def should_advance(self) -> bool:
        """Check if agent is ready to advance to next stage."""
        cfg = CURRICULUM_STAGES[self.stage]
        if self.stage >= self.num_stages:
            return False
        if self.stage_episode_count < cfg["min_episodes"]:
            return False
        return self.get_success_rate() >= cfg["success_threshold"]

    def advance_stage(self) -> bool:
        """Attempt to advance to next stage. Returns True if advanced."""
        if self.should_advance():
            old_stage = self.stage
            self.stage += 1
            self.stage_episode_count = 0
            self.stage_history.append({
                "from": old_stage,
                "to": self.stage,
                "total_episodes": self.total_episodes,
                "success_rate": self.get_success_rate(),
            })
            print(f"\n[CURRICULUM] Stage {old_stage} -> Stage {self.stage}")
            print(f"            Success rate: {self.get_success_rate():.2%}")
            return True
        return False

    def get_stats(self) -> Dict:
        return {
            "stage": self.stage,
            "stage_name": CURRICULUM_STAGES[self.stage]["name"],
            "success_rate": self.get_success_rate(),
            "mean_return": self.get_mean_return(),
            "mean_length": self.get_mean_length(),
            "stage_episodes": self.stage_episode_count,
            "total_episodes": self.total_episodes,
        }


# =============================================================================
# Curriculum Learning Wrapper
# =============================================================================

class CurriculumWrapper(gym.Wrapper):
    """
    Curriculum learning wrapper that manages stage transitions based on
    agent performance. Wraps an underlying ViZDoomEnv or similar dict-obs env.

    The wrapper is observation-space-agnostic — it forwards observations
    from the inner env, but injects goal-relative coordinates into the dict.

    Key features:
      - 4-stage progressive curriculum (easy -> medium -> hard -> expert)
      - Automatic stage advancement based on success rate
      - Stage-specific goal sampling (radius, max_steps)
      - Unified dict observation with goal_rel injected
      - Sim-to-real augmentation passthrough
    """

    def __init__(
        self,
        env: gym.Env,
        num_stages: int = 4,
        stage: int = 1,
        reward_shaping: bool = True,
        eval_mode: bool = False,
        goal_pickle_path: str = "/home/kojogyaase/Projects/Research/recovery-from-failure/goal_loc_images.pkl",
    ):
        super().__init__(env)
        self.num_stages = num_stages
        self.eval_mode = eval_mode
        self.reward_shaping = reward_shaping
        self.goal_pickle_path = goal_pickle_path

        # Load goal data if available (for OfficeEnv compatibility)
        self._goal_data = None
        if os.path.exists(goal_pickle_path):
            try:
                with open(goal_pickle_path, "rb") as f:
                    self._goal_data = pickle.load(f)
            except Exception:
                pass

        # Curriculum state
        self.metrics = CurriculumMetrics(num_stages=num_stages)
        if not eval_mode and stage > 1:
            self.metrics.stage = min(stage, num_stages)

        self.current_goal = None
        self.current_goal_idx = None
        self.distance_to_goal = None
        self.stop_count = 0

        # Build goal_rel observation space
        goal_rel_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32,
        )

        if isinstance(env.observation_space, spaces.Dict):
            new_spaces = dict(env.observation_space.spaces)
            new_spaces["goal_rel"] = goal_rel_space
            self.observation_space = spaces.Dict(new_spaces)
        else:
            self.observation_space = spaces.Dict({
                "obs": env.observation_space,
                "goal_rel": goal_rel_space,
            })

        self._last_info = {}

    # ------------------------------------------------------------------
    # Goal sampling
    # ------------------------------------------------------------------

    def _sample_goal(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Sample a goal position and optional goal image for current stage."""
        cfg = CURRICULUM_STAGES[self.metrics.stage]

        # Sample goal in polar coords, then convert to cartesian
        angle = np.random.uniform(-np.pi, np.pi)
        dist = np.random.uniform(cfg["goal_radius_min"], cfg["goal_radius_max"])
        goal_pos = np.array(
            [dist * np.cos(angle), dist * np.sin(angle), 0.0],
            dtype=np.float32
        )

        # For image goals, the goal image is captured from the env
        # (set by inner env's _sample_goal)
        goal_image = None
        if cfg["image_goals"] and self._goal_data is not None:
            idx = np.random.randint(len(self._goal_data))
            self.current_goal_idx = idx

        return goal_pos, goal_image

    def _build_goal_rel_obs(self, obs: Dict, goal_rel: np.ndarray) -> Dict:
        """Inject goal_rel into observation dict."""
        if isinstance(obs, dict):
            obs = dict(obs)
            obs["goal_rel"] = goal_rel
        else:
            obs = {"obs": obs, "goal_rel": goal_rel}
        return obs

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Sample new goal
        self.current_goal, _ = self._sample_goal()
        self.stop_count = 0

        # Get current position from info
        curr_pos = np.asarray(
            info.get("pos", [0.0, 0.0, 0.0]), dtype=np.float32
        )
        goal_rel = self.current_goal - curr_pos
        self.distance_to_goal = float(np.linalg.norm(goal_rel))

        obs = self._build_goal_rel_obs(obs, goal_rel)
        self._last_info = info
        return obs, info

    def step(self, action) -> Tuple[Dict, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_info = info

        cfg = CURRICULUM_STAGES[self.metrics.stage]

        # Compute goal-relative position
        curr_pos = np.asarray(info.get("pos", [0.0, 0.0, 0.0]), dtype=np.float32)
        goal_rel = self.current_goal - curr_pos

        # Reward shaping
        if self.reward_shaping:
            reward = cfg["step_penalty"]
            dist = np.linalg.norm(goal_rel)

            # Goal proximity reward
            reward += max(0.0, (cfg["goal_radius_max"] - dist) * 0.05)

            # Forward progress reward (velocity)
            vel = info.get("forward_vel", 0.0)
            reward += vel * 0.1

            # Goal reached
            if dist < 1.0:
                terminated = True
                reward += cfg["goal_reward"]

            # Collision / wall hit
            hit_type = info.get("hit", "none")
            if hit_type != "none":
                terminated = True
                reward += cfg["collision_penalty"]

            # Stall detection
            if vel < 0.01:
                self.stop_count += 1
            if self.stop_count > 20:
                truncated = True

        obs = self._build_goal_rel_obs(obs, goal_rel)
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Curriculum control
    # ------------------------------------------------------------------

    def record_episode_result(self, episode_return: float, episode_length: int,
                              goal_reached: bool = False, collision: bool = False):
        """Record episode result and check for stage advancement."""
        self.metrics.add_episode(episode_return, episode_length,
                                goal_reached, collision)
        if not self.eval_mode:
            self.metrics.advance_stage()

    def get_stage_config(self) -> Dict:
        """Get current stage configuration."""
        return CURRICULUM_STAGES[self.metrics.stage]

    def get_metrics(self) -> CurriculumMetrics:
        return self.metrics

    def force_stage(self, stage: int):
        """Manually set the curriculum stage (for eval or replay)."""
        self.metrics.stage = max(1, min(stage, self.num_stages))
        self.metrics.stage_episode_count = 0
        print(f"[CURRICULUM] Forced to stage {stage}: {CURRICULUM_STAGES[stage]['name']}")


# =============================================================================
# OfficeEnv Curriculum Wrapper (for Donkey Car sim transfer)
# =============================================================================

class OfficeEnvCurriculumWrapper(gym.Wrapper):
    """
    Curriculum wrapper specifically designed for OfficeEnv / Donkey Car sim.

    Wraps the donkey-warehouse-v0 style env and adds:
      - Goal-relative coordinates (goal_rel) to obs dict
      - Curriculum-based goal difficulty (easy/medium/hard goal positions)
      - Progressive goal distance scaling

    Goals are sampled from 3 difficulty regions:
      Easy  : Within 3m of start position
      Medium: 3-6m from start
      Hard  : 6-10m from start
    """

    def __init__(
        self,
        env: gym.Wrapper,
        curriculum_stage: int = 1,
        goal_radius_easy: float = 3.0,
        goal_radius_medium: float = 5.0,
        goal_radius_hard: float = 8.0,
        reward_shaping: bool = True,
    ):
        super().__init__(env)
        self.curriculum_stage = curriculum_stage
        self.goal_radius_easy = goal_radius_easy
        self.goal_radius_medium = goal_radius_medium
        self.goal_radius_hard = goal_radius_hard
        self.reward_shaping = reward_shaping

        self.current_goal = None
        self.stop_count = 0

        # Goal-relative observation space
        goal_rel_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32,
        )

        if isinstance(env.observation_space, spaces.Dict):
            new_spaces = dict(env.observation_space.spaces)
            new_spaces["goal_rel"] = goal_rel_space
            self.observation_space = spaces.Dict(new_spaces)
        else:
            self.observation_space = spaces.Dict({
                "obs": env.observation_space,
                "goal_rel": goal_rel_space,
            })

    def _sample_goal(self) -> np.ndarray:
        """Sample goal position based on curriculum stage."""
        # Sample in polar coords
        angle = np.random.uniform(-np.pi, np.pi)

        if self.curriculum_stage == 1:
            dist = np.random.uniform(0.5, self.goal_radius_easy)
        elif self.curriculum_stage == 2:
            dist = np.random.uniform(self.goal_radius_easy, self.goal_radius_medium)
        elif self.curriculum_stage == 3:
            dist = np.random.uniform(self.goal_radius_medium, self.goal_radius_hard)
        else:
            dist = np.random.uniform(3.0, self.goal_radius_hard)

        goal = np.array([dist * np.cos(angle), dist * np.sin(angle), 0.0],
                       dtype=np.float32)
        return goal

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_goal = self._sample_goal()
        self.stop_count = 0

        curr_pos = np.asarray(info.get("pos", [0.0, 0.0, 0.0]), dtype=np.float32)
        goal_rel = self.current_goal - curr_pos

        obs = self._inject_goal_rel(obs, goal_rel)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        curr_pos = np.asarray(info.get("pos", [0.0, 0.0, 0.0]), dtype=np.float32)
        goal_rel = self.current_goal - curr_pos

        if self.reward_shaping:
            reward = -0.01
            dist = np.linalg.norm(goal_rel)

            if dist < 1.0:
                terminated = True
                reward += 100.0

            vel = info.get("forward_vel", 0.0)
            reward += vel * 0.5

            hit = info.get("hit", "none")
            if hit != "none":
                terminated = True
                reward -= 100.0

            if vel < 0.01:
                self.stop_count += 1
            if self.stop_count > 20:
                truncated = True

        obs = self._inject_goal_rel(obs, goal_rel)
        return obs, reward, terminated, truncated, info

    def _inject_goal_rel(self, obs, goal_rel):
        if isinstance(obs, dict):
            obs = dict(obs)
            obs["goal_rel"] = goal_rel
        else:
            obs = {"obs": obs, "goal_rel": goal_rel}
        return obs

    def set_stage(self, stage: int):
        """Update curriculum stage."""
        self.curriculum_stage = max(1, min(stage, 4))
