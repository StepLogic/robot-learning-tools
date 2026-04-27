#!/usr/bin/env python
"""
ViZDoom gym wrapper for image/heading-goal navigation RL.

This module provides a gymnasium-compatible ViZDoom environment that outputs
dict observations compatible with the existing stacking/feature wrapper stack in
wrappers.py (StackingWrapper → MobileNetFeatureWrapper → GoalRelObservationWrapper).

Scenarios implemented:
  - SCENARIO_MAZE_EASY   : Open arena, single visible heading goal (colored marker)
  - SCENARIO_MAZE_MEDIUM : Wall obstacles, heading goal behind a corner
  - SCENARIO_MAZE_HARD   : Narrow corridor, distant heading goal, obstacles
  - SCENARIO_IMG_GOAL    : Image-goal matching: show target frame, agent must navigate to match it

Observation dict (compatible with StackingWrapper):
    obs["image"]      : np.ndarray (H,W,3) uint8  — RGB frame from ViZDoom
    obs["heading"]    : np.ndarray (3,) float32    — [sin(yaw), cos(yaw), yaw_rate]
    obs["goal_image"] : np.ndarray (H,W,3) uint8   — target goal image (Stage 4 only)

Action space: Box(2,) — [steering, throttle]  (same as donkey car)

Reward shaping:
    - Goal reached        : +100
    - Collision / wallhit: -10
    - Step penalty        : -0.01
    - Forward progress    : +velocity * 0.1
    - Goal proximity       : +0.5 * improvement in distance-to-goal
"""

import os
import sys
import uuid
from typing import Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Try to import ViZDoom; fall back to a dummy env if not installed
try:
    from vizdoom import DoomGame, GameVariable
    _VIZDOOM_AVAILABLE = True
except ImportError:
    _VIZDOOM_AVAILABLE = False
    print("[ViZDoom] vizdoom package not found — using DummyViZDoomEnv stub")


# =============================================================================
# Scenario configurations
# =============================================================================

_VIZDOOM_SCENARIOS = "/home/kojogyaase/anaconda3/envs/real-robot-env/lib/python3.12/site-packages/vizdoom/scenarios"

# Use bundled scenarios that support navigation:
#  - my_way_home: navigation in a single corridor
#  - health_gathering_supreme: open arena, good for goal navigation
#  - take_cover: corridor with obstacles
SCENARIO_MAZE_EASY = os.path.join(_VIZDOOM_SCENARIOS, "my_way_home.cfg")
SCENARIO_MAZE_MEDIUM = os.path.join(_VIZDOOM_SCENARIOS, "health_gathering_supreme.cfg")
SCENARIO_MAZE_HARD = os.path.join(_VIZDOOM_SCENARIOS, "take_cover.cfg")
SCENARIO_DEFEND = os.path.join(_VIZDOOM_SCENARIOS, "defend_the_center.cfg")
SCENARIO_DEATHMATCH = os.path.join(_VIZDOOM_SCENARIOS, "deathmatch.cfg")


# =============================================================================
# Helper: encode heading as 3D vector
# =============================================================================

def heading_to_vec(yaw_rad: float, yaw_rate: float = 0.0) -> np.ndarray:
    """Encode yaw angle as [sin, cos, rate] — same format expected by donkey wrappers."""
    return np.array(
        [np.sin(yaw_rad), np.cos(yaw_rad), yaw_rate],
        dtype=np.float32
    )


# =============================================================================
# DummyViZDoomEnv — fallback when vizdoom is not installed
# =============================================================================

class DummyViZDoomEnv(gym.Env):
    """
    Random-image stub that produces the same dict-observation interface as
    ViZDoomEnv. Useful for testing the wrapper stack locally without ViZDoom.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario_path: str = SCENARIO_MAZE_EASY,
        frame_skip: int = 4,
        resolution: Tuple[int, int] = (60, 108),
        num_goal_views: int = 8,
        heading_goals: bool = True,
        image_goals: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.scenario_path = scenario_path
        self.frame_skip = frame_skip
        self.resolution = resolution
        self.num_goal_views = num_goal_views
        self.heading_goals = heading_goals
        self.image_goals = image_goals
        self._seed = seed if seed is not None else np.random.randint(1 << 30)

        self.height, self.width = resolution
        self.num_channels = 3

        # Fake position/heading state
        self._pos = np.zeros(3, dtype=np.float32)
        self._yaw = 0.0
        self._yaw_rate = 0.0
        self._vel = 0.0
        self._hit = False
        self._step_count = 0
        self._max_steps = 1000

        # Goal state
        self._goal_pos = np.array([5.0, 0.0, 0.0], dtype=np.float32)
        self._goal_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._goal_heading = 0.0
        self._goal_view_idx = 0

        # Observation space
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, high=255,
                shape=(self.height, self.width, 3),
                dtype=np.uint8,
            ),
            "heading": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(3,), dtype=np.float32,
            ),
        })
        if self.image_goals:
            self.observation_space.spaces["goal_image"] = spaces.Box(
                low=0, high=255,
                shape=(self.height, self.width, 3),
                dtype=np.uint8,
            )

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32
        )
        self._info = {}

    def _generate_goal_image(self):
        """Generate a synthetic goal image based on goal heading."""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        hue = int((self._goal_heading / (2 * np.pi)) * 180) % 180
        img[:, :, 0] = hue
        img[:, :, 1] = 200
        img[:, :, 2] = 200
        self._goal_image = img

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._seed = seed
        np.random.seed(self._seed + self._step_count)

        self._pos = np.zeros(3, dtype=np.float32)
        self._yaw = np.random.uniform(-np.pi, np.pi)
        self._yaw_rate = 0.0
        self._vel = 0.0
        self._hit = False
        self._step_count = 0

        # Sample goal
        angle = np.random.uniform(-np.pi, np.pi)
        dist = np.random.uniform(3.0, 8.0)
        self._goal_pos = np.array(
            [dist * np.cos(angle), dist * np.sin(angle), 0.0],
            dtype=np.float32
        )
        self._goal_heading = angle
        self._goal_view_idx = np.random.randint(0, self.num_goal_views)
        self._generate_goal_image()

        obs = self._build_obs()
        self._info = {
            "pos": self._pos.copy(),
            "forward_vel": self._vel,
            "hit": "none",
        }
        return obs, self._info

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        steer, throttle = float(action[0]), float(action[1])

        # Very crude physics
        self._yaw_rate = steer * 2.0
        self._yaw += self._yaw_rate * 0.1
        self._vel = throttle * 2.0
        self._pos[0] += np.cos(self._yaw) * self._vel * 0.1
        self._pos[1] += np.sin(self._yaw) * self._vel * 0.1

        self._step_count += 1
        self._hit = np.random.rand() < 0.01  # fake collision chance
        wall_hit = np.linalg.norm(self._pos[:2]) > 12.0  # out of bounds

        reward = -0.01 + self._vel * 0.1
        dist_to_goal = np.linalg.norm(self._pos - self._goal_pos)
        if dist_to_goal < 1.0:
            reward += 100.0
        if self._hit or wall_hit:
            reward -= 10.0

        terminated = dist_to_goal < 1.0 or self._hit or wall_hit
        truncated = self._step_count >= self._max_steps

        obs = self._build_obs()
        self._info = {
            "pos": self._pos.copy(),
            "forward_vel": self._vel,
            "hit": "wall" if wall_hit else ("collision" if self._hit else "none"),
            "goal_pos": self._goal_pos.copy(),
        }
        return obs, reward, terminated, truncated, self._info

    def _build_obs(self) -> Dict[str, np.ndarray]:
        # Fake RGB frame
        noise = np.random.randint(0, 30, (self.height, self.width, 3), dtype=np.uint8)
        img = np.clip(noise + np.array([self._yaw * 30, 0, 0], dtype=np.uint8), 0, 255)

        obs = {
            "image": img,
            "heading": heading_to_vec(self._yaw, self._yaw_rate),
        }
        if self.image_goals:
            obs["goal_image"] = self._goal_image
        return obs

    def close(self):
        pass

    def render(self, mode="human"):
        pass

    def seed(self, s=None):
        self._seed = s
        np.random.seed(s)


# =============================================================================
# ViZDoomEnv — real gym wrapper around ViZDoom
# =============================================================================

class ViZDoomEnv(gym.Env):
    """
    Gymnasium wrapper for ViZDoom that produces dict observations compatible
    with the StackingWrapper → MobileNetFeatureWrapper → GoalRelObservationWrapper
    pipeline.

    Key features:
      - Frame skipping (default 4)
      - Dict observation: {"image": RGB, "heading": 3D vector}
      - Optional image-goal mode where goal_image is included in obs
      - Episode terminates on: goal reached, collision, timeout, or wall-hit
      - Heading goal tracking via GameVariable.ANGLE
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        scenario_path: str = SCENARIO_MAZE_EASY,
        frame_skip: int = 4,
        resolution: Tuple[int, int] = (60, 108),
        heading_goals: bool = True,
        image_goals: bool = False,
        seed: Optional[int] = None,
        visible: bool = False,
    ):
        """
        Args:
            scenario_path: Path to ViZDoom scenario .cfg file.
            frame_skip: Number of game ticks per step() call.
            resolution: (height, width) for the screen buffer.
            heading_goals: If True, track a heading (angle) goal using
                           GameVariable.ANGLE and expose it in the obs dict.
            image_goals: If True, include goal_image in observations (target frame
                         the agent must navigate to).
            seed: Random seed for the underlying DoomGame.
            visible: If True, show the ViZDoom window.
        """
        super().__init__()
        if not _VIZDOOM_AVAILABLE:
            raise RuntimeError(
                "ViZDoom is not installed. Install with: pip install vizdoom"
            )

        self.scenario_path = scenario_path
        self.frame_skip = frame_skip
        self.resolution = resolution
        self.heading_goals = heading_goals
        self.image_goals = image_goals
        self._visible = visible

        self.height, self.width = resolution

        # ------------------------------------------------------------------
        # Action space: [steering, throttle] — same as donkey car
        # ------------------------------------------------------------------
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # ------------------------------------------------------------------
        # Observation space
        # ------------------------------------------------------------------
        obs_spaces = {
            "image": spaces.Box(
                low=0, high=255,
                shape=(self.height, self.width, 3),
                dtype=np.uint8,
            ),
            "heading": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(3,), dtype=np.float32,
            ),
        }
        if self.image_goals:
            obs_spaces["goal_image"] = spaces.Box(
                low=0, high=255,
                shape=(self.height, self.width, 3),
                dtype=np.uint8,
            )

        self.observation_space = spaces.Dict(obs_spaces)

        # ------------------------------------------------------------------
        # Initialise DoomGame
        # ------------------------------------------------------------------
        self._game = DoomGame()
        self._game.load_config(scenario_path)
        self._game.set_seed(seed if seed is not None else np.random.randint(1 << 30))

        # Screen resolution
        self._game.set_screen_resolution(
            getattr(__import__("vizdoom", fromlist=["ScreenResolution"]),
                    "ScreenResolution").RES_160X120)
        self._game.set_screen_format(
            getattr(__import__("vizdoom", fromlist=["ScreenFormat"]),
                    "ScreenFormat").RGB24)
        self._game.set_window_visible(visible)

        # Enable relevant game variables
        self._game.add_available_game_variable(GameVariable.ANGLE)
        self._game.add_available_game_variable(GameVariable.POSITION_X)
        self._game.add_available_game_variable(GameVariable.POSITION_Y)
        self._game.add_available_game_variable(GameVariable.POSITION_Z)
        self._game.add_available_game_variable(GameVariable.VELOCITY_X)
        self._game.add_available_game_variable(GameVariable.VELOCITY_Y)

        # Available buttons: MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT
        # We will map our 2D action [steer, throttle] to these.
        self._game.set_available_buttons([
            getattr(__import__("vizdoom", fromlist=["Button"]), "Button").MOVE_FORWARD,
            getattr(__import__("vizdoom", fromlist=["Button"]), "Button").MOVE_BACKWARD,
            getattr(__import__("vizdoom", fromlist=["Button"]), "Button").TURN_LEFT,
            getattr(__import__("vizdoom", fromlist=["Button"]), "Button").TURN_RIGHT,
        ])

        self._game.init()

        # Internal state
        self._current_yaw = 0.0
        self._current_yaw_rate = 0.0
        self._last_yaw = 0.0
        self._hit = False
        self._step_count = 0
        self._max_steps = 1000  # will be overridden by cfg if present
        self._goal_pos = np.zeros(3, dtype=np.float32)
        self._goal_heading = 0.0
        self._goal_image = np.zeros(
            (self.height, self.width, 3), dtype=np.uint8
        )
        self._info = {}

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._game.set_seed(seed)
        self._game.new_episode()
        self._step_count = 0
        self._hit = False
        self._last_yaw = self._game.get_game_variable(GameVariable.ANGLE)
        self._current_yaw = self._last_yaw
        self._current_yaw_rate = 0.0
        self._sample_goal()

        obs, _ = self._build_obs()
        return obs, self._info

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict, float, bool, bool, Dict]:
        steer, throttle = float(action[0]), float(action[1])

        # Map to ViZDoom buttons
        buttons = self._action_to_buttons(steer, throttle)
        # print(buttons)
        self._game.set_action([steer, throttle])
        # for _ in range(self.frame_skip):
        #     self._game.advance_task()

        self._step_count += 1
        self._last_yaw = self._current_yaw
        self._current_yaw = self._game.get_game_variable(GameVariable.ANGLE)
        self._current_yaw_rate = self._current_yaw - self._last_yaw

        # Normalise yaw rate (ViZDoom can give large values on wrap-around)
        self._current_yaw_rate = np.clip(self._current_yaw_rate, -5.0, 5.0)

        reward = self._compute_reward()
        done = self._is_done()
        obs, info = self._build_obs()
        self._info.update(info)

        terminated = done and not info.get("TimeLimit.truncated", False)
        truncated = done and info.get("TimeLimit.truncated", False)
        return obs, reward, terminated, self._info

    def close(self):
        self._game.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return self._game.get_state().screen_buffer

    def seed(self, s=None):
        self._game.set_seed(s)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _action_to_buttons(self, steer: float, throttle: float):
        """
        Map [-1,1] steer × [0,1] throttle to a button index.
        Button indices: 0=MOVE_FORWARD, 1=MOVE_BACKWARD, 2=TURN_LEFT, 3=TURN_RIGHT
        """
        buttons = []
        if throttle > 0.05:
            buttons.append(0)  # MOVE_FORWARD
        elif throttle < -0.05:
            buttons.append(1)  # MOVE_BACKWARD
        if steer > 0.05:
            buttons.append(3)  # TURN_RIGHT
        elif steer < -0.05:
            buttons.append(2)  # TURN_LEFT
        # Return first button or empty
        return buttons[0] if buttons else -1

    def _sample_goal(self):
        """Sample a new random goal position and heading."""
        # Sample from reachable area — simple uniform disk
        angle = np.random.uniform(-np.pi, np.pi)
        dist = np.random.uniform(4.0, 9.0)
        self._goal_pos = np.array(
            [dist * np.cos(angle), dist * np.sin(angle), 0.0],
            dtype=np.float32
        )
        self._goal_heading = angle

        # Capture goal image from current view rotated to goal_heading
        # (simplified: store the current frame as the goal image)
        state = self._game.get_state()
        if state is not None:
            self._goal_image = np.transpose(
                state.screen_buffer, (1, 2, 0)
            ).copy()  # (3, H, W) → (H, W, 3)

    def _compute_reward(self) -> float:
        """Reward logic matching the existing RewardWrapper contract."""
        reward = -0.01  # step penalty

        x = self._game.get_game_variable(GameVariable.POSITION_X)
        y = self._game.get_game_variable(GameVariable.POSITION_Y)
        vel = self._game.get_game_variable(GameVariable.VELOCITY_X)

        current_pos = np.array([x, y, 0.0], dtype=np.float32)

        # Forward progress reward
        reward += float(np.clip(vel / 10.0, -0.5, 1.0))

        # Proximity to goal
        dist = np.linalg.norm(current_pos - self._goal_pos)
        reward += max(0.0, (10.0 - dist) * 0.05)

        # Termination rewards / penalties
        if dist < 1.0:          # goal reached
            reward += 100.0
        if self._game.is_episode_finished():
            # crashed or fell off map
            reward -= 100.0

        return reward

    def _is_done(self) -> bool:
        if self._game.is_episode_finished():
            return True
        if self._step_count >= self._max_steps:
            self._info["TimeLimit.truncated"] = True
            return True
        x = self._game.get_game_variable(GameVariable.POSITION_X)
        y = self._game.get_game_variable(GameVariable.POSITION_Y)
        dist = np.linalg.norm(np.array([x, y]) - self._goal_pos[:2])
        if dist < 1.0:
            return True
        return False

    def _build_obs(self) -> Tuple[Dict, Dict]:
        state = self._game.get_state()
        if state is None:
            # Episode finished; return zeros
            image = np.zeros(
                (self.height, self.width, 3), dtype=np.uint8
            )
        else:
            # (C, W, H) → (H, W, C)
            image = np.transpose(state.screen_buffer, (2, 1, 0)).copy()

        x = self._game.get_game_variable(GameVariable.POSITION_X)
        y = self._game.get_game_variable(GameVariable.POSITION_Y)
        current_pos = np.array([x, y, 0.0], dtype=np.float32)

        obs = {
            "image": image,
            "heading": heading_to_vec(
                np.deg2rad(self._current_yaw),
                self._current_yaw_rate
            ),
        }
        if self.image_goals:
            obs["goal_image"] = self._goal_image

        info = {
            "pos": current_pos,
            "forward_vel": float(
                np.linalg.norm([self._game.get_game_variable(GameVariable.VELOCITY_X)])
            ),
            "hit": "collision" if self._game.is_episode_finished() else "none",
            "goal_pos": self._goal_pos.copy(),
            "goal_heading": self._goal_heading,
        }
        return obs, info


# =============================================================================
# Convenience factory
# =============================================================================

def make_vizdoom_env(
    scenario: str = "easy",
    frame_skip: int = 4,
    resolution: Tuple[int, int] = (60, 108),
    heading_goals: bool = True,
    image_goals: bool = False,
    seed: Optional[int] = None,
    visible: bool = False,
) -> gym.Env:
    """
    Factory that returns the appropriate ViZDoom or dummy env.

    Args:
        scenario: One of {"easy", "medium", "hard", "defend", "deathmatch"}.
                  Maps to the bundled ViZDoom scenario files.
        frame_skip: Frame skip for the environment.
        resolution: (H, W) for the screen buffer.
        heading_goals: Include heading goal vector in obs.
        image_goals: Include goal_image in obs.
        seed: Random seed.
        visible: Show ViZDoom window.

    Returns:
        A ViZDoomEnv (or DummyViZDoomEnv if vizdoom is not installed).
    """
    scenario_map = {
        "easy": SCENARIO_MAZE_EASY,
        "medium": SCENARIO_MAZE_MEDIUM,
        "hard": SCENARIO_MAZE_HARD,
        "defend": SCENARIO_DEFEND,
        "deathmatch": SCENARIO_DEATHMATCH,
    }
    path = scenario_map.get(scenario.lower(), SCENARIO_MAZE_EASY)

    if not _VIZDOOM_AVAILABLE:
        return DummyViZDoomEnv(
            scenario_path=path,
            frame_skip=frame_skip,
            resolution=resolution,
            heading_goals=heading_goals,
            image_goals=image_goals,
            seed=seed,
        )

    return ViZDoomEnv(
        scenario_path=path,
        frame_skip=frame_skip,
        resolution=resolution,
        heading_goals=heading_goals,
        image_goals=image_goals,
        seed=seed,
        visible=visible,
    )
