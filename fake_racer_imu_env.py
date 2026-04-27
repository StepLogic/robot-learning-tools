"""
racer_imu_env.py  —  Dummy version for offline IQL training.

Mirrors the public API of the real RacerEnv / StackingWrapper / RewardWrapper
so that train_iql.py imports work unchanged, but:
  - No TCP socket, no hardware
  - Observations are random noise matching the real spaces
  - Episodes terminate after a fixed number of steps
"""

import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
from collections import deque


# ── Constants matching the real env ───────────────────────────────────────────

SCREEN_H, SCREEN_W = 480, 640
IMAGE_H,  IMAGE_W  = 120, 160
FPS = 30


# ─── Dummy base environment ───────────────────────────────────────────────────

class RacerEnv(gym.Env):
    """
    Drop-in replacement for the real RacerEnv.
    Observations are random noise; rewards are random.
    Used for offline IQL training where no real robot is available.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode: str = "human", max_episode_steps: int = 200):
        super().__init__()

        self.render_mode       = render_mode
        self.max_episode_steps = max_episode_steps
        self._step_count       = 0

        # ── Match real env spaces exactly ─────────────────────────────────────
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, high=255,
                shape=(IMAGE_H, IMAGE_W, 3),
                dtype=np.uint8,
            ),
            "imu": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(6,),
                dtype=np.float32,
            ),
        })

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 0.3], dtype=np.float32),
            dtype=np.float32,
        )

        # Dummy state
        self.current_image = np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8)
        self.current_imu   = np.zeros(6, dtype=np.float32)

        self.current_velocity  = {"cms": 0.0, "ms": 0.0, "method": "dummy"}
        self.current_collision = {
            "detected": False,
            "distance_cm": float("inf"),
            "threshold_cm": 15.0,
        }
        self.current_blocked = False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _random_obs(self) -> dict:
        return {
            "image": self.np_random.integers(0, 255,
                         size=(IMAGE_H, IMAGE_W, 3), dtype=np.uint8),
            "imu":   self.np_random.standard_normal(6).astype(np.float32),
        }

    def _get_info(self) -> dict:
        return {
            "trajectory":      {"position": []},
            "low_accel_count": 0,
            "velocity":        self.current_velocity,
            "collision":       self.current_collision,
            "blocked":         self.current_blocked,
        }

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        obs = self._random_obs()
        self.current_image = obs["image"]
        self.current_imu   = obs["imu"]
        return obs, self._get_info()

    def step(self, action):
        self._step_count += 1
        obs = self._random_obs()
        self.current_image = obs["image"]
        self.current_imu   = obs["imu"]

        reward     = float(self.np_random.standard_normal())
        terminated = False
        truncated  = self._step_count >= self.max_episode_steps
        return obs, reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "rgb_array":
            return self.current_image

    def close(self):
        pass


# ─── StackingWrapper (identical API to real version) ─────────────────────────

class StackingWrapper(gym.Wrapper):
    """Stack RGB frames, actions, and IMU data for temporal modelling."""

    def __init__(self, env: gym.Env, num_stack: int = 4):
        super().__init__(env)
        self.num_stack  = num_stack
        self.action_dim = env.action_space.shape[0]

        self.action_history = deque(maxlen=num_stack)
        self.rgb_history    = deque(maxlen=num_stack)
        self.imu_history    = deque(maxlen=num_stack)

        self.observation_space = spaces.Dict({
            "pixels": spaces.Box(
                low=0, high=255,
                shape=(IMAGE_H, IMAGE_W, 3 * num_stack),
                dtype=np.uint8,
            ),
            "actions": spaces.Box(
                low=np.tile(env.action_space.low,  num_stack),
                high=np.tile(env.action_space.high, num_stack),
                shape=(num_stack * self.action_dim,),
                dtype=np.float32,
            ),
            "imu": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(6 * num_stack,),
                dtype=np.float32,
            ),
        })

    def _stacked_obs(self) -> dict:
        actions = list(self.action_history)
        while len(actions) < self.num_stack:
            actions.insert(0, np.zeros(self.action_dim, dtype=np.float32))

        imus = list(self.imu_history)
        while len(imus) < self.num_stack:
            imus.insert(0, np.zeros(6, dtype=np.float32))

        return {
            "pixels":  np.concatenate(list(self.rgb_history), axis=-1).astype(np.uint8),
            "actions": np.concatenate(actions).astype(np.float32),
            "imu":     np.concatenate(imus).astype(np.float32),
        }

    def _push(self, obs: dict, action: np.ndarray | None = None):
        rgb = cv2.cvtColor(obs["image"], cv2.COLOR_BGR2RGB)
        self.rgb_history.append(rgb)
        self.imu_history.append(obs["imu"])
        if action is not None:
            self.action_history.append(action.astype(np.float32))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.action_history.clear()
        self.rgb_history.clear()
        self.imu_history.clear()
        for _ in range(self.num_stack):
            self._push(obs)
            self.action_history.append(np.zeros(self.action_dim, dtype=np.float32))
        return self._stacked_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._push(obs, action)
        return self._stacked_obs(), reward, terminated, truncated, info


# ─── RewardWrapper (pass-through for dummy env) ───────────────────────────────

class RewardWrapper(gym.Wrapper):
    """
    Reward wrapper — pass-through for the dummy env.
    Mirrors the real RewardWrapper's __init__ signature so imports work.
    """

    MIN_VELOCITY_MS   = 0.05
    STALL_LIMIT       = 100
    PROXIMITY_WARN_CM = 30.0

    def __init__(self, env):
        super().__init__(env)
        self.velocity     = 0.0
        self.stop_count   = 0
        self.move_count   = 0
        self.last_steering = 0.0

    def step(self, action):
        # In dummy mode just pass the reward through unchanged
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_steering = float(action[0])
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.velocity      = 0.0
        self.stop_count    = 0
        self.move_count    = 0
        self.last_steering = 0.0
        return self.env.reset(**kwargs)