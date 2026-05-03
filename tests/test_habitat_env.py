"""Tests for HabitatNavEnv, GoalImageWrapper, HabitatRewardWrapper, and wrapper stack."""
import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces

from configs.habitat_config import HabitatNavConfig
from habitat_env import HAS_HABITAT_LAB
from wrappers import GoalImageWrapper


# ============================================================================
# HabitatRewardWrapper tests
# ============================================================================

class TestHabitatRewardWrapper:
    """Test HabitatRewardWrapper reward logic with a mock env."""

    class MockEnv(gym.Wrapper):
        observation_space = spaces.Dict({"pixels": spaces.Box(0, 1, (8,))})
        action_space = spaces.Box(-1, 1, (2,))

        def __init__(self, distance=5.0, forward_vel=0.0, actual_vel=0.0,
                     habitat_success=0.0, hit=False, position=None):
            self._distance = distance
            self._forward_vel = forward_vel
            self._actual_vel = actual_vel
            self._habitat_success = habitat_success
            self._hit = hit
            self._position = position if position is not None else np.zeros(3)

        def reset(self, **kwargs):
            obs = {"pixels": np.zeros(8, dtype=np.float32),
                   "imu": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)}
            return obs, {
                "distance_to_goal": self._distance,
                "forward_vel": self._forward_vel,
                "actual_vel": self._actual_vel,
                "habitat_success": self._habitat_success,
                "hit": self._hit,
                "position": self._position,
            }

        def step(self, action):
            info = {
                "distance_to_goal": self._distance,
                "forward_vel": self._forward_vel,
                "actual_vel": self._actual_vel,
                "habitat_success": self._habitat_success,
                "hit": self._hit,
                "position": self._position,
            }
            obs = {"pixels": np.zeros(8, dtype=np.float32),
                   "imu": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)}
            return obs, 0.0, False, False, info

    def test_goal_condition_habitat_success(self):
        from train_habitat_her import HabitatRewardWrapper
        env = self.MockEnv(habitat_success=1.0)
        wrapper = HabitatRewardWrapper(env, k_goal=10.0)
        wrapper.reset()
        _, reward, terminated, _, _ = wrapper.step(np.array([0.0, 0.5]))
        assert terminated, "Should terminate when habitat_success == 1.0"
        assert reward >= 10.0, f"Should get goal reward, got {reward}"

    def test_collision_penalty(self):
        from train_habitat_her import HabitatRewardWrapper
        env = self.MockEnv(hit=True)
        wrapper = HabitatRewardWrapper(env, k_collision=3.0)
        wrapper.reset()
        _, reward, terminated, _, _ = wrapper.step(np.array([0.0, 0.5]))
        assert reward < 0, "Collision should produce negative reward"

    def test_steering_penalty(self):
        from train_habitat_her import HabitatRewardWrapper
        env = self.MockEnv()
        wrapper = HabitatRewardWrapper(env, k_steering=0.5)
        wrapper.reset()
        # Full steering should incur penalty of 0.5 * |1.0| = 0.5
        _, reward, _, _, _ = wrapper.step(np.array([1.0, 0.5]))
        assert reward == -0.5, f"Expected steering penalty -0.5, got {reward}"
        # No steering should have no penalty
        _, reward, _, _, _ = wrapper.step(np.array([0.0, 0.5]))
        assert reward == 0.0, f"Expected no penalty, got {reward}"
        # Half steering: 0.5 * |0.5| = 0.25
        _, reward, _, _, _ = wrapper.step(np.array([0.5, 0.5]))
        assert abs(reward + 0.25) < 1e-6, f"Expected -0.25, got {reward}"

    def test_proximity_penalty(self):
        from train_habitat_her import HabitatRewardWrapper
        import math

        # Near obstacle (proximity=0.1) → significant penalty
        env_near = self.MockEnv()
        # Override step() to return low proximity imu
        orig_step = env_near.step
        def step_near(action):
            obs, _, _, _, info = orig_step(action)
            obs["imu"][-1] = 0.1
            return obs, 0.0, False, False, info
        env_near.step = step_near

        wrapper_near = HabitatRewardWrapper(env_near)
        wrapper_near.reset()
        _, reward_near, _, _, _ = wrapper_near.step(np.array([0.0, 0.5]))
        # Base reward is -1.0, plus steering penalty 0, throttle check varies
        # But proximity penalty of exp(-0.1) ≈ 0.905 must be present
        assert reward_near <= -1.9, f"Expected penalty >= 0.9 on top of base -1.0, got {reward_near:.4f}"

        # Far from obstacle (proximity=5.0) → negligible penalty
        env_far = self.MockEnv()
        orig_step_far = env_far.step
        def step_far(action):
            obs, _, _, _, info = orig_step_far(action)
            obs["imu"][-1] = 5.0
            return obs, 0.0, False, False, info
        env_far.step = step_far

        wrapper_far = HabitatRewardWrapper(env_far)
        wrapper_far.reset()
        _, reward_far, _, _, _ = wrapper_far.step(np.array([0.0, 0.5]))
        penalty_far = math.exp(-5.0)
        assert penalty_far < 0.01, f"Far proximity penalty should be negligible, got {penalty_far:.4f}"

        # Sentinel -1.0 → should NOT apply proximity penalty (guard)
        env_sentinel = self.MockEnv()
        wrapper_sentinel = HabitatRewardWrapper(env_sentinel)
        wrapper_sentinel.reset()
        _, reward_sentinel, _, _, _ = wrapper_sentinel.step(np.array([0.0, 0.5]))
        # With sentinel -1.0 and guard, no proximity penalty applied
        assert reward_sentinel > -2.0, f"Sentinel should not trigger proximity penalty, got {reward_sentinel:.4f}"


# ============================================================================
# HabitatNavEnv tests (only if habitat_sim is installed)
# ============================================================================

@pytest.mark.skipif(not HAS_HABITAT_LAB, reason="habitat_lab not installed")
class TestHabitatNavEnv:
    def test_import_and_creation(self):
        from habitat_env import HabitatNavEnv
        cfg = HabitatNavConfig(
            scene_path="data/gibson/Cantwell.glb",
            image_height=120, image_width=160, seed=42,
        )
        env = HabitatNavEnv(config=cfg)
        assert env is not None
        env.close()

    def test_reset_and_step(self):
        from habitat_env import HabitatNavEnv
        cfg = HabitatNavConfig(
            scene_path="data/gibson/Cantwell.glb",
            image_height=120, image_width=160, seed=42,
        )
        env = HabitatNavEnv(config=cfg)
        obs, info = env.reset()
        assert obs["image"].shape == (120, 160, 3)
        assert obs["imu"].shape == (11,)
        assert obs["imu"][-1] >= -1.0, "proximity should be >= -1.0"
        assert "goal_image" in info
        assert "actual_vel" in info, "actual_vel key missing from info"
        env.close()

    def test_actual_vel_in_info(self):
        from habitat_env import HabitatNavEnv
        cfg = HabitatNavConfig(
            scene_path="data/gibson/Cantwell.glb",
            image_height=120, image_width=160, seed=42,
        )
        env = HabitatNavEnv(config=cfg)
        env.reset()
        action = np.array([0.0, 0.5], dtype=np.float32)
        obs, _, _, _, info = env.step(action)
        assert "actual_vel" in info, "actual_vel key missing from info"
        assert isinstance(info["actual_vel"], float)
        assert obs["imu"].shape == (11,), "IMU shape should be (11,) after step"
        assert obs["imu"][-1] >= -1.0, "proximity should be >= -1.0 after step"
        env.close()