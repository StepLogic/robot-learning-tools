import numpy as np
import pytest
import cv2
import gymnasium as gym
from gymnasium import spaces

from optical_flow_velocity_wrapper import OpticalFlowVelocityWrapper


def make_synthetic_frame(size=(480, 640), offset_x=0):
    """Create a synthetic frame with textured ground plane."""
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    # Checkerboard ground texture
    block = 20
    for y in range(0, size[0], block):
        for x in range(0, size[1], block):
            if ((x + offset_x) // block + y // block) % 2 == 0:
                img[y:y+block, x:x+block] = 200
            else:
                img[y:y+block, x:x+block] = 50
    return img


class MockEnv(gym.Env):
    """Minimal mock env for wrapper unit tests."""

    def __init__(self, reset_return, step_returns):
        super().__init__()
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
            "imu": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
        })
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._reset_return = reset_return
        self._step_returns = step_returns if isinstance(step_returns, list) else [step_returns]
        self._step_idx = 0

    def reset(self, **kwargs):
        self._step_idx = 0
        return self._reset_return

    def step(self, action):
        ret = self._step_returns[self._step_idx % len(self._step_returns)]
        self._step_idx += 1
        return ret


def test_forward_motion_velocity():
    """Synthetic forward motion: image shifts up (ground moves down)."""
    frame0 = make_synthetic_frame(offset_x=0)
    reset_ret = (
        {"image": frame0, "imu": np.zeros(6)},
        {"raw_image": frame0, "velocity": {"ms": 0.0}}
    )
    # Shift frame down by 5 pixels to simulate forward motion
    frame1 = np.roll(frame0, 5, axis=0)
    step_ret = (
        {"image": frame1, "imu": np.zeros(6)},
        0.0, False, False,
        {"raw_image": frame1, "velocity": {"ms": 0.0}}
    )
    env = MockEnv(reset_ret, step_ret)

    wrapper = OpticalFlowVelocityWrapper(
        env,
        camera_height_m=0.15,
        camera_matrix=np.array([[135.59, 0, 320],
                                [0, 135.31, 240],
                                [0, 0, 1]], dtype=np.float32),
        dist_coeffs=np.zeros(4),
    )

    wrapper.reset()
    obs, reward, terminated, truncated, info = wrapper.step(np.array([0.0, 0.15]))

    # With Z=0.15, f=135, pixel motion dy=5, dt=1/30:
    # expected_v ≈ (5 * 0.15) / (135 * 0.033) ≈ 0.17 m/s
    assert "of_velocity" in info
    assert info["of_tracking_state"] == "OK"
    assert info["of_velocity"] > 0.05


def test_pure_rotation_zero_translation():
    """With gyro-provided rotation, translational velocity should be near zero."""
    frame0 = make_synthetic_frame()
    reset_ret = (
        {"image": frame0, "imu": np.zeros(6)},
        {"raw_image": frame0, "velocity": {"ms": 0.0}}
    )
    # Simulate small yaw rotation: rotate image by 0.5 degrees around center
    h, w = frame0.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 0.5, 1.0)
    frame1 = cv2.warpAffine(frame0, M, (w, h))
    gyro = np.array([0, 0, 0, 0, 0, np.radians(15)], dtype=np.float32)
    step_ret = (
        {"image": frame1, "imu": gyro},
        0.0, False, False,
        {"raw_image": frame1, "velocity": {"ms": 0.0}}
    )
    env = MockEnv(reset_ret, step_ret)

    wrapper = OpticalFlowVelocityWrapper(
        env,
        camera_height_m=0.15,
        camera_matrix=np.array([[135.59, 0, 320],
                                [0, 135.31, 240],
                                [0, 0, 1]], dtype=np.float32),
        dist_coeffs=np.zeros(4),
    )

    wrapper.reset()
    obs, reward, terminated, truncated, info = wrapper.step(np.array([0.0, 0.15]))

    # First-order approximation is not perfect; allow some residual
    assert abs(info["of_velocity"]) < 0.5  # translation should be mostly suppressed


def test_tracking_lost_fallback():
    """When tracking is lost, velocity should fall back to zero."""
    reset_ret = (
        {"image": make_synthetic_frame(), "imu": np.zeros(6)},
        {"raw_image": make_synthetic_frame(), "velocity": {"ms": 0.0}}
    )
    # Use different random noise frames each step; LK error exceeds threshold
    step_rets = []
    for i in range(4):
        np.random.seed(i + 100)
        noise = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        step_rets.append((
            {"image": noise, "imu": np.zeros(6)},
            0.0, False, False,
            {"raw_image": noise, "velocity": {"ms": 0.0}}
        ))
    env = MockEnv(reset_ret, step_rets)

    wrapper = OpticalFlowVelocityWrapper(
        env,
        camera_height_m=0.15,
        camera_matrix=np.array([[135.59, 0, 320],
                                [0, 135.31, 240],
                                [0, 0, 1]], dtype=np.float32),
        dist_coeffs=np.zeros(4),
    )

    wrapper.reset()
    # Need at least 3 consecutive lost frames to trigger LOST state
    for _ in range(4):
        obs, reward, terminated, truncated, info = wrapper.step(np.array([0.0, 0.15]))

    assert info["of_tracking_state"] == "LOST"
    assert info["of_velocity"] == 0.0
    assert info["velocity"]["ms"] == 0.0
