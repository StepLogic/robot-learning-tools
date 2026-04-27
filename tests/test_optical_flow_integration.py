import numpy as np
import gymnasium as gym
from gymnasium import spaces

from optical_flow_velocity_wrapper import OpticalFlowVelocityWrapper


class MockEnv(gym.Env):
    """Minimal mock env for wrapper integration tests."""

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


def test_wrapper_integration():
    """End-to-end: wrapper processes a sequence of frames and produces velocity."""
    np.random.seed(42)
    frame0 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    reset_ret = (
        {"image": frame0, "imu": np.zeros(6)},
        {"raw_image": frame0, "velocity": {"ms": 0.0}}
    )
    # Forward motion: shift down 5 pixels
    frame1 = np.roll(frame0, 5, axis=0)
    step_ret = (
        {"image": frame1, "imu": np.zeros(6)},
        0.0, False, False,
        {"raw_image": frame1, "velocity": {"ms": 0.0}}
    )
    env = MockEnv(reset_ret, step_ret)

    K = np.array([[135.59, 0, 320], [0, 135.31, 240], [0, 0, 1]], dtype=np.float32)
    wrapper = OpticalFlowVelocityWrapper(env, camera_matrix=K, dist_coeffs=np.zeros(4))

    obs, info = wrapper.reset()
    assert info["of_tracking_state"] == "INITIALIZING"

    obs, reward, terminated, truncated, info = wrapper.step(np.array([0.0, 0.15]))
    assert "of_velocity" in info
    assert info["of_tracking_state"] in ("OK", "LOST")
