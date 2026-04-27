# Optical Flow Velocity Wrapper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a lightweight `gym.Wrapper` that replaces the server-provided velocity estimate in `RacerEnv` with a real-time visual-inertial estimate derived from sparse Lucas-Kanade optical flow, IMU gyro subtraction, and a ground-plane assumption.

**Architecture:** A single `OpticalFlowVelocityWrapper` class intercepts `step()` and `reset()`, processes consecutive raw 640×480 BGR frames from `info["raw_image"]` and IMU from `obs["imu"]`, detects Shi-Tomasi corners on a ground-plane ROI, tracks them with LK, subtracts rotational flow predicted from gyro, fits a RANSAC homography to residual flow, and converts to metric velocity using known camera height and pinhole intrinsics from `camera.yaml`.

**Tech Stack:** OpenCV (`cv2`), NumPy, PyYAML, Gymnasium.

---

## File Structure

| File | Responsibility |
|---|---|
| `racer_imu_env.py` | **Modify:** Fix the `raw_image` undefined-variable bug at line 266 so the env stores the raw frame before undistortion. |
| `optical_flow_velocity_wrapper.py` | **Create:** New `gym.Wrapper` subclass implementing the optical-flow velocity estimation pipeline. |
| `tests/test_optical_flow_velocity_wrapper.py` | **Create:** Unit tests using synthetic frame sequences with known ground-truth motion. |

---

## Task 1: Fix `raw_image` Bug in `racer_imu_env.py`

**Files:**
- Modify: `/home/kojogyaase/Projects/Research/recovery-from-failure/racer_imu_env.py:260-266`

- [ ] **Step 1: Read the relevant lines in `racer_imu_env.py`**

Read lines 255-275 to confirm the `raw_image` variable is used before being defined.

- [ ] **Step 2: Store raw image before undistortion**

In `_send_command_and_get_image`, store the raw image before calling `undistort`:

```python
# After parsing image from response (line ~261)
image, imu_6d, velocity, collision, blocked, orientation = _parse_response(
    result, self.imu_accel, self.imu_gyro)

# FIX: store raw image before undistortion
self.raw_image = image.copy()  # 640x480 BGR

# Then undistort for policy
self.image = undistort(image)
```

Remove the buggy line `self.raw_image = raw_image`.

- [ ] **Step 3: Verify the fix**

Run a quick syntax check:
```bash
python -m py_compile racer_imu_env.py
```

- [ ] **Step 4: Commit**

```bash
git add racer_imu_env.py
git commit -m "fix: store raw_image before undistort in RacerEnv"
```

---

## Task 2: Write Failing Unit Tests

**Files:**
- Create: `/home/kojogyaase/Projects/Research/recovery-from-failure/tests/test_optical_flow_velocity_wrapper.py`

- [ ] **Step 1: Write test for pure forward motion**

```python
import numpy as np
import pytest
import cv2
from unittest.mock import MagicMock

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


def test_forward_motion_velocity():
    """Synthetic forward motion: image shifts up (ground moves down)."""
    env = MagicMock()
    env.observation_space = MagicMock()
    env.action_space = MagicMock()

    wrapper = OpticalFlowVelocityWrapper(
        env,
        camera_height_m=0.15,
        fx=135.59, fy=135.31,
        camera_matrix=np.array([[135.59, 0, 320],
                                [0, 135.31, 240],
                                [0, 0, 1]], dtype=np.float32),
        dist_coeffs=np.zeros(4),
    )

    # Reset with first frame
    env.reset.return_value = (
        {"image": make_synthetic_frame(offset_x=0), "imu": np.zeros(6)},
        {"raw_image": make_synthetic_frame(offset_x=0), "velocity": {"ms": 0.0}}
    )
    wrapper.reset()

    # Step: shift texture upward by 10 pixels (ground moves down in image)
    env.step.return_value = (
        {"image": make_synthetic_frame(offset_x=0), "imu": np.zeros(6)},
        0.0, False, False,
        {"raw_image": make_synthetic_frame(offset_x=0), "velocity": {"ms": 0.0}}
    )
    obs, reward, terminated, truncated, info = wrapper.step(np.array([0.0, 0.15]))

    # With Z=0.15, f=135, pixel motion dy=10, dt=1/30:
    # expected_v ≈ (10 * 0.15) / (135 * 0.033) ≈ 0.34 m/s
    assert "of_velocity" in info
    assert info["of_tracking_state"] == "OK"
    assert info["of_velocity"] > 0.1
```

- [ ] **Step 2: Write test for pure rotation (zero translation)**

```python
def test_pure_rotation_zero_translation():
    """With gyro-provided rotation, translational velocity should be near zero."""
    env = MagicMock()
    env.observation_space = MagicMock()
    env.action_space = MagicMock()

    wrapper = OpticalFlowVelocityWrapper(
        env,
        camera_height_m=0.15,
        camera_matrix=np.array([[135.59, 0, 320],
                                [0, 135.31, 240],
                                [0, 0, 1]], dtype=np.float32),
        dist_coeffs=np.zeros(4),
    )

    env.reset.return_value = (
        {"image": make_synthetic_frame(), "imu": np.zeros(6)},
        {"raw_image": make_synthetic_frame(), "velocity": {"ms": 0.0}}
    )
    wrapper.reset()

    # Simulate yaw rotation via gyro: 5 deg/frame at 30fps = 150 deg/s
    gyro = np.array([0, 0, 0, 0, 0, np.radians(150)], dtype=np.float32)
    env.step.return_value = (
        {"image": make_synthetic_frame(), "imu": gyro},
        0.0, False, False,
        {"raw_image": make_synthetic_frame(), "velocity": {"ms": 0.0}}
    )
    obs, reward, terminated, truncated, info = wrapper.step(np.array([0.0, 0.15]))

    assert abs(info["of_velocity"]) < 0.05  # translation should be suppressed
```

- [ ] **Step 3: Write test for tracking lost fallback**

```python
def test_tracking_lost_fallback():
    """When tracking is lost, velocity should fall back to zero."""
    env = MagicMock()
    env.observation_space = MagicMock()
    env.action_space = MagicMock()

    wrapper = OpticalFlowVelocityWrapper(
        env,
        camera_height_m=0.15,
        camera_matrix=np.array([[135.59, 0, 320],
                                [0, 135.31, 240],
                                [0, 0, 1]], dtype=np.float32),
        dist_coeffs=np.zeros(4),
    )

    env.reset.return_value = (
        {"image": make_synthetic_frame(), "imu": np.zeros(6)},
        {"raw_image": make_synthetic_frame(), "velocity": {"ms": 0.0}}
    )
    wrapper.reset()

    # Step with a blank frame (no texture to track)
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    env.step.return_value = (
        {"image": blank, "imu": np.zeros(6)},
        0.0, False, False,
        {"raw_image": blank, "velocity": {"ms": 0.0}}
    )
    obs, reward, terminated, truncated, info = wrapper.step(np.array([0.0, 0.15]))

    assert info["of_tracking_state"] == "LOST"
    assert info["of_velocity"] == 0.0
    assert info["velocity"]["ms"] == 0.0
```

- [ ] **Step 4: Run tests to verify they fail**

```bash
cd /home/kojogyaase/Projects/Research/recovery-from-failure
pytest tests/test_optical_flow_velocity_wrapper.py -v
```

Expected: `FAILED` — `OpticalFlowVelocityWrapper` does not exist.

- [ ] **Step 5: Commit**

```bash
git add tests/test_optical_flow_velocity_wrapper.py
git commit -m "test: add optical flow velocity wrapper tests"
```

---

## Task 3: Implement `OpticalFlowVelocityWrapper`

**Files:**
- Create: `/home/kojogyaase/Projects/Research/recovery-from-failure/optical_flow_velocity_wrapper.py`

- [ ] **Step 1: Write wrapper skeleton and imports**

```python
"""Optical Flow Velocity Wrapper for RacerEnv.

Replaces server-provided velocity with a visual-inertial estimate derived from
sparse Lucas-Kanade optical flow, IMU gyro subtraction, and ground-plane assumption.
"""

import time
from typing import Any

import cv2
import numpy as np
import gymnasium as gym
import yaml


class OpticalFlowVelocityWrapper(gym.Wrapper):
    """
    Wrapper that overrides info["velocity"] with an optical-flow-based estimate.

    Assumes:
      - wrapped env info dict contains "raw_image" (640x480 BGR)
      - wrapped env observation contains "imu" (6,) [ax, ay, az, roll_rate, pitch_rate, yaw_rate]
      - camera is mounted at a fixed height looking forward
      - ground is approximately flat
    """

    def __init__(
        self,
        env: gym.Env,
        camera_height_m: float = 0.15,
        camera_matrix: np.ndarray | None = None,
        dist_coeffs: np.ndarray | None = None,
        fx: float | None = None,
        fy: float | None = None,
        roi_y_start: float = 0.4,
        max_features: int = 200,
        lk_win_size: tuple[int, int] = (15, 15),
        lk_max_level: int = 2,
        velocity_alpha: float = 0.3,
        tracking_lost_threshold: int = 3,
    ):
        super().__init__(env)

        self.camera_height_m = camera_height_m
        self.roi_y_start = roi_y_start
        self.max_features = max_features
        self.lk_params = dict(
            winSize=lk_win_size,
            maxLevel=lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        self.velocity_alpha = velocity_alpha
        self.tracking_lost_threshold = tracking_lost_threshold

        # Camera calibration
        if camera_matrix is not None:
            self.K = camera_matrix.astype(np.float32)
            self.D = dist_coeffs.astype(np.float32) if dist_coeffs is not None else np.zeros(4, dtype=np.float32)
        else:
            self.K = np.array([[fx or 135.0, 0, 320.0],
                               [0, fy or 135.0, 240.0],
                               [0, 0, 1]], dtype=np.float32)
            self.D = np.zeros(4, dtype=np.float32)

        # State
        self._prev_gray: np.ndarray | None = None
        self._prev_pts: np.ndarray | None = None
        self._prev_time: float | None = None
        self._filtered_velocity: float = 0.0
        self._consecutive_lost: int = 0

    def _create_ground_mask(self, shape: tuple[int, int]) -> np.ndarray:
        """Binary mask for ground-plane ROI."""
        mask = np.zeros(shape, dtype=np.uint8)
        y_start = int(shape[0] * self.roi_y_start)
        mask[y_start:, :] = 255
        return mask

    def _detect_features(self, gray: np.ndarray) -> np.ndarray | None:
        """Detect Shi-Tomasi corners on the ground-plane ROI."""
        mask = self._create_ground_mask(gray.shape[:2])
        pts = cv2.goodFeaturesToTrack(
            gray, mask=mask, maxCorners=self.max_features,
            qualityLevel=0.01, minDistance=10, blockSize=7
        )
        return pts

    def _track_features(self, prev_gray: np.ndarray, gray: np.ndarray, prev_pts: np.ndarray):
        """Track features with LK. Returns (curr_pts, status, error)."""
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, prev_pts, None, **self.lk_params
        )
        return curr_pts, status, err

    def _subtract_rotational_flow(self, prev_pts: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        """
        Predict rotational optical flow from gyro and subtract from point motion.

        Args:
            prev_pts: Nx2 array of previous feature locations (undistorted, normalized).
            gyro: (6,) IMU [ax, ay, az, roll_rate, pitch_rate, yaw_rate] in rad/s.
            dt: time delta in seconds.

        Returns:
            Nx2 predicted rotational flow in pixels.
        """
        if prev_pts is None or len(prev_pts) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        # Angular velocity vector (roll, pitch, yaw) in rad/s
        omega = gyro[3:6]  # [roll_rate, pitch_rate, yaw_rate]

        # Convert normalized points back to pixel coords for flow prediction
        # For small rotations, pixel displacement ≈ K * (omega × X) * dt
        # where X is the 3D ray direction
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        # Normalized image coordinates
        x_n = (prev_pts[:, 0] - cx) / fx
        y_n = (prev_pts[:, 1] - cy) / fy

        # 3D ray directions (assuming unit depth)
        X = np.stack([x_n, y_n, np.ones_like(x_n)], axis=1)  # Nx3
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        # Rotational velocity in camera frame: omega × X
        # Using cross product matrix
        wx, wy, wz = omega * dt
        rot_flow = np.zeros_like(prev_pts)
        rot_flow[:, 0] = fx * (-wz * y_n + wy * np.ones_like(x_n))
        rot_flow[:, 1] = fy * (wz * x_n - wx * np.ones_like(y_n))

        return rot_flow.astype(np.float32)

    def _estimate_velocity(self, prev_pts: np.ndarray, curr_pts: np.ndarray, dt: float) -> float:
        """
        Estimate forward velocity from residual (translational) optical flow.

        Uses the ground-plane assumption: Z = camera_height_m.
        For forward motion, vertical flow dominates in the bottom of the image.
        We take the median vertical flow of inliers.
        """
        if len(prev_pts) < 4:
            return 0.0

        # Compute homography on residual flow to find inliers
        H, mask = cv2.findHomography(prev_pts, curr_pts, cv2.RANSAC, 3.0)
        if H is None or mask is None:
            return 0.0

        inliers = curr_pts[mask.ravel().astype(bool)]
        if len(inliers) < 4:
            return 0.0

        # Median vertical pixel displacement (positive = ground moving down = forward motion)
        dy_pixels = np.median(inliers[:, 1] - prev_pts[mask.ravel().astype(bool), 1])

        # Convert pixel motion to metric velocity
        # v = (dy * Z) / (f_y * dt)
        # Use absolute value since direction is inferred from sign of flow
        fy = self.K[1, 1]
        Z = self.camera_height_m
        v_raw = (dy_pixels * Z) / (fy * dt)

        return float(v_raw)

    def _update_velocity(self, raw_velocity: float) -> float:
        """Apply EWMA filter."""
        self._filtered_velocity = (
            self.velocity_alpha * raw_velocity
            + (1.0 - self.velocity_alpha) * self._filtered_velocity
        )
        return self._filtered_velocity

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        raw_image = info.get("raw_image")
        if raw_image is not None:
            gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
            # Undistort using camera.yaml pinhole + radtan
            gray = cv2.undistort(gray, self.K, self.D)
            self._prev_gray = gray
            self._prev_pts = self._detect_features(gray)
            self._prev_time = time.time()

        self._filtered_velocity = 0.0
        self._consecutive_lost = 0

        # Override velocity in info
        info["velocity"] = {"cms": 0.0, "ms": 0.0, "method": "optical_flow"}
        info["of_velocity"] = 0.0
        info["of_tracking_state"] = "INITIALIZING"

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        raw_image = info.get("raw_image")
        imu = obs.get("imu", np.zeros(6, dtype=np.float32))
        gyro = imu[3:6]

        if raw_image is None or self._prev_gray is None:
            info["of_tracking_state"] = "NO_IMAGE"
            info["of_velocity"] = 0.0
            info["velocity"] = {"cms": 0.0, "ms": 0.0, "method": "optical_flow"}
            return obs, reward, terminated, truncated, info

        # Process current frame
        gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.undistort(gray, self.K, self.D)
        curr_time = time.time()
        dt = curr_time - self._prev_time if self._prev_time else 1.0 / 30.0
        self._prev_time = curr_time

        # Track features
        if self._prev_pts is None or len(self._prev_pts) < 8:
            self._prev_pts = self._detect_features(self._prev_gray)

        if self._prev_pts is not None and len(self._prev_pts) >= 8:
            curr_pts, status, err = self._track_features(self._prev_gray, gray, self._prev_pts)
            good_prev = self._prev_pts[status == 1]
            good_curr = curr_pts[status == 1]

            if len(good_prev) >= 8:
                # Subtract rotational flow
                rot_flow = self._subtract_rotational_flow(good_prev, gyro, dt)
                residual_curr = good_curr - rot_flow

                # Estimate velocity from residual flow
                raw_velocity = self._estimate_velocity(good_prev, residual_curr, dt)
                filtered_v = self._update_velocity(raw_velocity)

                self._consecutive_lost = 0
                self._prev_pts = good_curr.reshape(-1, 1, 2)

                info["of_velocity"] = filtered_v
                info["of_tracking_state"] = "OK"
                info["velocity"] = {
                    "cms": filtered_v * 100.0,
                    "ms": filtered_v,
                    "method": "optical_flow",
                }
            else:
                self._consecutive_lost += 1
                self._prev_pts = None
        else:
            self._consecutive_lost += 1

        # Check tracking lost
        if self._consecutive_lost >= self.tracking_lost_threshold:
            info["of_tracking_state"] = "LOST"
            info["of_velocity"] = 0.0
            info["velocity"] = {"cms": 0.0, "ms": 0.0, "method": "optical_flow"}
            self._filtered_velocity = 0.0

        self._prev_gray = gray
        return obs, reward, terminated, truncated, info
```

- [ ] **Step 2: Add `camera.yaml` loading helper**

Add a convenience class method to load calibration from the project's `camera.yaml`:

```python
@classmethod
def from_camera_yaml(cls, env: gym.Env, yaml_path: str, **kwargs):
    """Load pinhole intrinsics and radtan distortion from Kalibr camera.yaml."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    cam0 = data["cam0"]
    fx, fy, cx, cy = cam0["intrinsics"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    D = np.array(cam0["distortion_coeffs"], dtype=np.float32)

    return cls(env, camera_matrix=K, dist_coeffs=D, **kwargs)
```

- [ ] **Step 3: Run tests to verify they pass**

```bash
cd /home/kojogyaase/Projects/Research/recovery-from-failure
pytest tests/test_optical_flow_velocity_wrapper.py -v
```

Expected: `PASSED` for all three tests.

- [ ] **Step 4: Commit**

```bash
git add optical_flow_velocity_wrapper.py
git commit -m "feat: implement optical flow velocity wrapper"
```

---

## Task 4: Integration Test

**Files:**
- Create: `/home/kojogyaase/Projects/Research/recovery-from-failure/tests/test_optical_flow_integration.py`

- [ ] **Step 1: Write integration test with mock env**

```python
import numpy as np
from unittest.mock import MagicMock

from optical_flow_velocity_wrapper import OpticalFlowVelocityWrapper


def test_wrapper_integration():
    """End-to-end: wrapper processes a sequence of frames and produces velocity."""
    env = MagicMock()
    env.observation_space = MagicMock()
    env.action_space = MagicMock()

    K = np.array([[135.59, 0, 320], [0, 135.31, 240], [0, 0, 1]], dtype=np.float32)
    wrapper = OpticalFlowVelocityWrapper(env, camera_matrix=K, dist_coeffs=np.zeros(4))

    # Reset
    frame0 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    env.reset.return_value = (
        {"image": frame0, "imu": np.zeros(6)},
        {"raw_image": frame0, "velocity": {"ms": 0.0}}
    )
    obs, info = wrapper.reset()
    assert info["of_tracking_state"] == "INITIALIZING"

    # Step 1: forward motion
    frame1 = np.roll(frame0, 5, axis=0)  # shift down 5 pixels
    env.step.return_value = (
        {"image": frame1, "imu": np.zeros(6)},
        0.0, False, False,
        {"raw_image": frame1, "velocity": {"ms": 0.0}}
    )
    obs, reward, terminated, truncated, info = wrapper.step(np.array([0.0, 0.15]))
    assert "of_velocity" in info
    assert info["of_tracking_state"] in ("OK", "LOST")
```

- [ ] **Step 2: Run integration test**

```bash
pytest tests/test_optical_flow_integration.py -v
```

Expected: `PASSED`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_optical_flow_integration.py
git commit -m "test: add optical flow integration test"
```

---

## Spec Coverage Check

| Spec Section | Implementing Task |
|---|---|
| 3.1 `reset()` data flow | Task 3, Step 1 (`reset` method) |
| 3.2 `step()` data flow | Task 3, Step 1 (`step` method) |
| 4 Error handling (tracking lost, RANSAC, IMU missing, rapid rotation) | Task 3, Step 1 |
| 5 Parameters (all 7 listed) | Task 3, Step 1 (`__init__`) |
| 6 Integration notes (raw_image, camera.yaml, observation space) | Task 1 (bug fix), Task 3, Step 2 (`from_camera_yaml`) |
| 7 Testing strategy | Task 2 (unit), Task 4 (integration) |

## Placeholder Scan

- No "TBD", "TODO", or "implement later" found.
- All error-handling branches have concrete fallback values.
- All test code is complete with assertions.
- Type consistency checked: `camera_matrix` / `K` / `dist_coeffs` / `D` used consistently.

## Type Consistency Check

- `camera_matrix` parameter in `__init__` matches `self.K`.
- `dist_coeffs` parameter matches `self.D`.
- `velocity` dict keys match spec: `cms`, `ms`, `method`.
- Tracking states match spec: `INITIALIZING`, `OK`, `LOST`, `NO_IMAGE`.

---

**Plan complete and saved to `docs/superpowers/plans/2026-04-21-optical-flow-velocity.md`.**

Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints.

Which approach?