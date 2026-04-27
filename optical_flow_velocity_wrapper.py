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
        self.lk_error_threshold = 30.0  # Reject LK tracks with error above this
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

    def _subtract_rotational_flow(self, prev_pts: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
        """
        Predict rotational optical flow from gyro and subtract from point motion.

        Args:
            prev_pts: Nx2 array of previous feature locations (undistorted, normalized).
            omega: (3,) angular velocity [roll_rate, pitch_rate, yaw_rate] in rad/s.
            dt: time delta in seconds.

        Returns:
            Nx2 predicted rotational flow in pixels.
        """
        if prev_pts is None or len(prev_pts) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        # Convert normalized points back to pixel coords for flow prediction
        # For small rotations, pixel displacement ≈ K * (omega × X) * dt
        # where X is the 3D ray direction
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        # Normalized image coordinates
        x_n = (prev_pts[:, 0] - cx) / fx
        y_n = (prev_pts[:, 1] - cy) / fy

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
            # Filter by LK error to reject bad tracks
            valid = (status.flatten() == 1) & (err.flatten() < self.lk_error_threshold)
            good_prev = self._prev_pts[valid].reshape(-1, 2)
            good_curr = curr_pts[valid].reshape(-1, 2)

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
