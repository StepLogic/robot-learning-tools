# Optical Flow Velocity Wrapper — Design Specification

## 1. Overview

Replace the server-provided velocity estimate in `RacerEnv` with a real-time, lightweight visual-inertial ground odometry system based on **sparse Lucas-Kanade optical flow**, IMU gyro subtraction, and a ground-plane assumption.

**Scope:** A single `gym.Wrapper` that intercepts `step()` / `reset()`, processes consecutive raw camera frames + IMU, and overrides `info["velocity"]`.

## 2. Architecture

```
RacerEnv → [existing wrappers] → OpticalFlowVelocityWrapper → [downstream wrappers]
                                           │
                                           ├─ overrides info["velocity"]
                                           ├─ adds info["of_velocity"]
                                           └─ adds info["of_tracking_state"]
```

## 3. Components

| Component | Purpose | Key OpenCV / NumPy Functions |
|---|---|---|
| **Feature Tracker** | Detects Shi-Tomasi corners on a ground-plane ROI and tracks them with LK. Maintains a mask to ignore sky/horizon. | `cv2.goodFeaturesToTrack`, `cv2.calcOpticalFlowPyrLK` |
| **Gyro Subtractor** | Predicts rotational optical flow from IMU gyro and camera intrinsics, subtracts it from tracked point motion. | `cv2.projectPoints`, `cv2.undistortPoints` |
| **Motion Estimator** | Fits a homography to residual (translational) flow using RANSAC. Derives metric translation using the ground-plane assumption + known camera height. | `cv2.findHomography` |
| **Velocity Filter** | Simple EWMA low-pass filter on computed velocity to reduce jitter. | `numpy` |
| **State Manager** | Tracks internal state: previous grayscale frame, previous feature points, timestamps, tracking quality counters. | — |

## 4. Data Flow

### 4.1 `reset()`
1. Receive first frame from wrapped env.
2. Convert to grayscale.
3. Undistort using `camera.yaml` pinhole intrinsics (`K`, `D`) via `cv2.undistort`.
4. Detect initial features on ground-plane ROI.
5. Set velocity to zero in `info`, mark state as `INITIALIZING`.

### 4.2 `step(action)`
1. Receive new frame + IMU from wrapped env's `step()`.
2. Undistort new frame; track features with LK against previous frame.
3. Compute rotational optical flow from gyro (`roll_rate`, `pitch_rate`, `yaw_rate`) × `dt`.
4. Subtract rotational component from tracked point displacements.
5. Run RANSAC homography on residual displacements.
6. Derive metric velocity:
   ```
   v_x = (flow_x * Z) / (f_x * dt)
   v_y = (flow_y * Z) / (f_y * dt)
   ```
   where `Z` = camera mount height.
7. Apply EWMA filter: `v_filtered = alpha * v_raw + (1 - alpha) * v_prev`.
8. Override `info["velocity"]["ms"]` and set `info["velocity"]["method"] = "optical_flow"`.
9. If tracking is lost for > `tracking_lost_threshold` frames: fallback to zero velocity and flag `info["of_tracking_state"] = "LOST"`.

## 5. Error Handling

| Failure Mode | Response |
|---|---|
| < 8 tracked features | Re-detect Shi-Tomasi corners. If still < 8, mark `tracking_lost`, fallback to zero velocity. |
| RANSAC inliers < 50% | Reject frame, reuse previous filtered velocity. |
| IMU gyro missing | Use essential matrix estimate for rotation (less accurate, higher drift). |
| Rapid rotation (>30°/s) | Skip optical flow translation estimate for that frame; rely on gyro-only rotation + zero translation assumption. |

## 6. Parameters

| Parameter | Default | Description |
|---|---|---|
| `camera_height_m` | `0.15` | Camera mount height above ground (meters) |
| `roi_y_start` | `0.4` | Bottom fraction of image used for ground features (0.0 = top, 1.0 = bottom). 0.4 means bottom 60%. |
| `max_features` | `200` | Maximum Shi-Tomasi corners to detect |
| `lk_win_size` | `(15, 15)` | Lucas-Kanade search window size |
| `lk_max_level` | `2` | Pyramid levels for LK |
| `velocity_alpha` | `0.3` | EWMA smoothing factor (0 = no update, 1 = no smoothing) |
| `tracking_lost_threshold` | `3` | Consecutive lost frames before declaring tracking lost |

## 7. Integration Notes

- **Input:** Uses `info["raw_image"]` (640×480 BGR) and `obs["imu"]` (6-DOF) from the wrapped env.
- **Output:** Overrides `info["velocity"]["ms"]` and `info["velocity"]["method"]`. Adds `info["of_velocity"]`, `info["of_tracking_state"]`.
- **Prerequisite:** Fixes the `raw_image` undefined-variable bug in `racer_imu_env.py:266`.
- **Calibration:** Loads `camera.yaml` at wrapper init for `K` (intrinsics) and `D` (distortion coeffs). Uses `cv2.undistort` with the pinhole + radtan model.
- **Observation Space:** Unchanged.

## 8. Testing Strategy

- **Unit tests:** Mock frame sequences with known ground-truth pixel motion; assert computed velocity matches expected value.
- **Synthetic rotation:** Feed pure rotational IMU data with static scene; assert translational velocity ≈ 0.
- **Integration test:** Run wrapper with `RacerEnv` in simulation (CARLA); compare optical-flow velocity against ground-truth odometry.

## 9. Performance Budget

- Target: < 5 ms per frame on Raspberry Pi 4 (or equivalent host).
- Bottleneck: `cv2.calcOpticalFlowPyrLK` on ~200 points at 640×480.
- If too slow: reduce `max_features` to 100 or downsample to 320×240 before feature detection.
