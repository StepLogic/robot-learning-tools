# SLAM-Based Reward for RL Training — Design Spec

**Date:** 2026-04-14
**Status:** Approved

## Problem

The current RL training pipeline uses hardcoded pseudo-odometry (`dist_step = 0.01`) for position tracking and Hindsight Experience Replay (HER) for goal relabeling. This produces inaccurate distance estimates and requires the policy to observe a `goal_rel` key that couples training infrastructure to the observation space. The goal is to replace this with ORB_SLAM3 visual-inertial SLAM for accurate metric position, and use that position solely for reward computation — removing HER entirely.

## Architecture

### Wrapper Stack (Before → After)

**Before:**
```
RacerEnv → EnvCompatibility → StackingWrapper → RewardWrapper → MobileNetFeatureWrapper → RecordEpisodeStatistics → TimeLimit → GoalRelObservationWrapper
```

**After:**
```
RacerEnv → EnvCompatibility → StackingWrapper → MobileNetFeatureWrapper → RecordEpisodeStatistics → TimeLimit → SLAMRewardWrapper
```

Key changes:
- `RewardWrapper` and `GoalRelObservationWrapper` are replaced by `SLAMRewardWrapper`
- `HindsightReplayBuffer` is replaced by standard `ReplayBuffer`
- `goal_rel` observation key is removed

### SLAMRewardWrapper

A Gymnasium wrapper that:
1. Runs ORB_SLAM3 in a **dedicated background thread** using the existing `orbslam3` Python bindings
2. Each `step()`, pushes the camera image + accumulated IMU data to the SLAM thread queue
3. Reads the latest `(pose, velocity, tracking_state, timestamp)` from a thread-safe state dict
4. Computes **distance-based reward** from SLAM position
5. On reset: resamples goal, calls `slam.reset_active_map()` if tracking is lost

**Observation space (no `goal_rel`):**
```python
{"pixels": (num_frames * feature_dim,), "actions": (6,), "imu": (18,)}
```

### SLAM Thread

- Runs in a `threading.Thread` daemon
- Feeds frames and IMU to `orbslam3.System.track_monocular_imu_with_velocity()`
- Updates shared state: `latest_pose` (4x4), `latest_velocity` (3,), `latest_state` (string), `latest_timestamp`
- Thread-safe via `threading.Lock` on the shared state dict
- Graceful shutdown on `close()`

### Graceful Degradation

| SLAM State | Behavior |
|---|---|
| Not initialized (first 10-30 frames) | Fall back to pseudo-odometry (`dist_step=0.01`, yaw from server). Scale `k_dist` to 1.0 since pseudo-odometry is inaccurate |
| Tracking OK | Use SLAM position for distance-to-goal reward |
| Tracking lost >5 frames | `terminated=True, reward -= 30` |
| SLAM thread crash | Catch exception, fall back to pseudo-odometry, log error |

## ORB_SLAM3 Configuration

### Camera: KannalaBrandt8 (Fisheye)

The robot uses a fisheye camera. ORB_SLAM3 config file `donkeycar_fisheye_imu.yaml`:

```yaml
File.version: "1.0"
Camera.type: "KannalaBrandt8"

# From fisheye_calibration.json (640x480)
Camera1.fx: 163.251
Camera1.fy: 163.588
Camera1.cx: 319.161
Camera1.cy: 239.006
Camera1.k1: 3.5258
Camera1.k2: -92.1667
Camera1.k3: 601.312
Camera1.k4: -32.2121

Camera.width: 640
Camera.height: 480
Camera.newWidth: 640
Camera.newHeight: 480
Camera.fps: 20
Camera.RGB: 1

# IMU — body-to-camera transform (identity + small Z offset as approximation)
IMU.T_b_c1: !!opencv-matrix
   rows: 4
   cols: 4
   dt: f
   data: [1.0, 0.0, 0.0, 0.05,
         0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0]

# IMU noise — starting values, need calibration
IMU.NoiseGyro: 1.7e-4
IMU.NoiseAcc: 2.0e-3
IMU.GyroWalk: 1.9393e-5
IMU.AccWalk: 3.0e-3
IMU.Frequency: 20.0

# ORB extractor
ORBextractor.nFeatures: 1000
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7
```

### Critical: Image Path

- **ORB_SLAM3 receives the raw un-undistorted fisheye image** — ORB_SLAM3 handles undistortion internally using the KannalaBrandt8 model
- **MobileNetFeatureWrapper receives the undistorted image** (current `undistort()` in RacerEnv) — no change to the visual pipeline
- Implementation: `RacerEnv._parse_response()` currently calls `undistort(image)` and stores the result in `self.image`. We will store **both** the raw and undistorted image: `self.raw_image` (for SLAM) and `self.image` (for policy). The `_get_info()` method will expose `raw_image` so `SLAMRewardWrapper` can access it.

### IMU Data Format

The `orbslam3` Python binding expects IMU data as an Nx7 numpy array:
```python
imu_data = np.array([[ax, ay, az, gx, gy, gz, timestamp], ...])
```

The robot server provides IMU at ~20Hz (one reading per frame). The SLAM thread accumulates IMU readings between frames and sends them as a batch.

### Prerequisite: Recalibration

The existing `fisheye_calibration.json` has RMS error of 102.5 — this is unusable. A proper fisheye calibration (RMS < 1.0) is required before ORB_SLAM3 will work reliably. Use OpenCV's `cv2.fisheye.calibrate()` with a large chessboard and 50+ images.

## Reward Structure

### Formula

```
r = -k_step + k_dist * delta_dist + k_goal * 1[dist < threshold]
```

Where:
- `delta_dist = prev_dist_to_goal - current_dist_to_goal` — positive when moving toward goal
- `current_dist_to_goal = ||slam_pos - goal_pos||_2` — Euclidean distance in SLAM world frame (2D x,y)
- `1[dist < threshold]` — indicator function for goal reached

### Default Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `k_dist` | 5.0 | Scale for distance delta reward |
| `k_goal` | 50.0 | Sparse bonus for reaching goal |
| `k_step` | 0.1 | Per-step time penalty |
| `goal_threshold` | 0.5 | Meters to consider goal reached |
| `stall_limit` | 100 | Steps of zero velocity before truncation |

### Termination Conditions

| Condition | Reward modifier | terminated |
|---|---|---|
| Goal reached | `+k_goal (50.0)` | True (resample goal) |
| Collision detected | `-100.0` | True |
| Stall (vel < 0.01 for 100 steps) | `-50.0` | True |
| SLAM lost > 5 frames | `-30.0` | True |

### Goal Sampling

- On episode reset: `agent_start = slam_position`, goal sampled in polar coords `(angle, distance)` relative to start
- On goal reached: resample new goal relative to current SLAM position
- Goal range: `0.5m` to `20.0m` distance, `[-pi, pi]` angle

## Training Script Changes

### New File: `train_slam_robot.py`

Derived from `train_her_robot.py` with these changes:

**Removed:**
- `HindsightReplayBuffer` class — replaced by standard `ReplayBuffer`
- `GoalRelObservationWrapper` class — replaced by `SLAMRewardWrapper`
- HER flags: `her_fraction`, `her_strategy`, `goal_range` (old meaning), `use_goal_masking`, `mask_probability`
- `goal_rel` from observation space

**Added:**
- `SLAMRewardWrapper` (new file `slam_reward_wrapper.py`)
- SLAM flags: `slam_vocab`, `slam_settings`, `k_dist`, `k_goal`, `k_step`, `goal_threshold`
- SLAM state tracking in info dict: `slam_pos`, `slam_vel`, `slam_state`, `dist_to_goal`, `goal_pos`

**Modified:**
- Wrapper stack order (see Architecture section)
- ReplayBuffer: standard `ReplayBuffer` instead of `HindsightReplayBuffer`
- No `end_episode()` calls on replay buffer
- Transition dict no longer needs `info`/`next_info` for HER relabeling

### New File: `slam_reward_wrapper.py`

The `SLAMRewardWrapper` class implementing the Gymnasium wrapper interface:

```python
class SLAMRewardWrapper(gym.Wrapper):
    def __init__(self, env, vocab_path, settings_path, 
                 goal_threshold=0.5, k_dist=5.0, k_goal=50.0, k_step=0.1):
        # Initialize ORB_SLAM3 in background thread
        # Start SLAM thread daemon
        pass
    
    def reset(self, **kwargs):
        # Reset goal, pseudo-odometry fallback state
        # If SLAM initialized, reset goal relative to SLAM position
        # If SLAM tracking lost, call slam.reset_active_map()
        pass
    
    def step(self, action):
        # Push raw image + IMU to SLAM thread queue
        # Read latest SLAM state (pose, velocity, tracking_state)
        # Compute dist_to_goal
        # Compute reward: -k_step + k_dist * delta_dist + k_goal * goal_reached
        # Handle SLAM-lost termination
        # Return (obs, reward, terminated, truncated, info)
        pass
    
    def close(self):
        # Shutdown SLAM, join thread
        pass
```

### Modified: `racer_imu_env.py`

Minimal change: expose the **raw (un-undistorted)** image in the `info` dict so `SLAMRewardWrapper` can feed it to ORB_SLAM3.

```python
# In _send_command_and_get_image, after undistort:
info["raw_image"] = image_before_undistort  # BGR, 640x480
```

Also expose raw IMU data with timestamps for SLAM accumulation:

```python
info["imu_raw"] = imu_6d  # shape (6,), already available
info["timestamp"] = time.time()  # wall-clock timestamp for SLAM
```

## Calibration Pipeline

### Location

New folder `calibration/` in the project root.

### Hardware Context (from robot-server codebase)

- **Camera:** Raspberry Pi Camera Module via `picamera2` (libcamera). Currently configured at 120x120 in `server.py` — must be reconfigured to 640x480 for SLAM.
- **IMU:** WitMotion sensor (likely WT901/WT61) via USB serial, `witmotion` Python library. Angular velocity outputs in **deg/s** (not rad/s). Sample rate ~10-30Hz.
- **Server:** HTTP on port 8000 (`server.py`) OR TCP on port 9000 (the version used by `racer_imu_env.py`). IMU data is NOT integrated into `server.py` — it exists standalone in `test_imu.py`.
- **Existing calibration:** `fisheye_calibration.json` has RMS error 102.5 — unusable. `rover_orbslam_imu.yaml` has WitMotion noise estimates (NoiseGyro: 0.001, NoiseAcc: 0.01) — rough starting points, not calibrated.

### Script 1: `calibration/calibrate_camera.py` — Fisheye Camera Calibration

**Purpose:** Compute KannalaBrandt8 (fisheye) camera intrinsics and distortion coefficients.

**Procedure:**
1. Connect to the robot camera (either via the server or directly on the Pi with `picamera2`)
2. Capture 30-50 chessboard images at 640x480 resolution, covering all regions of the field of view
3. Detect chessboard corners with `cv2.findChessboardCorners()` + `cv2.cornerSubPix()`
4. Run `cv2.fisheye.calibrate()` for KannalaBrandt8 model
5. Validate: reprojection error must be < 1.0 RMS. Visual check: undistort a test image
6. Output `calibration/camera_calib.json` with: `fx, fy, cx, cy, k1, k2, k3, k4, rms_error, image_size`

**Critical:** Must capture raw un-undistorted fisheye images. The existing `undistort()` in `RacerEnv` must be bypassed. If using the server, the server must be configured to send raw images at 640x480.

**Chessboard:** 8x6 inner corners, square size 25mm (printed on A4 paper). Must fill >30% of the image area in each frame.

### Script 2: `calibration/calibrate_imu.py` — IMU Allan Variance Calibration

**Purpose:** Extract WitMotion IMU noise parameters for ORB_SLAM3.

**Procedure:**
1. Place robot on a flat, stable surface (no vibration)
2. Record static IMU data for 2-4 hours via the WitMotion `witmotion` library
3. Log timestamped accel (m/s^2) + gyro (deg/s) readings to `calibration/imu_log.csv`
4. Convert gyro from deg/s to rad/s: `gyro_rad = gyro_deg * pi / 180`
5. Compute Allan deviation curves for each axis
6. Extract noise parameters from the Allan deviation slopes:
   - **White noise** (slope -0.5): `NoiseGyro`, `NoiseAcc`
   - **Random walk / bias instability** (slope +0.5): `GyroWalk`, `AccWalk`
7. Output `calibration/imu_calib.json` with: `NoiseGyro, NoiseAcc, GyroWalk, AccWalk, Frequency, raw_data_path`

**WitMotion unit conversions:**
- Angular velocity: `deg/s → rad/s` (multiply by `pi/180`)
- Acceleration: already in `m/s^2` (no conversion needed)
- ORB_SLAM3 noise units: `NoiseGyro` in `rad/s/sqrt(Hz)`, `NoiseAcc` in `m/s^2/sqrt(Hz)`, `GyroWalk` in `rad/s^2/sqrt(Hz)`, `AccWalk` in `m/s^3/sqrt(Hz)`

### Script 3: `calibration/calibrate_extrinsics.py` — Camera-IMU Extrinsic Calibration

**Purpose:** Estimate the rigid body transform `T_b_c1` (body/IMU frame to camera frame).

**Procedure:**
1. Print and mount a calibration target (AprilTag 6x6 recommended by Kalibr)
2. Record synchronized camera frames + IMU data while slowly moving the robot in front of the target (~60 seconds of data)
3. Save as ROS bag format for Kalibr input
4. Run Kalibr `kalibr_calibrate_imu_camera` with:
   - Camera model: KannalaBrandt8 (from `camera_calib.json`)
   - IMU noise params (from `imu_calib.json`)
   - Target description (AprilTag config)
5. Extract `T_imu_camera` from Kalibr output
6. Convert to ORB_SLAM3's `IMU.T_b_c1` format (body-to-camera = inverse of camera-to-body)
7. Output `calibration/extrinsics_calib.json` with: `T_b_c1` as 4x4 matrix

**Kalibr requirements:**
- Docker container (recommended) or native install
- ROS bag with synchronized `/cam0/image_raw` and `/imu0` topics
- Target YAML (AprilTag 6x6, tag size in meters)

**Fallback if Kalibr is unavailable:** Manually measure the physical offset between IMU and camera mounting points. The Donkey Car camera is typically ~0.05m forward and ~0.05m above the IMU. This gives an approximate transform — less accurate but functional.

### Script 4: `calibration/generate_orbslam3_yaml.py` — ORB_SLAM3 Config Generator

**Purpose:** Assemble all calibration results into a ready-to-use ORB_SLAM3 YAML config.

**Procedure:**
1. Read `calibration/camera_calib.json`
2. Read `calibration/imu_calib.json`
3. Read `calibration/extrinsics_calib.json`
4. Assemble `donkeycar_fisheye_imu.yaml` with:
   - Camera section: KannalaBrandt8, intrinsics, distortion
   - IMU section: noise parameters, frequency, T_b_c1 transform
   - ORB extractor: `nFeatures=1000`, `scaleFactor=1.2`, `nLevels=8`, `iniThFAST=20`, `minThFAST=7`
   - Viewer section: default parameters
5. Output `calibration/donkeycar_fisheye_imu.yaml`
6. Validate YAML by attempting to initialize `orbslam3.System` with it

### Server Modifications Required

The robot server must support SLAM calibration and operation:

1. **Camera resolution:** Change from 120x120 to 640x480 in `server.py` (line with `camera.create_still_configuration`)
2. **IMU integration:** Add WitMotion IMU data to the server response. The `server.py` response JSON needs an `observation.imu` field with the structure:
   ```json
   {
     "acceleration": {"x": float, "y": float, "z": float},
     "angular_velocity": {"roll_rate": float, "pitch_rate": float, "yaw_rate": float},
     "orientation": {"roll": float, "pitch": float, "yaw": float}
   }
   ```
3. **Raw image exposure:** For calibration, the server must send raw (un-undistorted) fisheye images. The existing `undistort()` should be bypassed or made optional.

These server changes are a prerequisite for both calibration and the SLAM reward wrapper.

**Note on server implementations:** There are two server versions — `server.py` (HTTP, port 8000, no IMU) and the TCP server used by `racer_imu_env.py` (port 9000, with IMU). The calibration scripts target the **TCP server** (port 9000) since it already provides IMU data. The HTTP server (`server.py`) needs the IMU integration and resolution changes for standalone calibration without the training pipeline.

## Implementation Order

1. **Server modifications** — increase camera resolution, integrate IMU, expose raw image
2. **Camera calibration** (`calibration/calibrate_camera.py`) — must be done on the real robot
3. **IMU Allan variance calibration** (`calibration/calibrate_imu.py`) — 2-4 hour static recording
4. **Extrinsic calibration** (`calibration/calibrate_extrinsics.py`) — Kalibr with synchronized data
5. **Generate ORB_SLAM3 YAML** (`calibration/generate_orbslam3_yaml.py`) — assemble all calibrations
6. **Modify `racer_imu_env.py`** — expose raw image and timestamp in info dict
7. **Create `slam_reward_wrapper.py`** — the core wrapper with SLAM thread
8. **Create `train_slam_robot.py`** — training script with standard ReplayBuffer
9. **Test SLAM pipeline standalone** — verify ORB_SLAM3 initializes, tracks, returns poses
10. **Integration test** — run full training loop with SLAM reward