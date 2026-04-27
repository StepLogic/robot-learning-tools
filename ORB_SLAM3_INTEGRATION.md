# ORB_SLAM3 Integration for RL Training

This document describes the integration of ORB_SLAM3 for accurate velocity and position estimation in reinforcement learning training.

## Overview

The integration replaces the existing pseudo-odometry approach (`dist_step = 0.01`) and Hindsight Experience Replay (HER) with ORB_SLAM3 visual-inertial SLAM, providing accurate metric-scale position estimates for reward calculation.

## Key Benefits

1. **Accurate Position Estimation**: Metric-scale SLAM position instead of pseudo-odometry
2. **Simpler Architecture**: Eliminates HER and goal relabeling infrastructure
3. **Better Training Signal**: More accurate distance-to-goal rewards
4. **Real-world Transfer**: Improved policy performance on physical robot

## Architecture Changes

### Before (HER-based)
```
RacerEnv → EnvCompatibility → StackingWrapper → RewardWrapper → MobileNetFeatureWrapper → RecordEpisodeStatistics → TimeLimit → GoalRelObservationWrapper
```

### After (SLAM-based)
```
RacerEnv → EnvCompatibility → StackingWrapper → MobileNetFeatureWrapper → RecordEpisodeStatistics → TimeLimit → SLAMRewardWrapper
```

## Implementation Components

### 1. Calibration Pipeline (`calibration/`)

#### `calibrate_camera.py`
- Calibrates KannalaBrandt8 fisheye camera model
- Target: RMS reprojection error < 1.0
- Output: `calibration/camera_calib.json`

#### `calibrate_imu.py`
- Performs Allan variance analysis on WitMotion IMU
- Records 2-4 hours of static data
- Extracts noise parameters: `NoiseGyro`, `NoiseAcc`, `GyroWalk`, `AccWalk`
- Output: `calibration/imu_calib.json`

#### `calibrate_extrinsics.py`
- Calibrates camera-IMU extrinsic transform using Kalibr
- Uses AprilTag calibration target
- Output: `calibration/extrinsics_calib.json`

#### `generate_orbslam3_yaml.py`
- Assembles all calibrations into ORB_SLAM3 configuration
- Output: `calibration/donkeycar_fisheye_imu.yaml`

### 2. Environment Modifications (`racer_imu_env.py`)

Added to info dict:
- `raw_image`: Raw un-undistorted fisheye image (640x480, BGR)
- `imu_raw`: Raw IMU data for SLAM accumulation
- `timestamp`: Wall-clock timestamp for synchronization

### 3. SLAM Reward Wrapper (`slam_reward_wrapper.py`)

Key features:
- Background thread for ORB_SLAM3 processing
- Thread-safe state management
- Graceful degradation when SLAM tracking is lost
- SLAM-based reward calculation

#### Reward Structure
```
r = -k_step + k_dist * delta_dist + k_goal * 1[dist < threshold]
```

Where:
- `delta_dist = prev_dist_to_goal - current_dist_to_goal`
- `current_dist_to_goal = ||slam_pos - goal_pos||_2`

#### Default Parameters
- `k_dist = 5.0`: Distance delta reward weight
- `k_goal = 50.0`: Goal completion reward weight  
- `k_step = 0.1`: Per-step time penalty
- `goal_threshold = 0.5`: Goal completion distance (meters)

### 4. Training Script (`train_slam_robot.py`)

Replaces `train_her_robot.py` with:
- Standard `ReplayBuffer` instead of `HindsightReplayBuffer`
- `SLAMRewardWrapper` instead of `GoalRelObservationWrapper`
- Removed `goal_rel` from observation space
- Added SLAM-specific flags and statistics

## Prerequisites

### Hardware Requirements
- Raspberry Pi Camera Module (fisheye lens)
- WitMotion IMU (WT901/WT61 or similar)
- AprilTag calibration target (6x6 recommended)
- Chessboard for camera calibration (8x6, 25mm squares)

### Software Requirements
- ORB_SLAM3 source code with Python bindings
- OpenCV for camera calibration
- Kalibr for extrinsic calibration (Docker recommended)
- PyYAML for configuration

## Setup Instructions

### 1. Camera Calibration
```bash
# Capture calibration images (50+ with chessboard visible)
# Place images in calibration_images/ directory

# Run calibration
python calibration/calibrate_camera.py \
    --images calibration_images/*.jpg \
    --output calibration/camera_calib.json \
    --width 8 --height 6 --square-size 0.025
```

### 2. IMU Calibration
```bash
# Connect WitMotion IMU and run calibration (2-4 hours)
python calibration/calibrate_imu.py \
    --duration 7200 \
    --output calibration/imu_calib.json \
    --raw-data calibration/imu_raw.npy
```

### 3. Extrinsic Calibration
```bash
# Record synchronized camera+IMU data with AprilTag visible
# Move robot slowly to capture all degrees of freedom

# Run Kalibr calibration
python calibration/calibrate_extrinsics.py \
    --camera-calib calibration/camera_calib.json \
    --imu-calib calibration/imu_calib.json \
    --output calibration/extrinsics_calib.json
```

### 4. Generate ORB_SLAM3 Configuration
```bash
python calibration/generate_orbslam3_yaml.py \
    --camera-calib calibration/camera_calib.json \
    --imu-calib calibration/imu_calib.json \
    --extrinsics-calib calibration/extrinsics_calib.json \
    --output calibration/donkeycar_fisheye_imu.yaml
```

### 5. Update Robot Server
Ensure server provides:
- Camera resolution: 640x480 (not 120x120)
- Raw un-undistorted images
- IMU data at ~20Hz with proper timestamps

## Training

### Start SLAM-based Training
```bash
python train_slam_robot.py \
    --slam_vocab /path/to/ORBvoc.txt \
    --slam_settings calibration/donkeycar_fisheye_imu.yaml \
    --k_dist 5.0 --k_goal 50.0 --k_step 0.1 \
    --goal_threshold 0.5
```

### Training Parameters
- `--slam_vocab`: Path to ORB vocabulary file
- `--slam_settings`: Path to ORB_SLAM3 YAML configuration
- `--k_dist`: Weight for distance delta reward (default: 5.0)
- `--k_goal`: Weight for goal completion reward (default: 50.0)
- `--k_step`: Weight for per-step penalty (default: 0.1)
- `--goal_threshold`: Distance threshold for goal completion in meters (default: 0.5)

## Graceful Degradation

The system handles various SLAM states:

| SLAM State | Behavior |
|------------|----------|
| Not initialized (first 10-30 frames) | Fall back to pseudo-odometry |
| Tracking OK | Use SLAM position for distance-to-goal reward |
| Tracking lost >5 frames | Terminate episode with -30 reward penalty |
| SLAM thread crash | Catch exception, fall back to pseudo-odometry, log error |

## Monitoring and Debugging

### SLAM Statistics
The training script logs:
- `slam_ok_count`: Number of frames with successful SLAM tracking
- `slam_lost_count`: Number of frames with lost tracking
- Reprojection errors and tracking quality

### Common Issues

1. **High RMS error in camera calibration**
   - Solution: Capture more diverse images covering entire field of view
   - Target: RMS < 1.0

2. **SLAM tracking lost frequently**
   - Check camera focus and lighting conditions
   - Verify IMU calibration quality
   - Reduce robot speed during critical maneuvers

3. **Performance issues**
   - Ensure ORB_SLAM3 runs in background thread
   - Limit frame rate to 20Hz if needed
   - Reduce ORB features if necessary

## Migration from HER-based Training

### Key Changes
1. **Removed**:
   - `HindsightReplayBuffer` → `ReplayBuffer`
   - `GoalRelObservationWrapper` → `SLAMRewardWrapper`
   - `goal_rel` observation key
   - HER flags: `her_fraction`, `her_strategy`, `goal_range`, `use_goal_masking`, `mask_probability`

2. **Added**:
   - SLAM flags: `slam_vocab`, `slam_settings`, `k_dist`, `k_goal`, `k_step`, `goal_threshold`
   - SLAM state tracking in info dict

3. **Modified**:
   - Wrapper stack order
   - Reward calculation (SLAM-based distance instead of pseudo-odometry)
   - Observation space (no `goal_rel`)

## Performance Optimization

### ORB_SLAM3 Parameters
Adjust in `calibration/donkeycar_fisheye_imu.yaml`:
- `ORBextractor.nFeatures`: 500-1500 (tradeoff between accuracy and performance)
- `ORBextractor.scaleFactor`: 1.1-1.3 (scale pyramid spacing)
- `ORBextractor.nLevels`: 6-8 (scale pyramid levels)

### Training Parameters
- Reduce `batch_size` if memory constrained
- Adjust `k_dist` and `k_goal` for desired reward shaping
- Modify `goal_threshold` based on robot size and environment

## Future Improvements

1. **Visualization**: Add SLAM map visualization during training
2. **Loop Closure**: Better handling of loop closure events in reward calculation
3. **Multi-session**: Support for loading/saving SLAM maps between sessions
4. **Dynamic Parameters**: Adaptive reward parameters based on environment complexity

## References

- ORB_SLAM3: https://github.com/UZ-SLAMLab/ORB_SLAM3
- Kalibr: https://github.com/ethz-asl/kalibr
- OpenCV Camera Calibration: https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html

## Troubleshooting

### "ORB_SLAM3 not available" Error
Ensure ORB_SLAM3 Python bindings are properly installed and the module is in your Python path.

### Poor Tracking Performance
1. Check camera calibration quality (RMS < 1.0)
2. Verify IMU noise parameters
3. Ensure proper camera-IMU synchronization
4. Improve lighting conditions
5. Add more texture to environment

### High Memory Usage
1. Reduce `ORBextractor.nFeatures`
2. Limit replay buffer size
3. Reduce frame stacking depth

## Support

For issues with the integration, check:
1. Calibration quality metrics
2. SLAM thread logs
3. Environment info dict contents
4. Reward component breakdown in logs