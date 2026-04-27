# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Robotics RL research project for training navigation policies that recover from failures on a physical Donkey Car robot. Uses DrQ (Data-regularized Q-learning) with JAX, MobileNetV3 visual encoders, and various environment wrappers. Supports both simulation (Donkey Car, CARLA, ViZDoom) and real-robot deployment.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_optical_flow_velocity_wrapper.py -v

# Run jaxrl2's own test suite
pytest jaxrl2/tests/ -v
```

## Running Training & Evaluation

All training scripts use `absl.flags` + `app.run(main)`. Example invocations:

```bash
# Real-robot DrQ + HER training
python train_her_robot.py --env_name donkey-warehouse-v0 --port 9091 --max_steps 1000000

# SLAM-based real-robot training (requires ORB_SLAM3)
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/path/to/ORB_SLAM3/lib python train_slam_robot.py \
    --slam_vocab /path/to/ORBvoc.txt \
    --slam_settings calibration/donkeycar_fisheye_imu.yaml

# Simulation HER training
python train_sim_her.py --env_name donkey-warehouse-v0 --sim path/to/simulator

# ViZDoom curriculum training
python train_vizdoom_curriculum.py --max_steps 2000000 --num_stages 4 --init_stage 1

# Offline IQL training
python train_iql.py --teleop_buffer_path teleop_buffer.pkl

# Evaluation
python eval_carla.py --checkpoint_dir ./checkpoints/...
python eval_robot.py --checkpoint_dir ./checkpoints/...

# Config selection (all training scripts)
--config ./configs/drq_default.py   # default
--config ./configs/drq_robot.py     # robot-specific (init_temperature=0.1, num_qs=2)
--config ./configs/iql_default.py   # IQL offline
```

## Architecture

### Data Flow

```
Camera + IMU → RacerEnv → EnvCompatibility → StackingWrapper → MobileNetFeatureWrapper → RewardWrapper/SLAMRewardWrapper → DrQ Learner
```

Observation space is a `Dict` with keys: `pixels` (stacked visual features), `actions` (stacked action history), `imu` (stacked IMU readings). Action space is `Box(2,)` for `[steering, throttle]`.

### Wrapper Stacks

Two primary wrapper stacks exist:

**HER-based** (older, `train_her_robot.py`):
```
RacerEnv → EnvCompatibility → StackingWrapper → RewardWrapper → MobileNetFeatureWrapper → RecordEpisodeStatistics → TimeLimit → GoalRelObservationWrapper
```

**SLAM-based** (newer, `train_slam_robot.py`):
```
RacerEnv → EnvCompatibility → StackingWrapper → MobileNetFeatureWrapper → RecordEpisodeStatistics → TimeLimit → SLAMRewardWrapper
```

### Key Subsystems

- **Environments**: `racer_env.py` (HTTP, port 8000), `racer_imu_env.py` (TCP, port 9000, IMU+undistort+velocity+collision), `carla_env.py`, `vizdoom_env.py`, `gym-donkeycar/`
- **Wrappers**: `wrappers.py` (`StackingWrapper`, `MobileNetFeatureWrapper`, `RewardWrapper`, `EnvCompatibility`, `Logger`), `curriculum_wrappers.py`, `slam_reward_wrapper.py`, `optical_flow_velocity_wrapper.py`
- **RL algorithms**: `jaxrl2/` (vendored fork) — primary is `drq/`, also `iql/`, `sac/`, `ppo/`, `bc/`
- **Navigation baselines**: `navigation_policies/` (vendored ViNT/NoMaD/GNM)
- **Diffusion policy**: `diffusion_policy/` (vendored)
- **Calibration**: `calibration/kalibr/` for camera-IMU extrinsic calibration; `cvar_cam.yaml` at root is the ORB-SLAM3 config with Kalibr-calibrated parameters

### IMU Convention

IMU data is a 6D vector: `[ax, ay, az, gx, gy, gz]` (linear acceleration in m/s², angular velocity in rad/s). StackingWrapper stacks `num_stack` frames of both actions and IMU at 2x the pixel frame stack depth for longer proprioceptive memory.

### Training Script Pattern

All `train_*.py` scripts follow the same structure: absl flags → construct env → build wrapper stack → instantiate DrQ/IQL learner → collect transitions → periodic gradient updates → periodic checkpoint saves. Human-in-the-loop override is available via keyboard (`pygame` keys A/D/W/S).

## Conda Environments

- `real-robot-env` — primary environment for real-robot training
- `pyslam` — environment with ORB_SLAM3 Python bindings
- `ai-agents` — alternative environment (Python 3.10)

## Key Data Files

- `teleop_buffer.pkl` — current human teleoperation replay buffer (~9.6MB)
- `goal_loc_images.pkl` — goal location images for HER (~55MB)
- `online_teleop.pkl` and copies — larger historical buffers (~7GB each, avoid loading casually)
- `topomap/` — 22 PNG frames for topological map building (FAISS SIFT+VLAD)
- `checkpoints/` — saved model checkpoints in Orbax format
- `robot_policy/` — additional saved model checkpoints

## ORB_SLAM3 Integration

See `ORB_SLAM3_INTEGRATION.md` for full details. Key points:
- SLAM runs in a background thread (`SLAMThread` in `slam_reward_wrapper.py`)
- When SLAM tracking is lost for >5 frames, episode terminates with -30 reward penalty
- Graceful degradation: falls back to pseudo-odometry when SLAM is unavailable
- Camera-IMU calibration pipeline: `calibration/calibrate_camera.py` → `calibrate_imu.py` → `calibrate_extrinsics.py` → `calibration/generate_orbslam3_yaml.py`