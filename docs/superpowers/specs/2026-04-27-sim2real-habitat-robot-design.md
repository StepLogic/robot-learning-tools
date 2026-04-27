# Sim2Real: Habitat Sim Checkpoint to Real Robot Transfer

Date: 2026-04-27

## Problem

Habitat sim-trained DrQ checkpoints cannot be directly used on the real robot because:

1. **IMU dimension mismatch**: Habitat uses 10D IMU `[angular_vel_cmd, linear_vel_cmd, ax, ay, gx, gy, mean_resultant_accel_20, mean_throttle_20, geodesic_distance, distance_mask]`, while the real robot uses 6D `[ax, ay, az, roll_rate, pitch_rate, yaw_rate]`. After stacking 3 frames, the agent's `encoder_imu` input is 30D (Habitat) vs 18D (real), so checkpoint loading fails.

2. **No evaluation script**: There is no script to deploy a Habitat checkpoint on the real robot for inference-only evaluation.

3. **Training script mismatch**: `train_image_goal.py` has `--pretrained_checkpoint` support but its wrapper stack doesn't produce 10D IMU, so Habitat checkpoints can't be loaded.

## Design

### 1. `Sim2RealIMUWrapper` — Observation space adapter

A gym wrapper that maps the real robot's 6D IMU to the Habitat 10D format. Inserted between `RacerEnv` and `StackingWrapper`.

**IMU mapping table:**

| Index | Habitat field | Real robot source |
|-------|--------------|-------------------|
| 0 | angular_vel_cmd | Last steering action |
| 1 | linear_vel_cmd | Last throttle action |
| 2 | ax (forward accel) | Real `ax` |
| 3 | ay (lateral accel) | Real `ay` |
| 4 | gx (angular vel x) | Real `roll_rate` |
| 5 | gy (angular vel y) | Real `pitch_rate` |
| 6 | mean_resultant_accel_20 | Rolling mean of `sqrt(ax²+ay²+az²)` over 20 steps |
| 7 | mean_throttle_20 | Rolling mean of throttle over 20 steps |
| 8 | geodesic_distance | -1.0 (unknown on real robot) |
| 9 | distance_mask | 1.0 (True — goal features always visible) |

**Key behavior:**
- On `reset()`: clears rolling histories, sets last action to zeros, sets geodesic_distance=-1.0 and distance_mask=1.0
- On `step(action)`: stores `action` as the last command (used for imu[0:2] on the NEXT step), updates rolling means, maps real IMU to 10D
- Observation space: changes `imu` from `Box(6,)` to `Box(10,)`, all other keys unchanged

**File:** `sim2real_wrappers.py` (new file — keeps sim2real logic separate from core wrappers)

### 2. `eval_habitat_robot.py` — Deploy sim checkpoint for inference

Evaluation script that loads a Habitat-trained checkpoint and runs it on the real robot.

**Wrapper stack** (mirrors Habitat training):
```
RacerEnv → Sim2RealIMUWrapper → StackingWrapper(image_format="bgr") → MobileNetFeatureWrapper → GoalImageWrapper
```

**Features:**
- Loads Habitat checkpoint via `--checkpoint_path`
- Uses same `num_blocks` and `input_size` as Habitat training (default 13 and 84)
- Human-in-the-Loop keyboard control for override (same as `train_image_goal.py`)
- Goal image pool loaded from `robot_policy/goal_image_pool.pkl`
- Prints per-step diagnostics: feature distance to goal, velocity, collision state
- Saves evaluation metrics (episode returns, success rate, distances)

**Absl flags:** `--checkpoint_path`, `--host`, `--port`, `--ws_port`, `--num_episodes`, `--max_episode_steps`, `--mobilenet_blocks`, `--mobilenet_input_size`, `--frame_stack`, `--goal_feature_threshold`, `--enable_hitl`, `--goal_pool_path`

### 3. `train_image_goal.py` improvements

**3a. Add `Sim2RealIMUWrapper` to the wrapper stack when using sim checkpoints:**

When `--pretrained_checkpoint` or `--checkpoint_path` points to a Habitat checkpoint, insert `Sim2RealIMUWrapper` between `RacerEnv` and `StackingWrapper`. This is controlled by a new flag `--sim2real_imu` (default True when a checkpoint is provided, False otherwise).

**3b. Fix `sim2real_a` augmentation at inference time:**

`MobileNetV3Encoder.encode()` applies `sim2real_a` transforms (RandomResizedCrop, GaussianBlur, RandomRedLightFilter, ColorJitter) even during inference because the transform is bound in `__init__`. Add a `deterministic` flag to `MobileNetV3Encoder` that bypasses stochastic transforms and uses only deterministic preprocessing (resize + normalize) during inference.

**3c. Better checkpoint loading diagnostics:**

When loading a Habitat checkpoint, print the observation/action space shapes to verify compatibility before starting training.

### 4. GoalImageWrapper IMU masking fix

`GoalImageWrapper.step()` at `wrappers.py:445` checks `if obs["imu"][5] < 0.0` to zero out goal features. In Habitat's 10D IMU format, index 5 is `gy` (angular velocity y), which can legitimately be negative — causing spurious goal feature zeroing. The intent is to use the `distance_mask` field (index 9) which indicates whether the geodesic distance is masked.

**Habitat distance_mask semantics:** `imu[9] = 1.0` means distance IS masked (goal features should be hidden); `imu[9] = 0.0` means distance IS available (goal features visible). With `Sim2RealIMUWrapper`, `imu[9] = 1.0` always (distance always unknown on real robot, but we want goal features visible anyway — this is a design choice that the agent should always see the goal on the real robot).

**Fix:** Change `obs["imu"][5] < 0.0` to `obs["imu"].shape[0] >= 10 and obs["imu"][-1] > 0.5`. Only applies the mask when IMU is 10D (Habitat/Sim2Real format) and the mask flag is True. For 6D real robot (without Sim2RealIMUWrapper), goal features are never masked.

## Files Changed

| File | Changes |
|------|---------|
| `sim2real_wrappers.py` | NEW — `Sim2RealIMUWrapper` class |
| `eval_habitat_robot.py` | NEW — evaluation/deployment script |
| `train_image_goal.py` | Add `Sim2RealIMUWrapper` when loading sim checkpoints, add `--sim2real_imu` flag |
| `wrappers.py` | Fix `GoalImageWrapper.step()` IMU mask check; add `deterministic` flag to `MobileNetV3Encoder` |

## Out of Scope

- Changes to Habitat training script (`train_habitat_her.py`)
- Changes to RacerEnv or the TCP protocol
- Changes to the JAX DrQ agent architecture
- Converting MobileNetV3 from PyTorch to JAX (the current PyTorch-side encoding approach is retained)