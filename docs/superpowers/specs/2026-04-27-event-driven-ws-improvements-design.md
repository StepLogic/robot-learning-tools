# Event-Driven WebSocket Improvements

Date: 2026-04-27

## Problem

The robot server's WebSocket events have three gaps:

1. **IMU events are acceleration-only** — `_imu_callback` only publishes `imu_sample` on `AccelerationMessage`. Gyro, orientation, and magnetic messages update `imu_data` but don't fire a WS event. Clients that need paired accel+gyro (e.g., ORB-SLAM3 preintegration) only get the acceleration rate, missing inter-frame gyro updates.

2. **Obstacle events are transition-only** — `collision_event` only fires when the detected state changes (cleared→detected or detected→cleared). Clients get no continuous proximity data between transitions, preventing proactive braking.

3. **Exception swallowing** — `_imu_callback` uses `except Exception: pass`, silently dropping all errors with no visibility.

## Design

### 1. Batched IMU snapshots on every message type

**Server (`robot_server.py`)**: Modify `_imu_callback` so every message type (acceleration, angular velocity, orientation, magnetic) publishes a full `imu_sample` via WebSocket. Each event carries the complete `imu_data` snapshot via `get_imu_snapshot()`, so acceleration and gyro are always present in each message.

The `should_publish` flag is removed. Every callback invocation publishes.

This means when an acceleration message arrives, the event includes the latest gyro from a previous `AngularVelocityMessage`. When a gyro message arrives, the event includes the latest acceleration. Clients always get paired accel+gyro in every `imu_sample`.

**No client changes required** — the `imu_sample` event structure is identical (full snapshot dict). `drain_imu_buffer()` now returns more frequent, already-paired samples, which is exactly what ORB-SLAM3 preintegration needs.

### 2. Periodic `obstacle_update` events

**Server (`robot_server.py`)**: Every obstacle poll cycle (10Hz) publishes an `obstacle_update` event containing the current distance and detected state, regardless of whether the state changed. This provides continuous proximity data.

```json
{"type": "obstacle_update", "ts": 1234567890.123, "data": {"detected": false, "distance_cm": 45.3, "threshold_cm": 15.0}}
```

The existing `collision_event` (transition-based) is retained — it's useful for immediate "something changed" notification. `obstacle_update` is the continuous stream.

**Client (`robot_event_client.py`)**: Add `_obstacle_state` dict, `obstacle_state` property, and `on_obstacle` callback. Handle `obstacle_update` messages in `_handle_message`.

**StubEventClient**: Add matching no-op attributes.

**Consumers**:
- `racer_imu_env.py`: Expose `obstacle_state_event` in `_get_info()` for continuous proximity data.
- `test_orbslam_robot.py`: Can use `event_client.obstacle_state` for proactive braking before collision threshold.

### 3. Fix exception swallowing

**Server (`robot_server.py`)**: Replace `except Exception: pass` in `_imu_callback` with `except Exception as e: print(f"IMU callback error: {e}")`. Errors are now visible in server logs.

## Files Changed

| File | Changes |
|------|---------|
| `robot_server.py` | Remove `should_publish`, publish on all IMU message types, add `obstacle_update` broadcasts, fix exception handling |
| `robot_event_client.py` | Add `obstacle_update` handling, `obstacle_state` property, `on_obstacle` callback |
| `racer_imu_env.py` | Add `obstacle_state_event` to `_get_info()` info dict |
| `test_orbslam_robot.py` | Add optional proactive braking using `event_client.obstacle_state` |

## Data Flow (After Changes)

```
WitMotion IMU ──callback──> _imu_callback ──publish──> ws_publish("imu_sample", full_snapshot)
                                                      (on EVERY message type: accel, gyro, orientation, magnetic)

HC-SR04 poll (10Hz) ──> _obstacle_state update ──> ws_publish("obstacle_update", {detected, distance_cm, threshold_cm})
                                               ──> ws_publish("collision_event", {...})  (on state transition only)

Velocity estimator (20Hz) ──> ws_publish("velocity_update", {...})

Client ──ws──> RobotEventClient
                  .drain_imu_buffer()    → paired accel+gyro at ~200Hz
                  .collision_detected    → transition events (~10ms latency)
                  .obstacle_state        → continuous proximity at 10Hz
                  .velocity              → velocity at 20Hz
```

## Out of Scope

- GPIO interrupt-driven ultrasonic echo detection (10Hz polling is sufficient per user requirement)
- Changes to TCP protocol or response format
- Changes to velocity estimator
- Changes to `test_orbslam_robot.py`'s ORB-SLAM3 integration (it already uses `drain_imu_buffer()` correctly)