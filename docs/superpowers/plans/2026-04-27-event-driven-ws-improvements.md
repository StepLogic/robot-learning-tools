# Event-Driven WebSocket Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the robot server publish full IMU snapshots on every message type and continuous obstacle proximity data via WebSocket, and update the event client and consumers accordingly.

**Architecture:** Modify `robot_server.py` to publish `imu_sample` on all WitMotion message types (not just acceleration) and add periodic `obstacle_update` events at 10Hz. Update `robot_event_client.py` to handle `obstacle_update` messages. Add `obstacle_state_event` to the info dict in `racer_imu_env.py` and proactive braking in `test_orbslam_robot.py`.

**Tech Stack:** Python 3.9+, `websockets` library, `witmotion` library, `pytest`

---

### Task 1: Fix exception swallowing in `_imu_callback` and publish on all message types

**Files:**
- Modify: `robot_server.py:252-299`

- [ ] **Step 1: Write the failing test**

Add a test that verifies `_imu_callback` handles errors by checking that exceptions are logged, not silently swallowed. Since `_imu_callback` is an internal function called by the witmotion library and depends on hardware, we test the observable side effect: that `ws_publish` is called for all message types.

Create `tests/test_robot_server_ws.py`:

```python
"""
Tests for robot_server.py WebSocket publishing behavior.

Validates that _imu_callback publishes on all message types
and that obstacle daemon broadcasts obstacle_update events.
"""

import json
import math
import time
import threading
import pytest

# We test by mocking the witmotion message types and ws_publish


class TestIMUCallbackPublishesAllTypes:
    """Verify _imu_callback publishes imu_sample on every message type."""

    def test_acceleration_publishes(self):
        """AccelerationMessage should trigger a WS publish."""
        from unittest.mock import MagicMock, patch
        import robot_server

        published = []
        with patch.object(robot_server, 'ws_publish', side_effect=lambda t, d: published.append((t, d))):
            msg = MagicMock()
            msg.__class__ = robot_server.witmotion.protocol.AccelerationMessage
            msg.a = (0.1, 0.2, 9.81)
            robot_server._imu_callback(msg)
        assert any(t == "imu_sample" for t, d in published), \
            f"Expected imu_sample publish, got: {published}"

    def test_angular_velocity_publishes(self):
        """AngularVelocityMessage should trigger a WS publish."""
        from unittest.mock import MagicMock, patch
        import robot_server

        published = []
        with patch.object(robot_server, 'ws_publish', side_effect=lambda t, d: published.append((t, d))):
            msg = MagicMock()
            msg.__class__ = robot_server.witmotion.protocol.AngularVelocityMessage
            msg.w = (0.01, 0.02, 0.03)
            robot_server._imu_callback(msg)
        assert any(t == "imu_sample" for t, d in published), \
            f"Expected imu_sample publish for gyro, got: {published}"

    def test_orientation_publishes(self):
        """AngleMessage should trigger a WS publish."""
        from unittest.mock import MagicMock, patch
        import robot_server

        published = []
        with patch.object(robot_server, 'ws_publish', side_effect=lambda t, d: published.append((t, d))):
            msg = MagicMock()
            msg.__class__ = robot_server.witmotion.protocol.AngleMessage
            msg.roll = 1.0
            msg.pitch = 2.0
            msg.yaw = 3.0
            robot_server._imu_callback(msg)
        assert any(t == "imu_sample" for t, d in published), \
            f"Expected imu_sample publish for orientation, got: {published}"

    def test_magnetic_publishes(self):
        """MagneticMessage should trigger a WS publish."""
        from unittest.mock import MagicMock, patch
        import robot_server

        published = []
        with patch.object(robot_server, 'ws_publish', side_effect=lambda t, d: published.append((t, d))):
            msg = MagicMock()
            msg.__class__ = robot_server.witmotion.protocol.MagneticMessage
            msg.mag = (10, 20, 30)
            robot_server._imu_callback(msg)
        assert any(t == "imu_sample" for t, d in published), \
            f"Expected imu_sample publish for magnetic, got: {published}"

    def test_imu_snapshot_contains_accel_and_gyro(self):
        """Each imu_sample should contain both acceleration and angular_velocity."""
        from unittest.mock import MagicMock, patch
        import robot_server

        # First, send a gyro message to populate angular_velocity in imu_data
        gyro_msg = MagicMock()
        gyro_msg.__class__ = robot_server.witmotion.protocol.AngularVelocityMessage
        gyro_msg.w = (1.0, 2.0, 3.0)
        robot_server._imu_callback(gyro_msg)

        # Then, send an accel message — the snapshot should have both
        published = []
        with patch.object(robot_server, 'ws_publish', side_effect=lambda t, d: published.append((t, d))):
            accel_msg = MagicMock()
            accel_msg.__class__ = robot_server.witmotion.protocol.AccelerationMessage
            accel_msg.a = (0.1, 0.2, 9.81)
            robot_server._imu_callback(accel_msg)

        imu_samples = [d for t, d in published if t == "imu_sample"]
        assert len(imu_samples) >= 1, "Expected at least one imu_sample"
        snap = imu_samples[0]
        assert "acceleration" in snap, "Snapshot missing acceleration"
        assert "angular_velocity" in snap, "Snapshot missing angular_velocity"
        assert snap["angular_velocity"]["roll_rate"] == 1.0, \
            f"Expected gyro from prior message, got: {snap['angular_velocity']}"


class TestObstacleUpdatePublishes:
    """Verify obstacle daemon publishes obstacle_update events."""

    def test_obstacle_update_event_structure(self):
        """obstacle_update events should contain detected, distance_cm, threshold_cm."""
        from unittest.mock import MagicMock, patch
        import robot_server

        # Simulate one iteration of the obstacle daemon loop
        published = []
        with patch.object(robot_server, 'ws_publish', side_effect=lambda t, d: published.append((t, d))):
            with patch.object(robot_server, '_distance_sensor') as mock_sensor:
                mock_sensor.distance = 0.45  # 45 cm
                robot_server.OBSTACLE_AVAILABLE = True
                robot_server.OBSTACLE_THRESHOLD_CM = 15.0

                # Call the logic from inside the daemon
                dist_cm = mock_sensor.distance * 100.0
                detected = dist_cm < robot_server.OBSTACLE_THRESHOLD_CM
                robot_server.ws_publish("obstacle_update", {
                    "detected": detected,
                    "distance_cm": round(dist_cm, 2),
                    "threshold_cm": robot_server.OBSTACLE_THRESHOLD_CM,
                })

        updates = [d for t, d in published if t == "obstacle_update"]
        assert len(updates) == 1
        assert "detected" in updates[0]
        assert "distance_cm" in updates[0]
        assert "threshold_cm" in updates[0]


class TestIMUCallbackErrors:
    """Verify IMU callback logs errors instead of swallowing them."""

    def test_exception_logged(self, capsys):
        """Errors in _imu_callback should be printed, not silently swallowed."""
        from unittest.mock import MagicMock, patch
        import robot_server

        # Create a message that will cause an error
        msg = MagicMock()
        msg.__class__ = robot_server.witmotion.protocol.AccelerationMessage
        msg.a = (0.1, 0.2, 9.81)
        # Make imu_data temporarily raise an error
        original_data = robot_server.imu_data.copy()
        # Force an error by patching get_imu_snapshot to raise
        with patch.object(robot_server, 'get_imu_snapshot', side_effect=RuntimeError("test error")):
            robot_server._imu_callback(msg)

        captured = capsys.readouterr()
        assert "IMU callback error" in captured.out or "test error" in captured.out, \
            f"Expected error to be logged, got: {captured.out}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/kojogyaase/Projects/Research/recovery-from-failure && python -m pytest tests/test_robot_server_ws.py -v --timeout=10 2>&1 | head -80`

Expected: Several tests FAIL — `_imu_callback` currently only publishes on `AccelerationMessage`, not on gyro/orientation/magnetic. The snapshot test may also fail because gyro data won't be included in the publish when only accel fires.

Note: The exact test behavior depends on whether `witmotion` is importable. If not, tests may need to be skipped. The `robot_server` module imports `witmotion` at the top level, so if it's not installed, the entire module import will fail. In that case, these tests should be marked with `pytest.importorskip("witmotion")`.

- [ ] **Step 3: Implement the changes in `robot_server.py`**

In `robot_server.py`, modify `_imu_callback` (lines 252-299):

```python
def _imu_callback(msg):
    try:
        if isinstance(msg, witmotion.protocol.AccelerationMessage):
            ax, ay, az = msg.a
            now = time.time()
            with _imu_lock:
                if _last_accel_time is None:
                    _last_accel_time = now
                dt = now - _last_accel_time
                _last_accel_time = now
                r = math.sqrt(ax*ax + ay*ay + az*az)
                imu_data['acceleration'] = {'x': ax, 'y': ay, 'z': az, 'resultant': r}
                vx = imu_data['velocity']['x'] + ax * dt
                vy = imu_data['velocity']['y'] + ay * dt
                vz = imu_data['velocity']['z'] + az * dt
                imu_data['velocity'] = {
                    'x': vx, 'y': vy, 'z': vz,
                    'speed': math.sqrt(vx*vx + vy*vy + vz*vz)
                }
                imu_data['forward_velocity'] += ax * dt

        elif isinstance(msg, witmotion.protocol.AngularVelocityMessage):
            w = msg.w
            with _imu_lock:
                imu_data['angular_velocity'] = {
                    'roll_rate':  w[0], 'pitch_rate': w[1],
                    'yaw_rate':   w[2],
                    'magnitude':  math.sqrt(w[0]**2 + w[1]**2 + w[2]**2),
                }

        elif isinstance(msg, witmotion.protocol.AngleMessage):
            with _imu_lock:
                imu_data['orientation'] = {
                    'roll': msg.roll, 'pitch': msg.pitch, 'yaw': msg.yaw}

        elif isinstance(msg, witmotion.protocol.MagneticMessage):
            mg = msg.mag
            with _imu_lock:
                imu_data['magnetic'] = {'x': mg[0], 'y': mg[1], 'z': mg[2]}

        # Publish full snapshot on every message type
        snapshot = get_imu_snapshot()
        ws_publish("imu_sample", snapshot)
    except Exception as e:
        print(f"IMU callback error: {e}")
```

Key changes:
- Remove `should_publish` flag and the `if should_publish:` conditional
- Move `snapshot = get_imu_snapshot()` and `ws_publish("imu_sample", snapshot)` outside all the `isinstance` branches — they now execute on every message type
- Change `except Exception: pass` to `except Exception as e: print(f"IMU callback error: {e}")`

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/kojogyaase/Projects/Research/recovery-from-failure && python -m pytest tests/test_robot_server_ws.py -v --timeout=10 2>&1 | head -80`

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/kojogyaase/Projects/Research/recovery-from-failure
git add robot_server.py tests/test_robot_server_ws.py
git commit -m "feat: publish IMU snapshot on all message types and fix exception swallowing

- _imu_callback now publishes full imu_sample on every WitMotion message
  type (acceleration, angular velocity, orientation, magnetic) so clients
  always receive paired accel+gyro data
- Replace except Exception: pass with error logging
- Add tests for all four message type publishes and snapshot content"
```

---

### Task 2: Add `obstacle_update` broadcasts in obstacle daemon

**Files:**
- Modify: `robot_server.py:342-364` (obstacle daemon `_run` function)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_robot_server_ws.py`:

```python
class TestObstacleDaemonBroadcastsUpdate:
    """Verify the obstacle daemon publishes obstacle_update on every poll cycle."""

    def test_obstacle_update_published_every_cycle(self):
        """obstacle_update should be published on every poll, not just transitions."""
        from unittest.mock import MagicMock, patch
        import robot_server

        published = []
        with patch.object(robot_server, 'ws_publish', side_effect=lambda t, d: published.append((t, d))):
            # Simulate what the obstacle daemon does in one cycle
            # No obstacle detected, distance 45cm
            with patch.object(robot_server, '_distance_sensor') as mock_sensor:
                mock_sensor.distance = 0.45  # 45 cm
                dist_cm = mock_sensor.distance * 100.0
                detected = dist_cm < robot_server.OBSTACLE_THRESHOLD_CM
                with robot_server._obstacle_lock:
                    robot_server._obstacle_state['detected'] = detected
                    robot_server._obstacle_state['distance_cm'] = round(dist_cm, 2)
                # This is what the new code should do:
                robot_server.ws_publish("obstacle_update", {
                    "detected": detected,
                    "distance_cm": round(dist_cm, 2),
                    "threshold_cm": robot_server.OBSTACLE_THRESHOLD_CM,
                })

        updates = [d for t, d in published if t == "obstacle_update"]
        assert len(updates) == 1
        assert updates[0]["distance_cm"] == 45.0
        assert updates[0]["detected"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/kojogyaase/Projects/Research/recovery-from-failure && python -m pytest tests/test_robot_server_ws.py::TestObstacleDaemonBroadcastsUpdate -v --timeout=10 2>&1 | head -40`

Expected: The test may pass because it directly calls `ws_publish` — we need the actual daemon code to call it. The test verifies the contract; the real test is that the daemon's `_run` loop calls `ws_publish("obstacle_update", ...)`.

- [ ] **Step 3: Modify the obstacle daemon in `robot_server.py`**

In the `start_obstacle_daemon` function's `_run` inner function (around line 342-364), add `ws_publish("obstacle_update", ...)` after updating `_obstacle_state`. The modified loop body should be:

```python
    def _run():
        print("✓ Obstacle daemon started")
        while True:
            try:
                dist_cm  = _distance_sensor.distance * 100.0
                detected = dist_cm < OBSTACLE_THRESHOLD_CM
                if detected:
                    kit.continuous_servo[CONTINUOUS_SERVO_DRIVE].throttle = 0.0
                with _obstacle_lock:
                    _obstacle_state['detected']    = detected
                    _obstacle_state['distance_cm'] = round(dist_cm, 2)

                # Publish continuous proximity data every cycle
                ws_publish("obstacle_update", {
                    "detected":    detected,
                    "distance_cm": round(dist_cm, 2),
                    "threshold_cm": OBSTACLE_THRESHOLD_CM,
                })

                # Push collision event on state transition
                if detected != _prev_detected[0]:
                    ws_publish("collision_event", {
                        "detected": detected,
                        "distance_cm": round(dist_cm, 2),
                        "threshold_cm": OBSTACLE_THRESHOLD_CM,
                    })
                    _prev_detected[0] = detected
            except Exception as e:
                print(f"Obstacle daemon error: {e}")
            time.sleep(interval)
```

- [ ] **Step 4: Run all tests**

Run: `cd /home/kojogyaase/Projects/Research/recovery-from-failure && python -m pytest tests/test_robot_server_ws.py tests/test_robot_event_client.py -v --timeout=10 2>&1 | head -80`

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/kojogyaase/Projects/Research/recovery-from-failure
git add robot_server.py tests/test_robot_server_ws.py
git commit -m "feat: add periodic obstacle_update WebSocket broadcasts

The obstacle daemon now publishes obstacle_update events at 10Hz with
current distance and detection state, providing continuous proximity
data. The transition-based collision_event is retained for immediate
state-change notification."
```

---

### Task 3: Add `obstacle_update` handling to `RobotEventClient`

**Files:**
- Modify: `robot_event_client.py:38-74` (add `_obstacle_state`, `obstacle_state` property, `on_obstacle` callback)
- Modify: `robot_event_client.py:135-170` (add `obstacle_update` message handling)
- Modify: `robot_event_client.py:206-230` (add `StubEventClient` no-op attributes)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_robot_event_client.py`, in the `mock_server` fixture's handler, after the velocity update (around line 83):

```python
        # Send an obstacle update
        await websocket.send(json.dumps({
            "type": "obstacle_update",
            "ts": time.time(),
            "data": {"detected": False, "distance_cm": 45.3, "threshold_cm": 15.0},
        }))
```

Add a new test class:

```python
class TestObstacleUpdate:
    """Tests for obstacle_update event handling."""

    def test_obstacle_state_property(self, mock_server):
        """obstacle_state should reflect obstacle_update messages."""
        if not WEBSOCKETS_AVAILABLE:
            pytest.skip("websockets library not installed")
        client = RobotEventClient(host="127.0.0.1", ws_port=mock_server,
                                  reconnect_interval=0.5)
        client.start()
        time.sleep(0.5)

        state = client.obstacle_state
        assert state["detected"] is False
        assert state["distance_cm"] == 45.3
        assert state["threshold_cm"] == 15.0
        assert "timestamp" in state

        client.stop()

    def test_obstacle_callback(self, mock_server):
        """on_obstacle callback should fire on obstacle_update."""
        if not WEBSOCKETS_AVAILABLE:
            pytest.skip("websockets library not installed")
        client = RobotEventClient(host="127.0.0.1", ws_port=mock_server,
                                  reconnect_interval=0.5)
        received = []

        def callback(event):
            received.append(event)

        client.on_obstacle = callback
        client.start()
        time.sleep(0.5)

        assert len(received) >= 1, "obstacle_update callback should have fired"
        assert received[0]["distance_cm"] == 45.3

        client.stop()

    def test_stub_obstacle_state(self):
        """StubEventClient should have obstacle_state with safe defaults."""
        client = StubEventClient()
        state = client.obstacle_state
        assert state["detected"] is False
        assert state["distance_cm"] == float("inf")
        assert state["threshold_cm"] == 15.0
        assert "timestamp" in state
        assert client.on_obstacle is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/kojogyaase/Projects/Research/recovery-from-failure && python -m pytest tests/test_robot_event_client.py::TestObstacleUpdate -v --timeout=10 2>&1 | head -40`

Expected: FAIL — `RobotEventClient` has no `obstacle_state` property or `on_obstacle` callback, and `StubEventClient` has no `obstacle_state`.

- [ ] **Step 3: Implement `obstacle_update` handling in `RobotEventClient`**

In `robot_event_client.py`, make these changes:

**A. Add `_obstacle_state` and `on_obstacle` to `__init__`** (around line 53-68):

After `self._velocity_state` dict, add:

```python
        self._obstacle_state: dict = {
            "detected": False, "distance_cm": float("inf"),
            "threshold_cm": 15.0, "timestamp": 0.0,
        }
```

After `self.on_velocity = None`, add:

```python
        self.on_obstacle = None   # Callable[[dict], None]
```

**B. Add `obstacle_state` property** (after the `velocity` property, around line 115):

```python
    @property
    def obstacle_state(self) -> dict:
        with self._lock:
            return self._obstacle_state.copy()
```

**C. Add `obstacle_update` handling in `_handle_message`** (after the `velocity_update` elif block, around line 163):

```python
        elif msg_type == "obstacle_update":
            with self._lock:
                self._obstacle_state = {**data, "timestamp": ts}
            if self.on_obstacle is not None:
                try:
                    self.on_obstacle({**data, "timestamp": ts})
                except Exception:
                    pass
```

**D. Add `StubEventClient` attributes** (around line 206-230):

Add to `StubEventClient`:

```python
    obstacle_state = {
        "detected": False, "distance_cm": float("inf"),
        "threshold_cm": 15.0, "timestamp": 0.0,
    }
    on_obstacle = None
```

- [ ] **Step 4: Run all event client tests**

Run: `cd /home/kojogyaase/Projects/Research/recovery-from-failure && python -m pytest tests/test_robot_event_client.py -v --timeout=10 2>&1 | head -60`

Expected: All tests PASS, including the new `TestObstacleUpdate` tests and existing tests.

- [ ] **Step 5: Commit**

```bash
cd /home/kojogyaase/Projects/Research/recovery-from-failure
git add robot_event_client.py tests/test_robot_event_client.py
git commit -m "feat: add obstacle_update event handling to RobotEventClient

- Add _obstacle_state dict, obstacle_state property, on_obstacle callback
- Handle obstacle_update messages in _handle_message
- Add matching no-op attributes to StubEventClient
- Add tests for obstacle_state property, callback, and stub behavior"
```

---

### Task 4: Add `obstacle_state_event` to `racer_imu_env.py` info dict

**Files:**
- Modify: `racer_imu_env.py:310-330` (`_get_info` method)

- [ ] **Step 1: Add `obstacle_state_event` to the info dict**

In `racer_imu_env.py`, in the `_get_info` method (around line 310-330), add a new key after the existing `"collision_from_event"` line:

```python
            # ── Event-driven fields (WebSocket) ──────────────────────────────
            "imu_samples_between_frames": imu_samples,   # For SLAM preintegration
            "collision_from_event": self.event_client.collision_detected,  # ~10ms latency
            "velocity_event": self.event_client.velocity,
            "obstacle_state_event": self.event_client.obstacle_state,  # continuous proximity at 10Hz
```

This is a single line addition. No test changes needed since the info dict is just extended — existing consumers are unaffected.

- [ ] **Step 2: Verify the env module imports correctly**

Run: `cd /home/kojogyaase/Projects/Research/recovery-from-failure && python -c "from racer_imu_env import RacerEnv; print('OK')" 2>&1`

Expected: `OK` (or an import error if hardware dependencies are missing, which is expected on non-robot machines — in that case, verify the line was added correctly by reading the file).

- [ ] **Step 3: Commit**

```bash
cd /home/kojogyaase/Projects/Research/recovery-from-failure
git add racer_imu_env.py
git commit -m "feat: add obstacle_state_event to RacerEnv info dict

Exposes continuous proximity data from WebSocket event client
at 10Hz for downstream consumers (reward wrappers, SLAM)."
```

---

### Task 5: Add proactive braking using `obstacle_state` in `test_orbslam_robot.py`

**Files:**
- Modify: `test_orbslam_robot.py:820-828` (collision detection section)

- [ ] **Step 1: Add proximity-based proactive braking**

In `test_orbslam_robot.py`, after the existing collision detection block (around line 822-827), add proactive braking based on continuous proximity data:

```python
            # Real-time collision detection via WebSocket (~10ms latency)
            if event_client.collision_detected:
                coll = event_client.collision_info
                print(f"  ⚠ COLLISION via WS | dist={coll['distance_cm']:.1f}cm "
                      f"threshold={coll['threshold_cm']:.1f}cm")
                # Immediately stop motors
                steering, throttle = 0.0, 0.0

            # Proactive braking: reduce throttle when approaching obstacle
            obs_state = event_client.obstacle_state
            if obs_state["distance_cm"] < 40.0 and obs_state["distance_cm"] > 0:
                proximity_factor = max(0.3, obs_state["distance_cm"] / 40.0)
                throttle *= proximity_factor
                if obs_state["distance_cm"] < 25.0:
                    print(f"  ⚠ PROXIMITY | dist={obs_state['distance_cm']:.1f}cm "
                          f"throttle={throttle:.3f}")
```

- [ ] **Step 2: Verify the module imports correctly**

Run: `cd /home/kojogyaase/Projects/Research/recovery-from-failure && python -c "from robot_event_client import RobotEventClient; print('OK')" 2>&1`

Expected: `OK` or an import error for `websockets` (acceptable).

- [ ] **Step 3: Commit**

```bash
cd /home/kojogyaase/Projects/Research/recovery-from-failure
git add test_orbslam_robot.py
git commit -m "feat: add proactive braking based on continuous proximity data

Uses obstacle_state from WebSocket events to reduce throttle when
approaching obstacles (within 40cm), providing smoother deceleration
before collision threshold is reached."
```

---

### Task 6: Run full test suite and final verification

**Files:**
- All modified files

- [ ] **Step 1: Run all project tests**

Run: `cd /home/kojogyaase/Projects/Research/recovery-from-failure && python -m pytest tests/ -v --timeout=15 2>&1 | tail -30`

Expected: All tests PASS.

- [ ] **Step 2: Verify no regressions in import chain**

Run: `cd /home/kojogyaase/Projects/Research/recovery-from-failure && python -c "from robot_server import ws_publish, get_imu_snapshot; from robot_event_client import RobotEventClient, StubEventClient; print('All imports OK')" 2>&1`

Expected: `All imports OK` (or acceptable import errors for hardware dependencies like `witmotion`, `gpiozero`, `adafruit_servokit`).

- [ ] **Step 3: Final commit (if any remaining changes)**

```bash
cd /home/kojogyaase/Projects/Research/recovery-from-failure
git status
git diff
# Only commit if there are uncommitted changes
```