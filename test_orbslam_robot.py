#!/usr/bin/env python3
"""
ORB-SLAM3 Monocular-Inertial Test with Real Robot (TCP)

Connects to the robot via TCP, receives images and IMU data,
and runs ORB-SLAM3 for visual-inertial SLAM.

Keyboard controls (while OpenCV window is focused):
    W/S     Throttle forward/backward
    A/D     Steer left/right
    SPACE   Stop (zero controls)
    Q       Quit
    R       Reset SLAM
    T       Save trajectory

Usage:
    python test_orbslam_robot.py
    python test_orbslam_robot.py --no-imu
    python test_orbslam_robot.py --host 10.42.0.1 --port 9000
"""

import sys
import os
import time
import argparse
import base64
import json
import socket
import struct
import numpy as np
import cv2
from collections import deque
from scipy.spatial.transform import Rotation as R

from robot_event_client import RobotEventClient

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_ORBSLAM_LIB = "/media/kojogyaase/disk_two/Research/ORB_SLAM3/python"
DEFAULT_VOCAB_PATH = "/media/kojogyaase/disk_two/Research/ORB_SLAM3/Vocabulary/ORBvoc.txt"
DEFAULT_SETTINGS_PATH = "cvar_cam.yaml"
DEFAULT_HOST = "10.42.0.1"
DEFAULT_PORT = 9000

# Timeshift from Kalibr: camera is 0.604s ahead of IMU
TIMESHIFT_CAM_IMU = 0.6039

# WitMotion sensor outputs gyro in deg/s — must convert to rad/s
DEG_TO_RAD = np.pi / 180.0


def parse_args():
    parser = argparse.ArgumentParser(description="ORB-SLAM3 with real robot via TCP")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--viewer", action="store_true", help="Show ORB-SLAM3 viewer")
    parser.add_argument("--no-imu", action="store_true", help="Disable IMU (pure monocular)")
    parser.add_argument("--vocab", type=str, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--settings", type=str, default=DEFAULT_SETTINGS_PATH)
    parser.add_argument("--orbslam-lib", type=str, default=DEFAULT_ORBSLAM_LIB)
    parser.add_argument("--max-steps", type=int, default=0, help="0 = unlimited")
    parser.add_argument("--max-init-steps", type=int, default=300)
    parser.add_argument("--save-trajectory", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Log IMU data and diagnostics")
    parser.add_argument("--ws-port", type=int, default=9001, help="WebSocket event port")
    return parser.parse_args()


# ============================================================================
# TCP Protocol
# ============================================================================

_HEADER = struct.Struct("<I")


def _recv_exactly(sock, n):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        chunk = sock.recv_into(view[got:], n - got)
        if not chunk:
            raise ConnectionResetError("Server closed connection")
        got += chunk
    return bytes(buf)


def _send_message(sock, payload):
    sock.sendall(_HEADER.pack(len(payload)) + payload)


def _recv_message(sock):
    length = _HEADER.unpack(_recv_exactly(sock, 4))[0]
    return _recv_exactly(sock, length)


def robot_command(sock, steering, throttle):
    """Send one control command, receive observation dict."""
    cmd = json.dumps(
        {"steering": float(steering), "throttle": float(throttle)},
        separators=(",", ":"),
    ).encode()
    _send_message(sock, cmd)
    raw = _recv_message(sock)
    return json.loads(raw)


# ============================================================================
# Data Extraction
# ============================================================================

def parse_robot_image(obs):
    """Decode base64 JPEG image from robot observation."""
    img_b64 = obs["img"]
    img_bytes = base64.b64decode(img_b64)
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        # Fallback: raw pixel array
        img = np.array(obs["img"], dtype=np.uint8)
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def parse_imu(obs):
    """
    Extract IMU from robot observation.

    Returns (accel, gyro) as numpy float64 arrays.
    Gyro is converted from deg/s to rad/s.
    """
    imu_raw = obs.get("imu", {})

    accel_d = imu_raw.get("acceleration", {})
    accel = np.array([
        accel_d.get("x", 0.0),
        accel_d.get("y", 0.0),
        accel_d.get("z", 0.0),
    ], dtype=np.float64)

    gyro_d = imu_raw.get("angular_velocity", {})
    gyro_deg = np.array([
        gyro_d.get("roll_rate", 0.0),
        gyro_d.get("pitch_rate", 0.0),
        gyro_d.get("yaw_rate", 0.0),
    ], dtype=np.float64)

    # WitMotion outputs deg/s, ORB-SLAM3 expects rad/s
    gyro = gyro_deg * DEG_TO_RAD

    return accel, gyro


# ============================================================================
# Velocity Estimator
# ============================================================================

class VelocityEstimator:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.pose_history = deque(maxlen=window_size)
        self.time_history = deque(maxlen=window_size)
        self.linear_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.speed = 0.0

    def update(self, pose, timestamp):
        try:
            R_mat = pose[:3, :3]
            t_vec = pose[:3, 3]
            position = -R_mat.T @ t_vec
        except Exception:
            return self._result()

        self.pose_history.append(pose.copy())
        self.time_history.append(timestamp)

        if len(self.pose_history) < 2:
            return self._result()

        dt = self.time_history[-1] - self.time_history[-2]
        if dt < 1e-6:
            return self._result()

        prev_pose = self.pose_history[-2]
        prev_R = prev_pose[:3, :3]
        prev_t = prev_pose[:3, 3]
        prev_position = -prev_R.T @ prev_t

        displacement = position - prev_position
        instant_vel = displacement / dt

        alpha = 0.3
        self.linear_velocity = alpha * instant_vel + (1 - alpha) * self.linear_velocity
        self.speed = np.linalg.norm(self.linear_velocity)

        try:
            R_rel = R_mat @ prev_R.T
            rotvec = R.from_matrix(R_rel).as_rotvec()
            self.angular_velocity = alpha * (rotvec / dt) + (1 - alpha) * self.angular_velocity
        except Exception:
            pass

        return self._result()

    def _result(self):
        return {
            "linear_velocity": self.linear_velocity.copy(),
            "angular_velocity": self.angular_velocity.copy(),
            "speed": self.speed,
        }

    def reset(self):
        self.pose_history.clear()
        self.time_history.clear()
        self.linear_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.speed = 0.0


# ============================================================================
# ORB-SLAM3 Tracker
# ============================================================================

class ORBSLAM3Tracker:
    STATE_NAMES = {
        -1: "SYSTEM_NOT_READY", 0: "NO_IMAGES_YET", 1: "NOT_INITIALIZED",
        2: "OK", 3: "RECENTLY_LOST", 4: "LOST", 5: "OK_KLT",
    }

    def __init__(self, vocab_path, settings_path, use_viewer=False,
                 lib_path=None, use_imu=True, verbose=False):
        if lib_path:
            sys.path.insert(0, lib_path)

        self.use_imu = use_imu
        self.verbose = verbose
        sensor_type = "imu_monocular" if use_imu else "mono"

        print(f"\n{'='*60}")
        print("Initializing ORB-SLAM3")
        print(f"  Vocab: {vocab_path}")
        print(f"  Settings: {settings_path}")
        print(f"  Sensor: {sensor_type}")
        print(f"{'='*60}\n")

        import orbslam3
        self.slam = orbslam3.System(vocab_path, settings_path, sensor_type, use_viewer)

        self.frame_count = 0
        self.poses = []
        self.positions = []
        self.timestamps = []
        self.is_initialized = False
        self.imu_buffer = []
        self.last_timestamp = 0.0
        self.velocity_estimator = VelocityEstimator()

        print("ORB-SLAM3 system created!\n")

    def add_imu(self, accel, gyro, timestamp):
        """Add one IMU measurement. Format: [ax,ay,az,gx,gy,gz,ts]"""
        if accel is None or gyro is None:
            return
        vals = np.array([
            float(accel[0]), float(accel[1]), float(accel[2]),
            float(gyro[0]), float(gyro[1]), float(gyro[2]),
            float(timestamp),
        ], dtype=np.float64)
        # Reject NaN/Inf or extreme values to prevent preintegration divergence
        if not np.all(np.isfinite(vals)):
            return
        if np.any(np.abs(vals[:3]) > 1000.0):  # accel > 1000 m/s^2
            return
        if np.any(np.abs(vals[3:6]) > 100.0):  # gyro > 100 rad/s
            return
        self.imu_buffer.append(vals)

    def process(self, image, timestamp, accel=None, gyro=None):
        if self.verbose and accel is not None and gyro is not None:
            print(f"  IMU: acc=[{accel[0]:.3f},{accel[1]:.3f},{accel[2]:.3f}] "
                  f"gyr=[{gyro[0]:.5f},{gyro[1]:.5f},{gyro[2]:.5f}] "
                  f"ts={timestamp:.4f} dt={timestamp - self.last_timestamp if self.last_timestamp > 0 else 0:.4f} "
                  f"buf={len(self.imu_buffer)}")
        if accel is not None and gyro is not None:
            self.add_imu(accel, gyro, timestamp)

        dt = timestamp - self.last_timestamp if self.last_timestamp > 0 else 0.05
        self.last_timestamp = timestamp

        slam_velocity = np.zeros(3)

        if self.use_imu and len(self.imu_buffer) > 0:
            imu_array = np.vstack(self.imu_buffer).astype(np.float64)
            if imu_array.ndim == 1:
                imu_array = imu_array.reshape(1, -1)
            imu_array = imu_array[imu_array[:, 6].argsort()]
            try:
                pose, slam_velocity = self.slam.track_monocular_imu_with_velocity(
                    image, timestamp, imu_array)
            except AttributeError:
                pose = self.slam.track_monocular_imu(image, timestamp, imu_array)
            self.imu_buffer = [m for m in self.imu_buffer if m[6] > timestamp]
        elif not self.use_imu:
            try:
                pose, slam_velocity = self.slam.track_monocular_with_velocity(image, timestamp)
            except AttributeError:
                pose = self.slam.track_monocular(image, timestamp)
        else:
            # IMU mode but no IMU data — fall back to monocular
            try:
                pose, slam_velocity = self.slam.track_monocular_with_velocity(image, timestamp)
            except AttributeError:
                pose = self.slam.track_monocular(image, timestamp)

        state = self.slam.get_tracking_state_int()
        state_str = self.STATE_NAMES.get(state, "UNKNOWN")
        self.frame_count += 1

        if not self.is_initialized and state == 2:
            self.is_initialized = True
            print(f"\n  ORB-SLAM3 INITIALIZED at frame {self.frame_count}!\n")

        position = np.zeros(3)
        vel_info = {
            "speed": 0.0, "linear_velocity": np.zeros(3),
            "angular_velocity": np.zeros(3), "slam_velocity": slam_velocity.copy(),
        }

        if state in (2, 5):
            try:
                R_mat = pose[:3, :3]
                t_vec = pose[:3, 3]
                position = -R_mat.T @ t_vec
                self.poses.append(pose.copy())
                self.positions.append(position.copy())
                self.timestamps.append(timestamp)
                vel_info.update(self.velocity_estimator.update(pose, timestamp))
                if np.linalg.norm(slam_velocity) > 1e-6:
                    vel_info["linear_velocity"] = slam_velocity.copy()
                    vel_info["speed"] = np.linalg.norm(slam_velocity)
            except Exception:
                pass

        return {
            "pose": pose, "position": position, "state": state,
            "state_str": state_str, "is_initialized": self.is_initialized,
            "frame_count": self.frame_count, "timestamp": timestamp, "dt": dt,
            "num_keyframes": self.slam.get_num_keyframes(),
            "num_map_points": self.slam.get_num_map_points(),
            "imu_buffer_size": len(self.imu_buffer),
            **vel_info,
        }

    def get_trajectory(self):
        if not self.positions:
            return np.zeros((0, 3))
        return np.array(self.positions)

    def reset(self):
        self.slam.reset()
        self.frame_count = 0
        self.poses.clear()
        self.positions.clear()
        self.timestamps.clear()
        self.is_initialized = False
        self.imu_buffer.clear()
        self.last_timestamp = 0.0
        self.velocity_estimator.reset()

    def shutdown(self):
        self.slam.shutdown()


# ============================================================================
# Visualization
# ============================================================================

def draw_overlay(image, info, fps, accel=None, gyro=None, steering=0.0, throttle=0.0):
    img = image.copy()
    h, w = img.shape[:2]

    # Panel
    cv2.rectangle(img, (5, 5), (430, 340), (0, 0, 0), -1)
    cv2.rectangle(img, (5, 5), (430, 340), (255, 255, 255), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 25

    state = info["state"]
    state_color = (0, 255, 0) if state in (2, 5) else (0, 255, 255) if state == 1 else (0, 0, 255)

    cv2.putText(img, "ORB-SLAM3 + Robot", (10, y), font, 0.6, (255, 255, 255), 2); y += 25
    cv2.putText(img, f"State: {info['state_str']}", (10, y), font, 0.5, state_color, 1); y += 20
    cv2.putText(img, f"Frame: {info['frame_count']} | FPS: {fps:.1f} | dt: {info.get('dt',0)*1000:.1f}ms",
                (10, y), font, 0.45, (255, 255, 255), 1); y += 18
    cv2.putText(img, f"t: {info.get('timestamp',0):.3f}s", (10, y), font, 0.45, (255, 255, 255), 1); y += 18
    cv2.putText(img, f"KF: {info['num_keyframes']} | MP: {info['num_map_points']}",
                (10, y), font, 0.45, (255, 255, 255), 1); y += 22

    # Position
    pos = info["position"]
    cv2.putText(img, f"Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]",
                (10, y), font, 0.45, (200, 255, 200), 1); y += 22

    # Velocity
    cv2.putText(img, "--- Velocity ---", (10, y), font, 0.5, (0, 255, 255), 1); y += 20
    speed = info.get("speed", 0.0)
    slam_vel = info.get("slam_velocity", np.zeros(3))
    lin_vel = info.get("linear_velocity", np.zeros(3))
    ang_vel = info.get("angular_velocity", np.zeros(3))

    cv2.putText(img, f"Speed: {speed:.2f} m/s", (10, y), font, 0.5, (255, 255, 0), 1); y += 18
    cv2.putText(img, f"SLAM: [{slam_vel[0]:.2f}, {slam_vel[1]:.2f}, {slam_vel[2]:.2f}]",
                (10, y), font, 0.45, (0, 255, 0), 1); y += 18
    cv2.putText(img, f"Est:  [{lin_vel[0]:.2f}, {lin_vel[1]:.2f}, {lin_vel[2]:.2f}]",
                (10, y), font, 0.45, (200, 200, 200), 1); y += 18
    cv2.putText(img, f"Ang:  [{ang_vel[0]:.2f}, {ang_vel[1]:.2f}, {ang_vel[2]:.2f}]",
                (10, y), font, 0.45, (200, 200, 200), 1); y += 22

    # IMU
    cv2.putText(img, "--- IMU ---", (10, y), font, 0.5, (0, 255, 255), 1); y += 18
    if accel is not None:
        cv2.putText(img, f"Acc: [{accel[0]:.2f}, {accel[1]:.2f}, {accel[2]:.2f}]",
                    (10, y), font, 0.45, (200, 200, 200), 1)
    else:
        cv2.putText(img, "Acc: N/A", (10, y), font, 0.45, (128, 128, 128), 1)
    y += 16
    if gyro is not None:
        cv2.putText(img, f"Gyr: [{gyro[0]:.4f}, {gyro[1]:.4f}, {gyro[2]:.4f}] rad/s",
                    (10, y), font, 0.45, (200, 200, 200), 1)
    else:
        cv2.putText(img, "Gyr: N/A", (10, y), font, 0.45, (128, 128, 128), 1)
    y += 16
    cv2.putText(img, f"IMU buf: {info.get('imu_buffer_size', 0)}",
                (10, y), font, 0.45, (200, 200, 200), 1); y += 22

    # Controls
    cv2.putText(img, f"Steer: {steering:+.2f} | Throttle: {throttle:+.2f}",
                (10, y), font, 0.45, (255, 200, 0), 1)

    # Tracking indicator
    if info["is_initialized"]:
        cv2.putText(img, "TRACKING", (w - 120, 30), font, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(img, "INIT...", (w - 100, 30), font, 0.7, (0, 255, 255), 2)

    # Speed bar
    max_speed = 5.0
    bar_w = int(min(speed / max_speed, 1.0) * 150)
    cv2.rectangle(img, (w - 170, 50), (w - 170 + bar_w, 65), (0, 255, 0), -1)
    cv2.rectangle(img, (w - 170, 50), (w - 20, 65), (255, 255, 255), 1)
    cv2.putText(img, f"{speed:.1f}m/s", (w - 165, 80), font, 0.4, (255, 255, 255), 1)

    return img


def draw_trajectory(positions, size=300):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    if len(positions) < 2:
        cv2.putText(img, "Waiting...", (size // 4, size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        return img

    xz = positions[:, [0, 2]]
    xz_min, xz_max = xz.min(axis=0), xz.max(axis=0)
    xz_range = np.maximum(xz_max - xz_min, 0.1)
    margin = 30
    scale = (size - 2 * margin) / xz_range.max()
    center = np.array([size // 2, size // 2])
    xz_center = (xz_min + xz_max) / 2
    pixels = ((xz - xz_center) * scale + center).astype(np.int32)

    for i in range(1, len(pixels)):
        t = i / len(pixels)
        color = (int(255 * (1 - t)), 0, int(255 * t))
        cv2.line(img, tuple(pixels[i - 1]), tuple(pixels[i]), color, 2)

    cv2.circle(img, tuple(pixels[0]), 5, (0, 255, 0), -1)
    cv2.circle(img, tuple(pixels[-1]), 6, (0, 0, 255), -1)
    cv2.putText(img, "Trajectory", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, f"N={len(positions)}", (10, size - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
    return img


# ============================================================================
# Robot Connection
# ============================================================================

class RobotConnection:
    """Manages TCP connection to the robot with auto-reconnect."""

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self._sock = None

    def _connect(self):
        s = socket.create_connection((self.host, self.port), timeout=5)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.settimeout(2.0)
        self._sock = s
        print(f"Connected to {self.host}:{self.port}")

    def _disconnect(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def step(self, steering, throttle):
        """Send command, receive (image, accel, gyro, velocity_info, collision_info)."""
        for attempt in range(2):
            try:
                if self._sock is None:
                    self._connect()
                result = robot_command(self._sock, steering, throttle)
                break
            except (ConnectionResetError, BrokenPipeError, OSError, TimeoutError) as e:
                print(f"  Socket error ({e}), reconnecting (attempt {attempt + 1})")
                self._disconnect()
                if attempt == 1:
                    return None, None, None, {}, {}
        else:
            return None, None, None, {}, {}

        obs = result.get("observation", {})

        # Image
        try:
            image = parse_robot_image(obs)
        except Exception as e:
            print(f"  Image decode error: {e}")
            image = None

        # IMU
        try:
            accel, gyro = parse_imu(obs)
        except Exception as e:
            print(f"  IMU parse error: {e}")
            accel, gyro = None, None

        # Velocity / collision
        vel_raw = obs.get("velocity", {})
        velocity_info = {
            "cms": float(vel_raw.get("cms", 0.0)),
            "ms": float(vel_raw.get("ms", 0.0)),
        }
        obs_raw = obs.get("obstacle", {})
        collision_info = {
            "detected": bool(obs_raw.get("detected", False)),
            "distance_cm": float(obs_raw.get("distance_cm", float("inf"))),
        }

        return image, accel, gyro, velocity_info, collision_info

    def close(self):
        if self._sock is not None:
            try:
                robot_command(self._sock, 0.0, 0.0)
            except Exception:
                pass
        self._disconnect()


# ============================================================================
# Keyboard Control
# ============================================================================

def get_keyboard_controls():
    """Read keyboard state from OpenCV window. Returns (steering, throttle, should_quit, should_reset, should_save)."""
    steering = 0.0
    throttle = 0.0
    should_quit = False
    should_reset = False
    should_save = False

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:  # ESC
        should_quit = True
    elif key == ord("r"):
        should_reset = True
    elif key == ord("t"):
        should_save = True
    elif key == ord(" "):  # SPACE - stop
        pass  # steering=throttle=0
    else:
        # Continuous key tracking
        pressed = set()
        flags = cv2.getWindowProperty("ORB-SLAM3 Robot", cv2.WND_PROP_VISIBLE)
        # OpenCV doesn't have great key state tracking; use key events
        if key == ord("w"):
            throttle = 0.15
        elif key == ord("s"):
            throttle = -0.15
        elif key == ord("a"):
            steering = -0.5
        elif key == ord("d"):
            steering = 0.5

    return steering, throttle, should_quit, should_reset, should_save


# ============================================================================
# Initialization Phase
# ============================================================================

def initialize_slam(robot, tracker, use_imu, max_steps=300, event_client=None):
    """Drive the robot to initialize ORB-SLAM3."""
    print("\n" + "=" * 60)
    print(f"Initializing ORB-SLAM3 (IMU: {use_imu})...")
    print("=" * 60 + "\n")

    start_time = time.time()
    step = 0

    info = {
        "state": 0, "state_str": "STARTING", "is_initialized": False,
        "frame_count": 0, "num_keyframes": 0, "num_map_points": 0,
        "position": np.zeros(3), "imu_buffer_size": 0, "speed": 0.0,
        "linear_velocity": np.zeros(3), "angular_velocity": np.zeros(3),
        "slam_velocity": np.zeros(3), "timestamp": 0.0, "dt": 0.0,
    }

    while not tracker.is_initialized and step < max_steps:
        # Exploration: alternate steering to help SLAM initialize
        if step < 20:
            steering, throttle = 0.0, 0.15
        elif step < 50:
            steering, throttle = -0.3, 0.15
        elif step < 80:
            steering, throttle = 0.3, 0.15
        else:
            steering = np.random.uniform(-0.4, 0.4)
            throttle = 0.15

        image, accel, gyro, vel, coll = robot.step(steering, throttle)
        if image is None:
            step += 1
            continue

        timestamp = time.time() - start_time

        # Drain IMU buffer from WebSocket during init
        if event_client is not None:
            imu_samples = event_client.drain_imu_buffer()
            for sample in imu_samples:
                s_accel = np.array([
                    sample.get("acceleration", {}).get("x", 0.0),
                    sample.get("acceleration", {}).get("y", 0.0),
                    sample.get("acceleration", {}).get("z", 0.0),
                ], dtype=np.float64)
                s_gyro_deg = np.array([
                    sample.get("angular_velocity", {}).get("roll_rate", 0.0),
                    sample.get("angular_velocity", {}).get("pitch_rate", 0.0),
                    sample.get("angular_velocity", {}).get("yaw_rate", 0.0),
                ], dtype=np.float64)
                s_gyro = s_gyro_deg * DEG_TO_RAD
                s_ts = sample.get("timestamp", timestamp)
                adjusted_ts = s_ts - TIMESHIFT_CAM_IMU
                tracker.add_imu(s_accel, s_gyro, adjusted_ts)

        info = tracker.process(image, timestamp, accel, gyro)
        step += 1

        if step % 10 == 0:
            print(f"  Init {step:3d}/{max_steps} | {info['state_str']:15s} | "
                  f"t={timestamp:.2f}s | KF:{info['num_keyframes']:3d} | "
                  f"MP:{info['num_map_points']:5d}")

        # Visualize
        vis = draw_overlay(image, info, 0.0, accel, gyro, steering, throttle)
        cv2.imshow("ORB-SLAM3 Robot", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return info, start_time, False

    success = tracker.is_initialized
    print(f"\n{'Initialized' if success else 'Not initialized'} after {step} steps")
    return info, start_time, success


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    use_imu = not args.no_imu

    print("\n" + "=" * 60)
    print(f"ORB-SLAM3 Robot Test - {'Mono-Inertial' if use_imu else 'Monocular'}")
    print("=" * 60 + "\n")

    # Initialize tracker
    tracker = ORBSLAM3Tracker(
        vocab_path=args.vocab,
        settings_path=args.settings,
        use_viewer=args.viewer,
        lib_path=args.orbslam_lib,
        use_imu=use_imu,
        verbose=args.verbose,
    )

    # Connect to robot via TCP
    robot = RobotConnection(args.host, args.port)

    # Connect to WebSocket for high-frequency IMU streaming
    event_client = RobotEventClient(host=args.host, ws_port=args.ws_port)
    event_client.start()
    time.sleep(0.5)  # Let initial IMU data buffer

    # Initialize SLAM
    info, start_time, init_ok = initialize_slam(robot, tracker, use_imu,
                                                  args.max_init_steps,
                                                  event_client=event_client)

    print("\nControls: W/S=throttle, A/D=steer, SPACE=stop, Q=quit, R=reset, T=save")
    print("=" * 60 + "\n")

    # Main loop
    step = 0
    steering = 0.0
    throttle = 0.0
    last_time = time.time()
    fps = 0.0
    running = True

    # Smoothed controls for continuous key input
    target_steering = 0.0
    target_throttle = 0.0

    try:
        while running:
            if args.max_steps > 0 and step >= args.max_steps:
                break

            # Keyboard input
            key_steer, key_throttle, should_quit, should_reset, should_save = get_keyboard_controls()

            if should_quit:
                break
            if should_reset:
                print("\nResetting SLAM...")
                tracker.reset()
                info, start_time, _ = initialize_slam(robot, tracker, use_imu,
                                                       args.max_init_steps,
                                                       event_client=event_client)
                step = 0
                continue
            if should_save:
                traj = tracker.get_trajectory()
                if len(traj) > 0:
                    fname = f"trajectory_{int(time.time())}.npz"
                    np.savez(fname, positions=traj, timestamps=np.array(tracker.timestamps))
                    print(f"\nSaved: {fname}")

            # Update controls smoothly
            if key_steer != 0 or key_throttle != 0:
                target_steering = key_steer
                target_throttle = key_throttle
            elif any(k & 0xFF in (ord("w"), ord("s"), ord("a"), ord("d"))
                     for k in [cv2.waitKey(1)]):
                pass  # Keep current targets
            else:
                # Decay to zero
                target_steering *= 0.8
                target_throttle *= 0.8
                if abs(target_steering) < 0.01:
                    target_steering = 0.0
                if abs(target_throttle) < 0.01:
                    target_throttle = 0.0

            # Clamp throttle to safe range for the robot
            throttle = float(np.clip(target_throttle, -0.2, 0.2))
            steering = float(np.clip(target_steering, -1.0, 1.0))

            # Get observation from robot via TCP
            image, accel, gyro, vel_info, coll_info = robot.step(steering, throttle)
            if image is None:
                time.sleep(0.05)
                continue

            timestamp = time.time() - start_time
            step += 1

            # Drain ALL IMU samples between frames from WebSocket (~200Hz)
            # This is the key fix for ORB-SLAM3 preintegration:
            #   - Previously: 1 IMU per frame (from TCP snapshot)
            #   - Now: ~10-30 IMU samples per frame (from WS stream)
            #   - Enables correct application of camera-IMU timeshift (0.604s)
            imu_samples = event_client.drain_imu_buffer()
            for sample in imu_samples:
                s_accel = np.array([
                    sample.get("acceleration", {}).get("x", 0.0),
                    sample.get("acceleration", {}).get("y", 0.0),
                    sample.get("acceleration", {}).get("z", 0.0),
                ], dtype=np.float64)
                s_gyro_deg = np.array([
                    sample.get("angular_velocity", {}).get("roll_rate", 0.0),
                    sample.get("angular_velocity", {}).get("pitch_rate", 0.0),
                    sample.get("angular_velocity", {}).get("yaw_rate", 0.0),
                ], dtype=np.float64)
                s_gyro = s_gyro_deg * DEG_TO_RAD
                s_ts = sample.get("timestamp", timestamp)
                # Apply camera-IMU timeshift (camera is 0.604s ahead of IMU)
                adjusted_ts = s_ts - TIMESHIFT_CAM_IMU
                tracker.add_imu(s_accel, s_gyro, adjusted_ts)

            # Process camera frame with SLAM (IMU data already buffered)
            # NOTE: We pass timestamp (not adjusted_ts) for the camera frame
            # because the camera timestamp is the ground truth for the image
            info = tracker.process(image, timestamp)

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

            # FPS
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - last_time, 0.001)
            last_time = now

            # Visualize
            vis = draw_overlay(image, info, fps, accel, gyro, steering, throttle)
            traj = tracker.get_trajectory()
            traj_view = draw_trajectory(traj)

            h, w = vis.shape[:2]
            scale = min(h / 300, 0.5)
            traj_small = cv2.resize(traj_view, None, fx=scale, fy=scale)
            th, tw = traj_small.shape[:2]
            vis[h - th - 10:h - 10, w - tw - 10:w - 10] = traj_small

            cv2.imshow("ORB-SLAM3 Robot", vis)

            # Periodic logging
            if step % 50 == 0:
                pos = info["position"]
                print(f"Step {step:4d} | {info['state_str']:12s} | "
                      f"t={timestamp:.2f}s | speed={info['speed']:.2f}m/s | "
                      f"pos=[{pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}]")

    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        print("\n" + "=" * 60)
        print("Done!")
        print("=" * 60)

        traj = tracker.get_trajectory()
        print(f"Stats: {tracker.frame_count} frames, {len(traj)} poses")

        if len(traj) > 1:
            dist = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
            print(f"Distance: {dist:.2f}m")

        if args.save_trajectory and len(traj) > 0:
            fname = f"trajectory_final_{int(time.time())}.npz"
            np.savez(fname, positions=traj, timestamps=np.array(tracker.timestamps))
            print(f"Saved: {fname}")

        cv2.destroyAllWindows()
        tracker.shutdown()
        event_client.stop()
        robot.close()


if __name__ == "__main__":
    main()