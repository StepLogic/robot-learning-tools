import json
import os
import socket
import struct
import sys
import time
import numpy as np
import cv2
import pygame
import gymnasium as gym
from gymnasium import spaces
from collections import deque


INITIAL_SCREEN_WIDTH, INITIAL_SCREEN_HEIGHT = 640, 480
FPS = 30
ROBOT_HOST = "10.42.0.1"
ROBOT_PORT = 9000

_HEADER = struct.Struct('<I')


# ─── TCP helpers ──────────────────────────────────────────────────────────────

def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        chunk = sock.recv_into(view[got:], n - got)
        if not chunk:
            raise ConnectionResetError("Server closed connection")
        got += chunk
    return bytes(buf)


def _send_message(sock: socket.socket, payload: bytes):
    sock.sendall(_HEADER.pack(len(payload)) + payload)


def _recv_message(sock: socket.socket) -> bytes:
    length = _HEADER.unpack(_recv_exactly(sock, 4))[0]
    return _recv_exactly(sock, length)


# ─── Undistort ────────────────────────────────────────────────────────────────

def undistort(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    fish_K = np.array([[60, 0, 80], [0, 60, 60], [0, 0, 1]], dtype=np.float32)
    fish_D = np.array([[-0.0018], [0], [0], [0]], dtype=np.float32)
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        fish_K, fish_D, (w, h), np.eye(3), 0.75)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        fish_K, fish_D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
    return cv2.remap(img, map1, map2, cv2.INTER_LINEAR)


# ─── Server response parser ───────────────────────────────────────────────────

def _parse_response(result: dict, prev_accel: np.ndarray, prev_gyro: np.ndarray):
    """
    Parse a full server response dict into components.

    Returns:
        image       np.ndarray  (H, W, 3) uint8
        imu_6d      np.ndarray  (6,) float32  — [ax, ay, az, roll_rate, pitch_rate, yaw_rate]
        velocity    dict        {cms, ms, method}
        collision   dict        {detected, distance_cm, threshold_cm}
        blocked     bool        throttle was hard-zeroed by server
    """
    obs = result.get("observation", {})

    # ── Image ─────────────────────────────────────────────────────────────────
    image = np.array(obs["img"], dtype=np.uint8)
    image = undistort(image)

    # ── IMU ───────────────────────────────────────────────────────────────────
    imu_raw = obs.get("imu", {})

    accel_d = imu_raw.get("acceleration", {})
    if accel_d:
        accel = np.array(
            [accel_d.get("x", 0.0), accel_d.get("y", 0.0), accel_d.get("z", 0.0)],
            dtype=np.float32,
        )
    else:
        accel = prev_accel.copy()

    gyro_d = imu_raw.get("angular_velocity", {})
    if gyro_d:
        gyro = np.array(
            [gyro_d.get("roll_rate", 0.0),
             gyro_d.get("pitch_rate", 0.0),
             gyro_d.get("yaw_rate",  0.0)],
            dtype=np.float32,
        )
    else:
        gyro = prev_gyro.copy()

    imu_6d = np.concatenate([accel, gyro])

    # ── Velocity (from server velocity daemon) ────────────────────────────────
    vel_raw  = obs.get("velocity", {})
    velocity = {
        "cms":    float(vel_raw.get("cms",    0.0)),
        "ms":     float(vel_raw.get("ms",     0.0)),
        "method": str(vel_raw.get("method", "unknown")),
    }

    # ── Obstacle / collision ──────────────────────────────────────────────────
    obs_raw   = obs.get("obstacle", {})
    collision = {
        "detected":     bool(obs_raw.get("detected",     False)),
        "distance_cm":  float(obs_raw.get("distance_cm", float("inf"))),
        "threshold_cm": float(obs_raw.get("threshold_cm", 15.0)),
    }
    # print(collision)

    blocked = bool(result.get("blocked", False))

    return image, imu_6d, velocity, collision, blocked


# ─── Environment ──────────────────────────────────────────────────────────────

class RacerEnv(gym.Env):
    """
    Gymnasium environment for a Donkey Car robot.
    Uses a persistent TCP socket (4-byte length-prefixed JSON).

    Observation space (unchanged):
        image : (H, W, 3) uint8
        imu   : (6,) float32  — [ax, ay, az, roll_rate, pitch_rate, yaw_rate]

    Info dict (extended):
        trajectory      : {"position": [...]}
        low_accel_count : int
        velocity        : {"cms": float, "ms": float, "method": str}
        collision       : {"detected": bool, "distance_cm": float, "threshold_cm": float}
        blocked         : bool   — server hard-zeroed throttle due to obstacle
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': FPS}

    def __init__(self, render_mode: str = "human"):
        super().__init__()

        # ── Observation & action spaces (unchanged) ───────────────────────────
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, high=255,
                shape=(INITIAL_SCREEN_HEIGHT, INITIAL_SCREEN_WIDTH, 3),
                dtype=np.uint8,
            ),
            "imu": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(6,),
                dtype=np.float32,
            ),
        })

        self.action_space = spaces.Box(
            low=np.array([-1.0,  0.120], dtype=np.float32),
            high=np.array([ 1.0,  0.2], dtype=np.float32),
            dtype=np.float32,
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Pygame
        self.screen = None
        self.clock  = None
        self.first_image_received = False

        # Internal state
        self.trajectory    = {"position": []}
        self.frame_count   = 0
        self.prev_steering = 0.0

        self.image       = np.zeros((120, 160, 3), dtype=np.uint8)
        self.imu_accel   = np.zeros(3, dtype=np.float32)
        self.imu_gyro    = np.zeros(3, dtype=np.float32)
        self.current_image = self.image.copy()
        self.current_imu   = np.zeros(6, dtype=np.float32)

        # Velocity & collision — updated each step, exposed via info
        self.current_velocity  = {"cms": 0.0, "ms": 0.0, "method": "zeroed"}
        self.current_collision = {"detected": False, "distance_cm": float("inf"),
                                  "threshold_cm": 15.0}
        self.current_blocked   = False

        # Termination on sustained low acceleration
        self.low_accel_count     = 0
        self.LOW_ACCEL_THRESHOLD = 0.8   # m/s²
        self.LOW_ACCEL_STEPS     = 3

        # TCP socket (kept alive across steps)
        self._sock: socket.socket | None = None

    # ── Socket management ─────────────────────────────────────────────────────

    def _ensure_connected(self):
        if self._sock is not None:
            return
        self._sock = socket.create_connection((ROBOT_HOST, ROBOT_PORT), timeout=5)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock.settimeout(2.0)

    def _disconnect(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    # ── Communication ─────────────────────────────────────────────────────────

    def _send_command_and_get_image(
        self, steering: float, throttle: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Send one control command, receive full server response.
        Updates self.current_velocity, self.current_collision, self.current_blocked.
        Returns (image, imu_6d).
        """
        steering = float(steering)
        throttle = float(np.clip(throttle, -0.2, 0.2))

        payload = json.dumps(
            {'steering': steering, 'throttle': throttle},
            separators=(',', ':'),
        ).encode()

        for attempt in range(2):
            try:
                self._ensure_connected()
                _send_message(self._sock, payload)
                result = json.loads(_recv_message(self._sock))
                break
            except (ConnectionResetError, BrokenPipeError, OSError, TimeoutError) as e:
                print(f"  Socket error ({e}), reconnecting…")
                self._disconnect()
                if attempt == 1:
                    # Return stale data — velocity/collision unchanged
                    return self.image, np.concatenate([self.imu_accel, self.imu_gyro])

        # ── Parse full response ───────────────────────────────────────────────
        image, imu_6d, velocity, collision, blocked = _parse_response(
            result, self.imu_accel, self.imu_gyro)

        # Cache for stale-data fallback and info dict
        self.image             = image
        self.imu_accel         = imu_6d[:3]
        self.imu_gyro          = imu_6d[3:]
        self.current_velocity  = velocity
        self.current_collision = collision
        self.current_blocked   = blocked
        # print(c)
        return image, imu_6d

    # ── Gym API ───────────────────────────────────────────────────────────────

    def _get_obs(self) -> dict:
        return {"image": self.current_image, "imu": self.current_imu}

    def _get_info(self) -> dict:
        return {
            "trajectory":      self.trajectory,
            "low_accel_count": self.low_accel_count,
            # ── New fields ───────────────────────────────────────────────────
            "velocity":  self.current_velocity,    # {cms, ms, method}
            "collision": self.current_collision,   # {detected, distance_cm, threshold_cm}
            "blocked":   self.current_blocked,     # bool
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._send_command_and_get_image(0.0, 0.0)
        # for _ in range(5):
        #     self._send_command_and_get_image(0.0, -0.2)
        self.current_image, self.current_imu = \
            self._send_command_and_get_image(0.0, 0.0)

        self.trajectory      = {"position": []}
        self.frame_count     = 0
        self.low_accel_count = 0
        self.prev_steering   = 0.0
        self.first_image_received = False

        # Reset velocity/collision to safe defaults
        self.current_velocity  = {"cms": 0.0, "ms": 0.0, "method": "zeroed"}
        self.current_collision = {"detected": False, "distance_cm": float("inf"),
                                  "threshold_cm": 15.0}
        self.current_blocked   = False

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        steering, throttle = float(action[0]), float(action[1])

        reward     = -abs(self.prev_steering - steering)
        terminated = False

        self.current_image, self.current_imu = \
            self._send_command_and_get_image(steering, throttle)

        accel_magnitude = np.linalg.norm(self.current_imu[:3])

        # ── Collision penalty ─────────────────────────────────────────────────
        if self.current_collision["detected"]:
            reward     -= 20.0
            terminated  = True

        # ── Low-acceleration termination (unchanged logic) ────────────────────
        elif accel_magnitude < self.LOW_ACCEL_THRESHOLD:
            self.low_accel_count += 1
            reward -= 1.0
            if self.low_accel_count >= self.LOW_ACCEL_STEPS:
                terminated  = True
                reward     -= 10.0
        else:
            self.low_accel_count = 0
            reward += 1.0

        # ── Velocity reward shaping ───────────────────────────────────────────
        # Small bonus for forward motion; penalise if blocked with throttle applied
        vel_ms = self.current_velocity["ms"]
        if vel_ms > 0.05:
            reward += 0.5 * vel_ms          # reward forward progress
        if self.current_blocked and throttle > 0.05:
            reward -= 2.0                   # penalise trying to drive into obstacle

        self.prev_steering = steering
        self.frame_count  += 1

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, False, self._get_info()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_obs()["image"]
        self._render_frame()

    def _render_frame(self):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (INITIAL_SCREEN_WIDTH, INITIAL_SCREEN_HEIGHT))
            pygame.display.set_caption("Donkey Car Control")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.current_image is not None:
            self.first_image_received = True
            surface = pygame.surfarray.make_surface(
                cv2.resize(
                    self.current_image.swapaxes(0, 1),
                    (INITIAL_SCREEN_WIDTH, INITIAL_SCREEN_HEIGHT),
                    interpolation=cv2.INTER_AREA,
                )
            )
            self.screen.blit(surface, (0, 0))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        self._send_command_and_get_image(0.0, 0.0)
        self._disconnect()
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
        print("\nEnvironment closed.")


# ─── Stacking wrapper (unchanged) ─────────────────────────────────────────────

class StackingWrapper(gym.Wrapper):
    """Stack RGB frames, actions, and IMU data for temporal modelling."""

    def __init__(self, env: gym.Env, num_stack: int = 3):
        super().__init__(env)
        self.num_stack  = num_stack
        self.action_dim = env.action_space.shape[0]

        self.action_history = deque(maxlen=num_stack)
        self.rgb_history    = deque(maxlen=num_stack)
        self.imu_history    = deque(maxlen=num_stack)

        self.observation_space = spaces.Dict({
            "pixels": spaces.Box(
                low=0, high=255,
                shape=(120, 160, 3 * num_stack),
                dtype=np.uint8,
            ),
            "actions": spaces.Box(
                low=np.tile(env.action_space.low,  num_stack),
                high=np.tile(env.action_space.high, num_stack),
                shape=(num_stack * self.action_dim,),
                dtype=np.float32,
            ),
            "imu": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(6 * num_stack,),
                dtype=np.float32,
            ),
        })

    def _stacked_obs(self) -> dict:
        actions = list(self.action_history)
        while len(actions) < self.num_stack:
            actions.insert(0, np.zeros(self.action_dim, dtype=np.float32))

        imus = list(self.imu_history)
        while len(imus) < self.num_stack:
            imus.insert(0, np.zeros(6, dtype=np.float32))

        return {
            "pixels":  np.concatenate(list(self.rgb_history),  axis=-1).astype(np.uint8),
            "actions": np.concatenate(actions).astype(np.float32),
            "imu":     np.concatenate(imus).astype(np.float32),
        }

    def _push(self, obs: dict, action: np.ndarray | None = None):
        # breakpoint()
        rgb = cv2.cvtColor(obs["image"], cv2.COLOR_BGR2RGB)
        self.rgb_history.append(rgb)
        self.imu_history.append(obs["imu"])
        if action is not None:
            self.action_history.append(action.astype(np.float32))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.action_history.clear()
        self.rgb_history.clear()
        self.imu_history.clear()
        for _ in range(self.num_stack):
            self._push(obs)
            self.action_history.append(np.zeros(self.action_dim, dtype=np.float32))
        return self._stacked_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._push(obs, action)
        return self._stacked_obs(), reward, terminated, truncated, info
    


class RewardWrapper(gym.Wrapper):
    """
    Reward wrapper for RacerEnv.

    Reward signals (sourced from RacerEnv info dict):
      info["velocity"]   = {"cms": float, "ms": float, "method": str}
      info["collision"]  = {"detected": bool, "distance_cm": float, "threshold_cm": float}
      info["blocked"]    = bool  — server hard-zeroed throttle due to obstacle
    """

    # Tunable thresholds
    MIN_VELOCITY_MS   = 5.0   # m/s — below this counts as a stall
    STALL_LIMIT       = 100     # steps of stall before truncation
    PROXIMITY_WARN_CM = 30.0   # distance at which proximity penalty begins

    def __init__(self, env):
        super().__init__(env)
        self.velocity   = 0.0
        self.stop_count = 0
        self.move_count = 0
        self.last_steering=0
        self.velocity_history = deque(maxlen=100)
        self.prev_del=0
        self.distance=0
    def step(self, action):
        obs, _, _, _, info = self.env.step(action)

        terminated = False
        truncated  = False
        reward     = -.1

        # ── Unpack server-side signals ────────────────────────────────────────
        vel_info  = info.get("velocity",  {"ms": 0.0, "method": "zeroed"})
        coll_info = info.get("collision", {"detected": False,
                                           "distance_cm": 0,
                                           "threshold_cm": 15.0})
        blocked   = info.get("blocked", False)

        vel_ms   = float(vel_info["ms"])
        dist_cm  = float(coll_info["distance_cm"])
        collided = bool(coll_info["detected"])

        self.velocity = vel_ms
        self.velocity_history.append(vel_ms)
        del_dt=self.prev_del - dist_cm
        del_dt = np.nan_to_num(del_dt,nan=0,posinf=0,neginf=0)
        print("Dist Obs",dist_cm,del_dt)
        if action[1]<0.130:
            del_dt=0
            self.stop_count+=1
        else:
            self.stop_count=0
            # reward += action[1]

        self.distance+= del_dt

        reward  += np.clip(del_dt,-2,2)
        

        # ── 1. Forward motion ─────────────────────────────────────────────────
        # if not (del_dt > self.MIN_VELOCITY_MS and not blocked):
        #     self.stop_count += 1

        # ── 2. Proximity ramp penalty ─────────────────────────────────────────
        threshold_cm = float(coll_info["threshold_cm"])
        if  dist_cm < 50 :
            closeness = (self.PROXIMITY_WARN_CM - dist_cm) / (
                         self.PROXIMITY_WARN_CM - threshold_cm)
            reward   -= 2.0 * closeness
            
        # ── 3. Blocked by server obstacle guard ───────────────────────────────
        if blocked:
            reward -= 2.0

        if self.distance>100:
            reward+=100
            terminated=True
        # ── 4. Collision — hard termination ───────────────────────────────────
        if dist_cm < 40 or collided:
            reward    -= 100.0
            terminated = True

        # ── 5. Stall truncation ───────────────────────────────────────────────
        if self.stop_count > self.STALL_LIMIT:
            truncated = True
        self.last_steering=action[0]
        self.prev_del = dist_cm
        print("Dist Obs",dist_cm,del_dt,self.distance,reward,action[1])
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.velocity   = 0.0
        self.stop_count = 0
        self.move_count = 0
        self.distance=0
        obs, info = self.env.reset(**kwargs)
        coll_info = info.get("collision", {"detected": False,
                                        "distance_cm": 0.0,
                                        "threshold_cm": 15.0})

        dist_cm  = float(coll_info["distance_cm"])
        self.prev_del = dist_cm
        # print()
        return obs, info