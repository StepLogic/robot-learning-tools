#!/usr/bin/env python3
"""
TCP Vehicle Control Server
Protocol: 4-byte little-endian length prefix + JSON payload

Client sends:  {"steering": float, "throttle": float}
Server replies: {
    "steering": float,
    "throttle": float,       # actual throttle applied (0 if blocked)
    "blocked":  bool,        # true if obstacle halted movement
    "timestamp": int,        # ms since epoch
    "observation": {
        "img":          str,   # base64-encoded JPEG string
        "img_encoding": "jpeg_base64",
        "img_shape":    [height, width, 3],
        "imu":          {...}, # full IMU snapshot
        "obstacle": {
            "detected":     bool,
            "distance_cm":  float,
            "threshold_cm": float
        },
        "velocity": {
            "cms":    float,  # fused forward velocity cm/s
            "ms":     float,  # same in m/s
            "method": str     # "fused" | "accel_only" | "zeroed"
        }
    }
}

Image encoding:
  Frames are JPEG-compressed and base64-encoded before JSON serialisation.
  This reduces a 640x480 frame from ~3.5 MB (raw pixel array) to ~30-50 KB.

  Client decoding (Python):
      import base64, numpy as np, cv2
      jpg = base64.b64decode(obs["img"])
      frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
      rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

Daemons (all background threads):
  * IMU collector      - decodes witmotion packets, updates imu_data
  * Obstacle monitor   - polls HC-SR04 at 10 Hz, sets obstacle_state
  * Velocity estimator - complementary filter at 20 Hz fusing distance + accel
"""

import asyncio
import base64
import json
import math
import socket
import struct
import threading
import time

import cv2
import numpy as np
from witmotion import IMU
import witmotion

try:
    from websockets.asyncio.server import serve, broadcast as ws_broadcast
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

try:
    from gpiozero import DistanceSensor
    GPIOZERO_AVAILABLE = True
except Exception as e:
    print(e)
    GPIOZERO_AVAILABLE = False

try:
    from adafruit_servokit import ServoKit
except ImportError:
    ServoKit = None


# ==============================================================================
# CONFIG
# ==============================================================================
HOST           = '0.0.0.0'
PORT           = 9000
IMAGE_SIZE     = (640, 480)      # (width, height)
JPEG_QUALITY   = 85              # 0-100; lower = smaller payload, more artefacts
CAMERA_DEVICE  = 0
MAX_CLIENTS    = 4

CONTINUOUS_SERVO_DRIVE  = 0
STANDARD_SERVO_STEERING = 2

# Obstacle
TRIG_PIN               = 17
ECHO_PIN               = 27
OBSTACLE_THRESHOLD_CM  = 15.0
OBSTACLE_POLL_HZ       = 10

# WebSocket
WS_HOST           = '0.0.0.0'
WS_PORT           = 9001
WS_MAX_CLIENTS    = 4

# Velocity estimator
VEL_ESTIMATOR_HZ       = 20
COMPLEMENTARY_ALPHA    = 0.48
ACCEL_DEAD_BAND_CMS2   = 5.0
DISTANCE_DEAD_BAND_CM  = 0.5
VELOCITY_DECAY         = 0.85

# Pre-built JPEG params (avoids rebuilding the list every frame)
_JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]


# ==============================================================================
# SHARED STATE
# ==============================================================================

# -- IMU -----------------------------------------------------------------------
_imu_lock = threading.Lock()
imu_data  = {
    'acceleration':     {'x': 0.0, 'y': 0.0, 'z': 0.0, 'resultant': 0.0},
    'angular_velocity': {'roll_rate': 0.0, 'pitch_rate': 0.0,
                         'yaw_rate': 0.0,  'magnitude': 0.0},
    'orientation':      {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
    'magnetic':         {'x': 0, 'y': 0, 'z': 0},
    'velocity':         {'x': 0.0, 'y': 0.0, 'z': 0.0, 'speed': 0.0},
    'forward_velocity': 0.0,
}
_last_accel_time = None

# -- Obstacle ------------------------------------------------------------------
_obstacle_lock  = threading.Lock()
_obstacle_state = {
    'detected':    False,
    'distance_cm': float('inf'),
}

# -- Velocity estimator --------------------------------------------------------
_vel_lock  = threading.Lock()
_vel_state = {
    'cms':    0.0,
    'ms':     0.0,
    'method': 'zeroed',
}
_vel_internal = {
    'last_time':     None,
    'last_distance': None,
    'velocity_cms':  0.0,
}

# -- Camera --------------------------------------------------------------------
_camera_lock = threading.Lock()

# -- WebSocket broadcast -------------------------------------------------------
_ws_loop: asyncio.AbstractEventLoop | None = None
_ws_server = None
_ws_connections: set = set()
_ws_lock = threading.Lock()


# ==============================================================================
# HARDWARE INIT
# ==============================================================================

# -- Servos --------------------------------------------------------------------
SERVO_AVAILABLE = False
kit = None
if ServoKit:
    try:
        kit = ServoKit(channels=16)
        SERVO_AVAILABLE = True
        print("✓ Servo hardware initialized")
    except Exception as e:
        print(f"✗ Servo hardware not available: {e}")
else:
    print("✗ adafruit_servokit not installed")


# -- Camera --------------------------------------------------------------------
def _initialize_camera(device_id, width, height):
    candidates = [
        (device_id,     "requested index"),
        (0,             "index 0"),
        (1,             "index 1"),
        ("/dev/video0", "video0"),
        ("/dev/video1", "video1"),
    ]
    for device, label in candidates:
        try:
            print(f"  Trying {label}: {device}")
            cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
            if not cap.isOpened():
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS,          30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
            ret, frame = cap.read()
            if ret and frame is not None:
                for _ in range(10):
                    cap.grab()
                aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"    ✓ {aw}x{ah} @ {int(cap.get(cv2.CAP_PROP_FPS))} fps")
                return cap, (aw, ah), device
            cap.release()
        except Exception as e:
            print(f"    ✗ {e}")
    raise RuntimeError("Could not initialise camera on any device")

CAMERA_AVAILABLE = False
camera           = None
camera_device    = None
actual_size      = IMAGE_SIZE
print("\nCAMERA INITIALIZATION")
try:
    camera, actual_size, camera_device = _initialize_camera(
        CAMERA_DEVICE, IMAGE_SIZE[0], IMAGE_SIZE[1])
    CAMERA_AVAILABLE = True
    print(f"✓ Camera ready on {camera_device}")
except Exception as e:
    print(f"✗ Camera init failed: {e}")

# -- IMU -----------------------------------------------------------------------
IMU_AVAILABLE = False
imu_hw = None
try:
    imu_hw = IMU()
    IMU_AVAILABLE = True
    print("✓ IMU hardware initialized")
except Exception as e:
    print(f"✗ IMU not available: {e}")

# -- Distance sensor -----------------------------------------------------------
OBSTACLE_AVAILABLE = False
_distance_sensor   = None
if GPIOZERO_AVAILABLE:
    try:
        _distance_sensor   = DistanceSensor(echo=ECHO_PIN, trigger=TRIG_PIN, max_distance=4)
        OBSTACLE_AVAILABLE = True
        print("✓ Distance sensor initialized")
    except Exception as e:
        print(f"✗ Distance sensor not available: {e}")
else:
    print("✗ gpiozero / pigpio not available")


# ==============================================================================
# DAEMON 1 — IMU COLLECTOR
# ==============================================================================
def _imu_callback(msg):
    global _last_accel_time
    try:
        if isinstance(msg, witmotion.protocol.AccelerationMessage):
            ax, ay, az = msg.a
            now = time.time()
            with _imu_lock:
                if _last_accel_time is None:
                    _last_accel_time = now
                dt = now - _last_accel_time
                _last_accel_time = now
                r = math.sqrt(ax*ax + ay*ay + az*asz)
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


def _imu_zero_velocity():
    with _imu_lock:
        imu_data['velocity']         = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'speed': 0.0}
        imu_data['forward_velocity'] = 0.0


def get_imu_snapshot():
    with _imu_lock:
        return {k: (v.copy() if isinstance(v, dict) else v) for k, v in imu_data.items()}


def start_imu_daemon():
    if not IMU_AVAILABLE:
        print("⚠ IMU daemon skipped (hardware not available)")
        return
    def _run():
        try:
            imu_hw.subscribe(_imu_callback)
        except Exception as e:
            print(f"IMU daemon error: {e}")
    t = threading.Thread(target=_run, name="imu-daemon", daemon=True)
    t.start()
    print("✓ IMU daemon started")


# ==============================================================================
# DAEMON 2 — OBSTACLE MONITOR
# ==============================================================================
def get_obstacle_snapshot():
    with _obstacle_lock:
        return _obstacle_state.copy()


def start_obstacle_daemon():
    if not OBSTACLE_AVAILABLE:
        print("⚠ Obstacle daemon skipped (hardware not available)")
        return
    interval = 1.0 / OBSTACLE_POLL_HZ
    _prev_detected = [False]  # track state transitions (mutable list for closure)

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
    t = threading.Thread(target=_run, name="obstacle-daemon", daemon=True)
    t.start()


# ==============================================================================
# DAEMON 3 — VELOCITY ESTIMATOR
# ==============================================================================
def get_velocity_snapshot():
    with _vel_lock:
        return _vel_state.copy()


def _velocity_estimator_step():
    now  = time.time()
    snap = get_imu_snapshot()
    obs  = get_obstacle_snapshot()

    accel_x_g   = snap['acceleration']['x']
    accel_cms2  = accel_x_g * 100
    dist_cm     = obs['distance_cm']
    is_obstacle = obs['detected']
    iv          = _vel_internal

    if iv['last_time'] is None:
        iv['last_time']     = now
        iv['last_distance'] = dist_cm
        return

    dt = now - iv['last_time']
    if dt <= 0:
        return

    if is_obstacle:
        iv['velocity_cms']  = 0.0
        iv['last_time']     = now
        iv['last_distance'] = dist_cm
        _imu_zero_velocity()
        with _vel_lock:
            _vel_state.update({'cms': 0.0, 'ms': 0.0, 'method': 'zeroed'})
        return

    v_accel    = iv['velocity_cms'] + accel_cms2 * dt
    delta_d    = dist_cm - iv['last_distance']
    dist_valid = dist_cm < 350.0 and iv['last_distance'] < 350.0

    if dist_valid:
        v_dist  = delta_d / dt
        v_fused = COMPLEMENTARY_ALPHA * v_accel + (1 - COMPLEMENTARY_ALPHA) * v_dist
        method  = 'fused'
    else:
        v_fused = v_accel
        method  = 'accel_only'

    stationary = (
        abs(accel_cms2) < ACCEL_DEAD_BAND_CMS2 and
        abs(delta_d)    < DISTANCE_DEAD_BAND_CM
    )
    if stationary:
        v_fused *= VELOCITY_DECAY
        method   = 'zeroed'

    iv['velocity_cms']  = v_fused
    iv['last_time']     = now
    iv['last_distance'] = dist_cm

    with _vel_lock:
        _vel_state.update({
            'cms':    round(v_fused, 3),
            'ms':     round(v_fused / 100.0, 4),
            'method': method,
        })

    ws_publish("velocity_update", {
        'cms':    round(v_fused, 3),
        'ms':     round(v_fused / 100.0, 4),
        'method': method,
    })


def start_velocity_daemon():
    interval = 1.0 / VEL_ESTIMATOR_HZ
    def _run():
        print("✓ Velocity estimator daemon started")
        while True:
            try:
                _velocity_estimator_step()
            except Exception as e:
                print(f"Velocity daemon error: {e}")
            time.sleep(interval)
    t = threading.Thread(target=_run, name="velocity-daemon", daemon=True)
    t.start()


# ==============================================================================
# WEBSOCKET BROADCAST
# ==============================================================================

def ws_publish(event_type: str, data: dict):
    """Thread-safe: schedule a WebSocket broadcast from any daemon thread."""
    with _ws_lock:
        if _ws_loop is None or not _ws_connections:
            return
    msg = json.dumps({"type": event_type, "ts": time.time(), "data": data},
                     separators=(',', ':'))
    _ws_loop.call_soon_threadsafe(_ws_broadcast_sync, msg)


def _ws_broadcast_sync(msg: str):
    """Called on the WS event loop thread. Broadcasts to all connected clients."""
    with _ws_lock:
        conns = list(_ws_connections)
    if conns:
        asyncio.run_coroutine_threadsafe(
            _ws_broadcast_async(conns, msg), _ws_loop)


async def _ws_broadcast_async(conns, msg: str):
    """Async broadcast that handles individual send failures."""
    for ws in conns:
        try:
            await ws.send(msg)
        except Exception:
            pass


async def _ws_handler(websocket):
    """Handle a single WebSocket client connection."""
    with _ws_lock:
        _ws_connections.add(websocket)
    addr = websocket.remote_address if hasattr(websocket, 'remote_address') else 'unknown'
    print(f"  + WS client connected: {addr}")
    try:
        # Send initial status
        status_msg = json.dumps({
            "type": "server_status",
            "ts": time.time(),
            "data": {
                "imu_available": IMU_AVAILABLE,
                "camera_available": CAMERA_AVAILABLE,
                "obstacle_available": OBSTACLE_AVAILABLE,
                "servo_available": SERVO_AVAILABLE,
                "image_size": list(IMAGE_SIZE),
            }
        }, separators=(',', ':'))
        await websocket.send(status_msg)
        await websocket.wait_closed()
    except Exception:
        pass
    finally:
        with _ws_lock:
            _ws_connections.discard(websocket)
        print(f"  - WS client disconnected: {addr}")


async def _ws_server_main():
    """Main coroutine for the WebSocket server."""
    global _ws_server
    async with serve(_ws_handler, WS_HOST, WS_PORT, max_size=2**20) as server:
        _ws_server = server
        print(f"  WebSocket server on ws://{WS_HOST}:{WS_PORT}")
        await asyncio.Future()  # run forever


def start_ws_daemon():
    """Start WebSocket server in its own thread with its own asyncio loop."""
    global _ws_loop

    if not WEBSOCKETS_AVAILABLE:
        print("⚠ WebSocket daemon skipped (websockets library not installed)")
        print("  Install with: pip install websockets")
        return

    def _run():
        global _ws_loop
        _ws_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_ws_loop)
        try:
            _ws_loop.run_until_complete(_ws_server_main())
        except KeyboardInterrupt:
            pass
        finally:
            _ws_loop.close()

    t = threading.Thread(target=_run, name="ws-daemon", daemon=True)
    t.start()
    # Wait until _ws_loop is set
    for _ in range(100):
        if _ws_loop is not None:
            break
        time.sleep(0.01)
    print("✓ WebSocket daemon started")


# ==============================================================================
# CAMERA CAPTURE — JPEG + BASE64
# ==============================================================================

# Blank fallback: black frame pre-encoded as base64 JPEG
_blank_bgr    = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.uint8)
_, _blank_jpg = cv2.imencode('.jpg', _blank_bgr, _JPEG_PARAMS)
_blank_b64    = base64.b64encode(_blank_jpg.tobytes()).decode('ascii')


def capture_frame() -> str:
    """
    Capture one frame and return it as a base64-encoded JPEG string.

    Pipeline:
      1. grab + retrieve from camera (BGR native)
      2. resize if camera returned wrong resolution
      3. JPEG-compress at JPEG_QUALITY
      4. base64-encode -> ASCII string safe for JSON embedding

    Approximate payload sizes at JPEG_QUALITY=85:
      160x120  ->   ~5-8 KB
      640x480  ->  ~30-50 KB   (was ~3.5 MB as raw pixel array)
    """
    if not CAMERA_AVAILABLE:
        return _blank_b64
    try:
        with _camera_lock:
            camera.grab()
            ret, frame = camera.retrieve()
        if not ret or frame is None:
            return _blank_b64
        if frame.shape[1] != IMAGE_SIZE[0] or frame.shape[0] != IMAGE_SIZE[1]:
            frame = cv2.resize(frame, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        ok, jpg_buf = cv2.imencode('.jpg', frame, _JPEG_PARAMS)
        if not ok:
            return _blank_b64
        return base64.b64encode(jpg_buf.tobytes()).decode('ascii')
    except Exception as e:
        print(f"Camera error: {e}")
        return _blank_b64


# ==============================================================================
# SERVO CONTROL
# ==============================================================================
def apply_action(steering: float, throttle: float) -> tuple[bool, float]:
    obs     = get_obstacle_snapshot()
    blocked = obs['detected']
    dist_cm = obs['distance_cm']

    if not SERVO_AVAILABLE:
        return blocked, dist_cm

    try:
        angle = max(0, min(180, int(90 + 90 * steering)))
        kit.servo[STANDARD_SERVO_STEERING].angle = angle

        if blocked:
            kit.continuous_servo[CONTINUOUS_SERVO_DRIVE].throttle = 0
            print(f"BLOCKED {dist_cm:.1f} cm | "
                  f"steering={steering:.3f} -> {angle}deg  throttle=0 (forced)")
        else:
            t = max(-1.0, min(1.0, throttle))
            kit.continuous_servo[CONTINUOUS_SERVO_DRIVE].throttle = t
            vel = get_velocity_snapshot()
            print(f"  steering={steering:.3f} -> {angle}deg  "
                  f"throttle={t:.3f}  "
                  f"v={vel['cms']:.1f} cm/s [{vel['method']}]  "
                  f"dist={dist_cm:.1f} cm")
    except Exception as e:
        print(f"Servo error: {e}")

    return blocked, dist_cm


# ==============================================================================
# TCP FRAMING
# ==============================================================================
_HEADER = struct.Struct('<I')

def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    buf      = bytearray(n)
    view     = memoryview(buf)
    received = 0
    while received < n:
        chunk = sock.recv_into(view[received:], n - received)
        if not chunk:
            raise ConnectionResetError("Client disconnected")
        received += chunk
    return bytes(buf)

def send_message(sock: socket.socket, payload: bytes):
    sock.sendall(_HEADER.pack(len(payload)) + payload)

def recv_message(sock: socket.socket) -> bytes:
    length = _HEADER.unpack(_recv_exactly(sock, 4))[0]
    if length == 0 or length > 10_000_000:
        raise ValueError(f"Invalid message length: {length}")
    return _recv_exactly(sock, length)


# ==============================================================================
# PER-CLIENT HANDLER
# ==============================================================================
def handle_client(conn: socket.socket, addr):
    print(f"  + Client connected: {addr}")
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    conn.settimeout(30.0)

    try:
        while True:
            raw  = recv_message(conn)
            data = json.loads(raw)

            steering = float(data['steering'])
            throttle = float(data['throttle'])

            if not (-1.0 <= steering <= 1.0 and -1.0 <= throttle <= 1.0):
                send_message(conn, json.dumps({'error': 'values out of range'}).encode())
                continue

            blocked, dist_cm = apply_action(steering, throttle)

            imu_snap = get_imu_snapshot() if IMU_AVAILABLE else imu_data.copy()
            vel_snap = get_velocity_snapshot()
            obs_snap = get_obstacle_snapshot()

            response = {
                'steering':  steering,
                'throttle':  0.0 if blocked else throttle,
                'blocked':   blocked,
                'timestamp': int(time.time() * 1000),
                'observation': {
                    'img':          capture_frame(),      # base64 JPEG string
                    'img_encoding': 'jpeg_base64',
                    'img_shape':    [IMAGE_SIZE[1], IMAGE_SIZE[0], 3],
                    'imu': imu_snap,
                    'obstacle': {
                        'detected':     obs_snap['detected'],
                        'distance_cm':  obs_snap['distance_cm'],
                        'threshold_cm': OBSTACLE_THRESHOLD_CM,
                    },
                    'velocity': vel_snap,
                },
            }
            send_message(conn, json.dumps(response, separators=(',', ':')).encode())

    except (ConnectionResetError, BrokenPipeError, TimeoutError):
        print(f"  - Client disconnected: {addr}")
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        print(f"  ! Bad message from {addr}: {e}")
        try:
            send_message(conn, json.dumps({'error': str(e)}).encode())
        except Exception:
            pass
    except Exception as e:
        print(f"  ! Unexpected error from {addr}: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ==============================================================================
# CLEANUP
# ==============================================================================
def cleanup():
    print("\nCleaning up...")
    if _ws_loop is not None and _ws_server is not None:
        try:
            _ws_loop.call_soon_threadsafe(_ws_server.close)
            print("✓ WebSocket server closed")
        except Exception as e:
            print(f"✗ WebSocket cleanup: {e}")
    if SERVO_AVAILABLE:
        try:
            kit.continuous_servo[CONTINUOUS_SERVO_DRIVE].throttle = 0
            kit.servo[STANDARD_SERVO_STEERING].angle = 90
            print("✓ Servos stopped and centred")
        except Exception as e:
            print(f"✗ Servo cleanup: {e}")
    if CAMERA_AVAILABLE:
        try:
            camera.release()
            print("✓ Camera released")
        except Exception as e:
            print(f"✗ Camera cleanup: {e}")


# ==============================================================================
# MAIN
# ==============================================================================
def run_server():
    print("\nSTARTING DAEMONS")
    start_imu_daemon()
    start_obstacle_daemon()
    start_velocity_daemon()
    start_ws_daemon()
    time.sleep(0.5)

    if SERVO_AVAILABLE:
        kit.servo[STANDARD_SERVO_STEERING].angle = 90
        kit.continuous_servo[CONTINUOUS_SERVO_DRIVE].throttle = 0

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(MAX_CLIENTS)

    print(f"\n{'='*62}")
    print(f"  TCP Vehicle Server")
    print(f"{'='*62}")
    print(f"  Address  : {HOST}:{PORT}")
    print(f"  Protocol : 4-byte LE length prefix + JSON")
    print(f"  Image    : {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}  JPEG q={JPEG_QUALITY}  (~30-50 KB/frame)")
    print(f"  Hardware :")
    print(f"    Servo    : {'✓ ENABLED' if SERVO_AVAILABLE    else '✗ DISABLED'}")
    print(f"    Camera   : {'✓ ENABLED' if CAMERA_AVAILABLE   else '✗ DISABLED'}"
          + (f"  ({camera_device})" if CAMERA_AVAILABLE else ""))
    print(f"    IMU      : {'✓ ENABLED' if IMU_AVAILABLE      else '✗ DISABLED'}")
    print(f"    Obstacle : {'✓ ENABLED' if OBSTACLE_AVAILABLE else '✗ DISABLED'}"
          + (f"  (< {OBSTACLE_THRESHOLD_CM} cm -> STOP)" if OBSTACLE_AVAILABLE else ""))
    print(f"  Daemons  :")
    print(f"    imu-daemon       @ hardware rate")
    print(f"    obstacle-daemon  @ {OBSTACLE_POLL_HZ} Hz")
    print(f"    velocity-daemon  @ {VEL_ESTIMATOR_HZ} Hz  (a={COMPLEMENTARY_ALPHA})")
    print(f"    ws-daemon        @ ws://{WS_HOST}:{WS_PORT}"
          + ("  (events: imu_sample, collision_event, velocity_update)" if WEBSOCKETS_AVAILABLE else "  ✗ DISABLED"))
    print(f"{'='*62}\n")

    try:
        while True:
            conn, addr = server.accept()
            threading.Thread(
                target=handle_client,
                args=(conn, addr),
                daemon=True,
            ).start()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cleanup()
        server.close()


if __name__ == '__main__':
    run_server()
