"""
WebSocket event client for real-time robot sensor streaming.

Connects to the robot server's WebSocket endpoint (port 9001) and provides:
  - Buffered high-frequency IMU samples between camera frames (for SLAM preintegration)
  - Immediate collision event notification (~10ms vs 0-130ms TCP round-trip)
  - Continuous velocity updates at 20Hz

Usage:
    client = RobotEventClient(host="10.42.0.1", ws_port=9001)
    client.start()

    # In your step loop:
    imu_samples = client.drain_imu_buffer()  # all IMU samples since last drain

    # Register collision callback for immediate reaction:
    client.on_collision = lambda event: print("COLLISION!", event)

    # Check collision state:
    if client.collision_detected:
        handle_collision()

    client.stop()
"""

import asyncio
import json
import threading
import time

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False


class RobotEventClient:
    """
    Thread-safe WebSocket client that buffers IMU data and exposes collision events.

    Runs an asyncio event loop in a daemon thread. Auto-reconnects on disconnect.
    All public methods are thread-safe and can be called from any thread.
    """

    def __init__(self, host: str = "10.42.0.1", ws_port: int = 9001,
                 reconnect_interval: float = 2.0):
        self.host = host
        self.ws_port = ws_port
        self.reconnect_interval = reconnect_interval

        # Thread-safe state
        self._lock = threading.Lock()
        self._imu_buffer: list[dict] = []
        self._collision_state: dict = {
            "detected": False, "distance_cm": float("inf"),
            "threshold_cm": 15.0, "timestamp": 0.0,
        }
        self._velocity_state: dict = {
            "cms": 0.0, "ms": 0.0, "method": "zeroed", "timestamp": 0.0,
        }
        self._server_status: dict | None = None
        self._connected = threading.Event()

        # User callbacks (set by caller)
        self.on_collision = None  # Callable[[dict], None]
        self.on_velocity = None   # Callable[[dict], None]
        self.on_imu = None        # Callable[[dict], None]

        # Internal
        self._ws_loop: asyncio.AbstractEventLoop | None = None
        self._ws = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self):
        """Start the WebSocket client in a background thread."""
        if not WEBSOCKETS_AVAILABLE:
            print("Warning: websockets library not installed, event client disabled")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="ws-event-client", daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the WebSocket client and wait for the thread to finish."""
        self._stop_event.set()
        if self._ws_loop is not None and not self._ws_loop.is_closed():
            asyncio.run_coroutine_threadsafe(self._disconnect_ws(), self._ws_loop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    async def _disconnect_ws(self):
        if self._ws is not None:
            await self._ws.close()

    @property
    def collision_detected(self) -> bool:
        with self._lock:
            return self._collision_state["detected"]

    @property
    def collision_info(self) -> dict:
        with self._lock:
            return self._collision_state.copy()

    @property
    def velocity(self) -> dict:
        with self._lock:
            return self._velocity_state.copy()

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    def drain_imu_buffer(self) -> list[dict]:
        """Return all buffered IMU samples since the last call, clearing the buffer.

        Each entry is a dict with keys: acceleration, angular_velocity, orientation,
        magnetic, velocity, forward_velocity, timestamp (server-side).
        """
        with self._lock:
            samples = self._imu_buffer[:]
            self._imu_buffer.clear()
        return samples

    def get_imu_samples_since(self, since_ts: float) -> list[dict]:
        """Return IMU samples with timestamp > since_ts, without clearing."""
        with self._lock:
            return [s for s in self._imu_buffer if s.get("timestamp", 0) > since_ts]

    def _handle_message(self, msg: dict):
        """Process a single decoded JSON message from the WebSocket."""
        msg_type = msg.get("type")
        data = msg.get("data", {})
        ts = msg.get("ts", 0.0)

        if msg_type == "imu_sample":
            with self._lock:
                entry = {**data, "timestamp": ts}
                self._imu_buffer.append(entry)
                # Keep buffer bounded (last ~2.5s at 200Hz)
                if len(self._imu_buffer) > 500:
                    self._imu_buffer = self._imu_buffer[-400:]
            if self.on_imu is not None:
                try:
                    self.on_imu(data)
                except Exception:
                    pass

        elif msg_type == "collision_event":
            with self._lock:
                self._collision_state = {**data, "timestamp": ts}
            if self.on_collision is not None:
                try:
                    self.on_collision({**data, "timestamp": ts})
                except Exception:
                    pass

        elif msg_type == "velocity_update":
            with self._lock:
                self._velocity_state = {**data, "timestamp": ts}
            if self.on_velocity is not None:
                try:
                    self.on_velocity({**data, "timestamp": ts})
                except Exception:
                    pass

        elif msg_type == "server_status":
            with self._lock:
                self._server_status = data

    def _run(self):
        """Main loop: connect, receive messages, auto-reconnect on failure."""
        if not WEBSOCKETS_AVAILABLE:
            return

        async def _connect_and_listen():
            uri = f"ws://{self.host}:{self.ws_port}"
            async with websockets.connect(uri, max_size=2**20) as ws:
                self._ws = ws
                self._connected.set()
                async for raw_msg in ws:
                    if self._stop_event.is_set():
                        break
                    try:
                        msg = json.loads(raw_msg)
                        self._handle_message(msg)
                    except json.JSONDecodeError:
                        pass

        while not self._stop_event.is_set():
            try:
                asyncio.run(_connect_and_listen())
            except Exception:
                pass
            self._connected.clear()
            self._ws = None
            if not self._stop_event.is_set():
                time.sleep(self.reconnect_interval)


class StubEventClient:
    """No-op event client for offline/fake environments."""

    collision_detected = False
    collision_info = {
        "detected": False, "distance_cm": float("inf"),
        "threshold_cm": 15.0, "timestamp": 0.0,
    }
    velocity = {"cms": 0.0, "ms": 0.0, "method": "zeroed", "timestamp": 0.0}
    on_collision = None
    on_velocity = None
    on_imu = None

    def start(self):
        pass

    def stop(self):
        pass

    def drain_imu_buffer(self):
        return []

    def get_imu_samples_since(self, since_ts: float):
        return []

    @property
    def is_connected(self):
        return False