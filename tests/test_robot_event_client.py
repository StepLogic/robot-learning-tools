"""
Tests for robot_event_client.py — WebSocket event client.

Uses an in-process mock WebSocket server to verify:
  - IMU buffer drain and clear behavior
  - Collision event callback
  - Buffer size limit
  - Velocity and status messages
  - StubEventClient no-op behavior
"""

import json
import time
import threading
import pytest

from robot_event_client import RobotEventClient, StubEventClient, WEBSOCKETS_AVAILABLE


# ─── Helpers ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ws_port():
    """Find an available port for the test WebSocket server."""
    import socket as sock_mod
    with sock_mod.socket(sock_mod.AF_INET, sock_mod.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def mock_server(ws_port):
    """Start a mock WebSocket server that echoes back specific events."""
    if not WEBSOCKETS_AVAILABLE:
        pytest.skip("websockets library not installed")
        return

    import asyncio
    from websockets.asyncio.server import serve

    # Track received messages for verification
    received = []

    async def handler(websocket):
        # Send server_status on connect
        await websocket.send(json.dumps({
            "type": "server_status",
            "ts": time.time(),
            "data": {"imu_available": True, "camera_available": True,
                     "obstacle_available": True, "servo_available": True,
                     "image_size": [640, 480]},
        }))

        # Send a burst of IMU samples
        for i in range(10):
            msg = json.dumps({
                "type": "imu_sample",
                "ts": time.time(),
                "data": {
                    "acceleration": {"x": 0.1 * i, "y": 0.0, "z": 9.81, "resultant": 9.81},
                    "angular_velocity": {"roll_rate": 0.0, "pitch_rate": 0.0,
                                         "yaw_rate": 0.0, "magnitude": 0.0},
                    "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": float(i)},
                    "magnetic": {"x": 0, "y": 0, "z": 0},
                    "velocity": {"x": 0.0, "y": 0.0, "z": 0.0, "speed": 0.0},
                    "forward_velocity": 0.0,
                },
            })
            await websocket.send(msg)

        # Send a collision event
        await websocket.send(json.dumps({
            "type": "collision_event",
            "ts": time.time(),
            "data": {"detected": True, "distance_cm": 12.3, "threshold_cm": 15.0},
        }))

        # Send a velocity update
        await websocket.send(json.dumps({
            "type": "velocity_update",
            "ts": time.time(),
            "data": {"cms": 15.0, "ms": 0.15, "method": "fused"},
        }))

        # Keep connection open briefly to allow client to receive all messages
        await asyncio.sleep(0.5)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def run_server():
        async with serve(handler, "127.0.0.1", ws_port, max_size=2**20) as server:
            await asyncio.Future()  # run until cancelled

    def _run():
        loop.run_until_complete(run_server())

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    time.sleep(0.2)  # Let server start
    yield ws_port
    # Cleanup: cancel the future to stop the server
    for task in asyncio.all_tasks(loop):
        task.cancel()
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=2.0)


# ─── Tests ─────────────────────────────────────────────────────────────────────

class TestRobotEventClient:
    """Integration tests with a mock WebSocket server."""

    def test_connect_and_receive_status(self, mock_server):
        """Client should connect and receive server_status."""
        if not WEBSOCKETS_AVAILABLE:
            pytest.skip("websockets library not installed")
        client = RobotEventClient(host="127.0.0.1", ws_port=mock_server,
                                  reconnect_interval=0.5)
        client.start()
        time.sleep(0.3)
        assert client.is_connected, "Client should be connected after start"
        client.stop()

    def test_drain_imu_buffer(self, mock_server):
        """drain_imu_buffer should return buffered samples and clear the buffer."""
        if not WEBSOCKETS_AVAILABLE:
            pytest.skip("websockets library not installed")
        client = RobotEventClient(host="127.0.0.1", ws_port=mock_server,
                                  reconnect_interval=0.5)
        client.start()
        time.sleep(0.5)

        samples = client.drain_imu_buffer()
        assert len(samples) == 10, f"Expected 10 IMU samples, got {len(samples)}"

        # Each sample should have required keys
        for s in samples:
            assert "acceleration" in s, "Sample missing acceleration"
            assert "angular_velocity" in s, "Sample missing angular_velocity"
            assert "timestamp" in s, "Sample missing timestamp"

        # Buffer should be cleared after drain
        second_drain = client.drain_imu_buffer()
        assert len(second_drain) == 0, "Buffer should be empty after drain"

        client.stop()

    def test_collision_detected(self, mock_server):
        """collision_detected and collision_info should reflect collision events."""
        if not WEBSOCKETS_AVAILABLE:
            pytest.skip("websockets library not installed")
        client = RobotEventClient(host="127.0.0.1", ws_port=mock_server,
                                  reconnect_interval=0.5)
        client.start()
        time.sleep(0.5)

        assert client.collision_detected, "collision_detected should be True"
        info = client.collision_info
        assert info["detected"] is True
        assert info["distance_cm"] == 12.3
        assert info["threshold_cm"] == 15.0
        assert info["timestamp"] > 0

        client.stop()

    def test_collision_callback(self, mock_server):
        """on_collision callback should fire on collision_event."""
        if not WEBSOCKETS_AVAILABLE:
            pytest.skip("websockets library not installed")
        client = RobotEventClient(host="127.0.0.1", ws_port=mock_server,
                                  reconnect_interval=0.5)
        received_events = []

        def callback(event):
            received_events.append(event)

        client.on_collision = callback
        client.start()
        time.sleep(0.5)

        assert len(received_events) >= 1, "Collision callback should have fired"
        assert received_events[0]["detected"] is True

        client.stop()

    def test_velocity_update(self, mock_server):
        """velocity property should reflect velocity_update messages."""
        if not WEBSOCKETS_AVAILABLE:
            pytest.skip("websockets library not installed")
        client = RobotEventClient(host="127.0.0.1", ws_port=mock_server,
                                  reconnect_interval=0.5)
        client.start()
        time.sleep(0.5)

        vel = client.velocity
        assert vel["cms"] == 15.0
        assert vel["ms"] == 0.15
        assert vel["method"] == "fused"

        client.stop()

    def test_buffer_size_limit(self):
        """IMU buffer should not exceed 500 entries."""
        if not WEBSOCKETS_AVAILABLE:
            pytest.skip("websockets library not installed")
        import asyncio
        from websockets.asyncio.server import serve

        # Find a free port
        import socket as sock_mod
        with sock_mod.socket(sock_mod.AF_INET, sock_mod.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        sent_count = 0

        async def handler(websocket):
            nonlocal sent_count
            # Send 600 IMU samples (over the 500 limit)
            for i in range(600):
                msg = json.dumps({
                    "type": "imu_sample",
                    "ts": time.time(),
                    "data": {
                        "acceleration": {"x": 0.0, "y": 0.0, "z": 9.81, "resultant": 9.81},
                        "angular_velocity": {"roll_rate": 0.0, "pitch_rate": 0.0,
                                             "yaw_rate": 0.0, "magnitude": 0.0},
                        "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
                        "magnetic": {"x": 0, "y": 0, "z": 0},
                        "velocity": {"x": 0.0, "y": 0.0, "z": 0.0, "speed": 0.0},
                        "forward_velocity": 0.0,
                    },
                })
                await websocket.send(msg)
                sent_count += 1
            await asyncio.sleep(0.3)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def run_server():
            async with serve(handler, "127.0.0.1", port, max_size=2**20) as server:
                await asyncio.Future()

        def _run():
            loop.run_until_complete(run_server())

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        time.sleep(0.2)

        client = RobotEventClient(host="127.0.0.1", ws_port=port,
                                  reconnect_interval=0.5)
        client.start()
        time.sleep(0.5)

        samples = client.drain_imu_buffer()
        # Buffer should be capped at 500
        assert len(samples) <= 500, f"Buffer should be capped at 500, got {len(samples)}"

        client.stop()
        for task in asyncio.all_tasks(loop):
            task.cancel()
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=2.0)

    def test_reconnect(self):
        """Client should auto-reconnect when server disconnects."""
        if not WEBSOCKETS_AVAILABLE:
            pytest.skip("websockets library not installed")
        import asyncio
        from websockets.asyncio.server import serve

        import socket as sock_mod
        with sock_mod.socket(sock_mod.AF_INET, sock_mod.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Server that accepts one connection then stops
        async def short_lived_server():
            async with serve(lambda ws: None, "127.0.0.1", port, max_size=2**20) as server:
                await asyncio.sleep(0.3)

        def _run():
            loop.run_until_complete(short_lived_server())

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        time.sleep(0.1)

        client = RobotEventClient(host="127.0.0.1", ws_port=port,
                                  reconnect_interval=0.3)
        client.start()
        time.sleep(0.5)

        # Server already stopped, client should have tried to reconnect
        # The connection flag may be False, but the client shouldn't crash
        assert client._thread is not None and client._thread.is_alive(), \
            "Client thread should still be alive after disconnect"

        client.stop()
        t.join(timeout=2.0)


class TestStubEventClient:
    """Tests for StubEventClient (offline/fallback)."""

    def test_stub_properties(self):
        client = StubEventClient()
        assert client.collision_detected is False
        assert client.collision_info["detected"] is False
        assert client.velocity["cms"] == 0.0
        assert client.is_connected is False

    def test_stub_methods(self):
        client = StubEventClient()
        assert client.drain_imu_buffer() == []
        assert client.get_imu_samples_since(0) == []
        # These should not raise
        client.start()
        client.stop()
        assert client.on_collision is None
        assert client.on_velocity is None
        assert client.on_imu is None


class TestModuleLevel:
    """Module-level tests."""

    def test_import_available(self):
        """WEBSOCKETS_AVAILABLE should be a bool."""
        assert isinstance(WEBSOCKETS_AVAILABLE, bool)

    def test_stub_collision_info_structure(self):
        """StubEventClient collision_info should match expected structure."""
        client = StubEventClient()
        info = client.collision_info
        assert "detected" in info
        assert "distance_cm" in info
        assert "threshold_cm" in info
        assert "timestamp" in info
        assert info["detected"] is False
        assert info["distance_cm"] == float("inf")