import json
import socket
import struct
import time
import cv2
import numpy as np
import base64, numpy as np, cv2

HOST = '10.42.0.1'   # ← change to your robot's IP
PORT = 9000

_HEADER = struct.Struct('<I')

def undistort(img):
    w, h = img.shape[1], img.shape[0]
    fish_K = np.array([[60, 0, 80], [0, 60, 60], [0, 0, 1]], dtype=np.float32)
    fish_D = np.array([[-0.0018], [0], [0], [0]], dtype=np.float32)
    new_fish_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(fish_K, fish_D, (w, h), np.eye(3), 0.75)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(fish_K, fish_D, np.eye(3), new_fish_K, (w, h), cv2.CV_16SC2)
    return cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

def send_message(sock, payload: bytes):
    sock.sendall(_HEADER.pack(len(payload)) + payload)


def recv_exactly(sock, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        chunk = sock.recv_into(view[got:], n - got)
        if not chunk:
            raise ConnectionResetError("Server closed connection")
        got += chunk
    return bytes(buf)


def recv_message(sock) -> bytes:
    header = recv_exactly(sock, 4)
    length = _HEADER.unpack(header)[0]
    return recv_exactly(sock, length)


def control(sock, steering: float, throttle: float) -> dict:
    """Send one control command and return the observation."""
    cmd = json.dumps({'steering': steering, 'throttle': throttle},
                     separators=(',', ':')).encode()
    send_message(sock, cmd)
    raw = recv_message(sock)
    return json.loads(raw)


def main():
    with socket.create_connection((HOST, PORT), timeout=5) as sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"Connected to {HOST}:{PORT}")

        commands = [
            # (1.0, 0.0),   # straight, slow forward
            # (0.0,  0.0),   # straight, slow forward
            (-1.0,  0.0),   # straight, slow forward
        ]

        for steering, throttle in commands:
            (steering)
            t0 = time.perf_counter()
            obs = control(sock, steering, throttle)
            latency = (time.perf_counter() - t0) * 1000
            # print(obs["observation"]["img"].shape)
            # img = np.array(obs['observation']['img'], dtype=np.uint8)

            img = base64.b64decode(obs['observation']['img'])
            img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
            # img = cv2.cvtColor(im
            # print(img.shape)

            imu = obs['observation']['imu']
            cv2.imwrite(f"img_s{steering:+.2f}_t{throttle:+.2f}.jpg", img)


            print(f"s={steering:+.2f} t={throttle:+.2f}  "
                  f"latency={latency:.1f}ms  "
                  f"img={img.shape}  "
                  f"roll={imu['acceleration']}°")
            time.sleep(0.1)
main()