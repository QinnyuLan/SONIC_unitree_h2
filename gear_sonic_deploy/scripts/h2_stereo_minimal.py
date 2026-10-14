#!/usr/bin/env python3
"""Minimal Unitree H2 stereo viewer - single thread, no frills.

What the robot sends (doc "双目数据流接口"):
  * left RGB : RTP/H.264 pushed to this PC, UDP port 5004
  * right RGB: RTP/H.264 pushed to this PC, UDP port 5006
  * depth    : raw Y16 (uint16, millimeters), TCP server on the robot :5000,
               each frame = 24-byte frame header + 8-byte data header + payload

ffmpeg does the RTP/H.264 decoding; we just read raw BGR frames from its
stdout. Depth is parsed directly. Press Q or Ctrl+C to quit.
"""

import select
import socket
import struct
import subprocess

import cv2
import numpy as np

ROBOT_HOST = "192.168.123.161"
WIDTH, HEIGHT = 544, 448
RGB_BYTES = WIDTH * HEIGHT * 3

FRAME_HEADER = struct.Struct("<IIIQII")  # fourcc, seq, reserved, timestamp, w, h
DATA_HEADER = struct.Struct("<II")       # fourcc, payload size


def open_rgb_stream(port: int) -> subprocess.Popen:
    """Decode one RTP/H.264 UDP stream into raw BGR frames on stdout."""
    sdp = (
        "v=0\no=- 0 0 IN IP4 0.0.0.0\ns=h2\nc=IN IP4 0.0.0.0\nt=0 0\n"
        f"m=video {port} RTP/AVP 96\na=rtpmap:96 H264/90000\na=recvonly\n"
    )
    proc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-protocol_whitelist", "file,pipe,udp,rtp",
         "-fflags", "nobuffer", "-flags", "low_delay",
         "-analyzeduration", "4000000", "-probesize", "4000000",
         "-f", "sdp", "-i", "pipe:0",
         "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    proc.stdin.write(sdp.encode())
    proc.stdin.close()
    return proc


def read_exact(stream, size: int) -> bytes:
    # BufferedReader.read(n) blocks until n bytes; only EOF makes it short.
    data = stream.read(size)
    if len(data) != size:
        raise ConnectionError("stream closed")
    return data


def read_rgb_frame(proc: subprocess.Popen) -> np.ndarray:
    return np.frombuffer(read_exact(proc.stdout, RGB_BYTES), np.uint8).reshape(HEIGHT, WIDTH, 3)


def read_depth_frame(stream) -> np.ndarray:
    _fourcc, _seq, _res, _ts, w, h = FRAME_HEADER.unpack(read_exact(stream, FRAME_HEADER.size))
    _dfourcc, size = DATA_HEADER.unpack(read_exact(stream, DATA_HEADER.size))
    return np.frombuffer(read_exact(stream, size), dtype="<u2").reshape(h, w)


def colorize(depth: np.ndarray) -> np.ndarray:
    norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(255 - norm, cv2.COLORMAP_TURBO)


def main() -> None:
    left, right = open_rgb_stream(5004), open_rgb_stream(5006)
    try:
        depth_sock = socket.create_connection((ROBOT_HOST, 5000), timeout=5)
        depth_stream = depth_sock.makefile("rb")
    except OSError:  # depth service offline -> RGB only
        depth_sock = depth_stream = None
    depth_color = None
    try:
        while True:
            stereo = np.hstack((read_rgb_frame(left), read_rgb_frame(right)))
            if depth_sock and select.select([depth_sock], [], [], 0)[0]:
                depth_color = colorize(read_depth_frame(depth_stream))
            cv2.imshow("H2 stereo (left | right)", stereo)
            if depth_color is not None:
                cv2.imshow("H2 depth", depth_color)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except (ConnectionError, KeyboardInterrupt):
        pass
    finally:
        if depth_sock:
            depth_sock.close()
        for proc in (left, right):
            # ffmpeg blocked on a full stdout pipe ignores SIGTERM; give it
            # a moment, then kill.
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
