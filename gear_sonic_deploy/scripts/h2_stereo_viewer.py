#!/usr/bin/env python3
"""Play Unitree H2 stereo RGB and depth streams with OpenCV.

PC1 ``stereo patch`` must target this computer's eno1 IPv4 address.
RGB is available as the original UDP/5002 stream, separate left/right
UDP/5004 and UDP/5006 streams, or all three. Depth is Y16 over TCP/5000.
"""

from __future__ import annotations

import argparse
import logging
import queue
import signal
import socket
import struct
import threading
import time

import cv2
import numpy as np

LOGGER = logging.getLogger("h2_stereo_viewer")

ROBOT_HOST = "192.168.123.161"
ORIGINAL_RGB_PORT = 5002
LEFT_RGB_PORT = 5004
RIGHT_RGB_PORT = 5006
DEPTH_PORT = 5000
DEPTH_TIMEOUT = 5.0
RETRY_DELAY = 1.0

DEPTH_FRAME_HEADER = struct.Struct("<IQQIIII")
Y16_FOURCC = int.from_bytes(b"Y16 ", "big")
INVALID_DEPTH = np.iinfo(np.uint16).max

ORIGINAL_WINDOW = "Unitree H2 Original Stereo RGB"
LEFT_WINDOW = "Unitree H2 Left RGB"
RIGHT_WINDOW = "Unitree H2 Right RGB"
DEPTH_WINDOW = "Unitree H2 Depth"
ORIGINAL_WINDOW_SIZE = (960, 540)
EYE_WINDOW_SIZE = (544, 448)
WINDOW_STEP_X = 980
WINDOW_STEP_Y = 560


def put_latest(frames: queue.Queue[np.ndarray], frame: np.ndarray) -> None:
    try:
        frames.put_nowait(frame)
        return
    except queue.Full:
        pass
    try:
        frames.get_nowait()
    except queue.Empty:
        pass
    frames.put_nowait(frame)


def receive_exact(sock: socket.socket, size: int, stop: threading.Event) -> bytes:
    data = bytearray()
    last_data_at = time.monotonic()
    while len(data) < size:
        if stop.is_set():
            raise InterruptedError
        try:
            chunk = sock.recv(size - len(data))
        except socket.timeout:
            if time.monotonic() - last_data_at >= DEPTH_TIMEOUT:
                raise TimeoutError("depth stream stalled")
            continue
        if not chunk:
            raise ConnectionError("depth stream closed")
        data.extend(chunk)
        last_data_at = time.monotonic()
    return bytes(data)


def depth_loop(stop: threading.Event, frames: queue.Queue[np.ndarray]) -> None:
    while not stop.is_set():
        try:
            LOGGER.info("Connecting to depth TCP %s:%d", ROBOT_HOST, DEPTH_PORT)
            with socket.create_connection((ROBOT_HOST, DEPTH_PORT), timeout=3.0) as sock:
                sock.settimeout(0.5)
                announced = False
                while not stop.is_set():
                    header = receive_exact(sock, DEPTH_FRAME_HEADER.size, stop)
                    magic, seq, _timestamp, width, height, pixel_format, size = (
                        DEPTH_FRAME_HEADER.unpack(header)
                    )
                    expected = width * height * 2
                    if magic != Y16_FOURCC or pixel_format != Y16_FOURCC:
                        raise RuntimeError("unexpected depth pixel format")
                    if not (0 < width <= 8192 and 0 < height <= 8192) or size != expected:
                        raise RuntimeError(f"invalid depth frame: {width}x{height}, {size} bytes")
                    payload = receive_exact(sock, size, stop)
                    depth = np.frombuffer(payload, dtype="<u2").reshape(height, width)
                    if not announced:
                        LOGGER.info(
                            "Depth frame: %dx%d %s, sequence=%d",
                            width,
                            height,
                            depth.dtype,
                            seq,
                        )
                        announced = True
                    put_latest(frames, depth)
        except InterruptedError:
            return
        except Exception as exc:
            if not stop.is_set():
                LOGGER.warning("Depth unavailable: %s", exc)
                stop.wait(RETRY_DELAY)


def rgb_pipeline(port: int) -> str:
    return (
        f"udpsrc port={port} buffer-size=2097152 "
        'caps="application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000" ! '
        "rtpjitterbuffer latency=50 drop-on-latency=true ! "
        "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
        "video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false"
    )


def rgb_loop(
    label: str,
    port: int,
    stop: threading.Event,
    frames: queue.Queue[np.ndarray],
) -> None:
    while not stop.is_set():
        capture = cv2.VideoCapture(rgb_pipeline(port), cv2.CAP_GSTREAMER)
        announced = False
        try:
            if not capture.isOpened():
                raise ConnectionError("GStreamer pipeline did not open")
            LOGGER.info("Reading %s RGB on UDP %d with GStreamer", label, port)
            while not stop.is_set():
                ok, frame = capture.read()
                if not ok:
                    raise ConnectionError("GStreamer frame read failed")
                if not announced:
                    LOGGER.info(
                        "%s RGB frame: %dx%d %s",
                        label,
                        frame.shape[1],
                        frame.shape[0],
                        frame.dtype,
                    )
                    announced = True
                put_latest(frames, frame)
        except Exception as exc:
            if not stop.is_set():
                LOGGER.warning("%s RGB unavailable: %s", label, exc)
                stop.wait(RETRY_DELAY)
        finally:
            capture.release()


def colorize_depth(depth: np.ndarray, minimum: int, maximum: int) -> np.ndarray:
    valid = (depth != 0) & (depth != INVALID_DEPTH)
    clipped = np.clip(depth, minimum, maximum)
    normalized = (
        (clipped.astype(np.float32) - minimum) * (255.0 / (maximum - minimum))
    ).astype(np.uint8)
    colored = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rgb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable RGB streams selected by --rgb-mode",
    )
    parser.add_argument(
        "--rgb-mode",
        choices=("original", "eyes", "all"),
        default="eyes",
        help="original=UDP/5002, eyes=UDP/5004+5006, all=all RGB streams",
    )
    parser.add_argument(
        "--depth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable the Y16 depth stream",
    )
    parser.add_argument("--depth-min-mm", type=int, default=300)
    parser.add_argument("--depth-max-mm", type=int, default=5000)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="stop after every enabled stream processes N frames; 0 runs forever",
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    args = parser.parse_args()
    if not args.rgb and not args.depth:
        parser.error("at least one of --rgb or --depth must be enabled")
    if args.depth_min_mm >= args.depth_max_mm:
        parser.error("--depth-min-mm must be less than --depth-max-mm")
    if args.max_frames < 0:
        parser.error("--max-frames cannot be negative")
    return args


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(threadName)s: %(message)s",
    )

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    views: list[tuple[str, str, queue.Queue[np.ndarray], bool, tuple[int, int]]] = []
    workers: list[threading.Thread] = []
    if args.rgb:
        rgb_sources: list[tuple[str, str, int, tuple[int, int]]] = []
        if args.rgb_mode in ("original", "all"):
            rgb_sources.append(
                ("original", ORIGINAL_WINDOW, ORIGINAL_RGB_PORT, ORIGINAL_WINDOW_SIZE)
            )
        if args.rgb_mode in ("eyes", "all"):
            rgb_sources.extend(
                (
                    ("left", LEFT_WINDOW, LEFT_RGB_PORT, EYE_WINDOW_SIZE),
                    ("right", RIGHT_WINDOW, RIGHT_RGB_PORT, EYE_WINDOW_SIZE),
                )
            )
        for label, window, port, window_size in rgb_sources:
            frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
            views.append((label, window, frames, False, window_size))
            workers.append(
                threading.Thread(
                    target=rgb_loop,
                    args=(label, port, stop, frames),
                    name=f"h2-{label}-rgb",
                    daemon=True,
                )
            )
    if args.depth:
        depth_frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        views.append(("depth", DEPTH_WINDOW, depth_frames, True, EYE_WINDOW_SIZE))
        workers.append(
            threading.Thread(
                target=depth_loop,
                args=(stop, depth_frames),
                name="h2-depth",
                daemon=True,
            )
        )
    for worker in workers:
        worker.start()

    if not args.headless:
        for index, (
            _label,
            window,
            _frames,
            _is_depth,
            window_size,
        ) in enumerate(views):
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, *window_size)
            cv2.moveWindow(
                window,
                (index % 2) * WINDOW_STEP_X,
                (index // 2) * WINDOW_STEP_Y,
            )

    counts = {label: 0 for label, _window, _frames, _is_depth, _window_size in views}
    last_counts = counts.copy()
    positioned = args.headless
    last_log = time.monotonic()
    try:
        while not stop.is_set():
            for label, window, frames, is_depth, _window_size in views:
                try:
                    frame = frames.get_nowait()
                except queue.Empty:
                    continue
                counts[label] += 1
                if not args.headless:
                    if is_depth:
                        frame = colorize_depth(frame, args.depth_min_mm, args.depth_max_mm)
                    cv2.imshow(window, frame)

            if args.max_frames and all(count >= args.max_frames for count in counts.values()):
                break

            now = time.monotonic()
            if now - last_log >= 5.0:
                elapsed = now - last_log
                rates = {
                    label: round((count - last_counts[label]) / elapsed, 1)
                    for label, count in counts.items()
                }
                LOGGER.info("Processed frames: %s; FPS: %s", counts, rates)
                last_counts = counts.copy()
                last_log = now

            if args.headless:
                stop.wait(0.005)
                continue
            key = cv2.waitKey(1) & 0xFF
            if not positioned and any(counts.values()):
                for index, (
                    _label,
                    window,
                    _frames,
                    _is_depth,
                    _window_size,
                ) in enumerate(views):
                    cv2.moveWindow(
                        window,
                        (index % 2) * WINDOW_STEP_X,
                        (index // 2) * WINDOW_STEP_Y,
                    )
                positioned = True
            if key in (ord("q"), 27):
                break
    finally:
        stop.set()
        for worker in workers:
            worker.join(timeout=3.0)
        if not args.headless:
            cv2.destroyAllWindows()

    LOGGER.info("Stopped with frame counts: %s", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
