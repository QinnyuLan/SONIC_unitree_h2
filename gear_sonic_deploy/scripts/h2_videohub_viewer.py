#!/usr/bin/env python3
"""Play the Unitree H2 video_hub H.264 stream with OpenCV."""

from __future__ import annotations

import argparse
import time

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", default="eno1", help="robot network interface")
    parser.add_argument("--address", default="230.1.1.1", help="video_hub multicast address")
    parser.add_argument("--port", type=int, default=1720, help="video_hub UDP port")
    parser.add_argument("--headless", action="store_true", help="decode without opening a window")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N frames; 0 runs forever")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.max_frames < 0:
        parser.error("--max-frames cannot be negative")
    return args


def main() -> int:
    args = parse_args()
    pipeline = (
        f"udpsrc address={args.address} port={args.port} "
        f"multicast-iface={args.interface} ! queue ! "
        "application/x-rtp,media=video,encoding-name=H264 ! "
        "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
        "video/x-raw,width=1280,height=720,format=BGR ! "
        "appsink drop=1 max-buffers=1 sync=false"
    )

    print(
        f"Waiting for video_hub RTP/H.264 at {args.address}:{args.port} "
        f"on {args.interface}...",
        flush=True,
    )
    capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not capture.isOpened():
        raise SystemExit("Failed to open video_hub stream with OpenCV/GStreamer")

    frame_count = 0
    started_at = time.monotonic()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise SystemExit("video_hub stream stopped")

            frame_count += 1
            if frame_count == 1:
                height, width = frame.shape[:2]
                print(f"Receiving {width}x{height} from {args.address}:{args.port} on {args.interface}")

            if not args.headless:
                cv2.imshow("Unitree H2 video_hub", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

            if args.max_frames and frame_count >= args.max_frames:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    elapsed = time.monotonic() - started_at
    print(f"Decoded {frame_count} frames in {elapsed:.2f}s ({frame_count / elapsed:.1f} FPS)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
