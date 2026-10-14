#!/usr/bin/env python3
"""Relay the H2 monocular RTP/H.264 stream to XRoboToolkit Remote Vision.

XRoboToolkit 1.1.1 expects a side-by-side texture. This bridge decodes the
H2's UDP/5002 monocular stream once, scales it to one eye, duplicates it for
both eyes, and uses NVENC to send a length-framed H.264 stream to the headset.

Run this script with the system Python because Ubuntu's GStreamer bindings are
installed for /usr/bin/python3, independently of the SONIC virtual environment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import ipaddress
import logging
import signal
import socket
import struct
import threading
import time
from typing import Any

LOGGER = logging.getLogger("h2_xrobotoolkit_video_bridge")

CONTROL_PORT = 13579
VIDEO_PORT = 12345
H2_RTP_PORT = 5002
MAX_CONTROL_MESSAGE = 1024 * 1024
CAMERA_REQUEST_FIXED = struct.Struct("<2sB7i")


@dataclass(frozen=True)
class ControlMessage:
    command: str
    data: bytes


@dataclass(frozen=True)
class CameraRequest:
    width: int
    height: int
    fps: int
    bitrate: int
    enable_mv_hevc: int
    render_mode: int
    port: int
    camera: str
    ip: str


def receive_exact(sock: socket.socket, size: int, stop: threading.Event) -> bytes:
    data = bytearray()
    while len(data) < size:
        if stop.is_set():
            raise InterruptedError
        try:
            chunk = sock.recv(size - len(data))
        except socket.timeout:
            continue
        if not chunk:
            raise EOFError("peer closed the connection")
        data.extend(chunk)
    return bytes(data)


def receive_control_message(sock: socket.socket, stop: threading.Event) -> ControlMessage:
    body_size = struct.unpack(">I", receive_exact(sock, 4, stop))[0]
    if body_size < 8 or body_size > MAX_CONTROL_MESSAGE:
        raise ValueError(f"invalid XRoboToolkit control message size: {body_size}")
    return parse_control_message(receive_exact(sock, body_size, stop))


def parse_control_message(body: bytes) -> ControlMessage:
    if len(body) < 8:
        raise ValueError("XRoboToolkit control message is too short")

    command_size = struct.unpack_from("<i", body, 0)[0]
    if command_size < 0 or command_size > len(body) - 8:
        raise ValueError(f"invalid XRoboToolkit command size: {command_size}")
    command_end = 4 + command_size
    try:
        command = body[4:command_end].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("XRoboToolkit command is not valid UTF-8") from exc

    data_size = struct.unpack_from("<i", body, command_end)[0]
    data_start = command_end + 4
    if data_size < 0 or data_start + data_size != len(body):
        raise ValueError(f"invalid XRoboToolkit command payload size: {data_size}")
    return ControlMessage(command=command, data=body[data_start:])


def _read_compact_string(data: bytes, offset: int) -> tuple[str, int]:
    if offset >= len(data):
        raise ValueError("camera request is missing a string length")
    size = data[offset]
    offset += 1
    end = offset + size
    if end > len(data):
        raise ValueError("camera request contains a truncated string")
    try:
        value = data[offset:end].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("camera request string is not valid UTF-8") from exc
    return value, end


def parse_camera_request(data: bytes) -> CameraRequest:
    if len(data) < CAMERA_REQUEST_FIXED.size + 2:
        raise ValueError("camera request is too short")
    (
        magic,
        version,
        width,
        height,
        fps,
        bitrate,
        enable_mv_hevc,
        render_mode,
        port,
    ) = CAMERA_REQUEST_FIXED.unpack_from(data)
    if magic != b"\xca\xfe":
        raise ValueError(f"invalid camera request magic: {magic.hex()}")
    if version != 1:
        raise ValueError(f"unsupported camera request version: {version}")

    camera, offset = _read_compact_string(data, CAMERA_REQUEST_FIXED.size)
    ip, offset = _read_compact_string(data, offset)
    if offset != len(data):
        raise ValueError("camera request contains trailing bytes")
    if not 320 <= width <= 4096 or width % 2:
        raise ValueError(f"camera width must be even and in [320, 4096], got {width}")
    if not 240 <= height <= 2160:
        raise ValueError(f"camera height must be in [240, 2160], got {height}")
    if not 1 <= fps <= 240:
        raise ValueError(f"camera FPS must be in [1, 240], got {fps}")
    if not 1 <= bitrate <= 100_000_000:
        raise ValueError(f"camera bitrate must be in [1, 100000000], got {bitrate}")
    if not 1 <= port <= 65535:
        raise ValueError(f"camera port must be in [1, 65535], got {port}")
    if enable_mv_hevc:
        raise ValueError("XRoboToolkit requested HEVC, but this bridge outputs H.264")
    if ip:
        try:
            ipaddress.ip_address(ip)
        except ValueError as exc:
            raise ValueError(f"invalid headset IP in camera request: {ip}") from exc

    return CameraRequest(
        width=width,
        height=height,
        fps=fps,
        bitrate=bitrate,
        enable_mv_hevc=enable_mv_hevc,
        render_mode=render_mode,
        port=port,
        camera=camera,
        ip=ip,
    )


def build_pipeline(
    *,
    rtp_address: str,
    rtp_port: int,
    output_width: int,
    output_height: int,
    source_fps: int,
    bitrate: int,
    jitter_ms: int,
) -> str:
    eye_width = output_width // 2
    bitrate_kbps = max(500, min(100_000, round(bitrate / 1000)))
    gop_size = max(1, source_fps)
    return (
        "compositor name=mix background=black "
        f"sink_0::xpos=0 sink_0::ypos=0 sink_0::width={eye_width} "
        f"sink_0::height={output_height} sink_1::xpos={eye_width} sink_1::ypos=0 "
        f"sink_1::width={eye_width} sink_1::height={output_height} ! "
        f"video/x-raw,format=NV12,width={output_width},height={output_height},"
        f"framerate={source_fps}/1,pixel-aspect-ratio=1/1 ! "
        "nvh264enc preset=low-latency-hq zerolatency=true bframes=0 rc-lookahead=0 "
        f"rc-mode=cbr gop-size={gop_size} bitrate={bitrate_kbps} aud=true ! "
        "h264parse config-interval=-1 disable-passthrough=true ! "
        'video/x-h264,stream-format=byte-stream,alignment=au ! '
        "queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! "
        "appsink name=encoded emit-signals=true sync=false drop=true max-buffers=1 "
        f"udpsrc address={rtp_address} port={rtp_port} buffer-size=4194304 "
        'caps="application/x-rtp,media=video,encoding-name=H264,payload=96,'
        'clock-rate=90000" ! '
        f"rtpjitterbuffer latency={jitter_ms} drop-on-latency=true do-lost=true ! "
        "rtph264depay request-keyframe=true wait-for-keyframe=true ! h264parse ! "
        "nvh264dec max-display-delay=0 ! videoconvert ! videoscale add-borders=true ! "
        f"video/x-raw,format=NV12,width={eye_width},height={output_height},"
        f"framerate={source_fps}/1,pixel-aspect-ratio=1/1 ! tee name=duplicate "
        "duplicate. ! queue leaky=downstream max-size-buffers=2 "
        "max-size-bytes=0 max-size-time=0 ! mix.sink_0 "
        "duplicate. ! queue leaky=downstream max-size-buffers=2 "
        "max-size-bytes=0 max-size-time=0 ! mix.sink_1"
    )


def load_gstreamer() -> Any:
    try:
        import gi
    except ImportError as exc:
        raise RuntimeError(
            "GStreamer Python bindings are unavailable. Run with /usr/bin/python3, "
            "not .venv_teleop/bin/python."
        ) from exc
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    return Gst


class VideoStreamer:
    def __init__(
        self,
        *,
        gst: Any,
        request: CameraRequest,
        target_ip: str,
        rtp_address: str,
        rtp_port: int,
        source_fps: int,
        jitter_ms: int,
    ) -> None:
        self.gst = gst
        self.request = request
        self.target_ip = target_ip
        self.rtp_address = rtp_address
        self.rtp_port = rtp_port
        self.source_fps = source_fps
        self.jitter_ms = jitter_ms
        self.stop_requested = threading.Event()
        self.socket: socket.socket | None = None
        self.pipeline: Any | None = None
        self.monitor_thread: threading.Thread | None = None
        self.frames_sent = 0
        self.bytes_sent = 0
        self.started_at = 0.0

    def _connect(self) -> socket.socket:
        deadline = time.monotonic() + 5.0
        last_error: OSError | None = None
        while time.monotonic() < deadline and not self.stop_requested.is_set():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            sock.settimeout(1.0)
            try:
                sock.connect((self.target_ip, self.request.port))
                sock.settimeout(2.0)
                return sock
            except OSError as exc:
                last_error = exc
                sock.close()
                self.stop_requested.wait(0.2)
        raise ConnectionError(
            f"could not connect to XRoboToolkit decoder at "
            f"{self.target_ip}:{self.request.port}: {last_error}"
        )

    def start(self) -> None:
        self.socket = self._connect()
        pipeline_description = build_pipeline(
            rtp_address=self.rtp_address,
            rtp_port=self.rtp_port,
            output_width=self.request.width,
            output_height=self.request.height,
            source_fps=self.source_fps,
            bitrate=self.request.bitrate,
            jitter_ms=self.jitter_ms,
        )
        LOGGER.debug("GStreamer pipeline: %s", pipeline_description)
        self.pipeline = self.gst.parse_launch(pipeline_description)
        appsink = self.pipeline.get_by_name("encoded")
        if appsink is None:
            raise RuntimeError("GStreamer pipeline has no encoded appsink")
        appsink.connect("new-sample", self._on_new_sample)

        result = self.pipeline.set_state(self.gst.State.PLAYING)
        if result == self.gst.StateChangeReturn.FAILURE:
            self.stop()
            raise RuntimeError("failed to start the H2 video GStreamer pipeline")
        self.started_at = time.monotonic()
        self.monitor_thread = threading.Thread(
            target=self._monitor_bus,
            name="h2-xr-video-bus",
            daemon=True,
        )
        self.monitor_thread.start()
        LOGGER.info(
            "Streaming H2 mono as duplicated SBS %dx%d@%d to %s:%d at %.1f Mbps",
            self.request.width,
            self.request.height,
            self.source_fps,
            self.target_ip,
            self.request.port,
            self.request.bitrate / 1_000_000,
        )

    def _on_new_sample(self, sink: Any) -> Any:
        if self.stop_requested.is_set() or self.socket is None:
            return self.gst.FlowReturn.EOS
        sample = sink.emit("pull-sample")
        if sample is None:
            return self.gst.FlowReturn.ERROR
        buffer = sample.get_buffer()
        payload = buffer.extract_dup(0, buffer.get_size())
        try:
            self.socket.sendall(struct.pack(">I", len(payload)) + payload)
        except OSError as exc:
            if not self.stop_requested.is_set():
                LOGGER.error("Headset video connection failed: %s", exc)
            self.stop_requested.set()
            return self.gst.FlowReturn.ERROR
        self.frames_sent += 1
        self.bytes_sent += len(payload)
        return self.gst.FlowReturn.OK

    def _monitor_bus(self) -> None:
        assert self.pipeline is not None
        bus = self.pipeline.get_bus()
        last_report = time.monotonic()
        while not self.stop_requested.is_set():
            message = bus.timed_pop_filtered(
                200 * self.gst.MSECOND,
                self.gst.MessageType.ERROR | self.gst.MessageType.EOS,
            )
            if message is not None:
                if message.type == self.gst.MessageType.ERROR:
                    error, debug = message.parse_error()
                    LOGGER.error("GStreamer error: %s (%s)", error, debug)
                else:
                    LOGGER.warning("H2 video pipeline reached end of stream")
                self.stop_requested.set()
                break
            now = time.monotonic()
            if now - last_report >= 5.0:
                elapsed = max(now - self.started_at, 1e-6)
                LOGGER.info(
                    "Video sent: %d frames, %.1f FPS average, %.2f Mbps average",
                    self.frames_sent,
                    self.frames_sent / elapsed,
                    self.bytes_sent * 8 / elapsed / 1_000_000,
                )
                last_report = now

    def stop(self) -> None:
        self.stop_requested.set()
        if self.pipeline is not None:
            self.pipeline.set_state(self.gst.State.NULL)
            self.pipeline = None
        if self.socket is not None:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.socket.close()
            self.socket = None
        if (
            self.monitor_thread is not None
            and self.monitor_thread is not threading.current_thread()
        ):
            self.monitor_thread.join(timeout=2.0)
        if self.frames_sent:
            LOGGER.info("Stopped video after sending %d frames", self.frames_sent)


class XRoboToolkitVideoBridge:
    def __init__(self, args: argparse.Namespace, gst: Any) -> None:
        self.args = args
        self.gst = gst
        self.stop_requested = threading.Event()
        self.streamer: VideoStreamer | None = None

    def stop(self) -> None:
        self.stop_requested.set()
        self._stop_streamer()

    def _stop_streamer(self) -> None:
        if self.streamer is not None:
            self.streamer.stop()
            self.streamer = None

    def _open_camera(self, message: ControlMessage, peer_ip: str) -> None:
        request = parse_camera_request(message.data)
        target_ip = peer_ip
        if request.ip and request.ip != peer_ip:
            LOGGER.warning(
                "Headset advertised %s but connected from %s; using the reachable peer address",
                request.ip,
                peer_ip,
            )
        eye_ratio = (request.width / 2) / request.height
        if abs(eye_ratio - 16 / 9) > 0.05:
            LOGGER.warning(
                "Requested per-eye aspect ratio is %.3f, not 16:9. Select ZEDMINI "
                "in Remote Vision for an undistorted H2 mono image.",
                eye_ratio,
            )
        LOGGER.info(
            "OPEN_CAMERA from %s: source=%s, requested=%dx%d@%d, port=%d",
            peer_ip,
            request.camera,
            request.width,
            request.height,
            request.fps,
            request.port,
        )
        self._stop_streamer()
        streamer = VideoStreamer(
            gst=self.gst,
            request=request,
            target_ip=target_ip,
            rtp_address=self.args.rtp_address,
            rtp_port=self.args.rtp_port,
            source_fps=self.args.source_fps,
            jitter_ms=self.args.jitter_ms,
        )
        try:
            streamer.start()
        except Exception:
            streamer.stop()
            raise
        self.streamer = streamer

    def _handle_client(self, client: socket.socket, peer_ip: str) -> None:
        client.settimeout(0.5)
        while not self.stop_requested.is_set():
            try:
                message = receive_control_message(client, self.stop_requested)
            except socket.timeout:
                continue
            except EOFError:
                LOGGER.info("XRoboToolkit control client %s disconnected", peer_ip)
                return
            if message.command == "OPEN_CAMERA":
                try:
                    self._open_camera(message, peer_ip)
                except Exception:
                    LOGGER.exception("Could not start the H2 Remote Vision stream")
            elif message.command == "CLOSE_CAMERA":
                LOGGER.info("CLOSE_CAMERA from %s", peer_ip)
                self._stop_streamer()
            elif message.command == "AUDIO_SESSION":
                LOGGER.info("Audio session requested; this video-only bridge ignores it")
            else:
                LOGGER.warning("Ignoring XRoboToolkit command: %s", message.command)

    def serve(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.args.control_address, self.args.control_port))
            server.listen(2)
            server.settimeout(0.5)
            LOGGER.info(
                "Waiting for XRoboToolkit Remote Vision at %s:%d",
                self.args.control_address,
                self.args.control_port,
            )
            LOGGER.info(
                "On PICO select ZEDMINI, enable Listen, and enter this PC's Wi-Fi IP"
            )
            while not self.stop_requested.is_set():
                try:
                    client, peer = server.accept()
                except socket.timeout:
                    continue
                LOGGER.info("XRoboToolkit control client connected from %s:%d", *peer)
                with client:
                    try:
                        self._handle_client(client, peer[0])
                    except InterruptedError:
                        break
                    except Exception:
                        LOGGER.exception("XRoboToolkit control connection failed")
                self._stop_streamer()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-address", default="0.0.0.0")
    parser.add_argument("--control-port", type=int, default=CONTROL_PORT)
    parser.add_argument("--rtp-address", default="0.0.0.0")
    parser.add_argument("--rtp-port", type=int, default=H2_RTP_PORT)
    parser.add_argument("--source-fps", type=int, default=15)
    parser.add_argument("--jitter-ms", type=int, default=20)
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    args = parser.parse_args()
    for name in ("control_address", "rtp_address"):
        try:
            ipaddress.ip_address(getattr(args, name))
        except ValueError as exc:
            parser.error(f"--{name.replace('_', '-')} must be an IP address: {exc}")
    for name in ("control_port", "rtp_port"):
        if not 1 <= getattr(args, name) <= 65535:
            parser.error(f"--{name.replace('_', '-')} must be in [1, 65535]")
    if not 1 <= args.source_fps <= 240:
        parser.error("--source-fps must be in [1, 240]")
    if not 0 <= args.jitter_ms <= 1000:
        parser.error("--jitter-ms must be in [0, 1000]")
    return args


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(threadName)s: %(message)s",
    )
    gst = load_gstreamer()
    bridge = XRoboToolkitVideoBridge(args, gst)
    signal.signal(signal.SIGINT, lambda *_: bridge.stop_requested.set())
    signal.signal(signal.SIGTERM, lambda *_: bridge.stop_requested.set())
    try:
        bridge.serve()
    finally:
        bridge.stop()
    LOGGER.info("XRoboToolkit H2 video bridge stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
