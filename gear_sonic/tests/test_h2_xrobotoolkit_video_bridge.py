import importlib.util
from pathlib import Path
import socket
import struct
import sys
import threading
import unittest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "gear_sonic_deploy"
    / "scripts"
    / "h2_xrobotoolkit_video_bridge.py"
)
SPEC = importlib.util.spec_from_file_location("h2_xrobotoolkit_video_bridge", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BRIDGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BRIDGE
SPEC.loader.exec_module(BRIDGE)


def camera_request(
    *,
    width: int = 2560,
    height: int = 720,
    fps: int = 60,
    bitrate: int = 4_000_000,
    enable_mv_hevc: int = 0,
    port: int = 12345,
    camera: str = "ZED",
    ip: str = "192.168.56.176",
) -> bytes:
    camera_bytes = camera.encode()
    ip_bytes = ip.encode()
    return (
        struct.pack(
            "<2sB7i",
            b"\xca\xfe",
            1,
            width,
            height,
            fps,
            bitrate,
            enable_mv_hevc,
            2,
            port,
        )
        + bytes([len(camera_bytes)])
        + camera_bytes
        + bytes([len(ip_bytes)])
        + ip_bytes
    )


def control_packet(command: str, data: bytes) -> bytes:
    command_bytes = command.encode()
    body = (
        struct.pack("<i", len(command_bytes))
        + command_bytes
        + struct.pack("<i", len(data))
        + data
    )
    return struct.pack(">I", len(body)) + body


class H2XRoboToolkitVideoBridgeTest(unittest.TestCase):
    def test_parses_official_open_camera_message(self):
        payload = camera_request()
        message = BRIDGE.parse_control_message(control_packet("OPEN_CAMERA", payload)[4:])
        request = BRIDGE.parse_camera_request(message.data)

        self.assertEqual(message.command, "OPEN_CAMERA")
        self.assertEqual(request.width, 2560)
        self.assertEqual(request.height, 720)
        self.assertEqual(request.fps, 60)
        self.assertEqual(request.bitrate, 4_000_000)
        self.assertEqual(request.camera, "ZED")
        self.assertEqual(request.ip, "192.168.56.176")
        self.assertEqual(request.port, 12345)

    def test_receive_handles_fragmented_tcp_message(self):
        left, right = socket.socketpair()
        stop = threading.Event()
        packet = control_packet("OPEN_CAMERA", camera_request())
        try:
            for index in range(0, len(packet), 3):
                left.sendall(packet[index : index + 3])
            message = BRIDGE.receive_control_message(right, stop)
        finally:
            left.close()
            right.close()
        self.assertEqual(message.command, "OPEN_CAMERA")

    def test_rejects_hevc_and_invalid_sbs_width(self):
        with self.assertRaisesRegex(ValueError, "HEVC"):
            BRIDGE.parse_camera_request(camera_request(enable_mv_hevc=1))
        with self.assertRaisesRegex(ValueError, "even"):
            BRIDGE.parse_camera_request(camera_request(width=2559))

    def test_pipeline_duplicates_mono_and_outputs_h264_access_units(self):
        pipeline = BRIDGE.build_pipeline(
            rtp_address="0.0.0.0",
            rtp_port=5002,
            output_width=2560,
            output_height=720,
            source_fps=15,
            bitrate=4_000_000,
            jitter_ms=20,
        )
        self.assertIn("rtph264depay", pipeline)
        self.assertIn("nvh264dec", pipeline)
        self.assertIn("tee name=duplicate", pipeline)
        self.assertIn("sink_1::xpos=1280", pipeline)
        self.assertIn("nvh264enc", pipeline)
        self.assertIn("stream-format=byte-stream,alignment=au", pipeline)
        self.assertIn("appsink name=encoded", pipeline)


if __name__ == "__main__":
    unittest.main()
