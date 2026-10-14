"""H2 N1.7 observation, action, timing and RPC contract; no hardware side effects.

Feature conversion is a verbatim copy of Isaac-GR00T/gr00t/data/h2_sonic.py.
The model predicts native 6D Inspire commands. Only the outgoing SONIC wire
message uses 7D Dex3 references as an adapter to the existing binary backend.
"""

from dataclasses import dataclass
import json
import math

from gear_sonic.data.h2_collection.core import BODY_NAMES, HAND_NAMES
from gear_sonic.utils.inference.h2_vla_features import STATE_SLICES, pack_state
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message
import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq


PROMPT = "Pick up the orange plush toy with your left hand and place it in the cardboard box."
HORIZON = 40
FREQUENCY = 50
DECODER_SHA256 = "064f2c85346ab82f4e43dc5828db7e551879d061a667764b54e0b3e5ff610494"
ENCODER_SHA256 = "ecafb2f2479a7a6bc2139a54b46c8754435da7985d9c7e4b409fe7e4ec96c307"
CLOSED_REFERENCES = np.array(
    [[0, 0.7, 0.7, -1, -1.5, -1, -1.5], [0, -0.7, -0.7, 1, 1.5, 1, 1.5]],
    dtype=np.float32,
)
MAX_AGE = {"lowstate": 0.1, "hand_state": 0.3, "camera": 0.35, "g1_debug": 0.15}


def observation_from_arrays(raw_q, hand_q, quaternion, rgb, prompt=PROMPT):
    state = pack_state(np.concatenate((raw_q, hand_q)), BODY_NAMES + HAND_NAMES, quaternion)
    rgb = np.asarray(rgb)
    if rgb.shape != (360, 640, 3) or rgb.dtype != np.uint8:
        raise ValueError("Expected RGB uint8[360,640,3]")
    return {
        "video": {"ego_view": rgb[None, None]},
        "state": {k: state[None, None, a:b] for k, (a, b) in STATE_SLICES.items()},
        "language": {"annotation.human.task_description": [[prompt]]},
    }


def validate_samples(samples, now_ns, require_controller=True):
    ages = {}
    required = (
        MAX_AGE if require_controller else {k: v for k, v in MAX_AGE.items() if k != "g1_debug"}
    )
    for name, maximum in required.items():
        if name not in samples:
            raise ValueError(f"Missing {name}")
        age = (now_ns - samples[name].mono_ns) / 1e9
        if not 0 <= age <= maximum:
            raise ValueError(f"Stale {name}: {age:.3f}s > {maximum:.3f}s")
        ages[name] = age
    body, hand = samples["lowstate"].data, samples["hand_state"].data
    if not body.get("crc_ok", False):
        raise ValueError("lowstate CRC failed")
    if any(hand.get("lost", [True])):
        raise ValueError("Inspire feedback offline")
    if any(body.get("error", [])):
        raise ValueError("H2 motor error reported")
    for values in (body["q"], hand["q"], body["imu"]["quaternion"]):
        if not np.isfinite(values).all():
            raise ValueError("Non-finite robot feedback")
    return ages


def validate_controller(config):
    if not config or config.get("robot_profile") != "h2_native":
        raise ValueError("Need robot_config from the H2 native controller")
    if config.get("h2_vla_protocol") != 1 or not config.get("h2_vla_watchdog"):
        raise ValueError(
            "Controller is missing/enabled without the H2 VLA watchdog; use the dedicated deploy launcher"
        )
    if "gpu3_20260909_stage2_step005000" not in str(config.get("model_path", "")):
        raise ValueError("Controller is not using the dataset's SONIC decoder")
    if "h2_default_stand_only" not in str(config.get("reference_motion_path", "")):
        raise ValueError("Controller fallback must be the dedicated H2 standing reference")
    if not config.get("h2_vr_head_control"):
        raise ValueError("Fixed head command channel is disabled")


def validate_deployment(manifest):
    expected = {
        "embodiment_tag": "UNITREE_G1_SONIC",
        "state_width": 44,
        "action_width": 76,
        "action_horizon": 40,
        "control_frequency": 50,
        "sonic_decoder_sha256": DECODER_SHA256,
        "head_pitch_yaw_degrees": [40, 0],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Deployment bundle mismatch: {key}")


def validate_modality(config):
    expected = {
        "video": ["ego_view"],
        "state": list(STATE_SLICES),
        "action": ["motion_token", "left_hand_joints", "right_hand_joints"],
        "language": ["annotation.human.task_description"],
    }
    for name, keys in expected.items():
        if config.get(name, {}).get("modality_keys") != keys:
            raise ValueError(f"Policy modality mismatch: {name}")
        indices = list(range(HORIZON)) if name == "action" else [0]
        if config[name].get("delta_indices") != indices:
            raise ValueError(f"Policy horizon mismatch: {name}")


@dataclass
class ActionChunk:
    origin: float
    action: dict
    generation: int = 0

    def __post_init__(self):
        if not math.isfinite(self.origin):
            raise ValueError("Invalid action origin")
        for key, width in [("motion_token", 64), ("left_hand_joints", 6), ("right_hand_joints", 6)]:
            value = np.asarray(self.action[key], dtype=np.float32)
            if value.shape != (1, HORIZON, width) or not np.isfinite(value).all():
                raise ValueError(f"Invalid action {key}: {value.shape}")
            if key == "motion_token" and np.abs(value).max() > 1.25:
                raise ValueError("Token outside checked interface bounds")
            if "hand" in key and (value.min() < -0.05 or value.max() > 1.05):
                raise ValueError("Hand output outside normalized [0,1] range")
            self.action[key] = value

    @property
    def deadline(self):
        return self.origin + HORIZON / FREQUENCY

    def index(self, now):
        age = now - self.origin
        if not math.isfinite(age) or age < 0 or now >= self.deadline:
            raise ValueError("Action chunk expired; do not repeat its final action")
        return min(int(age * FREQUENCY), HORIZON - 1)


class BinaryHands:
    """Hysteresis: <=0.35 closes; >=0.65 opens. Starts from measured feedback."""

    def __init__(self, measured=None):
        self.open = np.ones(2, dtype=bool)
        if measured is not None:
            self.reset(measured)

    def reset(self, measured):
        value = np.asarray(measured).reshape(2, 6)
        if not np.isfinite(value).all():
            raise ValueError("Invalid hand feedback")
        self.open = value.mean(axis=1) >= 0.5

    def convert(self, left, right):
        values = np.stack((left, right))
        if values.shape != (2, 6) or not np.isfinite(values).all():
            raise ValueError("Expected finite H2 hand predictions, 2x6")
        mean = np.clip(values, 0, 1).mean(axis=1)
        self.open[mean <= 0.35] = False
        self.open[mean >= 0.65] = True
        return CLOSED_REFERENCES * (~self.open)[:, None]


def action_message(chunk, now, frame_index, hands):
    index = chunk.index(now)
    a = chunk.action
    proxy = hands.convert(a["left_hand_joints"][0, index], a["right_hand_joints"][0, index])
    payload = {
        "token_state": a["motion_token"][:, index],
        "frame_index": np.array([frame_index], dtype=np.int64),
        "left_hand_joints": proxy[0:1],
        "right_hand_joints": proxy[1:2],
        # Client and C++ receiver run on the SAME Linux host; both clocks use CLOCK_MONOTONIC.
        "h2_vla_deadline_monotonic": np.array([chunk.deadline], dtype=np.float64),
    }
    return pack_pose_message(payload, topic="pose", version=4), index, proxy


def _encode(value):
    if isinstance(value, np.ndarray) and value.dtype.kind == "O":
        raise ValueError("Object arrays are not permitted")
    return mnp.encode(value)


def _decode(value):
    if value.get(b"nd", value.get("nd")) and value.get(b"kind", value.get("kind")) in (b"O", "O"):
        raise ValueError("Object arrays are not permitted")
    if value.get("__ModalityConfig__") or value.get("__ModalityConfig_class__"):
        result = value["as_json"]
        return json.loads(result) if isinstance(result, str) else result
    return mnp.decode(value)


class PolicyRPC:
    """N1.7 PolicyServer wire-compatible client without importing torch or gr00t.

    One thread owns each client. REQ sockets are recreated after timeout.
    """

    def __init__(self, host="127.0.0.1", port=5550, timeout_ms=2000):
        self.context = zmq.Context()
        self.address = f"tcp://{host}:{port}"
        self.timeout_ms = timeout_ms
        self.socket = None

    def call(self, endpoint, data=None):
        if self.socket is None:
            self.socket = self.context.socket(zmq.REQ)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            self.socket.connect(self.address)
        try:
            self.socket.send(
                msgpack.packb({"endpoint": endpoint, "data": data or {}}, default=_encode)
            )
            result = msgpack.unpackb(self.socket.recv(), object_hook=_decode, raw=False)
        except Exception:
            self.socket.close(linger=0)
            self.socket = None
            raise
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(result["error"])
        return result

    def get_action(self, observation):
        result = self.call("get_action", {"observation": observation})
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise ValueError("Expected PolicyServer (action, info) response")
        return result[0]

    def close(self):
        if self.socket is not None:
            self.socket.close(linger=0)
        self.context.term()
