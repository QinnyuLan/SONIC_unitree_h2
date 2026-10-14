"""Replay recorded XRoboToolkit tracking data at the live SDK API boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
import threading
import time

import numpy as np

DEFAULT_FPS = 50.0

_POSE_KEYS = ("left_controller_pose", "right_controller_pose", "headset_pose")
_BUTTON_KEYS = (
    "a_button",
    "b_button",
    "x_button",
    "y_button",
    "left_menu_button",
    "right_menu_button",
    "left_axis_click",
    "right_axis_click",
)
_SCALAR_KEYS = ("left_trigger", "right_trigger", "left_grip", "right_grip")


def _key_lookup(mapping: Mapping, *names: str):
    normalized = {str(key).replace("_", "").lower(): value for key, value in mapping.items()}
    for name in names:
        key = name.replace("_", "").lower()
        if key in normalized:
            return normalized[key]
    return None


def _read_concatenated_json(path: Path) -> list[dict]:
    """Read the concatenated pretty-printed objects written by RobotDataRecorder."""
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    objects: list[dict] = []
    offset = 0
    while offset < len(text):
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset >= len(text):
            break
        try:
            value, offset = decoder.raw_decode(text, offset)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid XRoboToolkit JSON in {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON objects in {path}, got {type(value).__name__}")
        objects.append(value)
    return objects


def _numeric_vector(value, size: int) -> np.ndarray | None:
    if isinstance(value, str):
        tokens = value.replace(",", " ").split()
        if len(tokens) == size:
            try:
                return np.asarray([float(token) for token in tokens], dtype=np.float32)
            except ValueError:
                return None
    if isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        return array if array.shape == (size,) else None
    return None


def _extract_pose(value, *, context: str) -> np.ndarray:
    pose = _numeric_vector(value, 7)
    if pose is None and isinstance(value, Mapping):
        nested = _key_lookup(value, "pose", "p", "transform")
        if nested is not None and nested is not value:
            pose = _numeric_vector(nested, 7)

        if pose is None:
            position = _key_lookup(value, "position", "pos", "translation")
            rotation = _key_lookup(value, "rotation", "orientation", "quaternion", "quat")
            position_vec = _numeric_vector(position, 3)
            rotation_vec = _numeric_vector(rotation, 4)
            if position_vec is not None and rotation_vec is not None:
                pose = np.concatenate((position_vec, rotation_vec)).astype(np.float32)

        if pose is None:
            direct = [_key_lookup(value, key) for key in ("x", "y", "z", "qx", "qy", "qz", "qw")]
            if all(item is not None for item in direct):
                pose = np.asarray(direct, dtype=np.float32)

    if pose is None:
        raise ValueError(f"cannot parse {context} pose as [x,y,z,qx,qy,qz,qw]")
    if not np.all(np.isfinite(pose)):
        raise ValueError(f"{context} pose contains non-finite values")
    quat_norm = float(np.linalg.norm(pose[3:]))
    if quat_norm < 1e-8:
        raise ValueError(f"{context} pose contains a zero quaternion")
    pose[3:] /= quat_norm
    return pose


def _side_object(controller: Mapping, side: str) -> Mapping:
    value = _key_lookup(controller, side, f"{side}controller", f"controller{side}")
    if isinstance(value, Mapping):
        return value
    raise ValueError(f"controller.txt entry is missing the {side} controller object")


def _float_field(mapping: Mapping, *names: str) -> float:
    value = _key_lookup(mapping, *names)
    return 0.0 if value is None else float(value)


def _bool_field(mapping: Mapping, *names: str) -> bool:
    value = _key_lookup(mapping, *names)
    return False if value is None else bool(value)


def load_xrobotoolkit_recording(path: str | Path, *, fps: float = DEFAULT_FPS) -> dict[str, np.ndarray]:
    """Load an official RobotDataRecorder directory containing head/controller logs."""
    recording_dir = Path(path)
    head_path = recording_dir / "head.txt"
    controller_path = recording_dir / "controller.txt"
    if not head_path.is_file() or not controller_path.is_file():
        raise ValueError(
            f"{recording_dir} must contain head.txt and controller.txt from XRoboToolkit RobotDataRecorder"
        )
    if fps <= 0.0:
        raise ValueError("recording fps must be positive")

    heads = _read_concatenated_json(head_path)
    controllers = _read_concatenated_json(controller_path)
    if not heads or not controllers:
        raise ValueError("XRoboToolkit recording contains no headset/controller frames")
    if len(heads) != len(controllers):
        raise ValueError(
            "head.txt and controller.txt frame counts differ "
            f"({len(heads)} != {len(controllers)}); refusing to misalign three-point poses"
        )

    frame_count = len(heads)
    data: dict[str, np.ndarray] = {
        "fps": np.asarray(fps, dtype=np.float64),
        "timestamps_ns": np.rint(np.arange(frame_count) * 1e9 / fps).astype(np.int64),
        "headset_pose": np.stack(
            [
                _extract_pose(_key_lookup(entry, "pose", "p", "transform") or entry, context="headset")
                for entry in heads
            ]
        ),
    }

    left_entries = [_side_object(entry, "left") for entry in controllers]
    right_entries = [_side_object(entry, "right") for entry in controllers]
    data["left_controller_pose"] = np.stack(
        [_extract_pose(entry, context="left controller") for entry in left_entries]
    )
    data["right_controller_pose"] = np.stack(
        [_extract_pose(entry, context="right controller") for entry in right_entries]
    )

    for side, entries in (("left", left_entries), ("right", right_entries)):
        data[f"{side}_trigger"] = np.asarray(
            [_float_field(entry, "trigger", "triggerValue") for entry in entries], dtype=np.float32
        )
        data[f"{side}_grip"] = np.asarray(
            [_float_field(entry, "grip", "gripValue") for entry in entries], dtype=np.float32
        )
        data[f"{side}_axis"] = np.asarray(
            [
                [
                    _float_field(entry, "axisX", "joystickX", "thumbstickX"),
                    _float_field(entry, "axisY", "joystickY", "thumbstickY"),
                ]
                for entry in entries
            ],
            dtype=np.float32,
        )
        data[f"{side}_menu_button"] = np.asarray(
            [_bool_field(entry, "menuButton", "menu") for entry in entries], dtype=bool
        )
        data[f"{side}_axis_click"] = np.asarray(
            [_bool_field(entry, "axisClick", "joystickClick", "thumbstickClick") for entry in entries],
            dtype=bool,
        )

    # PICO primary/secondary buttons are X/Y on the left and A/B on the right.
    data["x_button"] = np.asarray(
        [_bool_field(entry, "primaryButton", "xButton", "x") for entry in left_entries], dtype=bool
    )
    data["y_button"] = np.asarray(
        [_bool_field(entry, "secondaryButton", "yButton", "y") for entry in left_entries], dtype=bool
    )
    data["a_button"] = np.asarray(
        [_bool_field(entry, "primaryButton", "aButton", "a") for entry in right_entries], dtype=bool
    )
    data["b_button"] = np.asarray(
        [_bool_field(entry, "secondaryButton", "bButton", "b") for entry in right_entries], dtype=bool
    )
    return data


def load_tracking_data_recording(path: str | Path) -> dict[str, np.ndarray]:
    """Load a PICO XRoboToolkit ``trackingData_*.txt`` full-body recording."""
    recording_path = Path(path)
    frames: list[dict] = []
    with recording_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON on line {line_number} of {recording_path}: {exc}"
                ) from exc
            # XRoboToolkit writes a camera/notice metadata object before the frames.
            if isinstance(entry, dict) and "Body" in entry:
                frames.append(entry)

    if not frames:
        raise ValueError(f"{recording_path} contains no XRoboToolkit full-body frames")

    timestamps = np.asarray(
        [int(_key_lookup(frame, "timeStampNs", "timestampNs")) for frame in frames],
        dtype=np.int64,
    )
    timestamps -= timestamps[0]
    deltas = np.diff(timestamps)
    if np.any(deltas <= 0):
        raise ValueError("trackingData frame timestamps must be strictly increasing")
    fps = 1e9 / float(np.median(deltas)) if len(frames) > 1 else DEFAULT_FPS

    controllers = [_key_lookup(frame, "Controller") for frame in frames]
    if not all(isinstance(controller, Mapping) for controller in controllers):
        raise ValueError("trackingData frame is missing the Controller object")
    left_entries = [_side_object(controller, "left") for controller in controllers]
    right_entries = [_side_object(controller, "right") for controller in controllers]

    body_frames: list[np.ndarray] = []
    for frame_index, frame in enumerate(frames):
        body = _key_lookup(frame, "Body")
        joints = _key_lookup(body, "joints") if isinstance(body, Mapping) else None
        if not isinstance(joints, list) or len(joints) != 24:
            count = len(joints) if isinstance(joints, list) else 0
            raise ValueError(
                f"trackingData frame {frame_index} must contain 24 body joints, got {count}"
            )
        body_frames.append(
            np.stack(
                [
                    _extract_pose(joint, context=f"body joint {joint_index}")
                    for joint_index, joint in enumerate(joints)
                ]
            )
        )

    data: dict[str, np.ndarray] = {
        "fps": np.asarray(fps, dtype=np.float64),
        "timestamps_ns": timestamps,
        "body_joints_pose": np.stack(body_frames),
        "headset_pose": np.stack(
            [
                _extract_pose(_key_lookup(frame, "Head"), context="headset")
                for frame in frames
            ]
        ),
        "left_controller_pose": np.stack(
            [_extract_pose(entry, context="left controller") for entry in left_entries]
        ),
        "right_controller_pose": np.stack(
            [_extract_pose(entry, context="right controller") for entry in right_entries]
        ),
    }

    for side, entries in (("left", left_entries), ("right", right_entries)):
        data[f"{side}_trigger"] = np.asarray(
            [_float_field(entry, "trigger", "triggerValue") for entry in entries],
            dtype=np.float32,
        )
        data[f"{side}_grip"] = np.asarray(
            [_float_field(entry, "grip", "gripValue") for entry in entries], dtype=np.float32
        )
        data[f"{side}_axis"] = np.asarray(
            [
                [
                    _float_field(entry, "axisX", "joystickX", "thumbstickX"),
                    _float_field(entry, "axisY", "joystickY", "thumbstickY"),
                ]
                for entry in entries
            ],
            dtype=np.float32,
        )
        data[f"{side}_menu_button"] = np.asarray(
            [_bool_field(entry, "menuButton", "menu") for entry in entries], dtype=bool
        )
        data[f"{side}_axis_click"] = np.asarray(
            [_bool_field(entry, "axisClick", "joystickClick", "thumbstickClick") for entry in entries],
            dtype=bool,
        )

    data["x_button"] = np.asarray(
        [_bool_field(entry, "primaryButton", "xButton", "x") for entry in left_entries], dtype=bool
    )
    data["y_button"] = np.asarray(
        [_bool_field(entry, "secondaryButton", "yButton", "y") for entry in left_entries], dtype=bool
    )
    data["a_button"] = np.asarray(
        [_bool_field(entry, "primaryButton", "aButton", "a") for entry in right_entries], dtype=bool
    )
    data["b_button"] = np.asarray(
        [_bool_field(entry, "secondaryButton", "bButton", "b") for entry in right_entries], dtype=bool
    )
    return data


class OfflineXRoboToolkit:
    """XRT-compatible replay object backed by recorded XR tracking data."""

    is_offline = True

    def __init__(
        self,
        dataset: Mapping[str, np.ndarray],
        *,
        speed: float = 1.0,
        loop: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ):
        if speed <= 0.0:
            raise ValueError("speed must be positive")
        self._data = {key: np.asarray(value).copy() for key, value in dataset.items()}
        self._speed = float(speed)
        self._loop = bool(loop)
        self._clock = clock
        self._lock = threading.Lock()
        self._armed_at: float | None = None
        self._initialized = False
        self._validate_and_fill_defaults()
        self.is_offline_3pt = "body_joints_pose" not in self._data

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        fps: float = DEFAULT_FPS,
        speed: float = 1.0,
        loop: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> "OfflineXRoboToolkit":
        source = Path(path)
        if source.is_dir():
            dataset = load_xrobotoolkit_recording(source, fps=fps)
        elif source.suffix.lower() == ".txt":
            dataset = load_tracking_data_recording(source)
        elif source.suffix.lower() == ".npz":
            with np.load(source, allow_pickle=False) as archive:
                dataset = {key: archive[key] for key in archive.files}
        else:
            raise ValueError(
                "offline VR input must be an XRoboToolkit trackingData .txt, "
                "RobotDataRecorder directory, or .npz"
            )
        return cls(dataset, speed=speed, loop=loop, clock=clock)

    @property
    def frame_count(self) -> int:
        return int(self._data["headset_pose"].shape[0])

    @property
    def fps(self) -> float:
        return float(self._data["fps"])

    def _frame_array(self, key: str, default, shape: tuple[int, ...], dtype) -> np.ndarray:
        value = np.asarray(self._data.get(key, default), dtype=dtype)
        if value.ndim == 0:
            value = np.full(shape, value, dtype=dtype)
        if value.shape != shape:
            raise ValueError(f"{key} must have shape {list(shape)}, got {list(value.shape)}")
        return value

    def _validate_and_fill_defaults(self):
        missing = [key for key in _POSE_KEYS if key not in self._data]
        if missing:
            raise ValueError(f"offline VR recording is missing required fields: {', '.join(missing)}")
        frames = int(np.asarray(self._data["headset_pose"]).shape[0])
        if frames == 0:
            raise ValueError("offline VR recording contains no frames")
        for key in _POSE_KEYS:
            poses = self._frame_array(key, 0.0, (frames, 7), np.float32)
            if not np.all(np.isfinite(poses)):
                raise ValueError(f"{key} contains non-finite values")
            norms = np.linalg.norm(poses[:, 3:], axis=1)
            if np.any(norms < 1e-8):
                raise ValueError(f"{key} contains a zero quaternion")
            poses[:, 3:] /= norms[:, None]
            self._data[key] = poses

        if "body_joints_pose" in self._data:
            body_poses = self._frame_array(
                "body_joints_pose", 0.0, (frames, 24, 7), np.float32
            )
            if not np.all(np.isfinite(body_poses)):
                raise ValueError("body_joints_pose contains non-finite values")
            norms = np.linalg.norm(body_poses[:, :, 3:], axis=2)
            if np.any(norms < 1e-8):
                raise ValueError("body_joints_pose contains a zero quaternion")
            body_poses[:, :, 3:] /= norms[:, :, None]
            self._data["body_joints_pose"] = body_poses

        fps = float(np.asarray(self._data.get("fps", DEFAULT_FPS)).reshape(()))
        if fps <= 0.0:
            raise ValueError("offline VR recording fps must be positive")
        self._data["fps"] = np.asarray(fps, dtype=np.float64)
        timestamps = np.rint(np.arange(frames) * 1e9 / fps).astype(np.int64)
        self._data["timestamps_ns"] = self._frame_array("timestamps_ns", timestamps, (frames,), np.int64)
        if np.any(np.diff(self._data["timestamps_ns"]) <= 0):
            raise ValueError("timestamps_ns must be strictly increasing")

        for key in _BUTTON_KEYS:
            self._data[key] = self._frame_array(key, False, (frames,), bool)
        for key in _SCALAR_KEYS:
            self._data[key] = self._frame_array(key, 0.0, (frames,), np.float32)
        for key in ("left_axis", "right_axis"):
            self._data[key] = self._frame_array(key, 0.0, (frames, 2), np.float32)

    def init(self):
        with self._lock:
            self._armed_at = None
            self._initialized = True

    def close(self):
        return None

    def _arm(self):
        if not self._initialized:
            self.init()
        with self._lock:
            if self._armed_at is None:
                self._armed_at = self._clock()

    def _raw_frame(self) -> int:
        with self._lock:
            armed_at = self._armed_at
        if armed_at is None:
            return 0
        elapsed_frames = (self._clock() - armed_at) * self.fps * self._speed
        return max(0, int(np.floor(elapsed_frames + 1e-9)))

    def _index(self) -> int:
        raw_frame = self._raw_frame()
        return raw_frame % self.frame_count if self._loop else min(raw_frame, self.frame_count - 1)

    def _timestamp_ns(self) -> int:
        raw_frame = self._raw_frame()
        index = raw_frame % self.frame_count if self._loop else min(raw_frame, self.frame_count - 1)
        timestamp = int(self._data["timestamps_ns"][index])
        if self._loop and raw_frame >= self.frame_count:
            step_ns = int(round(1e9 / self.fps))
            cycle_ns = int(self._data["timestamps_ns"][-1]) + step_ns
            timestamp += (raw_frame // self.frame_count) * cycle_ns
        return timestamp

    def _value(self, key: str):
        self._arm()
        value = self._data[key][self._index()]
        return value.copy() if isinstance(value, np.ndarray) else value.item()

    def is_body_data_available(self) -> bool:
        return "body_joints_pose" in self._data

    def is_vr_3pt_data_available(self) -> bool:
        return True

    def get_body_joints_pose(self) -> list[list[float]]:
        if not self.is_body_data_available():
            raise RuntimeError("offline recording does not contain full-body joints")
        return self._value("body_joints_pose").astype(np.float64).tolist()

    def get_time_stamp_ns(self) -> int:
        return self._timestamp_ns()

    def get_vr_3pt_poses(self) -> np.ndarray:
        """Return left, right, and headset poses from one recording frame."""
        self._arm()
        index = self._index()
        return np.stack([self._data[key][index] for key in _POSE_KEYS]).astype(np.float64)

    def get_left_controller_pose(self) -> list[float]:
        return self._value("left_controller_pose").astype(np.float64).tolist()

    def get_right_controller_pose(self) -> list[float]:
        return self._value("right_controller_pose").astype(np.float64).tolist()

    def get_headset_pose(self) -> list[float]:
        return self._value("headset_pose").astype(np.float64).tolist()

    def get_left_trigger(self) -> float:
        return float(self._value("left_trigger"))

    def get_right_trigger(self) -> float:
        return float(self._value("right_trigger"))

    def get_left_grip(self) -> float:
        return float(self._value("left_grip"))

    def get_right_grip(self) -> float:
        return float(self._value("right_grip"))

    def get_left_axis(self) -> list[float]:
        return self._value("left_axis").astype(np.float64).tolist()

    def get_right_axis(self) -> list[float]:
        return self._value("right_axis").astype(np.float64).tolist()

    def get_A_button(self) -> bool:
        return bool(self._value("a_button"))

    def get_B_button(self) -> bool:
        return bool(self._value("b_button"))

    def get_X_button(self) -> bool:
        return bool(self._value("x_button"))

    def get_Y_button(self) -> bool:
        return bool(self._value("y_button"))

    def get_left_menu_button(self) -> bool:
        return bool(self._value("left_menu_button"))

    def get_right_menu_button(self) -> bool:
        return bool(self._value("right_menu_button"))

    def get_left_axis_click(self) -> bool:
        return bool(self._value("left_axis_click"))

    def get_right_axis_click(self) -> bool:
        return bool(self._value("right_axis_click"))
