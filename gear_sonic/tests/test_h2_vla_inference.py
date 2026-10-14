import json
import time
from types import SimpleNamespace

from gear_sonic.data.h2_collection.core import BODY_NAMES, HAND_NAMES, unpack_wire
from gear_sonic.utils.inference.h2_vla import (
    CLOSED_REFERENCES,
    ActionChunk,
    BinaryHands,
    action_message,
    observation_from_arrays,
    validate_controller,
    validate_samples,
)
from gear_sonic.utils.inference.h2_vla_features import pack_state
import numpy as np
import pytest


def actions():
    return {
        "motion_token": np.zeros((1, 40, 64), np.float32),
        "left_hand_joints": np.zeros((1, 40, 6), np.float32),
        "right_hand_joints": np.ones((1, 40, 6), np.float32),
    }


def test_native_joint_order_and_gravity():
    q = np.arange(43, dtype=np.float32)
    state = pack_state(q, BODY_NAMES + HAND_NAMES, [1, 0, 0, 0])
    assert state.shape == (44,)
    np.testing.assert_array_equal(state[4:6], [5, 4])
    np.testing.assert_array_equal(state[10:12], [11, 10])
    np.testing.assert_array_equal(state[29:41], q[31:])
    np.testing.assert_allclose(state[-3:], [0, 0, -1])


def test_observation_keeps_measured_hands_and_rgb():
    rgb = np.full((360, 640, 3), [20, 40, 60], dtype=np.uint8)
    obs = observation_from_arrays(np.zeros(31), np.arange(12) / 12, [1, 0, 0, 0], rgb)
    assert obs["video"]["ego_view"].shape == (1, 1, 360, 640, 3)
    np.testing.assert_array_equal(obs["video"]["ego_view"][0, 0, 0, 0], [20, 40, 60])
    np.testing.assert_allclose(obs["state"]["left_hand"][0, 0], np.arange(6) / 12)
    assert sum(x.shape[-1] for x in obs["state"].values()) == 44


@pytest.mark.parametrize("age,index", [(0, 0), (0.10, 4), (0.4, 20), (0.79, 39)])
def test_elapsed_time_selects_action(age, index):
    # Floating-point .1 can be immediately below the tick boundary.
    chunk = ActionChunk(10, actions())
    assert abs(chunk.index(10 + age) - index) <= (1 if age == 0.1 else 0)


@pytest.mark.parametrize("now", [9.9, 10.8, 11, float("nan")])
def test_expired_chunk_never_repeats_last_action(now):
    with pytest.raises(ValueError):
        ActionChunk(10, actions()).index(now)


@pytest.mark.parametrize(
    "key,bad", [("motion_token", float("nan")), ("motion_token", 1.3), ("left_hand_joints", 2)]
)
def test_invalid_actions_are_rejected(key, bad):
    a = actions()
    a[key][0, 0, 0] = bad
    with pytest.raises(ValueError):
        ActionChunk(10, a)


def test_g1_hand_width_is_rejected():
    a = actions()
    a["left_hand_joints"] = np.zeros((1, 40, 7))
    with pytest.raises(ValueError):
        ActionChunk(10, a)


def test_binary_hands_round_trip_through_actual_inspire_projection():
    hands = BinaryHands()
    proxy = hands.convert(np.zeros(6), np.ones(6))
    closure = np.sum(proxy * CLOSED_REFERENCES, axis=1) / np.sum(CLOSED_REFERENCES**2, axis=1)
    np.testing.assert_allclose(1 - closure, [0, 1])
    proxy = hands.convert(np.full(6, 0.5), np.full(6, 0.5))
    np.testing.assert_array_equal(hands.open, [False, True])
    hands.convert(np.full(6, 0.8), np.full(6, 0.2))
    np.testing.assert_array_equal(hands.open, [True, False])


def test_wire_keeps_native_token_and_transports_deadline():
    a = actions()
    a["motion_token"][0, 20] = np.arange(64) / 100
    message, index, _ = action_message(ActionChunk(10, a), 10.4, 42, BinaryHands())
    topic, decoded = unpack_wire(message)
    assert topic == "pose" and index == 20
    np.testing.assert_allclose(decoded["token_state"], a["motion_token"][:, 20])
    assert decoded["left_hand_joints"].shape == (1, 7)
    assert decoded["right_hand_joints"].shape == (1, 7)
    assert decoded["h2_vla_deadline_monotonic"][0] == 10.8
    header = json.loads(message[4:1284].split(b"\0")[0])
    assert header["v"] == 4 and header["endian"] == "le"


def samples():
    now = time.monotonic_ns()
    values = {
        "lowstate": {
            "q": [0] * 31,
            "imu": {"quaternion": [1, 0, 0, 0]},
            "crc_ok": True,
            "error": [0] * 31,
        },
        "hand_state": {"q": [1] * 12, "lost": [False] * 12},
        "camera": {},
        "g1_debug": {},
    }
    return now, {k: SimpleNamespace(data=v, mono_ns=now) for k, v in values.items()}


def test_freshness_and_feedback_fail_closed():
    now, s = samples()
    validate_samples(s, now)
    s["camera"].mono_ns = now - int(0.36e9)
    with pytest.raises(ValueError, match="Stale camera"):
        validate_samples(s, now)
    now, s = samples()
    s["hand_state"].data["lost"][0] = True
    with pytest.raises(ValueError, match="offline"):
        validate_samples(s, now)
    now, s = samples()
    s["lowstate"].data["crc_ok"] = False
    with pytest.raises(ValueError, match="CRC"):
        validate_samples(s, now)


def test_old_controller_cannot_be_armed():
    with pytest.raises(ValueError):
        validate_controller({"robot_profile": "h2_native"})
    good = {
        "robot_profile": "h2_native",
        "h2_vla_protocol": 1,
        "h2_vla_watchdog": True,
        "h2_vr_head_control": True,
        "model_path": "../gpu3_20260909_stage2_step005000/model_decoder.onnx",
        "reference_motion_path": "reference/h2_default_stand_only",
    }
    validate_controller(good)
    good["model_path"] = "wrong_model.onnx"
    with pytest.raises(ValueError):
        validate_controller(good)


def test_live_stand_bootstraps_before_debug_but_run_requires_it(tmp_path, monkeypatch):
    """Exercise the actual client loop without DDS, camera, or an action socket."""
    import io
    import sys
    from types import ModuleType

    import cv2
    from gear_sonic.scripts import run_h2_vla_inference as client
    import zmq

    original_event = client.threading.Event
    now, feedback = samples()
    del feedback["g1_debug"]
    feedback["camera"].data["image"] = np.zeros((360, 640, 3), dtype=np.uint8)
    hub = SimpleNamespace(
        stopping=original_event(),
        config={
            "robot_profile": "h2_native",
            "h2_vla_protocol": 1,
            "h2_vla_watchdog": True,
            "h2_vr_head_control": True,
            "model_path": "../gpu3_20260909_stage2_step005000/model_decoder.onnx",
            "reference_motion_path": "reference/h2_default_stand_only",
        },
    )

    def snapshot():
        for sample in feedback.values():
            sample.mono_ns = time.monotonic_ns()
        return feedback

    hub.snapshot = snapshot
    collector = ModuleType("gear_sonic.data.h2_collection.collector")
    collector.Hub = lambda: hub
    collector.DDSInput = lambda *args: SimpleNamespace(close=lambda: None)
    collector.camera_input = collector.zmq_input = lambda *args: None
    monkeypatch.setitem(sys.modules, collector.__name__, collector)
    busy, ready = original_event(), original_event()
    ready.set()  # Model handshake is already complete in this scenario.
    worker_events = iter((busy, ready))
    monkeypatch.setattr(client.threading, "Event", lambda: next(worker_events))
    monkeypatch.setattr(
        client.threading, "Thread", lambda **kwargs: SimpleNamespace(start=lambda: None)
    )
    monkeypatch.setattr(client.signal, "signal", lambda *args: None)
    monkeypatch.setattr(client, "verify_sonic", lambda *args: None)
    monkeypatch.setattr(cv2, "getBuildInformation", lambda: "GStreamer:                   YES")
    monkeypatch.setattr(sys, "stdin", io.StringIO("stand\nrun\nquit\n"))
    monkeypatch.setattr(client.select, "select", lambda *args: ([sys.stdin], [], []))
    sent = []
    publisher = SimpleNamespace(
        setsockopt=lambda *args: None,
        bind=lambda *args: None,
        send=sent.append,
        close=lambda **kwargs: None,
    )
    monkeypatch.setattr(
        zmq, "Context", lambda: SimpleNamespace(socket=lambda *args: publisher, term=lambda: None)
    )
    client.live_or_shadow(
        SimpleNamespace(
            mode="live",
            sonic_model=tmp_path,
            rgb_port=0,
            domain=239,
            interface="unused",
            state_address="unused",
            prompt="test",
            inference_interval=0.4,
        ),
        tmp_path,
    )
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert any(e["event"] == "operator_command" and e["command"] == "stand" for e in events)
    assert any(
        e["event"] == "command_rejected"
        and e["command"] == "run"
        and e["reason"] == "Missing g1_debug"
        for e in events
    )
    assert sent and all(not message.startswith(b"pose") for message in sent)
