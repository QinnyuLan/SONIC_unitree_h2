import copy
import numpy as np
import pytest

from gear_sonic.commands.protocol import envelope, planner_command
from gear_sonic.commands.receiver import CommandInbox
from gear_sonic.planner.motion import Trajectory, h2_to_g1_mujoco, SEMANTIC_TO_MUJOCO
from gear_sonic.sources.planner import valid_state


def request(source="a", seq=0, kind="planner", **kwargs):
    message = envelope(kind, planner_command() if kind == "planner" else {},
        source=source, session=source, seq=seq, **kwargs)
    message["sent_at_ms"] = 1000
    return message


def test_planner_ownership_release_and_takeover():
    inbox = CommandInbox({"planner", "stand"})
    assert inbox.accept(request(), now=0, wall_ms=1000)
    assert not inbox.accept(request("b"), now=.01, wall_ms=1010)
    assert inbox.accept(request("b", takeover=True), now=.02, wall_ms=1020)
    assert not inbox.accept(request(seq=1, kind="stand"), now=.03, wall_ms=1030)
    assert inbox.active(.04)["source"] == "b"
    assert inbox.active(.25) is None
    assert inbox.accept(request(seq=2), now=.26, wall_ms=1100)


def test_planner_expiry_uses_sender_timestamp():
    inbox = CommandInbox({"planner", "stand"})
    assert inbox.accept(request(), now=5, wall_ms=1200)
    assert inbox.active(5.049) is not None
    assert inbox.active(5.051) is None
    assert not inbox.accept(request(seq=1), now=6, wall_ms=1250)
    assert not inbox.accept(request(seq=2), now=6, wall_ms=749)
    assert not inbox.accept(request(), now=6, wall_ms=1000)


@pytest.mark.parametrize("field,value", [
    ("seq", -1), ("seq", 2**64), ("seq", True), ("sent_at_ms", True),
    ("sent_at_ms", -1), ("ttl_ms", 1001), ("takeover", "yes"),
    ("source", []), ("kind", "token"), ("data", {"mode": 2**40}),
])
def test_planner_rejects_invalid_envelopes(field, value):
    message = request()
    message[field] = value
    inbox = CommandInbox({"planner", "stand"})
    assert not inbox.accept(message, now=0, wall_ms=1000)
    assert inbox.active(0) is None


def test_planner_context_joint_projection():
    dds = np.arange(31, dtype=float) / 20
    actual = h2_to_g1_mujoco(dds)
    assert actual[0] == pytest.approx(0)
    assert actual[1] == pytest.approx(.9847439527511597 * dds[1])
    assert actual[6] == pytest.approx(.8660245537757874 * dds[6])
    assert actual[7] == pytest.approx(.08700508624315262 * dds[6] + .9847439527511597 * dds[7])
    assert actual[4] == dds[5]  # H2 ankle pitch precedes roll in G1 MuJoCo order.
    assert actual[5] == dds[4]
    assert actual[28] == dds[28]


def prediction():
    data = np.zeros((64, 36))
    data[:, 3] = 1
    data[:, 7:] = np.arange(29)[None] + np.arange(64)[:, None] / 30
    data[:, 0] = np.arange(64) / 30
    return data


def test_resampling_sparse_time_spacing_and_terminal_velocity():
    motion = Trajectory.from_prediction(prediction(), 60)
    assert len(motion.q) == 100
    sample = motion.sparse(5)
    np.testing.assert_allclose(np.array(sample["joint_pos"])[:, 0], .1 + np.arange(10) * .1)
    np.testing.assert_allclose(sample["joint_vel"], 1)
    np.testing.assert_allclose(motion.q[0], SEMANTIC_TO_MUJOCO)
    assert np.count_nonzero(motion.sparse(500)["joint_vel"]) == 0
    context = motion.context(5)[0]
    np.testing.assert_allclose(context[:, 7], .1 + np.arange(4) / 30, atol=1e-7)
    np.testing.assert_allclose(context[:, 7:], np.arange(29)[None] + (.1 + np.arange(4) / 30)[:, None], atol=2e-6)


def test_quaternion_interpolation_and_merge():
    data = prediction()
    angle = np.arange(64) / 30
    data[:, 3] = np.cos(angle / 2)
    data[:, 6] = np.sin(angle / 2)
    motion = Trajectory.from_prediction(data, 60)
    np.testing.assert_allclose(2 * np.arctan2(motion.root[:50, 3], motion.root[:50, 0]), np.arange(50) / 50, atol=1e-6)
    context = motion.context(500)
    np.testing.assert_allclose(context[0, :, 3:7], np.tile(motion.root[-1], (4, 1)), atol=1e-7)
    new = copy.deepcopy(motion)
    new.q += 10
    merged = motion.merge(new, 3, 5)
    np.testing.assert_allclose(merged.q[0], motion.q[3])
    np.testing.assert_allclose(merged.q[10], new.q[8])
    np.testing.assert_allclose(np.linalg.norm(merged.root, axis=1), 1)


@pytest.mark.parametrize("count", [0, 1, 65])
def test_prediction_rejects_invalid_frame_count(count):
    with pytest.raises(ValueError):
        Trajectory.from_prediction(prediction(), count)


def test_prediction_rejects_nonfinite_or_zero_quaternions():
    data = prediction()
    data[4, 7] = np.nan
    with pytest.raises(ValueError):
        Trajectory.from_prediction(data, 60)
    data = prediction()
    data[4, 3:7] = 0
    with pytest.raises(ValueError):
        Trajectory.from_prediction(data, 60)


def test_runtime_state_requires_fresh_finite_running_pose():
    state = {"schema": "sonic.h2.state.v1", "sent_at_ms": 1000, "phase": "running",
             "joint_pos": [0] * 31, "root_quat": [1, 0, 0, 0]}
    assert valid_state(state, 1499)
    assert not valid_state(state, 1500)
    assert not valid_state(state, 749)
    assert not valid_state({**state, "phase": "initializing"}, 1000)
    assert not valid_state({**state, "root_quat": [0] * 4}, 1000)
    assert not valid_state({**state, "joint_pos": [float("nan")] * 31}, 1000)
