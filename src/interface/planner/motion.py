"""SONIC planner context, resampling and blending in NumPy.

Planner qpos uses G1 MuJoCo order; network commands use native G1 semantic order.
"""
from __future__ import annotations

import numpy as np
from interface.commands.protocol import array, g1_reference, planner_command

SEMANTIC_TO_MUJOCO = np.array([
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
    16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
])
POLICY_TO_H2_DDS = np.array([
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 5, 11,
    16, 23, 4, 10, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
])


def h2_to_g1_mujoco(joints):
    q = np.asarray(array(joints, (31,), "H2 joint_pos"))[POLICY_TO_H2_DDS]
    # Physical H2 hip pitch/roll axes projected into the G1 context coordinates.
    for pitch, roll, sign in ((0, 3, -1), (1, 4, 1)):
        q[pitch], q[roll] = (0.8660245537757874 * q[pitch],
            sign * 0.08700508624315262 * q[pitch] + 0.9847439527511597 * q[roll])
    result = np.empty(29)
    result[SEMANTIC_TO_MUJOCO] = q
    return result


def slerp(first, second, weight):
    """Pairwise shortest-path interpolation of wxyz quaternions."""
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    first = first / np.linalg.norm(first, axis=-1, keepdims=True)
    second = second / np.linalg.norm(second, axis=-1, keepdims=True)
    dot = np.sum(first * second, axis=-1, keepdims=True)
    second = np.where(dot < 0, -second, second)
    dot = np.clip(np.abs(dot), 0, 1)
    weight = np.asarray(weight)[..., None]
    theta = np.arccos(dot)
    sine = np.sin(theta)
    safe = np.where(sine > 1e-8, sine, 1)
    curved = (np.sin((1 - weight) * theta) * first + np.sin(weight * theta) * second) / safe
    result = np.where(dot > .9995, first + weight * (second - first), curved)
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


class Trajectory:
    """A 50 Hz G1 trajectory with semantic joints and world-frame root pose."""

    def __init__(self, q, dq, root, position):
        self.q, self.dq, self.root, self.position = [np.asarray(x, dtype=float) for x in (q, dq, root, position)]
        count = len(self.q)
        if not 2 <= count <= 1500:
            raise ValueError("Planner trajectory requires 2..1500 frames")
        for value, shape in ((self.q, (count, 29)), (self.dq, (count, 29)),
                             (self.root, (count, 4)), (self.position, (count, 3))):
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError("Invalid planner trajectory")
        norm = np.linalg.norm(self.root, axis=1, keepdims=True)
        if np.any(norm < 1e-6):
            raise ValueError("Planner produced a zero quaternion")
        self.root = self.root / norm

    @classmethod
    def from_prediction(cls, qpos, count):
        qpos = np.asarray(qpos, dtype=float)
        if not isinstance(count, int) or not 2 <= count <= 64 or qpos.shape != (64, 36):
            raise ValueError("Invalid planner prediction shape or frame count")
        qpos = qpos[:count]
        if not np.isfinite(qpos).all() or np.any(np.linalg.norm(qpos[:, 3:7], axis=1) < 1e-6):
            raise ValueError("Invalid planner prediction values")
        frames = np.arange(int(np.floor(count / 30 * 50))) * 30 / 50
        first = np.minimum(frames.astype(int), count - 1)
        second = np.minimum(first + 1, count - 1)
        weight = (frames - first)[:, None]
        values = (1 - weight) * qpos[first] + weight * qpos[second]
        root = slerp(qpos[first, 3:7], qpos[second, 3:7], weight[:, 0])
        q = values[:, 7:][:, SEMANTIC_TO_MUJOCO]
        dq = np.diff(q, axis=0) * 50
        return cls(q, np.concatenate((dq, dq[-1:])), root, values[:, :3])

    def context(self, start):
        frames = np.clip(start + np.arange(4) * 50 / 30, 0, len(self.q) - 1)
        first = frames.astype(int)
        second = np.minimum(first + 1, len(self.q) - 1)
        weight = (frames - first)[:, None]
        context = np.empty((4, 36), dtype=np.float32)
        context[:, :3] = (1 - weight) * self.position[first] + weight * self.position[second]
        context[:, 3:7] = slerp(self.root[first], self.root[second], weight[:, 0])
        context[:, 7 + SEMANTIC_TO_MUJOCO] = (1 - weight) * self.q[first] + weight * self.q[second]
        return context[None]

    def sparse(self, cursor):
        indices = np.clip(cursor + np.arange(10) * 5, 0, len(self.q) - 1)
        velocity = self.dq[indices].copy()
        velocity[indices == len(self.q) - 1] = 0
        return g1_reference(self.q[indices], velocity, self.root[indices])

    def merge(self, generated, cursor, generated_start):
        count = generated_start - cursor + len(generated.q)
        if not 2 <= count <= 1500:
            raise ValueError("Planner inference exceeded the available trajectory horizon")
        frames = np.arange(count)
        previous = np.clip(cursor + frames, 0, len(self.q) - 1)
        following = np.clip(cursor + frames - generated_start, 0, len(generated.q) - 1)
        weight = np.clip((frames - max(0, generated_start - cursor)) / 8, 0, 1)[:, None]
        def mix(name):
            return (1 - weight) * getattr(self, name)[previous] + weight * getattr(generated, name)[following]
        return Trajectory(mix("q"), mix("dq"),
            slerp(self.root[previous], generated.root[following], weight[:, 0]), mix("position"))


class G1Planner:
    """Python planner API; inference is exclusively through TensorRT Python."""

    def __init__(self, model, *, device=0):
        from interface.planner.engine import TensorRTEngine
        self.engine = TensorRTEngine(model, device=device)
        self.inputs = {
            "context_mujoco_qpos": np.zeros((1, 4, 36), np.float32),
            "target_vel": np.array([-1], np.float32), "mode": np.array([0], np.int64),
            "movement_direction": np.zeros((1, 3), np.float32),
            "facing_direction": np.array([[1, 0, 0]], np.float32),
            "random_seed": np.array([1234], np.int64),
            "has_specific_target": np.zeros((1, 1), np.int64),
            "specific_target_positions": np.zeros((1, 4, 3), np.float32),
            "specific_target_headings": np.zeros((1, 4), np.float32),
            "allowed_pred_num_tokens": np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]], np.int64),
            "height": np.array([-1], np.float32),
        }
        expected = {name: (value.shape, value.dtype) for name, value in self.inputs.items()}
        if self.engine.inputs != expected or self.engine.outputs != {
            "mujoco_qpos": ((1, 64, 36), np.dtype("float32")),
            "num_pred_frames": ((1,), np.dtype("int32")),
        }:
            self.close()
            raise ValueError("Planner model does not match the released G1 interface")

    def _infer(self, context, command):
        data = planner_command(**command)
        self.inputs["context_mujoco_qpos"] = np.asarray(context, dtype=np.float32)
        for name, field in (("mode", "mode"), ("target_vel", "speed"), ("height", "height"),
                            ("movement_direction", "movement"), ("facing_direction", "facing")):
            self.inputs[name][...] = data[field]
        output = self.engine.infer(self.inputs)
        return Trajectory.from_prediction(output["mujoco_qpos"][0], int(output["num_pred_frames"][0]))

    def initialize(self, h2_joints):
        context = np.zeros((1, 4, 36), np.float32)
        context[:, :, 2] = .788740
        context[:, :, 3] = 1  # Upstream planner begins in a zero-heading frame.
        context[:, :, 7:] = h2_to_g1_mujoco(h2_joints)
        return self._infer(context, planner_command())

    def update(self, motion: Trajectory, cursor: int, command: dict):
        generated_start = cursor + 2  # Same 40 ms look-ahead as upstream SONIC.
        return self._infer(motion.context(generated_start), command), generated_start

    def close(self):
        self.engine.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
