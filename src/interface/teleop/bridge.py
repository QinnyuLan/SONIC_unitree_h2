"""Adapt PICO/SMPL tracking to independent body, head and hand channels."""
from __future__ import annotations

import argparse
import json
import time
from contextlib import ExitStack
import numpy as np
import zmq
from scipy.spatial.transform import Rotation
from interface.commands import RuntimeClient, g1_reference, smpl_reference, vr_reference
from interface.commands.protocol import planner_command
from interface.teleop.transport.packets import HEADER_SIZE

DTYPES = {"f32": "<f4", "f64": "<f8", "i32": "<i4", "i64": "<i8", "u8": "u1", "bool": "?"}
TOPICS = ("head_pose", "command", "planner", "pose")
G1_DEFAULT = np.array([-.312, -.312, 0, 0, 0, 0, 0, 0, 0, .669, .669,
                       .2, .2, -.363, -.363, .2, -.2, 0, 0, 0, 0, .6, .6, 0, 0, 0, 0, 0, 0])
UPPER = np.array([2, 5, 8, 11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])


def standing_g1():
    return g1_reference(np.tile(G1_DEFAULT, (10, 1)), np.zeros((10, 29)), np.tile([1, 0, 0, 0], (10, 1)))


def unpack(message: bytes):
    topic = next((t for t in TOPICS if message.startswith(t.encode())), None)
    if topic is None or len(message) > 131072:
        raise ValueError("unknown topic or oversized tracking packet")
    start = len(topic)
    header = json.loads(message[start:start + HEADER_SIZE].split(b"\0", 1)[0])
    if header.get("endian") != "le":
        raise ValueError("expected little-endian packet")
    offset, result = start + HEADER_SIZE, {}
    for field in header["fields"]:
        dtype = np.dtype(DTYPES[field["dtype"]])
        shape = field["shape"]
        if not shape or len(shape) > 4 or any(type(n) is not int or n < 1 or n > 4096 for n in shape):
            raise ValueError("invalid tracking field shape")
        count = int(np.prod(shape))
        size = count * dtype.itemsize
        if offset + size > len(message) or field["name"] in result:
            raise ValueError("truncated or duplicate tracking field")
        value = np.frombuffer(message, dtype=dtype, count=count, offset=offset).reshape(shape)
        if not np.isfinite(value).all():
            raise ValueError("non-finite tracking data")
        result[field["name"]] = value
        offset += size
    if offset != len(message):
        raise ValueError("unexpected tracking payload bytes")
    stamp = float(result.get("timestamp_realtime", [0])[0])
    if stamp and abs(time.time() - stamp) > .5:
        raise ValueError("stale tracking packet or unsynchronized clocks")
    return topic, result


def pose_reference(data, kind, lower_body, heading=0.0):
    if kind == "vr":
        return vr_reference(data["vr_position"].reshape(3, 3), data["vr_orientation"].reshape(3, 4), lower_body)
    joints = data["smpl_joints"][-10:]
    roots = data["smpl_root_quat"][-10:].reshape(-1, 4)
    wrists = data["joint_pos"][-10:, 23:29]
    if heading:
        rotation = Rotation.from_euler("z", heading)
        joints = rotation.apply(joints.reshape(-1, 3)).reshape(joints.shape)
        roots = (rotation * Rotation.from_quat(roots, scalar_first=True)).as_quat(scalar_first=True)
    return smpl_reference(joints, roots, wrists)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking", default="tcp://127.0.0.1:5556")
    parser.add_argument("--runtime", default="tcp://127.0.0.1:5558")
    parser.add_argument("--head", default="tcp://127.0.0.1:5562")
    parser.add_argument("--hands", default="tcp://127.0.0.1:5563")
    parser.add_argument("--planner-input", default="tcp://127.0.0.1:5560")
    parser.add_argument("--planner-output", default="tcp://127.0.0.1:5561")
    parser.add_argument("--pose-kind", choices=["smpl", "vr"], default="smpl")
    parser.add_argument("--takeover", action="store_true")
    args = parser.parse_args()
    context = zmq.Context.instance()
    tracking = context.socket(zmq.SUB)
    tracking.setsockopt(zmq.LINGER, 0); tracking.setsockopt(zmq.RCVHWM, 16)
    for topic in TOPICS:
        tracking.setsockopt_string(zmq.SUBSCRIBE, topic)
    tracking.connect(args.tracking)
    planned = context.socket(zmq.SUB)
    planned.setsockopt(zmq.LINGER, 0); planned.setsockopt(zmq.CONFLATE, 1)
    planned.setsockopt_string(zmq.SUBSCRIBE, ""); planned.connect(args.planner_output)
    with ExitStack() as stack:
        body = stack.enter_context(RuntimeClient("teleop", args.runtime, takeover=args.takeover))
        head = stack.enter_context(RuntimeClient("teleop_head", args.head, takeover=args.takeover))
        hands = stack.enter_context(RuntimeClient("teleop_hands", args.hands, takeover=args.takeover))
        planner = stack.enter_context(RuntimeClient("teleop_planner", args.planner_input, ttl_ms=500))
        lower = standing_g1()
        lower_stamp = 0
        latest = {}
        use_planner = False
        control_stamp = 0
        heading = 0.0
        try:
            while True:
                now = time.monotonic()
                for _ in range(32):
                    if not tracking.poll(0):
                        break
                    try:
                        topic, data = unpack(tracking.recv())
                        latest[topic] = (data, now)
                        if topic == "command":
                            if bool(data["stop"][0]):
                                body.send("stop"); return
                            use_planner = bool(data["planner"][0]); control_stamp = now
                        elif topic == "pose":
                            heading += float(data.get("heading_increment", [0])[0])
                        elif topic == "head_pose":
                            pitch, yaw = data["head_pitch_yaw"]
                            head.send("head", {"pitch": pitch, "yaw": yaw})
                        if topic in {"pose", "planner"}:
                            hand_data = {side: data[f"{side}_hand_joints"].reshape(-1)
                                         for side in ("left", "right") if f"{side}_hand_joints" in data}
                            if hand_data:
                                hands.send("hands", hand_data)
                    except (ValueError, KeyError, TypeError) as error:
                        print(f"Tracking packet rejected: {error}")
                if planned.poll(0):
                    try:
                        message = planned.recv_json()
                        if message["kind"] == "g1" and 0 <= time.time() * 1000 - message["sent_at_ms"] < message["ttl_ms"]:
                            lower = g1_reference(**message["data"]); lower_stamp = now
                    except (ValueError, KeyError, TypeError) as error:
                        print(f"Planner packet rejected: {error}")
                lower_fresh = now - lower_stamp < .25
                if use_planner and now - control_stamp < .5:
                    entry = latest.get("planner")
                    if entry and now - entry[1] < .25:
                        data = entry[0]
                        planner.send("planner", planner_command(int(data["mode"][0]), data["movement"], data["facing"],
                                                                float(data.get("speed", [-1])[0]), float(data.get("height", [-1])[0])))
                        if "vr_position" in data:
                            body.send("vr", pose_reference(data, "vr", lower if lower_fresh else standing_g1()))
                        elif lower_fresh:
                            reference = dict(lower)
                            if "upper_body_position" in data:
                                q = np.array(lower["joint_pos"]); dq = np.array(lower["joint_vel"])
                                q[:, UPPER] = data["upper_body_position"]; dq[:, UPPER] = 0
                                reference = g1_reference(q, dq, lower["root_quat"])
                            body.send("g1", reference)
                else:
                    entry = latest.get("pose")
                    if entry and now - entry[1] < .25:
                        body.send(args.pose_kind, pose_reference(entry[0], args.pose_kind, standing_g1(), heading))
                time.sleep(.02)
        except KeyboardInterrupt:
            pass
        finally:
            tracking.close(); planned.close()


if __name__ == "__main__":
    main()
