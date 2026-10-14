"""Independent Python/TensorRT G1 planner service. No DDS or robot SDK imports."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
import signal
import time
import uuid

import numpy as np
import zmq
from interface.commands.protocol import array, envelope
from interface.commands.receiver import Receiver
from interface.planner.motion import G1Planner

from interface.paths import PLANNER_MODEL

DEFAULT_MODEL = PLANNER_MODEL


def valid_state(state, wall_ms=None):
    wall_ms = time.time_ns() // 1_000_000 if wall_ms is None else wall_ms
    try:
        sent = state["sent_at_ms"]
        if (state["schema"] != "sonic.h2.state.v1" or state["phase"] != "running"
                or type(sent) is not int or not -250 <= wall_ms - sent < 500):
            return False
        array(state["joint_pos"], (31,), "state joints")
        quat = array(state["root_quat"], (4,), "state quaternion")
        norm = np.linalg.norm(quat)
        return np.isfinite(norm) and norm > 1e-6
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def serve(args):
    running = True
    def stop(*_):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    with ExitStack() as stack:
        planner = stack.enter_context(G1Planner(args.model, device=args.device))
        context = zmq.Context()
        stack.callback(context.term)
        receiver = Receiver(context, args.input_bind, {"planner", "stand"})
        stack.callback(receiver.close)
        pub, push, sub = [context.socket(kind) for kind in (zmq.PUB, zmq.PUSH, zmq.SUB)]
        for socket in (pub, push, sub):
            socket.setsockopt(zmq.LINGER, 0)
            stack.callback(socket.close)
        pub.setsockopt(zmq.SNDHWM, 2)
        pub.bind(args.output_bind)
        push.setsockopt(zmq.SNDHWM, 2)
        push.setsockopt(zmq.IMMEDIATE, 1)
        if not args.publish_only:
            push.connect(args.runtime)
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        sub.setsockopt(zmq.CONFLATE, 1)
        sub.setsockopt(zmq.MAXMSGSIZE, 131072)
        sub.connect(args.state)
        state, motion, owner = None, None, None
        session, seq = uuid.uuid4().hex, 0
        timeline = last_plan = 0

        def send(kind, data):
            nonlocal seq
            message = envelope(kind, data, source="planner", session=session, seq=seq)
            seq += 1
            pub.send_json(message, flags=zmq.NOBLOCK)
            if not args.publish_only:
                try:
                    push.send_json(message, flags=zmq.NOBLOCK)
                except zmq.Again:
                    pass  # A missing/busy runtime cannot block the source service.

        def receive():
            nonlocal state
            receiver.poll()
            try:
                candidate = sub.recv_json(flags=zmq.NOBLOCK)
                state = candidate if valid_state(candidate) else None
            except zmq.Again:
                pass
            except (ValueError, UnicodeError):
                state = None

        print("Planner ready; waiting for runtime state and planner commands", flush=True)
        try:
            while running:
                tick = time.monotonic()
                receive()
                command = receiver.inbox.active()
                next_owner = (command["source"], command["session"]) if command else None
                if not command or not valid_state(state) or (owner is not None and owner != next_owner):
                    if motion is not None:
                        send("stand", {})
                    motion, owner = None, None
                    time.sleep(.02)
                    continue
                if motion is None:
                    motion = planner.initialize(state["joint_pos"])
                    owner = next_owner
                    session, seq = uuid.uuid4().hex, 0
                    timeline = time.monotonic()
                    last_plan = timeline - 1
                if time.monotonic() - last_plan >= .2:
                    cursor = max(0, int((time.monotonic() - timeline) * 50))
                    generated, start = planner.update(motion, cursor, command["data"])
                    advanced = max(0, int((time.monotonic() - timeline) * 50))
                    motion = motion.merge(generated, advanced, start)
                    timeline += advanced / 50
                    last_plan = time.monotonic()
                # Inference may outlast the request/state. Recheck before publishing.
                receive()
                current = receiver.inbox.active()
                if (current and (current["source"], current["session"]) == owner
                        and valid_state(state) and running):
                    cursor = max(0, int((time.monotonic() - timeline) * 50))
                    send("g1", motion.sparse(cursor))
                time.sleep(max(0, tick + .02 - time.monotonic()))
        finally:
            send("stand", {})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--input-bind", default="tcp://127.0.0.1:5560")
    parser.add_argument("--output-bind", default="tcp://127.0.0.1:5561")
    parser.add_argument("--runtime", default="tcp://127.0.0.1:5558")
    parser.add_argument("--state", default="tcp://127.0.0.1:5559")
    parser.add_argument("--publish-only", action="store_true")
    args = parser.parse_args()
    serve(args)


if __name__ == "__main__":
    main()
