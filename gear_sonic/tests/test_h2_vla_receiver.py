"""Exercise the actual C++ receiver on a private loopback port; never opens DDS."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time

from gear_sonic.utils.inference.h2_vla import ActionChunk, BinaryHands, action_message
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    build_head_pose_message,
    pack_pose_message,
)
import numpy as np
import zmq


def run(binary):
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    port = publisher.bind_to_random_port("tcp://127.0.0.1")
    with tempfile.TemporaryDirectory() as temporary:
        status = Path(temporary) / "status.jsonl"
        with (Path(temporary) / "cpp.log").open("w") as log:
            process = subprocess.Popen(
                [str(binary), str(port), str(status)],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
            )

            def latest():
                lines = status.read_text().splitlines() if status.exists() else []
                for line in reversed(lines):
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        continue
                return {}

            def expect(**values):
                state = latest()
                assert all(state.get(k) == v for k, v in values.items()), (values, state)

            def arm():
                publisher.send(build_command_message(True, False, False))
                time.sleep(0.12)

            def send_for(seconds):
                chunk = ActionChunk(
                    time.monotonic(),
                    {
                        "motion_token": np.full((1, 40, 64), 0.2, np.float32),
                        "left_hand_joints": np.zeros((1, 40, 6), np.float32),
                        "right_hand_joints": np.ones((1, 40, 6), np.float32),
                    },
                )
                stop = time.monotonic() + seconds
                while time.monotonic() < stop:
                    message, _, _ = action_message(chunk, time.monotonic(), 1, BinaryHands())
                    publisher.send(message)
                    time.sleep(0.02)

            try:
                time.sleep(0.5)
                arm()
                send_for(0.2)
                time.sleep(0.03)
                expect(has_token=True, has_left=True, start=True, play=True)
                assert abs(latest()["left_1"] - 0.7) < 1e-5
                # Head heartbeats must NOT refresh token validity.
                for _ in range(25):
                    publisher.send(build_head_pose_message(0.6981317, 0, time.monotonic()))
                    time.sleep(0.02)
                expect(has_token=False, has_left=False, start=True, play=False)
                send_for(0.2)
                expect(has_token=False, play=False)
                arm()
                send_for(0.2)
                time.sleep(0.03)
                expect(has_token=True, play=True)
                expired = {
                    "token_state": np.full((1, 64), 0.2, np.float32),
                    "h2_vla_deadline_monotonic": np.array([time.monotonic() - 1], np.float64),
                }
                publisher.send(pack_pose_message(expired, topic="pose", version=4))
                time.sleep(0.1)
                expect(has_token=False, play=False)
                arm()
                send_for(0.2)
                time.sleep(0.03)
                expect(has_token=True)
                invalid = {
                    "token_state": np.full((1, 64), np.nan, np.float32),
                    "h2_vla_deadline_monotonic": np.array([time.monotonic() + 0.5], np.float64),
                }
                publisher.send(pack_pose_message(invalid, topic="pose", version=4))
                time.sleep(0.1)
                expect(has_token=False, play=False)
                print(
                    "PASS: actual C++ packet/hand decoding, watchdog despite head heartbeat, latched hold, explicit re-arm, expired and NaN rejection; NO DDS opened."
                )
            except Exception:
                log.flush()
                print((Path(temporary) / "cpp.log").read_text())
                raise
            finally:
                process.terminate()
                process.wait(timeout=5)
                publisher.close(linger=0)
                context.term()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, required=True)
    run(p.parse_args().binary)
