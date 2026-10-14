#!/usr/bin/env python3
"""Launch an interactive replay or planner demo on loopback MuJoCo DDS."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import time

import zmq

from interface.motion_replay import Motion

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOTION = ROOT / "motions/example_g1_replay/walking_quip_360_R_002__A428"


class Processes:
    def __init__(self, directory, domain):
        self.directory = directory
        self.children = []
        self.environment = dict(os.environ, DDS_DOMAIN=str(domain), PYTHONUNBUFFERED="1",
                                OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def start(self, name, command, **environment):
        path = self.directory / f"{name}.log"
        with path.open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=dict(self.environment, **environment),
                                       stdout=log, stderr=log, start_new_session=True)
        self.children.append((name, process))
        return process

    def error(self, name, reason):
        path = self.directory / f"{name}.log"
        tail = path.read_text(errors="replace")[-2500:]
        return RuntimeError(f"{name}: {reason}\n{tail}\nLog: {path}")

    def completed(self, name, process, timeout):
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            raise self.error(name, f"startup timed out after {timeout:g} seconds") from None
        if process.returncode:
            raise self.error(name, f"exited with status {process.returncode}")

    @staticmethod
    def stop(process):
        if process.poll() is not None:
            return
        for sig, timeout in ((signal.SIGINT, 4), (signal.SIGKILL, 2)):
            try:
                os.killpg(process.pid, sig)
                process.wait(timeout=timeout)
                return
            except ProcessLookupError:
                return
            except subprocess.TimeoutExpired:
                continue
        print(f"PID {process.pid} did not exit after SIGKILL; check the GPU/driver state.",
              file=sys.stderr, flush=True)

    def close(self):
        for _, process in reversed(self.children):
            self.stop(process)


def latest_physics(path):
    """Read the latest complete diagnostic row without loading the whole recording."""
    if not path.is_file():
        return {}
    with path.open("rb") as stream:
        header = stream.readline().decode().strip().split(",")
        offset = stream.tell()
        stream.seek(0, 2)
        stream.seek(max(offset, stream.tell() - 65536))
        lines = stream.read().split(b"\n")
    if len(lines) < 2:
        return {}
    values = next(csv.reader([lines[-2].decode()]))
    return dict(zip(header, values)) if len(header) == len(values) else {}


def run(args, directory, endpoints, processes):
    runtime_args = ["--command-bind", endpoints["body"], "--state-bind", endpoints["state"],
                    "--head-bind", endpoints["head"], "--hands-bind", endpoints["hands"]]
    print("Checking GPU access…", flush=True)
    gpu = processes.start("gpu", ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    processes.completed("gpu", gpu, 8)
    print("Preparing the policy engines (first use can take several minutes)…", flush=True)
    warmup = processes.start("warmup", ["bash", "scripts/run_runtime.sh", "dry-run",
                                        "--seconds", "1", *runtime_args])
    processes.completed("warmup", warmup, args.startup_timeout)

    watched = []
    if args.demo == "planner":
        print("Preparing the planner engine…", flush=True)
        planner = processes.start("planner", ["bash", "scripts/run_planner.sh",
            "--input-bind", endpoints["planner"], "--output-bind", endpoints["planned"],
            "--runtime", endpoints["body"], "--state", endpoints["state"]])
        deadline = time.monotonic() + args.startup_timeout
        while "Planner ready;" not in (directory / "planner.log").read_text(errors="replace"):
            if planner.poll() is not None or time.monotonic() >= deadline:
                raise processes.error("planner", "did not become ready")
            time.sleep(.1)
        watched.append(("planner", planner))

    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt(zmq.CONFLATE, 1)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.connect(endpoints["state"])
    state = None
    last_state = time.monotonic()
    states = (directory / "runtime_states.jsonl").open("w")
    truth = directory / "physics/truth.csv"
    try:
        simulator = processes.start("simulator", ["bash", "scripts/run_h2_paper_sim.sh"],
            GEAR_SONIC_MUJOCO_ENABLE_ELASTIC="1",
            GEAR_SONIC_SIM_DIAGNOSTICS_DIR=str(directory / "physics"),
            GEAR_SONIC_SIM_DIAGNOSTICS_HZ="50")
        runtime = processes.start("runtime", ["bash", "scripts/run_runtime.sh", "sim", *runtime_args])
        watched.extend((("simulator", simulator), ("runtime", runtime)))

        def poll():
            nonlocal state, last_state
            for name, process in watched:
                if process.poll() is not None:
                    raise processes.error(name, f"exited with status {process.returncode}")
            if sub.poll(20):
                state = sub.recv_json()
                last_state = time.monotonic()
                states.write(json.dumps(state) + "\n")
                if time.time_ns() // 1_000_000 - state["sent_at_ms"] > 500:
                    raise RuntimeError("Runtime state is stale")
                _, x, y, _ = state["root_quat"]
                tilt = math.acos(max(-1., min(1., 1 - 2 * (x*x + y*y))))
                if tilt > math.radians(60):
                    raise RuntimeError("Robot tilt exceeded 60 degrees; stopping the demo")
            if time.monotonic() - last_state > 10:
                raise RuntimeError("No fresh runtime state for 10 seconds")

        def wait_until(predicate, timeout, reason):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                poll()
                if predicate():
                    return
            raise RuntimeError(reason)

        wait_until(lambda: state and state["phase"] == "running", 30, "Standing policy did not start")
        print("\nStanding policy is running. In the MuJoCo window, press 9 to release the elastic support.\n"
              "Observe unsupported standing, then return here and press Enter to start motion.\n"
              "Ctrl+C stops this demo and its child processes.", flush=True)
        while True:
            poll()
            if select.select([sys.stdin], [], [], 0)[0]:
                if not sys.stdin.readline():
                    raise RuntimeError("Interactive terminal input closed")
                physics = latest_physics(truth)
                if physics.get("elastic_enabled") == "0":
                    break
                print("Elastic support is still enabled. Press 9 in the viewer, wait a second, then press Enter.", flush=True)

        if args.demo == "replay":
            motion = Motion.load(args.motion)
            producer = processes.start("replay", [sys.executable, "-m", "interface.motion_replay",
                                                  str(args.motion), "--endpoint", endpoints["body"]])
            expected_source = "motion_replay"
            expected_kind = motion.reference_type
            duration = motion.duration + 5
        else:
            producer = processes.start("control", [sys.executable, "-m", "interface.planner.client",
                "--endpoint", endpoints["planner"], "--mode", "1", "--speed", str(args.speed),
                "--movement", "1", "0", "0"])
            expected_source, expected_kind = "planner", "g1"
            duration = args.seconds
        print(f"Running {args.demo}…", flush=True)
        started = time.monotonic()
        accepted = 0
        while time.monotonic() - started < duration:
            poll()
            if state and state["source"] == expected_source and state["body_kind"] == expected_kind:
                accepted += 1
            if producer.poll() is not None:
                if producer.returncode or args.demo == "planner":
                    raise RuntimeError(f"Command producer exited unexpectedly; see {directory}")
                break
            if not accepted and time.monotonic() - started > 5:
                raise RuntimeError("Runtime has not accepted the motion source within five seconds")
        if args.demo == "replay" and producer.poll() is None:
            raise RuntimeError("Replay did not finish within its motion duration")
        if not accepted:
            raise RuntimeError("Runtime did not acknowledge any motion commands")
        processes.stop(producer)
        wait_until(lambda: state and state["body_kind"] == "stand", 3, "Runtime did not return to standing")
        print("Motion ended; observing standing for three seconds…", flush=True)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            poll()
        print("Demo finished. Review the viewer and recorded physics to assess tracking quality.", flush=True)
    finally:
        states.close()
        sub.close()
        context.term()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo", choices=("replay", "planner"))
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION, help="G1, H2 or SMPL clip for replay")
    parser.add_argument("--speed", type=float, default=.3, help="planner walking speed, m/s")
    parser.add_argument("--seconds", type=float, default=12, help="planner walking duration")
    parser.add_argument("--domain", type=int, default=220, help="unused loopback DDS domain")
    parser.add_argument("--startup-timeout", type=float, default=600, help="engine preparation timeout")
    parser.add_argument("--check", action="store_true", help="check local files and replay data without GPU or DDS")
    args = parser.parse_args()
    if not 0 <= args.domain <= 230:
        parser.error("--domain must be between 0 and 230")
    if not all(math.isfinite(x) and x > 0 for x in (args.speed, args.seconds, args.startup_timeout)):
        parser.error("speed, seconds and startup-timeout must be finite and positive")
    for path in (ROOT / "build/runtime/bin/sonic_h2_runtime", ROOT / ".venv_sim/bin/python"):
        if not path.is_file():
            parser.error(f"Missing {path}; follow docs/DEPLOYMENT.md")
    if args.demo == "replay":
        args.motion = args.motion.resolve()
        motion = Motion.load(args.motion)
        print(f"Replay: {motion.reference_type.upper()}, {motion.duration:.2f} seconds, {args.motion}", flush=True)
    else:
        for path in (ROOT / ".venv_planner/bin/python", ROOT / "models/motion_planner_g1/planner_sonic.onnx"):
            if not path.is_file():
                parser.error(f"Missing {path}; follow docs/PLANNER.md")
    if args.check:
        print("Local demo prerequisites passed. GPU inference and closed-loop tracking were not tested.")
        return 0
    if not sys.stdin.isatty() or not os.environ.get("DISPLAY"):
        parser.error("Run from a terminal in a local graphical desktop (DISPLAY required)")

    output = ROOT / "outputs/demos"
    output.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f"{args.demo}-", dir=output))
    print(f"Logs: {directory}\nDDS: loopback lo, domain {args.domain}", flush=True)
    processes = Processes(directory, args.domain)
    status, error = "interrupted", None
    # Hold the lock until all children have been stopped. IPC paths are unique per run.
    with (output / f"domain-{args.domain}.lock").open("w") as lock, tempfile.TemporaryDirectory(prefix="sonic-demo-") as ipc:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"Another demo is using domain {args.domain}; choose a different --domain")
        endpoints = {name: f"ipc://{ipc}/{name}" for name in ("body", "state", "head", "hands", "planner", "planned")}
        def interrupt(*_):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupt)
        try:
            run(args, directory, endpoints, processes)
            status = "completed"
        except KeyboardInterrupt:
            print("Stopping demo…", flush=True)
        except (RuntimeError, OSError, ValueError) as exc:
            status, error = "failed", str(exc)
            print(error, file=sys.stderr, flush=True)
        finally:
            # A second interrupt must not abandon the other processes during cleanup.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            processes.close()
            (directory / "result.json").write_text(json.dumps({
                "demo": args.demo, "status": status, "error": error,
                "dds_domain": args.domain, "interface": "lo", "hardware_connected": False,
                "scope": "Interactive demonstration; completion is not a tracking-accuracy benchmark",
            }, indent=2) + "\n")
    return 1 if status == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
