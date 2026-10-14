#!/usr/bin/env python3
"""H2 N1.7 client. Default is shadow (observations/inference only, no commands).

Live commands are typed in this terminal: stand, run, pause, stop, quit.
No command starts automatically. C++ has an independent deadline watchdog.
"""

# ruff: noqa: E402

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import select
import signal
import socket
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external_dependencies/unitree_sdk2_python"))

from gear_sonic.utils.inference.h2_vla import (
    DECODER_SHA256,
    ENCODER_SHA256,
    PROMPT,
    ActionChunk,
    BinaryHands,
    PolicyRPC,
    action_message,
    observation_from_arrays,
    validate_controller,
    validate_deployment,
    validate_modality,
    validate_samples,
)
import numpy as np


def verify_sonic(model):
    for filename, expected in [
        ("model_decoder.onnx", DECODER_SHA256),
        ("model_encoder_full.onnx", ENCODER_SHA256),
    ]:
        digest = hashlib.sha256()
        with (model / filename).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"SONIC model mismatch: {model / filename}")


def replay(args, output):
    """Network-only smoke/latency check. Never imports DDS or opens action PUB."""
    fixture = np.load(args.fixture, allow_pickle=False)
    rpc = PolicyRPC(args.host, args.port, timeout_ms=30000)
    latencies, errors = [], []
    try:
        validate_modality(rpc.call("get_modality_config"))
        manifest = rpc.call("h2_deployment_info")
        validate_deployment(manifest)
        (output / "model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for i in range(args.samples):
            index = i % len(fixture["rgb"])
            obs = observation_from_arrays(
                fixture["body_q"][index],
                fixture["hand_q"][index],
                fixture["quaternion"][index],
                fixture["rgb"][index],
                args.prompt,
            )
            start = time.monotonic()
            action = rpc.get_action(obs)
            chunk = ActionChunk(start, action)
            elapsed = time.monotonic() - start
            latencies.append(elapsed)
            if "target_token" in fixture:
                errors.append(
                    float(
                        np.mean(
                            (action["motion_token"][0, 0] - fixture["target_token"][index]) ** 2
                        )
                    )
                )
            np.savez_compressed(output / f"replay_{i:03d}.npz", **action)
            print(
                json.dumps(
                    {
                        "sample": i,
                        "latency_s": elapsed,
                        "fresh_on_arrival": elapsed < 0.8,
                        "shapes": {k: list(v.shape) for k, v in chunk.action.items()},
                    }
                ),
                flush=True,
            )
        steady = latencies[1:] or latencies
        result = {
            "mode": "replay_no_hardware",
            "samples": args.samples,
            "latencies_s": latencies,
            "steady_mean_s": float(np.mean(steady)),
            "steady_p95_s": float(np.percentile(steady, 95)),
            "single_frame_token_mse": float(np.mean(errors)) if errors else None,
        }
        (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        rpc.close()


def live_or_shadow(args, output):
    import cv2
    from gear_sonic.data.h2_collection.collector import DDSInput, Hub, camera_input, zmq_input
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
        build_command_message,
        build_head_pose_message,
    )
    import zmq

    if "GStreamer:                   YES" not in cv2.getBuildInformation():
        raise RuntimeError("Use .venv_data_collection with its GStreamer-enabled cv2")
    live = args.mode == "live"
    if live:
        verify_sonic(args.sonic_model)
    # Match the collector's exclusive ownership of the RTP receive port.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("", args.rgb_port))
    lock = open(f"/tmp/h2_n17_vla_{os.getuid()}_{args.domain}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    hub = Hub()
    context, publisher, dds = None, None, None
    events = (output / "events.jsonl").open("a", buffering=1)
    requests, results = queue.Queue(maxsize=1), queue.Queue(maxsize=2)
    busy, ready = threading.Event(), threading.Event()
    mode, generation, cached = "idle", 0, None
    last_request, last_status, last_wait = 0.0, 0.0, 0.0
    last_send, frame, chunk_number = None, 0, 0
    hands = BinaryHands()

    def event(kind, **values):
        record = {
            "time": datetime.now().astimezone().isoformat(),
            "monotonic": time.monotonic(),
            "event": kind,
            **values,
        }
        events.write(json.dumps(record) + "\n")
        if kind != "action":
            print(json.dumps(record), flush=True)

    def worker():
        rpc = PolicyRPC(args.host, args.port)
        try:
            while not hub.stopping.is_set():
                if not ready.is_set():
                    try:
                        validate_modality(rpc.call("get_modality_config"))
                        manifest = rpc.call("h2_deployment_info")
                        validate_deployment(manifest)
                        (output / "model_manifest.json").write_text(
                            json.dumps(manifest, indent=2) + "\n"
                        )
                        ready.set()
                    except Exception as exc:
                        results.put((None, f"Policy handshake failed: {exc}"))
                        hub.stopping.wait(1)
                        continue
                try:
                    request = requests.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    obs, origin, epoch, number = request
                    chunk = ActionChunk(origin, rpc.get_action(obs), epoch)
                    np.savez_compressed(
                        output / f"chunk_{number:06d}.npz",
                        **chunk.action,
                        rgb=obs["video"]["ego_view"][0, 0],
                        state=np.concatenate(list(obs["state"].values()), axis=-1)[0, 0],
                        origin=np.array(origin),
                    )
                    results.put((chunk, None))
                except Exception as exc:
                    # A server restart must re-check model identity. The main
                    # loop latches live execution to hold on this error.
                    ready.clear()
                    results.put((None, str(exc)))
                finally:
                    busy.clear()
        finally:
            ready.clear()
            rpc.close()

    def hold(reason):
        nonlocal mode, generation, cached
        mode = "hold" if mode != "idle" else "idle"
        generation += 1
        cached = None
        event("hold", reason=reason, receiver_fallback_ms=250)

    def shutdown(*_):
        hub.stopping.set()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    try:
        if live:
            context = zmq.Context()
            publisher = context.socket(zmq.PUB)
            publisher.setsockopt(zmq.LINGER, 0)
            publisher.setsockopt(zmq.SNDHWM, 3)
            # Same-host receiver is mandatory for the monotonic deadline contract.
            publisher.bind("tcp://127.0.0.1:5556")
        dds = DDSInput(hub, args.interface, args.domain)
        for target, target_args in [
            (camera_input, (hub, args.rgb_port, 1920, 1080)),
            (zmq_input, (hub, "tcp://127.0.0.1:5556", args.state_address)),
            (worker, ()),
        ]:
            threading.Thread(target=target, args=target_args, daemon=True).start()
        event(
            "started",
            mode=args.mode,
            prompt=args.prompt,
            commands="stand / run / pause / stop / quit",
            publisher=bool(publisher),
        )
        while not hub.stopping.is_set():
            tick = time.monotonic()
            samples = hub.snapshot()
            valid, reason, ages = False, "waiting for observations", {}
            try:
                # C++ publishes robot_config while waiting for stand, but its
                # g1_debug heartbeat only starts after entering CONTROL.
                ages = validate_samples(
                    samples, time.monotonic_ns(), require_controller=live and mode != "idle"
                )
                if live:
                    validate_controller(hub.config)
                valid = True
            except Exception as exc:
                reason = str(exc)
            if not valid and mode == "run":
                hold(reason)
            if not valid and tick - last_wait >= 2:
                event("waiting", reason=reason)
                last_wait = tick
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline()
                if not line:
                    if live:
                        shutdown()
                    else:
                        time.sleep(0.02)
                command = line.strip().lower()
                if command in ("quit", "exit"):
                    shutdown()
                elif command == "stop" and live:
                    publisher.send(build_command_message(start=False, stop=True, planner=False))
                    event("stop", note="Explicit controller stop; operator must restart deploy")
                    shutdown()
                elif command == "pause" and live:
                    hold("operator pause")
                elif command in ("stand", "run") and live:
                    if not valid or not ready.is_set():
                        event(
                            "command_rejected",
                            command=command,
                            reason=reason if not valid else "Policy is not ready",
                        )
                    elif command == "run" and mode == "idle":
                        event(
                            "command_rejected",
                            command=command,
                            reason="Use stand first and verify stable standing",
                        )
                    elif command == "run" and (cached is None or tick >= cached.deadline):
                        event(
                            "command_rejected",
                            command=command,
                            reason="Wait for a fresh prediction after model warm-up before run",
                        )
                    elif (
                        command == "run"
                        and np.max(
                            np.abs(
                                np.array(samples["lowstate"].data["q"])[29:31] - [np.deg2rad(40), 0]
                            )
                        )
                        > 0.1
                    ):
                        event(
                            "command_rejected",
                            command=command,
                            reason="Head is not at training pitch=40deg/yaw=0deg",
                        )
                    else:
                        generation += 1
                        cached = None
                        hands.reset(samples["hand_state"].data["q"])
                        publisher.send(build_command_message(start=True, stop=False, planner=False))
                        mode = "run" if command == "run" else "hold"
                        last_request = 0
                        event("operator_command", command=command, mode=mode, generation=generation)
            if live and mode != "idle":
                publisher.send(build_head_pose_message(np.deg2rad(40), 0, tick))
            try:
                chunk, error = results.get_nowait()
                if error:
                    if mode == "run":
                        hold(error)
                    else:
                        event("inference_error", error=error)
                elif chunk.generation == generation:
                    cached = chunk
                    event(
                        "prediction",
                        latency_s=time.monotonic() - chunk.origin,
                        generation=generation,
                    )
            except queue.Empty:
                pass
            if (
                valid
                and ready.is_set()
                and not busy.is_set()
                and tick - last_request >= args.inference_interval
            ):
                image = cv2.resize(
                    samples["camera"].data["image"], (640, 360), interpolation=cv2.INTER_AREA
                )
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                body, hand = samples["lowstate"].data, samples["hand_state"].data
                origin = time.monotonic()
                obs = observation_from_arrays(
                    body["q"], hand["q"], body["imu"]["quaternion"], image, args.prompt
                )
                busy.set()
                requests.put_nowait((obs, origin, generation, chunk_number))
                chunk_number += 1
                last_request = origin
            if live and mode == "run" and cached is not None:
                try:
                    now = time.monotonic()
                    if last_send is not None and now - last_send > 0.2:
                        raise ValueError("Action publisher missed 200ms; explicit re-arm required")
                    message, index, proxy = action_message(cached, now, frame, hands)
                    publisher.send(message)
                    last_send = now
                    event(
                        "action",
                        frame=frame,
                        index=index,
                        age_s=now - cached.origin,
                        deadline=cached.deadline,
                        token=cached.action["motion_token"][0, index].tolist(),
                        inspire_open=hands.open.tolist(),
                        protocol_hand=proxy.tolist(),
                    )
                    frame += 1
                except Exception as exc:
                    hold(str(exc))
            else:
                last_send = None
            if tick - last_status >= 2:
                event(
                    "status",
                    mode=mode,
                    observation_valid=valid,
                    inference_ready=ready.is_set(),
                    frames_sent=frame,
                    observation_ages_s=ages,
                )
                last_status = tick
            time.sleep(max(0, 0.02 - (time.monotonic() - tick)))
    finally:
        hub.stopping.set()
        # Stop publishing: receiver independently latches to the standing reference.
        # Do not implicitly send C++ stop, which terminates its standing controller.
        if dds is not None:
            dds.close()
        if publisher is not None:
            publisher.close(linger=0)
            context.term()
        event(
            "client_exit",
            fallback="receiver standing watchdog" if live else "no commands published",
        )
        events.close()
        lock.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["shadow", "live", "replay", "check"], default="shadow")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5550)
    p.add_argument("--interface", default="eno1")
    p.add_argument("--domain", type=int, default=0)
    p.add_argument("--rgb-port", type=int, default=5002)
    p.add_argument("--state-address", default="tcp://127.0.0.1:5557")
    p.add_argument("--inference-interval", type=float, default=0.4)
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument(
        "--sonic-model",
        type=Path,
        default=ROOT / "sonic_h2_native_release/experiments/gpu3_20260909_stage2_step005000",
    )
    p.add_argument(
        "--fixture", type=Path, default=ROOT / "models/h2_n1d7/latest/replay_fixture.npz"
    )
    p.add_argument("--samples", type=int, default=12)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if not 0.05 <= args.inference_interval <= 0.4 or args.samples < 1:
        p.error("inference interval must be 0.05..0.4 seconds and samples positive")
    if args.mode == "check":
        verify_sonic(args.sonic_model)
        print(
            "H2 observation/command modules import successfully; SONIC hashes match; no hardware opened."
        )
        return
    output = (
        args.output
        or ROOT / "gear_sonic_deploy/logs" / f"h2_vla_{args.mode}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "run_config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    if args.mode == "replay":
        replay(args, output)
    else:
        live_or_shadow(args, output)


if __name__ == "__main__":
    main()
