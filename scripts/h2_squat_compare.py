#!/usr/bin/env python3
"""Explicit, manually operated H2 release squat comparison launchers."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "gear_sonic_deploy/logs/squat_compare"
RELEASE = ROOT / "sonic_h2_native_release/exported"
REFERENCE = ROOT / "gear_sonic_deploy/reference/example_h2/deploy"
MOTION = "squat_001__A359"


def file_hash(path):
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def model_reference_hashes():
    sources = [RELEASE/"model_encoder_full.onnx", RELEASE/"model_decoder.onnx",
               ROOT/"gear_sonic_deploy/policy/release/observation_config_h2_native.yaml",
               REFERENCE/"h2_lora_reference_manifest.json"]
    sources += sorted((REFERENCE/MOTION).glob("*.csv"))
    return {str(s): file_hash(s) for s in sources}


def launch_args(args):
    path = LOGS / args.run
    # Remove implicit experiment overrides, especially auto-start and skipped release.
    env = {k: v for k, v in os.environ.items() if not k.startswith("GEAR_SONIC_")}
    env['GEAR_SONIC_DOMAIN_ID'] = '0'
    if args.command == "sim":
        version = "h2_native_unitree_ros_loop" + ("_ankle_sysid" if args.variant == "identified" else "")
        env['GEAR_SONIC_SIM_DIAGNOSTICS_DIR'] = str(path / "sim")
        env['GEAR_SONIC_SIM_DIAGNOSTICS_HZ'] = '200'
        cmd = [str(ROOT/".venv_sim/bin/python"), str(ROOT/"gear_sonic/scripts/run_sim_loop.py"),
               "--wbc-version", version, "--interface", "lo", "--realtime-max-lag-ms", "20"]
    elif args.command == "record":
        cmd = [str(ROOT/".venv_sim/bin/python"), str(ROOT/"scripts/record_h2_squat_dds.py"),
               "--target", args.target, "--interface", "lo" if args.target == "sim" else args.interface,
               "--domain", "0", "--output", str(path/"dds")]
    else:
        env['GEAR_SONIC_INITIAL_MOTION_NAME'] = MOTION
        cmd = ["bash", str(ROOT/"gear_sonic_deploy/deploy.sh"),
               "sim" if args.target == "sim" else args.interface,
               "--robot-profile", "h2_native", "--no-planner", "--input-type", "keyboard",
               "--hand-backend", "none", "--encoder-file", str(RELEASE/"model_encoder_full.onnx"),
               "--decoder-file", str(RELEASE/"model_decoder.onnx"),
               "--obs-config", str(ROOT/"gear_sonic_deploy/policy/release/observation_config_h2_native.yaml"),
               "--motion-data", str(REFERENCE), "--enable-csv-logs", "--logs-dir", str(path/"deploy")]
        for flag in ("waist-pitch-kp", "waist-roll-kp", "ankle-pitch-kp", "ankle-roll-kp", "ankle-pitch-kd"):
            cmd += ["--h2-" + flag + "-scale", "1.0"]
    return path, env, cmd


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("sim", "deploy", "record"))
    p.add_argument("--run", required=True, help="unique trial label, same in all terminals")
    p.add_argument("--variant", choices=("baseline", "identified"), default="identified")
    p.add_argument("--target", choices=("sim", "real"), default="sim")
    p.add_argument("--interface", default="eno1")
    p.add_argument("--dry-run", action="store_true", help="print only; no files, DDS or controller")
    args = p.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run):
        p.error("run must be a simple alphanumeric label (hyphen/underscore allowed)")
    if args.command == "sim" and args.target != "sim":
        p.error("sim command requires --target sim")
    if args.target == "real" and (args.interface == "lo" or not re.fullmatch(r"[A-Za-z0-9_.:-]+", args.interface)):
        p.error("real target requires the robot NIC, e.g. eno1")
    path, env, cmd = launch_args(args)
    print("Logs:", path, flush=True)
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        print(json.dumps({k: v for k, v in env.items() if k.startswith("GEAR_SONIC_")}, indent=2))
        return
    path.mkdir(parents=True, exist_ok=True)
    marker = path/(args.command + "_launch.json")
    output = path/({"sim": "sim", "record": "dds", "deploy": "deploy"}[args.command])
    if marker.exists() or output.exists():
        p.error(f"component already exists: {output}; choose a new --run for another trial")
    for previous in path.glob("*_launch.json"):
        if json.loads(previous.read_text())['target'] != args.target:
            p.error("cannot mix real and sim recordings in one run")
    metadata = dict(target=args.target, command=cmd, variant=args.variant if args.command == "sim" else None,
                    reference_type="h2", motion=MOTION, gains="all five scaling arguments explicitly 1.0",
                    files=model_reference_hashes(),
                    overrides={k: v for k, v in env.items() if k.startswith("GEAR_SONIC_")})
    if args.command == 'sim':
        implementation = [ROOT/'gear_sonic/utils/mujoco_sim'/name for name in (
            'base_sim.py', 'realtime.py', 'unitree_sdk2py_bridge.py', 'squat_diagnostics.py',
            'wbc_configs/h2_native_sonic_unitree_ros_loop.yaml',
            'wbc_configs/h2_native_sonic_unitree_ros_loop_ankle_sysid.yaml')]
        metadata['simulation_implementation_files'] = {str(s): file_hash(s) for s in implementation}
        metadata['realtime'] = dict(scheduler='absolute_deadline_v1', max_lag_ms=20,
                                    physics_dt_s=.001, viewer_state_only=True)
    for previous in path.glob("*_launch.json"):
        if json.loads(previous.read_text())['files'] != metadata['files']:
            p.error("model/reference/config files changed within this run; use a new --run")
    with marker.open("x") as f:
        json.dump(metadata, f, indent=2)
    os.chdir(ROOT)
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
