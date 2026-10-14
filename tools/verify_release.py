#!/usr/bin/env python3
"""Fast, offline release preflight for SONIC_unitree_h2.

The check only verifies files and Python imports. It never starts DDS,
connects to a robot, opens a camera, or sends motor commands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_file(path: Path, required: bool = True) -> bool:
    ok = path.is_file()
    mark = "OK" if ok else ("MISSING" if required else "optional")
    print(f"[{mark:7}] {path.relative_to(ROOT)}")
    return ok or not required


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imports", action="store_true", help="also import the simulation packages")
    parser.add_argument("--sim", action="store_true", help="load H2 and run 100 headless physics steps without DDS")
    parser.add_argument("--teleop", action="store_true", help="load the VR mapper, XR binding, and G1/H2 kinematics without devices")
    parser.add_argument("--require-planner", action="store_true", help="require the separately downloaded G1 planner")
    args = parser.parse_args()

    print(f"SONIC_unitree_h2: {ROOT}")
    print(f"platform: {platform.system()} {platform.machine()} Python {platform.python_version()}")
    required = [
        "pyproject.toml",
        "assets/robots/h2/H2_loop.xml",
        "assets/robots/h2/H2_loop.urdf",
        "assets/robots/g1/g1_29dof_with_hand.urdf",
        "assets/human/human_joints_info.pkl",
        "third_party/unitree_sdk2_python/setup.py",
        "third_party/xrobotoolkit_pybind/CMakeLists.txt",
        "third_party/xrobotoolkit_sdk/manifest.json",
        "src/sim/__main__.py",
        "src/interface/teleop/pico.py",
        "scripts/build_controller.sh",
        "config/sim/h2.yaml",
        "scripts/run_h2_teleop_head_adjust.sh",
        "src/runtime/CMakeLists.txt",
        "src/runtime/src/inference/InferenceEngine.cpp",
        "src/runtime/third_party/unitree_sdk2/CMakeLists.txt",
        "config/h2_deploy.env",
        "scripts/run_runtime.sh",
        "models/released/model_encoder_full.onnx",
        "models/released/model_encoder.onnx",
        "models/released/model_decoder.onnx",
        "models/released/SHA256SUMS",
        "models/released/deployment_manifest.json",
        "src/runtime/src/main.cpp",
        "src/runtime/include/observations.hpp",
        "src/interface/replay.py",
        "src/interface/teleop/bridge.py",
        "src/interface/planner/service.py",
        "src/interface/planner/motion.py",
        "src/interface/planner/engine.py",
    ]
    good = all([check_file(ROOT / path) for path in required])
    if not (ROOT / "third_party/unitree_sdk2_python/setup.py").is_file() or not (
        ROOT / "third_party/xrobotoolkit_pybind/CMakeLists.txt"
    ).is_file():
        print("Initialize dependencies: git submodule update --init --recursive")
    sdk_manifest = ROOT / "third_party/xrobotoolkit_sdk/manifest.json"
    if sdk_manifest.is_file():
        for entry in json.loads(sdk_manifest.read_text())["files"]:
            path = sdk_manifest.parent / entry["path"]
            ok = path.is_file() and sha256(path) == entry["sha256"]
            print(f"[{'OK' if ok else 'FAIL':7}] SHA256 XR SDK {entry['path']}")
            good = good and ok
    # Use the same destination and pinned checksum as the download command.
    sys.path.insert(0, str(ROOT))
    from download_from_hf import PLANNER_FILE, PLANNER_SHA256
    planner = ROOT / "models" / PLANNER_FILE[1]
    planner_ok = check_file(planner, required=args.require_planner)
    if planner.is_file():
        planner_ok = sha256(planner) == PLANNER_SHA256
        print(f"[{'OK' if planner_ok else 'FAIL':7}] SHA256 G1 planner")
    else:
        print("G1 planner is downloaded separately; see docs/PLANNER.md.")
        print("  python3 download_from_hf.py --planner-only")
    good = good and planner_ok
    sums = ROOT / "models/released/SHA256SUMS"
    if sums.is_file():
        for line in sums.read_text().splitlines():
            expected, name = line.split(maxsplit=1)
            path = sums.parent / name
            ok = path.is_file() and sha256(path) == expected
            print(f"[{'OK' if ok else 'FAIL':7}] SHA256 {path.relative_to(ROOT)}")
            good = good and ok
    for script in ROOT.glob("scripts/*.sh"):
        if not os.access(script, os.X_OK):
            print(f"[WARN   ] not executable: {script.relative_to(ROOT)}")
    if args.imports:
        checks = [
            ("command interfaces", "interface"),
            ("simulator", "sim"),
            ("mujoco", "mujoco"),
            ("numpy", "numpy"),
            ("pyzmq", "zmq"),
            ("Unitree DDS bridge", "sim.dds"),
        ]
        for label, module in checks:
            try:
                __import__(module)
            except Exception as exc:  # pragma: no cover - diagnostic path
                good = False
                print(f"[FAIL   ] import {label}: {exc}")
            else:
                print(f"[OK     ] import {label}")
    if args.sim:
        try:
            import mujoco
            import numpy as np
            from sim.config import SimLoopConfig
            from sim.environment import DefaultEnv
            config = SimLoopConfig(enable_onscreen=False).load_wbc_yaml()
            env = DefaultEnv(config, onscreen=False, offscreen=False)
            assert env.unitree_bridge is None
            assert env.num_body_dof == 31
            for _ in range(100):
                mujoco.mj_step(env.mj_model, env.mj_data)
            assert np.isfinite(env.mj_data.qpos).all()
            print("[OK     ] H2 assets and 100 offline physics steps")
        except Exception as exc:
            good = False
            print(f"[FAIL   ] simulator: {exc}")
    if args.teleop:
        try:
            import xrobotoolkit_sdk  # Import only; do not initialize devices.
            from interface.teleop.pico import ThreePointPose
            from interface.kinematics.g1 import instantiate_g1_robot_model
            from interface.kinematics.h2 import instantiate_h2_robot_model
            instantiate_g1_robot_model()
            instantiate_h2_robot_model()
            print("[OK     ] VR mapper, XR binding, and G1/H2 kinematics")
        except Exception as exc:
            good = False
            print(f"[FAIL   ] teleoperation: {exc}")
    manifest = ROOT / "release/source_manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text())
        print(f"manifest: {data.get('file_count')} files, {data.get('total_bytes')} bytes")
    print("preflight: PASS" if good else "preflight: FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
