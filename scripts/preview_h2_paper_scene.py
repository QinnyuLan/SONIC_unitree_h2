#!/usr/bin/env python3
"""Render the actual H2/G1 paper scene at its reset pose, without DDS or control.

This is a style preview, not a policy rollout or evidence of tracking quality.
Use MUJOCO_GL=egl on a machine without a display.
"""

import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "test_logs/h2_paper_preview/paper.png")
    parser.add_argument("--style", choices=["default", "paper"], default="paper")
    parser.add_argument("--robot", choices=["h2", "g1"], default="h2")
    parser.add_argument("--azimuth", type=float, default=135)
    parser.add_argument("--elevation", type=float, default=-15)
    parser.add_argument("--distance", type=float)
    args = parser.parse_args()
    args.output = args.output.resolve()
    os.chdir(ROOT)

    import mujoco
    from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
    from gear_sonic.utils.mujoco_sim.presentation import configure_presentation_camera

    config = SimLoopConfig(
        wbc_version="h2_native_unitree_ros_loop" if args.robot == "h2" else "sonic_model12",
        visual_style=args.style,
        enable_onscreen=False, presentation_azimuth=args.azimuth,
        presentation_elevation=args.elevation,
        presentation_distance=args.distance if args.distance is not None else (4.5 if args.robot == "h2" else 3.2),
        presentation_lookat=(0.0, 0.0, 0.85 if args.robot == "h2" else 0.65),
    ).load_wbc_yaml()
    config["PRINT_SCENE_INFORMATION"] = False
    env = DefaultEnv(config, onscreen=False)
    env.mj_model.vis.global_.offwidth = 1920
    env.mj_model.vis.global_.offheight = 1080
    camera = mujoco.MjvCamera()
    configure_presentation_camera(camera, env.mj_model, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(env.mj_model, height=1080, width=1920) as renderer:
        renderer.update_scene(env.mj_data, camera=camera)
        pixels = renderer.render()
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "1920x1080", "-i", "pipe:0", "-frames:v", "1", str(args.output),
        ], input=pixels.tobytes(), check=True)
    print(args.output)


if __name__ == "__main__":
    main()
