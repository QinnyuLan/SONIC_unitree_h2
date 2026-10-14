#!/usr/bin/env python3
"""Native MuJoCo window for recorded ankle tests. NO DDS or robot connection."""

import argparse
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import sys
import time
import threading
from types import SimpleNamespace

import mujoco
import mujoco.viewer
import numpy as np

from sim_h2_ankle_sysid import FixtureEnv, make_config, ROOT, NAMES

SOURCES = {
    "left": ROOT/"gear_sonic_deploy/logs/ankle_dynamic_left_20260906_210301_532438",
    "right": ROOT/"gear_sonic_deploy/logs/ankle_dynamic_right_20260906_211438_540301",
}
MODEL = ROOT/"gear_sonic/data/assets/robot_description/h2_ankle_sysid/H2_loop_ankle_sysid.xml"


@contextmanager
def passive_viewer(model, data, key_callback):
    # MuJoCo 3.10 close() only signals its daemon render thread. Finish that
    # thread before Python's atexit GLFW cleanup can tear down its GL context.
    before = set(threading.enumerate())
    viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
    viewer_threads = set(threading.enumerate()) - before
    try:
        yield viewer
    finally:
        viewer.close()
        for thread in viewer_threads:
            thread.join(timeout=5)
        if any(thread.is_alive() for thread in viewer_threads):
            raise RuntimeError("MuJoCo viewer thread did not finish closing")


class ViewFixture(FixtureEnv):
    def _build_runtime_scene_xml(self, xml_path):
        # FixtureEnv expects meshes beside the source; identified derivatives
        # keep the original meshes in the official asset directory.
        result = super()._build_runtime_scene_xml(xml_path)
        import xml.etree.ElementTree as ET
        tree = ET.parse(result)
        tree.getroot().find("compiler").set("meshdir", str(ROOT/"gear_sonic/data/assets/robot_description/unitree_ros/h2_description"))
        tree.write(result, encoding="utf-8")
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=MODEL)
    p.add_argument("--side", choices=("left", "right", "both"), default="both")
    p.add_argument("--trial", default="6:1", help="amplitude degrees:frequency Hz, or all")
    p.add_argument("--baseline", action="store_true", help="disable identified friction for A/B")
    p.add_argument("--view", choices=("ankles", "full"), default="ankles")
    p.add_argument("--speed", type=float, default=1.)
    p.add_argument("--repeat", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--render-check", type=Path, help="save representative rendered images without a window")
    p.add_argument("--max-seconds", type=float, default=0.)
    a = p.parse_args()
    if not np.isfinite(a.speed) or a.speed <= 0 or a.speed > 4:
        p.error("speed must be in (0,4]")
    if not np.isfinite(a.max_seconds) or a.max_seconds < 0:
        p.error("max-seconds must be finite and nonnegative")
    if a.render_check and a.repeat:
        p.error("render-check cannot be combined with repeat")
    if a.render_check:
        from PIL import Image
    sides = ("left", "right") if a.side == "both" else (a.side,)
    config = make_config(73, SOURCES[sides[0]]/"metadata.json")
    config["ROBOT_SCENE"] = str(a.model.resolve())
    if a.baseline:
        config["ANKLE_PITCH_RESPONSE"] = {}
    env = ViewFixture(config, onscreen=False)
    if not a.baseline and env.ankle_response is None:
        raise ValueError("selected model contains no ankle response parameters")
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    command = unitree_hg_msg_dds__LowCmd_()
    order = np.array([env.body_joint_names.index(n) for n in NAMES])
    env.set_unitree_bridge(SimpleNamespace(low_cmd=command, num_body_motor=31, num_hand_motor=0,
        motor_to_sim_joint_index=order, joystick=None, use_sensor=False,
        body_low_cmd_available=lambda: True, body_low_cmd_is_fresh=lambda: True,
        PublishLowState=lambda obs: None))
    meta = json.loads((SOURCES[sides[0]]/"metadata.json").read_text())
    trials = [(s["amplitude_deg"], s["frequency_hz"]) for s in meta["stages"]]
    if a.trial != "all":
        try:
            selected = tuple(map(float, a.trial.split(":")))
        except ValueError:
            p.error("trial must use amplitude:frequency, for example 6:1")
        if selected not in trials:
            p.error("trial must be one of the 18 recorded amplitude/frequency pairs")
        trials = [selected]
    state = dict(paused=False, restart=False, next=False)

    def key(code):
        if code == 32:
            state["paused"] = not state["paused"]
        elif code in (82, 114):
            state["restart"] = True
        elif code in (78, 110):
            state["next"] = True

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    left_foot = env.mj_data.body("left_ankle_pitch_link").xpos
    right_foot = env.mj_data.body("right_ankle_pitch_link").xpos
    camera.lookat[:] = (left_foot+right_foot)/2+[0, 0, .12]
    camera.distance, camera.elevation, camera.azimuth = 1.45, -12, -150
    if a.view == "full":
        camera.lookat[:] = [0, 0, 1.2]
        camera.distance = 3.3
    viewer_context = nullcontext(None) if a.headless or a.render_check else passive_viewer(env.mj_model, env.mj_data, key)
    env.mj_model.vis.global_.offwidth = max(960, env.mj_model.vis.global_.offwidth)
    env.mj_model.vis.global_.offheight = max(720, env.mj_model.vis.global_.offheight)
    renderer = mujoco.Renderer(env.mj_model, height=720, width=960) if a.render_check else None
    if a.render_check:
        a.render_check.mkdir(parents=True, exist_ok=False)
    print(f"NO DDS connection. Model={'official baseline' if a.baseline else 'identified candidate'}, side={a.side}, trials={len(trials)}", flush=True)
    print("Space: pause/resume; R: restart trial; N: next trial; close window: exit.", flush=True)
    initial = env.mj_data.qpos.copy()
    total = 0.
    images = []
    try:
        with viewer_context as viewer:
            if viewer:
                viewer.cam.lookat[:] = camera.lookat
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = camera.distance, camera.azimuth, camera.elevation
            index = 0
            while index < len(trials):
                amplitude, frequency = trials[index]
                stem = f"{amplitude:g}deg_{frequency:g}Hz"
                data = {side: np.load(SOURCES[side]/"friction_hard"/(stem+".npz"))["commands"] for side in sides}
                duration = min(c[-1, 0] for c in data.values())
                mujoco.mj_resetData(env.mj_model, env.mj_data)
                env.mj_data.qpos[:] = initial
                mujoco.mj_forward(env.mj_model, env.mj_data)
                env._closed_loop_jacobian_cache_time = None
                if env.ankle_response:
                    env.ankle_response.reset()
                state["restart"], state["next"] = False, False
                begin = time.monotonic()
                print(f"{stem}, Kp=40 Kd=2, recorded targets, duration={duration:.2f}s", flush=True)
                step = 0
                while env.mj_data.time < duration:
                    if viewer and not viewer.is_running():
                        return
                    if state["restart"] or state["next"]:
                        break
                    if state["paused"]:
                        if viewer:
                            viewer.sync()
                        time.sleep(.02)
                        begin = time.monotonic()-env.mj_data.time/a.speed
                        continue
                    now = env.mj_data.time
                    sample = data[sides[0]]
                    ci = max(0, np.searchsorted(sample[:, 0], now, side="right")-1)
                    values = sample[ci, 3:].reshape(5, 31).copy()
                    if len(sides) == 2:
                        c = data["right"]
                        j = max(0, np.searchsorted(c[:, 0], now, side="right")-1)
                        values[:, 11] = c[j, 3:].reshape(5, 31)[:, 11]
                    for j in range(31):
                        m = command.motor_cmd[j]
                        m.q, m.dq, m.kp, m.kd, m.tau = values[:, j]
                    env.sim_step()
                    step += 1
                    total += env.sim_dt
                    if step % 20 == 0 and viewer:
                        viewer.sync()
                    if renderer and len(images) < 3 and now >= (2.25+len(images)*.25)/frequency:
                        renderer.update_scene(env.mj_data, camera=camera)
                        pixels = renderer.render().copy()
                        Image.fromarray(pixels).save(a.render_check/f"frame_{len(images)}.png")
                        images.append(pixels)
                    if not a.headless and not renderer:
                        remaining = begin+env.mj_data.time/a.speed-time.monotonic()
                        if remaining > 0:
                            time.sleep(remaining)
                    if a.max_seconds and total >= a.max_seconds:
                        return
                if not state["restart"]:
                    index += 1
                if index == len(trials) and a.repeat:
                    index = 0
    finally:
        if renderer:
            renderer.close()
            if len(images) < 3 and sys.exc_info()[0] is None:
                raise RuntimeError("render check ended before all three snapshots; increase max-seconds")
            elif len(images) >= 3:
                variance = [float(np.std(frame)) for frame in images]
                difference = float(np.mean(np.abs(images[0].astype(float)-images[-1].astype(float))))
                check = dict(frame_std=variance, mean_pixel_change=difference,
                             nonblank=min(variance)>5, moving=difference>.01)
                (a.render_check/"render_check.json").write_text(json.dumps(check, indent=2)+"\n")
                print(check, flush=True)
                if not check["nonblank"] or not check["moving"]:
                    raise RuntimeError("render check failed")


if __name__ == "__main__":
    main()
