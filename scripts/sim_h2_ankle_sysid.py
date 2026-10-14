#!/usr/bin/env python3
"""Offline command replay in an ideal unloaded official H2 closed-loop fixture.

Never connects DDS. Locks the pelvis; ready-pose logs PD-hold other joints,
while legacy sine logs constrain non-ankle outputs. Ankle PR joints, motors and
linkages remain dynamic. All contact is disabled.
This is an ideal fixture, NOT a validated model of a particular real support rig.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external_dependencies/unitree_sdk2_python"))

import numpy as np
import yaml

from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import (
    H2_DDS_TO_LOCAL_MUJOCO, H2_JOINT_NAMES_DDS, H2_JOINT_NAMES_LOCAL_MUJOCO,
)
from h2_ankle_sysid import CMD_FIELDS, NAMES, Recorder, State, save_json


CONFIG = ROOT / "gear_sonic/utils/mujoco_sim/wbc_configs/h2_native_sonic_unitree_ros_loop.yaml"
READY_SCHEMAS = {"h2_ankle_range_v1", "h2_ankle_dynamic_v1", "h2_ankle_dynamic_v2"}


class FixtureEnv(DefaultEnv):
    def _build_runtime_scene_xml(self, xml_path):
        root = ET.parse(xml_path).getroot()
        root.find("compiler").set("meshdir", str(xml_path.parent))
        pelvis = root.find("worldbody/body[@name='pelvis']")
        pelvis.set("pos", "0 0 1.5")
        pelvis.set("quat", " ".join(map(str, self.config["DEFAULT_BASE_QUAT"])))
        for geom in root.iter("geom"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
        equality = root.find("equality")
        ET.SubElement(equality, "weld", name="fixture_pelvis", body1="pelvis", solref="0.005 1")
        for name, q in zip(H2_JOINT_NAMES_LOCAL_MUJOCO, self.config["DEFAULT_DOF_ANGLES"]):
            if self.config.get("FIXTURE_LOCK_NON_ANKLE", True) and "ankle_" not in name:
                ET.SubElement(equality, "joint", name="fixture_"+name,
                    joint1=name, polycoef=f"{q} 0 0 0 0", solref="0.005 1")
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            ET.ElementTree(root).write(f, encoding="utf-8")
            return f.name


def make_config(domain, pose_from=None):
    config = yaml.safe_load(CONFIG.read_text())
    config.update(DOMAIN_ID=domain, INTERFACE="lo", PRINT_SCENE_INFORMATION=False,
                  ENABLE_ELASTIC_BAND=False, CALIBRATE_BASE_Z_TO_H2_SOLE_CONTACT=False,
                  DEFAULT_BASE_POS=[0, 0, 1.5])
    if pose_from:
        meta = json.loads(pose_from.read_text())
        if meta.get("schema") not in READY_SCHEMAS | {"h2_ankle_sysid_v1"} or meta.get("joints_dds") != H2_JOINT_NAMES_DDS:
            raise ValueError("expected H2 ankle sine/range metadata with the same DDS joint order")
        q = np.asarray(meta["base_q"], dtype=float)
        quat = np.asarray(meta["initial_quaternion"], dtype=float)
        if q.shape != (31,) or quat.shape != (4,) or not np.isfinite(q).all() or not np.isfinite(quat).all():
            raise ValueError("invalid captured pose")
        if not .9 < np.linalg.norm(quat) < 1.1:
            raise ValueError("invalid captured quaternion")
        angles = np.zeros(31)
        angles[H2_DDS_TO_LOCAL_MUJOCO] = q
        config["DEFAULT_DOF_ANGLES"] = angles.tolist()
        config["DEFAULT_BASE_QUAT"] = (quat/np.linalg.norm(quat)).tolist()
        config["FIXTURE_LOCK_NON_ANKLE"] = meta["schema"] not in READY_SCHEMAS
    return config


def apply_pitch_response(env, parameters):
    """Opt-in effective pitch friction hypothesis, not identified hardware constants."""
    if not isinstance(parameters, dict) or set(parameters) - {"left", "right"}:
        raise ValueError("pitch response expects only left/right frictionloss increments in Nm")
    changes = {}
    for side, value in parameters.items():
        options = value if isinstance(value, dict) else {"frictionloss_add_nm": value}
        if set(options) - {"frictionloss_add_nm", "hard_friction"}:
            raise ValueError("unknown pitch response option")
        increment = float(options["frictionloss_add_nm"])
        hard = options.get("hard_friction", False)
        if not isinstance(hard, bool):
            raise ValueError("hard_friction must be a boolean")
        if not np.isfinite(increment) or not 0 <= increment <= 10:
            raise ValueError("pitch frictionloss increment must be finite and within [0, 10] Nm")
        name = side + "_ankle_pitch_joint"
        joint = env.mj_model.joint(name)
        dof = int(joint.dofadr[0])
        original = float(env.mj_model.dof_frictionloss[dof])
        changes[name] = dict(original_nm=original, added_nm=increment, total_nm=original+increment,
                             hard_friction=hard,
                             original_solref=env.mj_model.dof_solref[dof].tolist(),
                             original_solimp=env.mj_model.dof_solimp[dof].tolist())
    # Validate every entry before changing any model property.
    for name, change in changes.items():
        dof = int(env.mj_model.joint(name).dofadr[0])
        env.mj_model.dof_frictionloss[dof] = change["total_nm"]
        if change["hard_friction"]:
            env.mj_model.dof_solref[dof] = [max(2*env.sim_dt, .002), 1]
            env.mj_model.dof_solimp[dof] = [.999, .999, .001, .5, 2]
        change["solref"] = env.mj_model.dof_solref[dof].tolist()
        change["solimp"] = env.mj_model.dof_solimp[dof].tolist()
    return changes


def replay(source, output, response_config=None):
    """Replay the recorded command timeline in simulation time, without any DDS."""
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_

    meta = json.loads((source / "metadata.json").read_text())
    config = make_config(73, source / "metadata.json")
    with (source / "commands.csv").open() as f:
        rows = list(csv.DictReader(f))
    if meta["schema"] in READY_SCHEMAS:
        # Compare from deploy ready; do not replay the operator's startup pose.
        start = next((i for i, r in enumerate(rows) if r["phase"] == "waiting"), None)
        if start is None:
            raise ValueError("run did not reach ready; no experiment to replay")
        rows = rows[start:]
    if not rows:
        raise ValueError("no commands to replay")
    timestamps = [int(r["send_begin_monotonic_ns"]) for r in rows]
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("command timestamps must increase")
    for row in rows:
        for prefix in ("q_des", "dq_des", "kp", "kd", "tau_ff"):
            if not all(np.isfinite(float(row[f"{prefix}_{i}"])) for i in range(31)):
                raise ValueError("non-finite recorded command")
    env = FixtureEnv(config, onscreen=False)
    response = (yaml.safe_load(response_config.read_text()) if response_config else {})
    changes = apply_pitch_response(env, response)
    cmd = unitree_hg_msg_dds__LowCmd_()
    bridge = SimpleNamespace(low_cmd=cmd, num_body_motor=31, num_hand_motor=0,
        motor_to_sim_joint_index=np.array([env.body_joint_names.index(n) for n in NAMES]),
        joystick=None, use_sensor=False, body_low_cmd_available=lambda: True, body_low_cmd_is_fresh=lambda: True,
        PublishLowState=lambda obs: None)
    env.set_unitree_bridge(bridge)
    rec = Recorder(output)
    meta["args"] = {**meta["args"], "target": "sim", "interface": "none (offline)", "output": str(output)}
    meta.update(source_log=str(source.resolve()), source_status=meta["status"], status="replaying",
                offline_replay=True, fixture=("pelvis fixed, remaining joints PD-held, no contacts"
                    if meta["schema"] in READY_SCHEMAS else "pelvis and non-ankle outputs fixed, no contacts"),
                timebase="synthetic monotonic timestamps aligned to original command timeline; no network delay",
                replay_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                replay_config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
                replay_asset_sha256=hashlib.sha256((ROOT / config["ROBOT_SCENE"]).read_bytes()).hexdigest(),
                commands_origin="original recorded command rows; associated source-state fields are not simulated")
    meta["pitch_response_hypothesis"] = changes
    if response_config:
        meta["pitch_response_config"] = str(response_config.resolve())
        meta["pitch_response_config_sha256"] = hashlib.sha256(response_config.read_bytes()).hexdigest()
    save_json(output / "metadata.json", meta)
    first, last = timestamps[0], timestamps[-1]
    pos, emitted = 0, 0
    dt_ns = round(env.sim_dt * 1e9)
    print(f"Offline replay: {(last-first)/1e9:.1f} simulated seconds; NO DDS connection", flush=True)
    progress = time.monotonic()
    try:
        for now in range(first, last + dt_ns, dt_ns):
            while pos + 1 < len(rows) and timestamps[pos+1] <= now:
                pos += 1
            while emitted <= pos:
                rec.put("commands", [rows[emitted][key] for key in CMD_FIELDS])
                emitted += 1
            row = rows[pos]
            for i in range(31):
                m = cmd.motor_cmd[i]
                m.mode = 1
                m.q, m.dq = float(row[f"q_des_{i}"]), float(row[f"dq_des_{i}"])
                m.kp, m.kd, m.tau = float(row[f"kp_{i}"]), float(row[f"kd_{i}"]), float(row[f"tau_ff_{i}"])
            obs = env.prepare_obs()
            order = bridge.motor_to_sim_joint_index
            state = State(now, 0, round(env.mj_data.time*1000), 0, meta["mode_machine"],
                tuple(obs["body_q"][order]), tuple(obs["body_dq"][order]), tuple(obs["body_tau_est"][order]),
                (0,)*62, (0,)*31, tuple(obs["floating_base_pose"][3:7]),
                tuple(obs["floating_base_vel_local"][3:6]), tuple(obs["floating_base_acc"][:3]), 0, False)
            rec.put("states", [state.receive_ns, state.source_ns, state.tick, state.mode_pr, state.mode_machine,
                0, 0, *state.q, *state.dq, *state.tau, *state.temperature, *state.errors,
                *state.quaternion, *state.gyro, *state.accel])
            if rec.error:
                raise RuntimeError(rec.error)
            env.sim_step()
            if time.monotonic()-progress > 20:
                print(f"Replay {(now-first)/1e9:.1f} / {(last-first)/1e9:.1f} s", flush=True)
                progress = time.monotonic()
        meta["status"] = "completed"
    except BaseException:
        meta["status"] = "replay_failed"
        raise
    finally:
        rec.close()
        meta["log_counts"] = rec.counts
        if rec.error:
            meta["status"] = rec.error
        save_json(output / "metadata.json", meta)
    if rec.error:
        raise RuntimeError(rec.error)
    print(f"Replay saved: {output}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay-log", type=Path, required=True, help="captured sysid log directory")
    p.add_argument("--output", type=Path, help="new directory for offline replay logs")
    p.add_argument("--response-config", type=Path,
                   help="optional YAML with left/right pitch frictionloss increments (Nm); no DDS")
    args = p.parse_args()
    replay(args.replay_log, args.output or args.replay_log / "sim_replay", args.response_config)


if __name__ == "__main__":
    main()
