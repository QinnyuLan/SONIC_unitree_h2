"""Read-only H2 solver telemetry; no extra forward/step or applied forces."""

from dataclasses import asdict
import time

import mujoco
import numpy as np

from sim.diagnostics.writer import DiagnosticLog
from sim.dds import H2_JOINT_NAMES_DDS

TIMING_FIELDS = ["time_monotonic_ms", "wall_elapsed_s", "physics_elapsed_s", "physics_steps",
                 "realtime_factor", "lag_ms", "max_lag_ms", "step_work_ms_mean",
                 "viewer_ms_mean", "other_ms_mean"]


def ground_wrench(model, data, foot_bodies):
    """Force ON the foot and moment about world origin, ground contacts only."""
    force, moment = np.zeros(3), np.zeros(3)
    count = 0
    for i in range(data.ncon):
        contact = data.contact[i]
        b1, b2 = (int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
        if b1 in foot_bodies and b2 == 0:
            sign = -1
        elif b2 in foot_bodies and b1 == 0:
            sign = 1
        else:
            continue
        local = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, local)
        rotation = contact.frame.reshape(3, 3).T
        f = sign * (rotation @ local[:3])
        torque = sign * (rotation @ local[3:])
        force += f
        moment += np.cross(contact.pos, f) + torque
        if local[0] > 0:
            count += 1
    # Intersection of resultant wrench with the horizontal z=0 plane.
    valid = force[2] > 1.0
    cop = np.array([-moment[1]/force[2], moment[0]/force[2], 0.]) if valid else np.full(3, np.nan)
    return count, force, moment, valid, cop


class SquatDiagnostics:
    def __init__(self, env, directory, hz=200):
        self.env = env
        m = env.mj_model
        if m.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4:
            raise ValueError("step diagnostics requires Euler/implicit integrator")
        if not np.isfinite(hz) or not 1 <= hz <= 1 / m.opt.timestep:
            raise ValueError("diagnostic Hz must be within [1, physics Hz]")
        self.order = np.array([env.body_joint_names.index(n) for n in H2_JOINT_NAMES_DDS])
        self.feet = {}
        for side in ("left", "right"):
            # The official sole mesh belongs to ankle_pitch_link, not ankle_roll_link.
            body = m.body(side + "_ankle_pitch_link").id
            subtree = {body}
            for i in range(body + 1, m.nbody):
                if int(m.body_parentid[i]) in subtree:
                    subtree.add(i)
            self.feet[side] = (body, subtree)
        self.stride = max(1, round(1 / (hz * m.opt.timestep)))
        self.step = 0
        self.epoch = 0
        self.pending = None
        self.origin = env.mj_data.xpos[env.root_body_id].copy()
        fields = ["time_monotonic_ms", "time_realtime_ms", "sim_time_s", "sim_step_end_s",
                  "reset_epoch", "elastic_enabled", "command_age_ms", "command_received"]
        for prefix, axes in (("root_pos_world", "xyz"), ("root_displacement_world", "xyz"),
                             ("root_quat_wxyz", "wxyz"), ("root_velocity_world", "xyz"),
                             ("com_world", "xyz"), ("band_force_world", "xyz")):
            fields += [f"{prefix}_{a}" for a in axes]
        for side in self.feet:
            for prefix in ("pos_world", "velocity_world", "force_world", "moment_world_origin", "cop_z0_world"):
                fields += [f"{side}_{prefix}_{a}" for a in "xyz"]
            fields += [f"{side}_contact_count", f"{side}_cop_valid"]
        self.groups = ("q", "dq", "q_target_used", "dq_target_used", "kp_used", "kd_used", "tau_ff_used",
                       "pd_requested", "motor_requested", "motor_response_prelimit",
                       "motor_response_requested", "motor_response_clipped", "motor_ctrl_clipped",
                       "motor_force_applied", "output_equivalent_motor_force", "motor_clipped", "extra_pitch_force")
        for group in self.groups:
            fields += [f"{group}_{i}" for i in range(31)]
        response = getattr(env, "ankle_response", None)
        params = [asdict(f.parameters) for _, _, f in response.axes] if response else []
        metadata = dict(schema="h2_squat_sim_truth_v1", source="mujoco_solver",
            config=env.config, joint_order=H2_JOINT_NAMES_DDS, requested_hz=hz,
            effective_sim_hz=1/(self.stride*m.opt.timestep), ankle_response_parameters=params,
            motor_response_candidate=dict(
                config=env.config.get("H2_MOTOR_RESPONSE", {}),
                motor_positions=getattr(env, "motor_response_groups", {}),
                status="effective candidate, not identified hardware parameters"),
            initial_qpos=env.mj_data.qpos.tolist(), initial_qvel=env.mj_data.qvel.tolist(),
            physics=dict(timestep=float(m.opt.timestep), integrator=int(m.opt.integrator),
                         gravity=m.opt.gravity.tolist(), body_mass=m.body_mass.tolist(),
                         geom_names=[m.geom(i).name for i in range(m.ngeom)],
                         geom_friction=m.geom_friction.tolist(),
                         actuator_ctrlrange=m.actuator_ctrlrange.tolist(),
                         actuator_forcerange=m.actuator_forcerange.tolist(),
                         body_motor_clip_limits=env.body_torque_limit.tolist()),
            motor_actuator_names=[m.actuator(int(env.body_actuator_index[i])).name for i in self.order],
            semantics={
                "clock": "host CLOCK_MONOTONIC milliseconds, aligns with deploy on this host",
                "state": "pre-integration q/dq; kinematics and solver forces from that step",
                "motor": "motor-side after closed-loop mapping; NOT output-joint measured torque",
                "motor_requested": "unmodified PD request after output-to-motor closed-chain mapping",
                "motor_response_prelimit": "candidate gain and exact-ZOH first-order lag applied to mapped motor request, before candidate cap",
                "motor_response_requested": "post-candidate motor request, after selected candidate effort caps and before native clipping; equals motor_requested when disabled",
                "motor_response_clipped": "candidate effort cap only; excludes gain or lag changes",
                "motor_clipped": "native clipping only, comparing motor_response_requested to final motor ctrl",
                "motor_response_state": "pre-limit effort, reset to zero; saturation does not feed back into lag state",
                "output_equivalent_motor_force": "pinv(J.T) times applied motor force using pre-step Jacobian; mapped estimate, excludes external/passive forces",
                "extra_pitch_force": "additional identified force on output DOF, separate from motor force",
                "cop": "ground-only resultant wrench at world z=0, NaN when Fz<=1 N; no contact estimate",
                "com": "root subtree mass center, world frame",
            })
        self.log = DiagnosticLog(directory, {"truth": fields, "realtime": TIMING_FIELDS}, metadata)
        print(f"[Diagnostics] Simulation truth: {directory} ({metadata['effective_sim_hz']:g} Hz)")

    def begin(self, pd_requested, motor_requested, motor_response_requested=None):
        self.step += 1
        if (self.step - 1) % self.stride:
            return
        e, o = self.env, self.order
        d = e.mj_data
        bridge = e.unitree_bridge
        last = getattr(bridge, "low_cmd_last_time", None)
        motor_ctrl = d.ctrl[e.body_actuator_index].copy()
        response = getattr(e, "motor_response", None)
        if motor_response_requested is None:
            motor_response_requested = motor_requested
        self.pending = dict(
            clock=time.monotonic_ns()/1e6, realtime=time.time_ns()/1e6, sim_time=d.time,
            age=(time.monotonic()-last)*1000 if last is not None else np.nan,
            received=int(getattr(bridge, "low_cmd_received", False)),
            elastic=int(bool(e.elastic_band and e.elastic_band.enable)),
            q=d.qpos[e.body_qpos_index][o].copy(), dq=d.qvel[e.body_dof_index][o].copy(),
            pd_requested=pd_requested[o].copy(), motor_requested=motor_requested[o].copy(),
            motor_response_prelimit=(response.last_prelimit[o].copy() if response
                                     else motor_requested[o].copy()),
            motor_response_requested=motor_response_requested[o].copy(),
            motor_response_clipped=(response.clipped[o].astype(int) if response else np.zeros(31, dtype=int)),
            motor_ctrl_clipped=motor_ctrl[o],
            motor_clipped=(np.abs(motor_response_requested[o]-motor_ctrl[o]) > 1e-8).astype(int),
            extra_pitch_force=(e.ankle_response.last_force[e.body_dof_index][o].copy()
                               if e.ankle_response else np.zeros(31)))
        for group, values in zip(("q_target_used", "dq_target_used", "kp_used", "kd_used", "tau_ff_used"),
                                 e.diagnostic_body_commands):
            self.pending[group] = values[o].copy()
        self.pending['mappings'] = [(g['output_positions'], g['motor_positions'],
                                    e._closed_loop_output_to_motor_jacobian(g).copy())
                                   for g in e.closed_loop_transmissions]

    def end(self):
        if self.pending is None:
            return
        e, p = self.env, self.pending
        self.pending = None
        m, d = e.mj_model, e.mj_data
        row = [p['clock'], p['realtime'], p['sim_time'], d.time, self.epoch,
               p['elastic'], p['age'], p['received']]
        root = e.root_body_id
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, root, velocity, 0)
        band = d.xfrc_applied[e.band_attached_link, :3] if e.elastic_band else np.zeros(3)
        row.extend(np.concatenate((d.xpos[root], d.xpos[root]-self.origin, d.xquat[root],
                                   velocity[3:], d.subtree_com[root], band)))
        for body, subtree in self.feet.values():
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
            count, force, moment, valid, cop = ground_wrench(m, d, subtree)
            row.extend(np.concatenate((d.xpos[body], velocity[3:], force, moment, cop)))
            row.extend((count, int(valid)))
        p['motor_force_applied'] = d.actuator_force[e.body_actuator_index][self.order]
        equivalent = d.actuator_force[e.body_actuator_index].copy()
        for outputs, motors, jacobian in p['mappings']:
            equivalent[outputs] = np.linalg.pinv(jacobian.T) @ d.actuator_force[e.body_actuator_index][motors]
        p['output_equivalent_motor_force'] = equivalent[self.order]
        for group in self.groups:
            row.extend(p[group])
        self.log.put("truth", row)

    def reset(self):
        self.epoch += 1
        self.pending = None
        self.origin = self.env.mj_data.xpos[self.env.root_body_id].copy()

    def record_timing(self, values):
        self.log.put("realtime", [values[name] for name in TIMING_FIELDS])

    def close(self, *, incomplete=False):
        self.log.close(incomplete=incomplete)
        print(f"[Diagnostics] Rows={self.log.counts}, dropped={self.log.dropped}, error={self.log.error}")
