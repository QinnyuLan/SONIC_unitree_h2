# Experimental ankle-pitch response model

`H2_loop_ankle_sysid.xml` preserves the official H2 meshes, inertias,
constraints and native actuator/passive parameters. It embeds additional
left/right dynamic-friction parameters in `h2_ankle_pitch_response_v1` custom
text, consumed by `gear_sonic.utils.mujoco_sim.ankle_response`.

Do not load this MJCF in a generic MuJoCo executable and assume the extra
friction is active: custom text alone does not apply forces.

From the SONIC-H2 root:

```bash
source .venv_sim/bin/activate
python scripts/view_h2_ankle_sysid.py --side right --trial 6:1 --repeat
```

Use `--baseline` to disable extra dynamics, `--side left` for the left ankle,
and `--trial all` for all 18 recorded conditions. This viewer initializes no
DDS connection. A separate full-body simulator profile is registered as
`h2_native_unitree_ros_loop_ankle_sysid`; it is not the default.

Validation scope: unloaded ready pose, original LowCmd, Kp=40/Kd=2.
See `docs/report/h2_ankle_pitch_identified_mujoco.md` for metrics and limitations.
