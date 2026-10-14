# H2 ankle-pitch dynamic-response candidate

This is an opt-in, unloaded closed-loop response model. It is not a uniquely
identified motor/transmission model and has not been validated in loaded gait.
Neither the release policy nor real-robot gains are changed.

## Inputs and identification

Hardware recordings:

- `gear_sonic_deploy/logs/ankle_dynamic_left_20260906_210301_532438`
- `gear_sonic_deploy/logs/ankle_dynamic_right_20260906_211438_540301`

Each contains 18 amplitude/frequency conditions: +/-1 to +/-25 degrees,
0.1 to 3 Hz, centered at -0.363 rad. These ranges are NOT a Cartesian grid:
the 2-3 Hz tests use +/-1 degree, not +/-25 degrees. Kp=40 Nm/rad,
Kd=2 Nm*s/rad, desired velocity=0, feedforward torque=0. Original 50 Hz
position targets and approximately 500 Hz published command timestamps are
replayed causally with zero-order hold in a 1 kHz MuJoCo physics loop.
The input is actual LowCmd, never an encoder reference trajectory.

The official H2 closed-chain model is kept, including its linkage inertia,
passive parameters, torque mapping and actuator limits. The pelvis is fixed
in an ideal fixture; contacts are disabled; other joints remain PD-held using
the recorded commands. The real support rig is not explicitly modeled.

Six conditions per side initialize a five-parameter fit on a reduced MuJoCo
model: 3deg/0.25Hz, 6deg/0.5Hz, 6deg/1Hz, 10deg/0.25Hz, 25deg/0.25Hz,
and 1deg/2Hz. Right-side parameters are subsequently refined inside the
full closed-chain model using three of those conditions. Final validation
replays all 18 conditions per side with the full model, not the surrogate.
Twelve conditions per side are not directly included in optimization, but
they were inspected during model development: this is not a blind test set.

The objective compares AC angle trajectories and velocity, without a fitted
time shift. Mean-position errors are reported separately, not silently
corrected in simulation. Neither measured q nor measured dq is injected into
the simulated dynamics. Numerical optimization is local and budget-limited;
the delivered parameters are a candidate, not proof of a global optimum.

## Additional dynamics

An extra resisting generalized torque is applied only to each configured
ankle-pitch output coordinate. It has a presliding bristle state and a
velocity-dependent Stribeck envelope:

```text
F_limit(q,v) = Fc + (Fs-Fc)*exp(-(v/vs)^2) + c*(q-q0)^2
dz/dt       = v - |v|*sigma0*z/F_limit(q,v)
F_extra     = sigma0*z
qfrc_applied[pitch] = -F_extra
```

The state equation follows the LuGre form with sigma1=0 and no additional
viscous term, since the official model already has damping. The quadratic
angle term is an empirical extension for this experiment, not an official
Unitree parameter. See [Canudas de Wit et al., 1995](https://lucris.lub.lu.se/ws/files/6363840/8498924.pdf).
Frozen-velocity integration is exact within each physics step; internal state
is reset at the beginning of each independent replay. The original MuJoCo
friction remains in place, so Fs/Fc are **additional**, not total friction.
The runtime owns `qfrc_applied` on these two DOFs; another force injector must
not independently overwrite the same two entries.

Parameters and original-asset checksum:
`gear_sonic/data/assets/robot_description/h2_ankle_sysid/identification.json`.
Full validation metrics and raw, unshifted q/dq plots:
`gear_sonic_deploy/logs/ankle_dynamic_identification_20260906/`.

## Selected result

The selected left model is `dynamic_fit/fit.json`, validated in
`dynamic_validated/`. The selected right model is `dynamic_refined/fit.json`,
validated in `dynamic_refined_validated/`. More aggressive right-side
full-shape and low-Stribeck-speed candidates were rejected: the former failed
6deg/0.25Hz (gain 0.081 versus measured 0.563), and the latter overpredicted
3deg/1.5Hz (gain 0.170 versus 0.043). Thus a lower fitting objective alone was
not used to choose the delivered model.

Across all 18 conditions per side:

| Metric | Left | Right |
| --- | ---: | ---: |
| Gain MAE, original official model | 0.1145 | 0.3201 |
| Gain MAE, previous dry-friction candidate | 0.0225 | 0.0313 |
| Gain MAE, new candidate | 0.0173 | 0.0226 |
| Mean phase error, previous dry friction, measured gain >0.2 | 5.75 deg | 7.14 deg |
| Mean phase error, new candidate, measured gain >0.2 | 1.03 deg | 3.97 deg |
| Median AC angle RMSE | 0.134 deg | 0.489 deg |
| Maximum AC angle RMSE | 0.427 deg | 0.857 deg |
| Maximum absolute mean-position error | 0.293 deg | 1.266 deg |

These are absolute gain errors, not percentages. AC RMSE removes each
signal's mean for the metric only; the rendered simulation and raw plots do
not receive mean corrections. At 6deg/1Hz the measured/new-sim phases are
-33.43/-33.67 degrees on the left and -37.72/-45.76 degrees on the right.
The right foot's remaining phase difference is not concealed by the average.

This improves the reproduction of small-motion suppression and asymmetric
left/right response, but does not reproduce every sticking/release event or
weak-response phase. Parameters are correlated effective-response values,
not calibrated measurements of five independent hardware properties.

## Native MuJoCo window

Run from the repository root in a desktop terminal:

```bash
source .venv_sim/bin/activate
python scripts/view_h2_ankle_sysid.py --side right --trial 6:1 --repeat
```

- `--side left`: replay the left experiment with its captured fixture pose.
- `--trial all`: run the complete 18-condition sweep in recorded order.
- `--trial 3:0.25`: inspect the right foot's small-motion suppression.
- `--trial 25:0.25`: inspect the largest recorded angle at 0.25 Hz.
- `--baseline`: disable extra dynamics for comparison with the official model.
- `--view full`: show the full supported robot instead of the feet close-up.
- `--speed 0.5`: half wall-clock playback speed; physics dt is unchanged.
- `--side both`: move both feet together in the left recording's fixture pose.
  This is a qualitative visualization, not the individually validated right
  experiment, whose captured body orientation differs.

Space pauses/resumes; R restarts the current trial; N advances to the next;
close the window to exit. Each trial resets independently to the ready pose.
No DDS is initialized and no robot is contacted. Do not start deploy or VR
for this viewer. Use a desktop terminal with its normal DISPLAY environment.

## Model integration

The separate MJCF is:
`gear_sonic/data/assets/robot_description/h2_ankle_sysid/H2_loop_ankle_sysid.xml`.
It refers to the existing official meshes and embeds the extra parameters in
MuJoCo custom text. Stateful friction cannot be represented faithfully by a
single fixed XML `frictionloss`. **Loading this XML in a generic viewer without
the SONIC-H2 runtime will silently omit the extra dynamic friction.** Use the
viewer above or the registered simulation profile below.

For later whole-body A/B experiments, the opt-in profile is
`h2_native_unitree_ros_loop_ankle_sysid`; the original
`h2_native_unitree_ros_loop` remains unchanged. For example:

```bash
python gear_sonic/scripts/run_sim_loop.py \
  --wbc-version h2_native_unitree_ros_loop_ankle_sysid --interface lo
```

This last command is a DDS-based simulator, unlike the offline ankle viewer.
Whole-body policy rollouts are a separate next step, not validation performed
by this identification exercise.

## Software verification

The 14 focused response/replay tests passed, including native-asset parameter
preservation, state reset, zero-velocity memory, dissipation, step-size
consistency and equivalence of the fast replay with the normal simulator PD
and closed-chain torque mapping. The final profile loads and resets both
ankle models. EGL pixel checks confirm nonblank moving official assets; the
native window was started and closed successfully for both candidate and
baseline modes. No viewer or controller is left running by this task.

Broader regression: 39 additional tests passed, three private DDS integration
tests were skipped, and one old export-contract test could not run because
`sonic_h2_native_release/exported/deployment_contract.json` is absent. The
missing deployment file was not fabricated or replaced for this task.

## Limits

- Weak 1-degree responses and their phases remain imperfectly reproduced.
  Phase at near-zero gain is not a reliable pure-delay estimate.
- Torque calibration, internal motor control, friction, preload, compliance
  and support motion cannot be uniquely separated from these unloaded tests.
- No new motor velocity cap or arbitrary command delay is introduced.
- Extrapolation to contact loads, other ready poses, gains or temperature is
  unvalidated. In particular, this does not yet establish the cause of forward
  steps during squat recovery.
