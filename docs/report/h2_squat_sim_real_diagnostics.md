# H2 release squat: simulation / hardware comparison

## Scope

The original change added logging only. The 2026-09-07 correction below also
fixes wall-clock pacing and removes simulator overhead. Policy, reference CSVs,
robot assets, floor friction, PD gains and the closed-loop torque mapping
equations are unchanged. The manual comparison launcher never automatically
arms a controller or starts playback.

## Realtime correction (2026-09-07)

The original `base01`/`id01` simulations advanced only about 5.5-5.6 physics
seconds during an 8.5-wall-second reference clip. They are not timing-equivalent
to hardware. Keep `real01`, but rerun the two simulations as `base02`/`id02`.

The simulator now paces every 1 ms physics step against cumulative monotonic
deadlines. Short overruns and oversleep are recovered by omitting later sleeps,
without skipping physics steps, changing dt, or rescaling reference velocity.
It remains wall-clock coupled to the external 50 Hz policy, not a lockstep
simulation that can run arbitrarily slower than real time.

Closed-chain Jacobian extraction uses MuJoCo's native sparse conversion, with
equality-row layout reuse after checking the layout each step. The PD equations,
pseudoinverses, constraint checks and actuator mappings remain unchanged. Tests
compare against the old extraction and require bit-identical q, dq and ctrl.
The same scalar PD expression is evaluated with array arithmetic on one command
packet snapshot. Diagnostic command fields are assembled only on sampled steps,
not on the intervening four physics steps where they would be discarded.
Empty hand-state packets are no longer published for a zero-hand-motor profile;
body LowState, torso IMU and odometry remain published every physics step.

H2-native uses the faster `viewer.sync(state_only=True)` with a full initial
sync and a 250 ms render warmup before the timing epoch. It still accepts GUI
events. Programmatic edits to model properties during a run need an explicit
full sync; this comparison does not make such edits. See the
[MuJoCo passive-viewer API](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer).
Initial observation/matrix work and local DDS codec serialization/deserialization
are also warmed before this epoch, without advancing physics or publishing a
command. This keeps one-time initialization out of the runtime lag budget.

The comparison launcher explicitly sets `--realtime-max-lag-ms 20`. Exceeding
that lag stops the simulator with a nonzero exit and marks its log `incomplete`,
with `realtime.valid=false` and the failure reason. Do not relax the limit just
to finish a comparison. Stop deploy and the recorder as well, keep the failed
log, and check host load before starting a fresh run. This is a simulator
validity check, not a real-robot safety mechanism.

Every 5 seconds the simulator prints `[Realtime] RTF=...`; expect approximately
1.000. `sim/realtime.csv` records elapsed wall and cumulative physics time, lag,
maximum lag and stage-average costs every second. `sim/metadata.json` records
the final clock-validity summary; `sim_launch.json` includes implementation
hashes. Model resets do not reset the pacing epoch or hide accumulated lag.

Use the commands below with `base02`/`id02` instead of the old labels. Do not
overwrite the old captures. After normal startup and band release, wait for
settling and a healthy realtime report before playing the clip.

The launcher selects the current non-v1.1 `h2_native` release, full encoder,
H2 reference / selector 0, no planner, keyboard input and no hand commands.
It selects `squat_001__A359` from the existing eight-motion H2 collection.
All five gain scaling flags are explicitly 1.0; ankle pitch is Kp=40, Kd=2.
It removes inherited `GEAR_SONIC_*` overrides, including automatic playback,
action overrides, skipped MotionSwitcher and IMU-bias experiments.

Each terminal uses the same `--run` label for one experiment. Existing component
logs are never appended/overwritten. Use a new label for another launch.
Model, observation config, reference manifest and squat CSV hashes are stored
in the launch JSON files; a within-run mismatch is rejected before launch.

If only the recorder exits during setup, leave the simulator running and keep
the failed logs. Start a new passive recording directory within the same run:

```bash
.venv_sim/bin/python scripts/record_h2_squat_dds.py \
  --target sim --interface lo --domain 0 \
  --output gear_sonic_deploy/logs/squat_compare/base01/dds_retry01
```

The deploy terminal still uses `--run base01`. Start playback only after the
replacement recorder is running. An absent LowCmd publisher before deploy
starts is expected. Simulated LowState may have an invalid/zero CRC; this is
recorded but does not stop the recorder. Reader data callbacks are installed
only after construction, and already-queued samples are drained immediately;
startup discovery/status callbacks do not dereference the reader argument.

## Simulation

Stop previous controllers/simulators yourself before starting this test. Do not
run the baseline and identified simulation simultaneously (both use lo/domain 0).
Run from `/home/qingyul/workspace/SONIC-H2` in each terminal.

Terminal 1, original official closed-chain model:

```bash
.venv_sim/bin/python scripts/h2_squat_compare.py sim --variant baseline --run base01
```

Terminal 2, passive DDS recorder (no commands, no RPC):

```bash
.venv_sim/bin/python scripts/h2_squat_compare.py record --target sim --run base01
```

Terminal 3, policy:

```bash
.venv_sim/bin/python scripts/h2_squat_compare.py deploy --target sim --run base01
```

At the deploy prompt confirm `y` and Enter. After `Init Done`, use `]` in the
deploy terminal to enter policy control. In the MuJoCo window press `9` to
disable the elastic band and confirm `ElasticBand enable: False`. Let the
robot stand/settle normally, then wait about 3 seconds before pressing `t` in
the deploy terminal. Do not press Enter (planner toggle), `z` (encoder toggle)
or motion-selection keys during the comparison.

The selected clip lasts about 8.5 seconds and contains several squat/stand-up
movements. After completion the existing controller resets to frame 0 and
pauses playback. When stable, wait 5 seconds, then press `t` again; record up
to three playbacks if stable. This is not an instruction to repeat instability.

Finish with the existing deploy `o` stop (holding torque is removed, damping
remains), then Ctrl+C in the recorder and simulator terminals to flush logs.

Repeat with a fresh simulator and change all run labels to `id01`:

```bash
# Terminal 1
.venv_sim/bin/python scripts/h2_squat_compare.py sim --variant identified --run id01
# Terminal 2
.venv_sim/bin/python scripts/h2_squat_compare.py record --target sim --run id01
# Terminal 3
.venv_sim/bin/python scripts/h2_squat_compare.py deploy --target sim --run id01
```

The original/identified model starts from the same configured pose and runtime
options. DDS/policy wall-clock scheduling is not deterministic: verify actual
pre-play q/dq, contact and support states in the logs before treating a pair as
matched initial conditions. A run with an active band is not an unsupported
whole-body comparison.

## Hardware: eno1

Use a clear floor and fall-arrest protection with an operator ready to stop.
For loaded measurements the feet must support the robot normally; a taut
suspension carrying body weight invalidates this comparison. This is NOT the
previous unloaded ankle identification test. Keep physical payloads and foot
placement consistent. Do not simultaneously run VR or another deploy process.
Stopping deploy removes pose holding, so secure the robot before stopping.

Terminal 1, recorder first:

```bash
.venv_sim/bin/python scripts/h2_squat_compare.py record --target real --interface eno1 --run real01
```

Terminal 2, policy (this WILL initiate the existing real-robot deployment after
the normal confirmation prompt):

```bash
.venv_sim/bin/python scripts/h2_squat_compare.py deploy --target real --interface eno1 --run real01
```

Normal deployment confirmation is preserved; do not confirm until the robot
and surroundings are ready. The real path retains CRC checking and
MotionSwitcher.ReleaseMode; the recorder never releases control or publishes.
Wait for `Init Done`, then `]`, settle for about 3 seconds, then `t`.
The selected motion must print `squat_001__A359`, encoder mode 0. Stop the
experiment on instability; do not continue repetitions just to finish a count.
After the controller stops, Ctrl+C the recorder separately. Stopping the
recorder alone does NOT stop or protect the robot.

Record a fixed side-view video showing both feet and pelvis. No hardware foot
force or absolute body-position sensor is available in the subscribed LowState
schema, so video/external tracking is necessary to corroborate forward steps.
No software-only logging change can create those measurements.

## Recorded quantities and limitations

All output is under `gear_sonic_deploy/logs/squat_compare/<run>/`:

| File | Source and meaning |
| --- | --- |
| `deploy/lowcmd_command.csv` | Actual published target q/dq/Kp/Kd/tau_ff; existing 500 Hz command logger |
| `deploy/q.csv`, `dq.csv`, IMU files | Existing policy-side feedback at 50 Hz |
| `deploy/motion_name.csv`, `motion_playing.csv`, `encoder_mode.csv` | Playback intervals and routing |
| `dds/lowstate.csv` | Every received LowState: 31 native DDS joints, tau_est, temperatures, motor errors, IMU, tick, CRC |
| `dds/lowcmd.csv` | Every received command, with arrival/source timestamps and publisher handle |
| `dds/secondary_imu.csv` | Raw secondary IMU, if published |
| `dds/odometry_unverified.csv` | Optional rt/odostate, NOT trusted hardware ground truth; may be empty |
| `sim/truth.csv` | 200 Hz simulation-time sampling of solved physics and used commands |
| `sim/realtime.csv` | Clock lag, realtime factor and loop stage-average costs |
| `sim/metadata.json` | Runtime config, masses, friction, limits, initial state and joint/actuator order |

Simulation truth includes root position/displacement/quaternion, whole-robot
subtree COM, left/right foot poses/velocities, ground forces and moments,
CoP on the world z=0 plane, elastic-band state/force, actual PR targets/gains
used by the PD calculation, requested PD effort, motor-side request after
closed-loop mapping, clipped ctrl, solver actuator_force and additional
identified ankle force. The motor clipping flag compares request with ctrl.
`output_equivalent_motor_force` maps solver motor force back through the same
pre-step Jacobian; it is not a separate joint-torque measurement and excludes
contact, passive forces and added pitch friction.

Truth q/dq are captured before integration; kinematics, contacts and forces
come from that solver step. Both start/end simulation times are explicit.
The diagnostic rejects RK4, whose intermediate evaluations do not have these
semantics. It does not invoke additional forward/step calculations or mutate
physics. Foot ground force is force ON the robot, expressed in world axes.
CoP is NaN with `cop_valid=0` when vertical force is <=1 N. No-contact force
is zero; unavailable real measurements are NOT replaced with zeros.

`time_monotonic_ms` uses the local host clock and aligns with deploy CSVs.
Receive timestamps are not sensor acquisition timestamps. Raw source
timestamps/ticks are retained to inspect delay, packet gaps and repeated
samples. No IMU double-integration displacement or guessed hardware contact
force is reported. Hardware `tau_est` remains a drive estimate, not a
calibrated output torque. Foot/body velocity from `mj_objectVelocity` is the
MuJoCo body-centered spatial velocity in world axes.

CSV writing is bounded and asynchronous. `metadata.json` records row counts,
overflow and writer errors. DDS metadata also records lost/rejected samples,
CRC failures, matched-publisher counts and missing required topics. Empty
optional topics are normal; empty LowState/LowCmd or dropped rows invalidate
high-resolution timing claims. The recorder is not a safety monitor.

## Offline Verification

```bash
.venv_sim/bin/python -m unittest \
  gear_sonic.tests.test_sim_realtime \
  gear_sonic.tests.test_h2_squat_diagnostics \
  gear_sonic.tests.test_h2_squat_dds \
  gear_sonic.tests.test_h2_ankle_response \
  gear_sonic.tests.test_h2_ankle_friction_hypothesis -v
```

The DDS integration test uses synthetic messages on lo/domain 187 only.
No tests start a policy or connect to eno1. Append `--dry-run` to a launcher
command to print its fully expanded arguments without files, DDS or execution.

The separate automatic full-chain timing smoke test DOES start a policy on
loopback/domain 188, a simulator and the passive recorder. It preloads the policy
before starting physics, then plays one squat clip after ready-pose initialization
and disables the simulated band first. `--repeats 3` collects three playbacks,
with approximately 5.5 seconds between clips. The three-repeat capture used
`base03_auto` and `id02_auto` on 2026-09-07; earlier failed captures are retained.
It has no real-interface option and cleans up only its own child processes.
It is for timing verification, not matched-state behavioral comparison:

```bash
.venv_sim/bin/python scripts/verify_h2_sim_realtime.py \
  --variant baseline --onscreen --output test_logs/realtime_baseline_unique
.venv_sim/bin/python scripts/verify_h2_sim_realtime.py \
  --variant identified --onscreen --output test_logs/realtime_identified_unique
```

The final identified-model windowed check (`test_logs/realtime_id_warm_20260907`)
completed 425 active playback frames with RTF 1.00000060 and maximum lag 7.520 ms.
It is a separate timing check, not part of the three-repeat A/B statistics.
