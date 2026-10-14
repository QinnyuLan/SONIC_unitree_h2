# H2 Ankle Ready-Pose Range Test

`scripts/h2_ankle_range_test.py` replaces the captured-pose workflow for range
experiments. It uses the release `h2_native` deploy ready angles and nominal
initialization Kp/Kd, then changes only one ankle pitch target at a time. All
other targets stay at ready, including ankle roll and the opposite ankle pitch.
PD holding is not a mechanical lock and actual positions will have errors.

For dynamic gain/phase identification around ready, use the same controller with
`--experiment dynamic`: [H2 Ankle Dynamic Identification](h2_ankle_dynamic_sysid.md).
The range mode below remains a low-speed workspace/static-response experiment.

This is an unloaded mechanism experiment, not a standing balance controller.
It does not start `deploy.sh`, a policy, planner, VR server, or model inference.
It does not modify the normal deployment or models. Preparation uses a smooth
speed/acceleration-limited trajectory lasting at least three seconds, not the
deploy controller's fixed three-second linear interpolation.

## Preconditions

- Secure the body with a rigid, suitable support. It must remain safe when ALL
  holding Kp is removed at exit. A fall-arrest tether alone is not sufficient.
- Both feet must be off the ground. Feet, linkages and wiring must have clearance.
  The support must allow hips, knees, waist, arms and head to move to ready;
  do not clamp a joint that the preparation trajectory will move.
- Stop deploy, Pico/VR and other custom LowCmd publishers. The robot's high-level
  mode is released through MotionSwitcher after the single start key `y`.
  Before release the tool permits at most one existing LowCmd publisher. It
  requires zero other publishers before creating its own control writer.
- Keep a trained operator at the independent emergency stop. Software checks
  cannot guarantee a stop under a frozen process, link loss, or hardware fault.
- Confirm H2 PR joint order: left pitch DDS 5, right pitch DDS 11, left/right
  roll DDS 4/10. A compatible DDS message type is not proof of robot identity.
- No real-hardware run or full-range hardware clearance has been validated by
  the automated tests. Do not defeat a guard to make a scan reach an endpoint.

## Run

Use one dedicated interactive terminal; no deploy or simulator process is needed.

```bash
cd /home/qingyul/workspace/SONIC-H2
source .venv_sim/bin/activate

# Preview: does not create DDS or send any motor command.
python scripts/h2_ankle_range_test.py --side both --full-range

RUN="/home/qingyul/workspace/SONIC-H2/gear_sonic_deploy/logs/ankle_range_$(date +%Y%m%d_%H%M%S)"
python scripts/h2_ankle_range_test.py \
  --target real --interface eno1 --domain 0 \
  --side both --full-range \
  --range-step-deg 15 --speed-deg-s 5 \
  --kp-scale 1.0 --kd-scale 1.0 \
  --execute --confirm-supported --output "$RUN"
```

`--kp-scale` and `--kd-scale` multiply BOTH ankle pitch gains, consistently across
preparation and side changes. Baseline Kp=40, Kd=2. All other gains remain the
release initialization values. The unchanged gains are not hardware safety
certification. `--side left` / `--side right` selects a single ankle.

1. Secure the stationary robot, then press `y` in THIS terminal, without Enter.
   There are no typed release/arm phrases. This key authorizes MotionSwitcher
   release AND preparation. The tool checks an empty high-level mode and zero
   LowCmd publishers continuously for one second, then checks the mode again.
   No test LowCmd writer exists before these checks succeed.
2. It captures current angles and ramps holding Kp from zero over three seconds,
   with Kd active immediately. All joints then move automatically to the release
   ready pose; no second `y` is needed.
3. Wait for `Init Done`. Both ankle pitch/roll errors must be below 0.10 rad
   (5.73 deg); other joints may have a static error below 0.15 rad (8.59 deg),
   matching their existing holding-error bound. Targets are still the release
   ready pose, not recaptured biased measurements. Inspect all clearances.
4. Press `n` for each stage: ready -> lower endpoint -> upper endpoint -> ready.
   Each endpoint and the return angle has a two-second dwell. Movements are
   quintic with a maximum target speed of 5 deg/s and acceleration of 10 deg/s^2.
5. Every expansion and the left-to-right switch requires another `n`; the robot
   holds ready while waiting. There is no automatic real-hardware expansion.
6. `q`, Escape, Ctrl+C or Unitree remote SELECT requests damping and exit. Normal
   completion also sends all-joint Kp=0 / Kd=8 for about one second and exits.
   This removes posture holding; it does not power off or retain standing balance.

## Requested Range

Both ready ankle pitch angles are -20.798 deg. With `--full-range`, bounds come
from the local official `unitree_ros/h2_description/H2_loop.xml`, with a 0.05 rad
(2.865 deg) margin at each end. The maximum requested interval is
[-62.135, +32.135] deg, NOT the mechanical stops or a verified hardware workspace.
Roll stays at its ready target of zero. This does not test all joint axes or the
two-dimensional pitch/roll coupled workspace.

For each selected ankle the default staged intervals are:

| Stage | Lower (deg) | Upper (deg) |
| --- | ---: | ---: |
| 1 | -35.80 | -5.80 |
| 2 | -50.80 | +9.20 |
| 3 | -62.13 | +24.20 |
| 4 | -62.13 | +32.14 |

Without `--full-range`, only ready +/-15 deg is requested. To use an independently
approved smaller absolute interval, replace `--full-range` with, for example,
`--range-deg -35 -5`. It must include the ready angle and stay within the model
margin. `--range-step-deg` controls expansion per manually approved stage.

## Guards

- Reuses the original tool's CRC, device tick, stale state (50 ms), command stall
  (20 ms), PR/machine mode, motor fault, temperature (65 C), IMU movement (10 deg),
  official joint bounds, logging and exclusive publisher checks.
- Both ankle pitch speeds are limited to 1 rad/s measured; other joint speeds
  to 3 rad/s. Pitch/roll estimated torque thresholds are 12/8 Nm on both feet.
  Predicted pitch PD torque must stay below 10 Nm. Estimates are not calibrated
  physical output torque measurements.
- During preparation, target tracking error must remain within 0.20 rad.
  During scans, tested pitch error must remain within 0.12 rad and other joint
  errors within 0.15 rad. This replaces the old small capture-displacement bound,
  not the other protections. A stalled foot cannot accumulate a full-range target.
- Initial ready settling requires both ankle pitch/roll errors below 0.10 rad,
  other joint errors below 0.15 rad, and every joint's speed below 0.15 rad/s.
  These must hold continuously for one second, after at least three seconds
  settling. The same position/speed criteria apply before a scan and at its
  return to ready. This accepts limited non-ankle PD static offsets without
  changing targets, gains, or the runtime tracking/torque guards.
- Failure to settle within ten seconds still aborts. While blocked, at most one
  diagnostic every two seconds identifies the joints, error/speed and limit;
  the timeout also includes this information. Confirmation waits time out at
  120 seconds; total armed duration is limited to 2400 seconds.
- A detected second writer or firmware mode change prevents competing damping
  commands. Damping is best effort, not a hardware emergency stop.
- Release uses the local Unitree SDK on the same configured domain/interface as
  LowState. The RPC timeout is two seconds (SDK send and response waits can each
  consume a timeout). CheckMode/ReleaseMode errors abort without motor commands.
  There are at most three successful release requests, with two seconds between
  attempts, and at most ten seconds for old LowCmd publishers to disappear.
  A retained publisher is not ignored even when CheckMode reports an empty mode.
- Cancelled confirmation never calls ReleaseMode. Release itself can remove body
  support BEFORE any test motor command is sent. On failure/cancellation after
  release, the high-level controller is NOT automatically restored. Restore it
  using the normal robot procedure only after securing the robot; this tool never
  calls SelectMode. Stop requests are checked between blocking RPC calls, so keep
  the independent emergency stop available.
- 50 Hz position targets, 500 Hz requested DDS publication; dq_des=0 and tau_ff=0.
  Real scheduling performance is recorded, not assumed.

## Analysis and Simulation

New logs use schema `h2_ankle_range_v1`, raw DDS-order absolute q/dq/tau_est,
published targets/gains, actual timestamps, stage events, ready/captured poses,
completed stages, MotionSwitcher checks/release results, and asset/profile/script
hashes. `ready_check` records the position/speed criteria; `ready_reached` records
actual angles, speeds and static errors. Wait/timeout events record the blockers.
Existing directories are not
overwritten. Never confuse `completed` with measured range coverage.

```bash
# Offline only; none of these commands connects to a robot.
python scripts/analyze_h2_ankle_range.py "$RUN"
python scripts/sim_h2_ankle_sysid.py --replay-log "$RUN"
python scripts/analyze_h2_ankle_range.py "$RUN" "$RUN/sim_replay"
```

`range_analysis.json` and `range_response.csv` report requested/published/actual
angle ranges, tracking RMSE and maximum error, endpoint dwell errors, peak speed
and estimated torque. Incomplete stages are marked. A slow range sweep measures
range-dependent tracking and static/nonlinear effects; it does not by itself
identify frequency response, bandwidth, or an optimal Kd.

Offline replay uses the official closed-chain H2 asset and the deployment's
existing PD/Jacobian torque mapping at 1 ms physics. Only the pelvis is welded;
other joints remain dynamic under recorded PD commands, and all contacts are
disabled. Replay starts at ready, after preparation. It is an ideal support
fixture without network/firmware delay, not an exact model of a real fixture.
The original small-signal log format remains supported with its original fixture.

## Verification

```bash
python -m unittest gear_sonic.tests.test_h2_ankle_release gear_sonic.tests.test_h2_ankle_sysid gear_sonic.tests.test_h2_ankle_range -v
H2_SYSID_DDS_TESTS=1 python -m unittest gear_sonic.tests.test_h2_ankle_release gear_sonic.tests.test_h2_ankle_range -v
```

The opt-in tests use synthetic LowState on loopback DDS domains 97 and 99, not
`eno1`. They verify preparation, left/right isolation, both stages, damping and
analysis, including a 6.08-degree static offset at both elbows. Targets remain
unchanged even in that offset test.
An additional loopback domain 98 test uses the actual SDK RPC client with a fake
MotionSwitcher service, checking mode release and publisher removal without
creating a test LowCmd writer. Neither test releases real robot control.
These tests do not certify real robot safety or successful full-range tracking.
