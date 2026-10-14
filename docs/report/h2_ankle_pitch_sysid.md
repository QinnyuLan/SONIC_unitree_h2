# H2 Ankle Pitch Identification: Phase 1

For dynamic identification using the deploy-ready posture, use
[H2 Ankle Dynamic Identification](h2_ankle_dynamic_sysid.md). For progressive
low-speed workspace tests, use [H2 Ankle Ready-Pose Range Test](h2_ankle_range_test.md).
This document retains the original captured-pose, small-signal sine experiment;
its separate legacy arming workflow has not changed.

This tool tests one unloaded ankle pitch at a time, without SONIC, a planner,
reference motions, or G1/H2 adaptation. It does not change the release models or
normal deployment configuration. This is an experimental diagnostic, not a
validated safety controller or an automatic gain tuner.

## Physical Preconditions

- Secure the H2 body AND legs mechanically. Both feet must remain clear of the
  ground and obstacles. A loose fall-arrest tether alone is not sufficient.
- The fixture must remain safe when ALL holding Kp becomes zero. Normal completion
  and operator/fault stops send damping-only commands for approximately one second
  and then exit. They do not retain a standing posture or power off the motors.
- Keep a trained operator at the robot's independent emergency stop. Python/Linux
  scheduling and a broken DDS link cannot guarantee delivery of a stop command.
- Stop deploy, VR control, and every other LowCmd publisher. Put the robot into
  the established low-level/debug condition with its high-level controller off.
  This script never calls MotionSwitcher ReleaseMode automatically.
- Confirm the hardware is H2 with the deployment's PR mapping: left ankle pitch
  DDS 5, right ankle pitch DDS 11. The common HG message type cannot prove robot
  identity. Software cannot detect whether a physical fixture is present.
- Use the same supported posture and fixed Kp/Kd across comparisons. First run
  only the lowest frequency, inspect it, then extend the frequency list. Do not
  relax guards or increase amplitude solely to force a failed test to finish.

## Real Hardware

Use a dedicated terminal. No `deploy.sh`, ONNX model, planner or Pico server is
needed. The existing simulation virtual environment contains the local DDS SDK.

```bash
cd /home/qingyul/workspace/SONIC-H2
source .venv_sim/bin/activate

# Preview only: no DDS participant or LowCmd writer is created.
python scripts/h2_ankle_sysid.py --side left --frequencies 0.25

# Read-only connection check and raw state recording. No motor commands.
python scripts/h2_ankle_sysid.py \
  --target real --interface eno1 --domain 0 \
  --side left --monitor-seconds 5

# First active experiment: +/-0.5 degree, 0.25 Hz, left ankle only.
RUN="/home/qingyul/workspace/SONIC-H2/gear_sonic_deploy/logs/ankle_left_$(date +%Y%m%d_%H%M%S)"
python scripts/h2_ankle_sysid.py \
  --target real --interface eno1 --domain 0 \
  --side left --amplitude-deg 0.5 --frequencies 0.25 \
  --kp-scale 1.0 --kd-scale 1.0 \
  --execute --confirm-supported --output "$RUN"
```

The default tested ankle gains are Kp=40, Kd=2. `--kp-scale` and `--kd-scale`
affect only the selected pitch joint. Select a previously approved fixed gain
pair for the fixture; the defaults are not a statement of hardware safety.
All other joints use the release nominal gains, including the other ankle.

1. Before enabling any motor commands, the tool waits for valid, stationary
   LowState and checks for other command publishers.
2. Type `ARM H2 LEFT` (or `ARM H2 RIGHT`) and Enter. The script captures fresh
   absolute joint angles, then ramps holding Kp from zero over three seconds.
   It does NOT move the robot to a predefined stand pose. Kd is active immediately.
3. After three more seconds settling, inspect the fixture and press `n` to start
   the first frequency. Commands remain at the captured pose while waiting.
4. Each frequency has two cycles ramping in, five full-amplitude measurement
   cycles, two cycles ramping out, and three seconds at the original target.
   At 0.25 Hz, excitation itself lasts 36 seconds. Every subsequent frequency
   requires a new `n`. Confirmation times out after 60 seconds to damping.
5. `q`, Escape, Ctrl+C, or Unitree remote SELECT requests an immediate damping
   stop. After the last frequency the tool also damps and exits automatically.

For the right ankle, restart with `--side right`, a new output directory, and
type `ARM H2 RIGHT`. After the first low-frequency experiment is reviewed, use
`--frequencies 0.25,0.5,1,1.5,2` for a manually gated sweep. The last frequency
is not necessarily sufficient to identify the full actuator bandwidth.

## Guards and Timing

- Command publication target: 500 Hz. Position target updates: 50 Hz, held between
  updates, matching the deployment. `dq_des=0`, `tau_ff=0` throughout.
- Measured state comes directly from DDS receive callbacks, not the policy's
  downsampled logger. Actual rates and jitter must be checked in each log.
- LowState CRC is mandatory on hardware. Only explicit `--target sim` on `lo`
  and a nonzero DDS domain bypasses the simulation bridge's missing CRC.
- Stops on stale state/device tick (50 ms), command loop/write stall (20 ms),
  non-finite state, PR/mode changes, motor faults, temperature >=65 C, or logging
  failure/overflow. These are software diagnostics, not safety-certified limits.
- Stops on official joint-limit violations, selected pitch speed >1 rad/s,
  other joint speed >3 rad/s, selected pitch displacement >0.10 rad from capture,
  other joint displacement >0.15 rad, pitch tracking error >0.08 rad, or body
  orientation change >10 degrees.
- Stops on selected pitch `tau_est` >12 Nm, same-foot roll `tau_est` >8 Nm, or
  estimated commanded pitch PD torque >10 Nm (absolute values). `tau_est` is an
  uncalibrated estimate; these guards do not establish true output torque limits.
- A second LowCmd publisher causes a stop without sending competing damping
  commands. A changed firmware mode similarly prevents a damping command.
- Arming is interactive on real hardware. `--auto` is rejected for real hardware.
  Do not run this tool concurrently with another controller, including during
  its captured-pose waiting phases.

## Logs and Offline Simulation Comparison

Each run creates a new directory; an existing output directory is refused:

- `states.csv`: all 31 absolute PR q/dq/tau_est values in DDS order, temperatures,
  motor faults, pelvis IMU, remote keys, CRC result, device tick, DDS source
  timestamp, workstation monotonic receive timestamp.
- `commands.csv`: actual published targets/gains, DDS send begin/end timestamps,
  phase/frequency and associated latest state tick. This is not an action log.
- `events.jsonl`: arming, frequency changes, stop reasons.
- `metadata.json`: initial pose, order, arguments, source asset/script hashes,
  outcome and sample counts. Simulator CRC bypass is recorded explicitly.

Do not combine DDS-order absolute angles with the normal policy-order, offset
state log. Do not subtract remote source timestamps from workstation monotonic
timestamps. Device tick units must be verified on hardware; the MuJoCo bridge's
tick is explicitly milliseconds. Logging is buffered through a bounded queue.

```bash
# No robot communication in either of these commands.
python scripts/analyze_h2_ankle_sysid.py "$RUN"
python scripts/sim_h2_ankle_sysid.py --replay-log "$RUN"
python scripts/analyze_h2_ankle_sysid.py "$RUN" "$RUN/sim_replay"
```

Offline replay uses the official `unitree_ros/h2_description/H2_loop.xml`, the
existing output-side PD / closed-loop Jacobian torque mapping, and 1 ms physics.
It replays recorded q/dq/Kp/Kd/tau_ff in their original command-time order. It
does not initialize DDS and does not depend on the computer sustaining real time.
Timing is quantized to the 1 ms physics grid; simulated feedback has no network
or firmware filtering delay. The original asset is never edited.

The virtual fixture fixes the pelvis and all non-ankle output joints with equality
constraints, leaves both ankle PR joints/motors/linkages dynamic, and disables
all contacts. It starts at the recorded joint pose and pelvis orientation. This
ideal fixture is only comparable to a sufficiently rigid real support setup;
fixture flexibility, clamp placement and load differences remain confounders.

Analysis produces `frequency_response.csv` and `analysis.json`: gain, phase,
pitch tracking error, roll cross-response, torque estimate extrema, measured
rates and fit-quality flags. It uses the constant-amplitude window and the actual
published target with causal zero-order hold. A very small or nonlinear response
may indicate stiction or insufficient resolution; do not interpret it as a
trustworthy linear actuator model or automatically increase excitation.

The real result includes command transport, firmware servo, mechanism, sensing
and receive delay. The simulated result lacks unmodeled hardware effects. Phase
is NOT a pure motor delay; these logs alone do not uniquely identify physical
inertia, friction, torque gain and damping. Aborted/incomplete runs are marked,
and incomplete frequency windows must not be treated as validated estimates.

## Verification

```bash
python -m unittest gear_sonic.tests.test_h2_ankle_sysid -v
H2_SYSID_DDS_TESTS=1 python -m unittest gear_sonic.tests.test_h2_ankle_sysid_dds -v
```

The opt-in integration tests launch synthetic state publishers only on `lo`, DDS
domains 91-96. They test command contents, both frequency transitions, monitor-only
behavior, operator interruption, state timeout, and competing publishers. They do not access `eno1`.
