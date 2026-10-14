# H2 Ankle Dynamic Identification

`scripts/h2_ankle_range_test.py --experiment dynamic` measures the local closed-loop
frequency response from published ankle pitch target to measured ankle pitch.
It reuses the release ready pose, PD holding, MotionSwitcher handoff, logging and
safety checks of the [ready-pose controller](h2_ankle_range_test.md).
It does not run a policy or modify normal deployment, models or gains.

## Protocol

- One ankle per run: `--side left` (DDS 5) or `--side right` (DDS 11).
- Both ankle pitch target centers are the release ready angle, -20.798 deg.
  All other joint targets remain at release ready during excitation. PD holding
  is not a mechanical lock; actual joint positions may have static errors.
- Default fixed peak amplitude: +/-3 deg at 0.25, 0.5, 0.75 and 1 Hz.
- Each frequency has two cosine-envelope ramp-in cycles, six constant-amplitude
  measurement cycles, two ramp-out cycles, then three seconds at ready.
  Only the constant-amplitude cycles enter the frequency-response fit.
- This schedule takes about 95 seconds, excluding preparation and manual waits.
  Frequency changes are automatic after one initial `n`. The two ankles are
  separate runs. There is no automatic side change. A single-amplitude run never
  expands amplitude; an explicit `--sweep-plan` advances all its listed bands.
- Position targets update at 50 Hz, publication is requested at 500 Hz, and
  dq_des=tau_ff=0, matching the deployment command contract. Kp/Kd are fixed.
- Baseline Kp=40, Kd=2. `--kp-scale` and `--kd-scale` multiply both ankle pitch
  gains throughout preparation and measurement; all other gains remain unchanged.
- Supported dynamic tests do not stop on body IMU yaw, roll or pitch changes.
  IMU samples and the captured initial quaternion are still logged; finite-value
  and quaternion-validity checks remain enabled. Metadata explicitly records
  `imu_orientation_stop_enabled: false`. Range tests and normal deploy are unchanged.

This first frequency band measures dynamic tracking, not the complete actuator
bandwidth. It does NOT uniquely identify motor inertia/friction, pure firmware
delay, open-loop damping, or optimal Kd. An unloaded supported response is also
not the loaded standing/squat response.

## Preconditions

Mechanically secure the body using a suitable support that remains safe after
all holding Kp is removed. Both feet and ankle linkages must be free of ground,
fixtures and wiring interference. The support must permit the preparation motion
of the hips, knees, waist, arms and head. A fall-arrest tether alone is insufficient.
Stop deploy, Pico/VR and every other custom LowCmd publisher. Keep an operator at
the independent robot emergency stop; software/DDS cannot guarantee stop delivery.
The automated tests do not establish real-hardware safety.

## Run

Use a dedicated terminal. No deploy or simulator process is needed for collection.

```bash
cd /home/qingyul/workspace/SONIC-H2
source .venv_sim/bin/activate

# Preview only: no DDS participant, release RPC, or motor commands.
python scripts/h2_ankle_range_test.py --experiment dynamic --side left

SIDE=left
RUN="/home/qingyul/workspace/SONIC-H2/gear_sonic_deploy/logs/ankle_dynamic_${SIDE}_$(date +%Y%m%d_%H%M%S)"
python scripts/h2_ankle_range_test.py \
  --experiment dynamic --target real --interface eno1 --domain 0 \
  --side "$SIDE" --amplitude-deg 3 --frequencies 0.25,0.5,0.75,1 --cycles 6 \
  --kp-scale 1 --kd-scale 1 \
  --execute --confirm-supported --output "$RUN"
```

1. Press `y` once, without Enter: release high-level control, verify publisher
   handoff, ramp holding gains, then move to deploy ready automatically.
   The former `RELEASE H2 CONTROL` and `ARM H2 READY` inputs have been removed.
2. Wait for `Init Done`, inspect support and clearances, then press `n` once.
   All listed frequencies run in order with no further confirmation.
3. `q`, Escape, Ctrl+C or Unitree remote SELECT requests stop. Completion/stop
   sends all-joint Kp=0/Kd=8 for about one second and exits. Posture holding ends.
   The high-level controller is never automatically restored. Release itself
   may remove support before this tool publishes its first motor command.

For the right ankle, set `SIDE=right` and rerun the RUN assignment and command.
Use a fresh log directory for every trial. Keep posture, support and gains fixed
for a real/simulation comparison. Repeat a baseline run before varying Kp or Kd;
change one gain at a time, retaining the same excitation.

## Extended Amplitude and Frequency Plan

The explicit `scripts/configs/h2_ankle_dynamic_large_range.json` plan keeps
Kp=40/Kd=2 and tests the following sequence around the same -20.798 deg center:

| Peak amplitude (deg) | Frequencies (Hz) |
| ---: | --- |
| +/-3 | 0.25, 0.5, 1, 1.5 |
| +/-6 | 0.25, 0.5, 0.75, 1 |
| +/-10 | 0.25, 0.4, 0.6 |
| +/-15 | 0.25, 0.4 |
| +/-25 | 0.1, 0.25 |
| +/-1 | 2, 2.5, 3 |

These are 18 trials taking 513 seconds with six measurement cycles, excluding
preparation and initial waits. The largest target interval is [-45.798, +4.202]
deg, a 50-degree peak-to-peak excursion. This range has NOT been dynamically
validated on hardware. Each trial ramps smoothly back to ready and holds for
three seconds before the next trial. A failed guard aborts the entire plan;
there is no skip, automatic retry or escalation after a failure.

The JSON explicitly raises the planned smooth-waveform speed/acceleration caps
from 0.4 rad/s and 3 rad/s^2 to 0.8 rad/s and 8 rad/s^2. Measured ankle speed
(1 rad/s), tested-joint tracking error (0.12 rad), predicted PD torque (10 Nm),
estimated torque, temperature (65 C), DDS, mode and all other guards are unchanged.
The high-frequency band has smaller amplitude because +/-25 deg at 3 Hz would
require about 8.2 rad/s peak target speed even without its envelope. Policy target
updates at 50 Hz do not mean the joint is expected to oscillate at 50 Hz.
Higher-frequency +/-1 deg responses may be too weak/nonlinear to identify; inspect
the measured amplitudes, residuals and quality flags rather than assuming a valid
linear bandwidth measurement. Keep an operator watching throughout all 8.5 minutes.

```bash
cd /home/qingyul/workspace/SONIC-H2
source .venv_sim/bin/activate

# Preview only: includes the full plan and its explicit target caps.
python scripts/h2_ankle_range_test.py --experiment dynamic --side left \
  --sweep-plan scripts/configs/h2_ankle_dynamic_large_range.json

SIDE=left
RUN="/home/qingyul/workspace/SONIC-H2/gear_sonic_deploy/logs/ankle_dynamic_extended_${SIDE}_$(date +%Y%m%d_%H%M%S)"
python scripts/h2_ankle_range_test.py \
  --experiment dynamic --target real --interface eno1 --domain 0 \
  --side "$SIDE" --sweep-plan scripts/configs/h2_ankle_dynamic_large_range.json \
  --cycles 6 --kp-scale 1 --kd-scale 1 \
  --execute --confirm-supported --output "$RUN"
```

The same single `y` then single `n` workflow applies. Before `n`, verify clearance
for the ENTIRE listed range; there is no repeated approval before a larger band.
No typed release/arm phrases have been restored. Choose `SIDE=right` in a separate
run after inspecting the left run. Remove `--execute --confirm-supported` to preview
without creating DDS or releasing robot control.

## Analysis and Simulation

```bash
# Offline only. These commands do not connect to the robot.
python scripts/analyze_h2_ankle_sysid.py "$RUN"
python scripts/sim_h2_ankle_sysid.py --replay-log "$RUN"
python scripts/analyze_h2_ankle_sysid.py "$RUN" "$RUN/sim_replay" \
  --compare-output "$RUN/real_vs_sim.csv"
```

`frequency_response.csv` and `analysis.json` contain gain, gain in dB, phase,
input/output amplitudes, static offset, tracking RMSE, roll coupling, estimated
torque, 2nd/3rd harmonic distortion and cycle-to-cycle gain/phase repeatability.
The fit includes a constant and linear drift; alignment uses the recorded actual
published target with causal zero-order hold on workstation monotonic timestamps.
Real phase includes transport/feedback delay. Estimated torque is not a calibrated
output torque measurement. Low response, incomplete windows, large receive gaps,
distortion and poor repeatability receive diagnostic flags, not a safety verdict.
Harmonic/repeatability thresholds are heuristics, not spectral coherence.

Multi-amplitude runs use `h2_ankle_dynamic_v2`: every amplitude/frequency pair has
a unique excitation phase in the unchanged raw command CSV format. The analyzer
never pools the 3-degree and 25-degree trials at the same frequency. Legacy
single-amplitude logs remain supported. Analysis also records actual position
min/max and peak-to-peak range, peak speed/error, low-speed sample fraction,
per-trial Kp/Kd and unmeasured/incomplete trials. `amplitude_comparison` in
`analysis.json` groups gains/phases at common frequencies by commanded amplitude.
The paired comparison CSV includes real/sim gain ratio, additional phase lag,
peak-to-peak range and harmonic distortion, with each side's quality flags.

Interpret common-frequency amplitude dependence before fitting a linear model:
if gain rises and the flat/stair-like regions shrink as amplitude increases,
investigate friction/deadband/load effects. If gain and phase stay consistent
across amplitude with low distortion, a local linear model is more defensible.
Neither pattern alone identifies the mechanism or justifies a Kd change. Repeat
baseline trials to check fixture, temperature and time-order effects; this plan
does not randomize amplitude order. A read-only/offline replay is an ideal-fixture
comparison, not a hardware validation.

Replay uses the local official `unitree_ros/h2_description/H2_loop.xml` closed-chain
asset and the existing PD/Jacobian torque mapping, with 1 ms physics. It replays
the same recorded targets and Kp/Kd starting at ready. Only the pelvis is welded;
the other joints remain dynamic and PD-held, with all contacts disabled. This is
an ideal fixture without firmware/network delay, not a calibrated real rig.

## Limits and Extension

The existing stale-state, CRC, mode, publisher, measured velocity, temperature,
tracking error and torque stops remain enabled. Only the body-orientation-change
stop is disabled for this supported dynamic experiment; IMU data-validity checks
remain enabled. This tool does not detect loss of body support. Dynamic targets must also
stay inside official joint bounds with a 0.05 rad margin. The sine-envelope
derivative bounds default to 0.4 rad/s and 3 rad/s^2; an explicit sweep plan may
specify at most 0.8 rad/s and 8 rad/s^2. This checks the intended smooth waveform,
while deployment-style targets are still zero-order-held. The selected caps are
printed before connection and saved in metadata.

Amplitude is never silently reduced with frequency. To extend the frequency band,
first inspect the lower-band signal quality, then configure a separate fixed,
lower-amplitude band. CLI permits 0.1-25 deg, 0.1-3 Hz, 3-12 measurement cycles and
at most 600 seconds of excitation/holds, subject to the stricter derivative caps.
These are experiment bounds, not certified safe limits. If responses show stiction,
large harmonics, drift or saturation, do not interpret their fundamental alone as
a linear motor model or extrapolate a new Kd from it.

## Verification

```bash
python -m unittest gear_sonic.tests.test_h2_ankle_release gear_sonic.tests.test_h2_ankle_sysid gear_sonic.tests.test_h2_ankle_range gear_sonic.tests.test_h2_ankle_dynamic -v
H2_SYSID_DDS_TESTS=1 python -m unittest gear_sonic.tests.test_h2_ankle_release gear_sonic.tests.test_h2_ankle_range gear_sonic.tests.test_h2_ankle_dynamic -v
```

The dynamic integration test uses an actual pseudo-terminal with one `y` and one
`n`, synthetic LowState/LowCmd on private lo/90, two frequencies, static elbow
bias, frequency analysis and offline official-asset replay. Release RPC tests use
a fake service on lo/98; range regressions use lo/97 and lo/99. No test connects
to `eno1`, releases real robot control, or validates hardware motion safety.
An additional multi-amplitude integration test covers +/-6 and +/-25 deg at the
same frequency, a 3 Hz trial, fixed gains on all joints, unique analysis groups,
paired replay export and unchanged simulated velocity/tracking guards.
An additional private-loopback test injects a 60-degree body IMU rotation after
arming and checks that all planned frequencies complete with unchanged gains,
the IMU samples remain in the log, and orientation-stop metadata is disabled.
