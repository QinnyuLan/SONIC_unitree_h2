# Conditional Loaded H2 Actuator Identification

Date: 2026-09-07.

## Outcome

The existing logs support a useful **conditional left-knee effective actuation
hypothesis**, but do not uniquely identify hardware efficiency, delay, or
transmission losses. Lowering the left-knee motor effort gain improves short
recorded-command replays substantially. It does **not** robustly reproduce the
real robot's persistent forward lean after all three rises in a complete
policy-controlled squat playback.

The final exploratory configuration is
[candidate_gain_only.json](../../gear_sonic_deploy/logs/actuator_identification_20260907/candidate_gain_only.json):

```json
{"config": {"left_knee": {"gain": 0.725}}}
```

This is simulation-only and opt-in. The existing bilateral ankle response is
retained, with no new ankle gain, lag, or cap. No default model, ONNX, reference,
friction setting, hardware Kp/Kd, or hardware controller was changed. In
particular, 0.725 must not be interpreted as a measured motor efficiency or as
a recommendation to multiply real Kp/Kd by 0.725 or its reciprocal.

All artifacts are under
[actuator_identification_20260907](../../gear_sonic_deploy/logs/actuator_identification_20260907).

## Inputs and Observability

The measured inputs are actual DDS `LowCmd` targets, target velocities, Kp, Kd,
and feed-forward efforts from `squat_compare/real01`. They are replayed using
causal zero-order hold on recorded host arrival time. Encoder references are
not substituted for actuator commands. The recorded state is joint q/dq and
pelvis IMU orientation/angular velocity.

The [effort audit](../../gear_sonic_deploy/logs/actuator_identification_20260907/effort_audit/README.md)
found one active command publisher during each playback. The second publisher
recorded elsewhere in the session had no commands inside the three active
playbacks. Mean active LowCmd intervals were 2.000 ms; causal command age at
LowState arrival had a 95th percentile of 1.908-1.921 ms. No competing publisher
is inferred from the recorder's session-wide discovery maximum.

Drive-reported `tau_est` closely follows reconstructed PD in these data:

| Joint | PD absolute p95 | tau_est absolute p95 | Zero-lag correlation | Zero-lag fitted slope |
| --- | ---: | ---: | ---: | ---: |
| Left knee | 163.778 | 162.825 | 0.994 | 0.986 |
| Right knee | 87.585 | 86.555 | 0.985 | 0.972 |
| Left ankle pitch | 57.959 | 57.885 | 0.993 | 0.988 |
| Right ankle pitch | 69.173 | 68.774 | 0.993 | 0.989 |

These are logged torque-like units, not an independent output-torque
calibration. The estimator may share controller calculations or filtering.
The roughly 3 ms best alignment in a host-time lag scan is not a measured motor
delay. Host arrival, DDS source timestamp, and hardware tick are different
clocks without established synchronization. Neither observed maxima nor
repeated torque samples establish a physical saturation threshold.

Missing quantities remain important: real root translation and linear
velocity, contact state and forces, sole pressure, output-torque calibration,
and transmission preload. There is no new real-robot recording in this work.

## Replay Method

[identify_h2_loaded_actuators.py](../../scripts/identify_h2_loaded_actuators.py)
uses the full official closed-chain MuJoCo robot with the previously fitted
bilateral ankle bristle response. Every command retains its recorded PD gains.
Joint inertia, native force limits, contact settings and the existing ankle
parameters are unchanged in the main response-gain experiments.

There are 12 independent 250 ms windows: a pre-motion window and three
rise-start windows in each of three squat repetitions. Repetitions 1 and 2 are
used for screening and fitting. Repetition 3 is a within-session holdout, not
blind or independent validation. Measurements in each window initialize state
once; no measured q/dq, root motion, or force is injected after initialization.

The initialization follows the existing linkage assembly branch and enforces
position/velocity equality constraints. Root height is inferred from the
lowest sole vertex, root velocity from an assumed no-slip support condition,
and initial ankle bristle preload is zero. A first-order motor state, when
present, starts at the initial steady requested effort, not a measured torque.
Warnings, resets, nonfinite values, and invalid assembly branches reject a
window rather than silently continuing.

The dimensionless composite score is the mean square of concatenated
residuals, divided by 5 degrees in radians: four joint position errors,
0.035 seconds times their velocity errors, and 0.4 times pelvis pitch error.
The four joints are both ankle pitches and both knees. It is a deliberately
limited diagnostic objective, not a full-body tracking or balance score.

## Screening and Identifiability

[screen03](../../gear_sonic_deploy/logs/actuator_identification_20260907/screen03/screen.json)
contains 29 cases: nominal plus one-group gain, first-order lag, and cap
perturbations for the four actuator groups. Each ankle group addresses both
actual motors in its closed-chain transmission, not a synthetic output-pitch
force. Each knee group addresses its actual knee motor.

| Candidate | Training score | Repetition-3 score |
| --- | ---: | ---: |
| Existing ankle model | 0.356068 | 0.367435 |
| Left-knee gain 0.70 | 0.113884 | 0.123667 |
| Right-knee gain 0.70 | 0.287308 | 0.265992 |
| Left-ankle gain 1.10 | 0.301756 | 0.307933 |
| Left-ankle gain 0.70 | 0.502030 | 0.511248 |
| Right-ankle gain 0.70 | 0.442771 | 0.457274 |

Adding ankle lag or reducing ankle effort did not explain these windows.
Reducing selected native caps to 60% was largely inactive, so these data do
not identify those limits. This does not prove the ankles are correct under
all loads; it argues against indiscriminately slowing both ankles to force
an apparent full-body failure.

The initial [fit01](../../gear_sonic_deploy/logs/actuator_identification_20260907/fit01/candidate.json)
returned left-knee gain 0.723279 and a 4.8995 ms first-order time constant.
[knee_profile](../../gear_sonic_deploy/logs/actuator_identification_20260907/knee_profile/README.md)
then showed that this numerical precision is not identification precision:

- Profiled time constants 0, 5, 10 and 15 ms have training losses within 1.8%.
- Ten gain/lag grid points with gains 0.70-0.75 lie within 5% of the minimum.
  This descriptive valley is not a confidence interval.
- Gain 0.725 with zero lag scores 0.112982/0.122738, essentially as well as
  0.112948/0.122009 for the prior gain/lag fit.

The final exploratory file therefore omits lag instead of claiming a measured
5 ms response time. A first-order lag is not a pure transport delay either.

Coarse alternate mechanical hypotheses also improve the same objective. A
left-knee motor friction increment of 20 Nm scores 0.223601/0.225872; a motor
damping increment of 5 Nm*s/rad scores 0.205896/0.214318. These large increments
are sensitivity probes, not proposed hardware constants. They do not match
the gain fit as closely in the tested range, but they do not rule out other
loss models or combinations. All temporary native parameter changes were
restored between candidates and did not change source assets.

The prior gain/lag candidate also improved holdout scores at 0.10, 0.25 and
0.50 s horizons by about 29%, 67% and 88%, respectively. Longer horizons cover
different samples and increasingly depend on unknown contact/root state; the
absolute scores should not be compared as if they used identical targets.

## Final Gain-Only Results

[summary_gain_only](../../gear_sonic_deploy/logs/actuator_identification_20260907/summary_gain_only/summary.json)
reports the following mean per-window errors on repetition 3 under the
nominal initialization assumption:

| Quantity | Existing ankle model | Plus left-knee gain 0.725 |
| --- | ---: | ---: |
| Composite score | 0.367435 | 0.122738 |
| Left-knee q RMSE | 3.762 deg | 1.153 deg |
| Right-knee q RMSE | 3.271 deg | 1.396 deg |
| Left-ankle pitch q RMSE | 5.052 deg | 2.262 deg |
| Right-ankle pitch q RMSE | 2.700 deg | 2.827 deg |
| Pelvis pitch RMSE | 1.581 deg | 1.390 deg |

The joint improvements are coupled system responses, not separately fitted
gains for every joint. The right ankle is slightly worse in this summary.

The candidate improves the all-window score by 65.9%-69.9% across the six
tested initialization variants: both-foot, left-foot, right-foot, or zero root
velocity, plus minimum sole clearance offsets of -1 and +2 mm. This supports
robustness to these particular nuisance changes, not to every unobserved
contact state. At nominal initialization, reconstructed left/right sole
heights differ by as much as 11.04 mm. Thus joint/IMU state alone does not
establish simultaneous flat-foot support in the model.

## Unloaded Regression Check

The [final gain-only regression](../../gear_sonic_deploy/logs/actuator_identification_20260907/unloaded_gain_only/summary.json)
completed 18 conditions per side: 36 conditions, each replayed with baseline
and candidate, using the original command ramp and two steady cycles. The
fixture retains the full closed chain with the pelvis supported, no ground
contact, and other joints PD-held. The raw command and measured arrays are
used; previously predicted trajectories are ignored.

| Side | Baseline median AC q RMSE | Candidate median AC q RMSE | Worst increase across conditions |
| --- | ---: | ---: | ---: |
| Left ankle | 0.134217 deg | 0.134189 deg | 0.000425 deg |
| Right ankle | 0.497352 deg | 0.497352 deg | 0.00000269 deg |

No condition regressed by more than 0.1 deg. The earlier gain/lag candidate
likewise passed all 36 conditions, with worst increases of 0.000429 deg left
and 0.00000250 deg right. These are regression checks on the original source
session, not independent validation of an ankle hardware model.

## Complete Policy Validation

The actual current release policy ran three complete `squat_001__A359`
playbacks for each candidate. The reference selector was H2/0, with no planner.
The comparison reused three valid `id02_auto` existing-ankle-model playbacks
and the three existing `real01` hardware playbacks. Both new candidate runs
used the same encoder, decoder, observation configuration, reference hashes,
and all five explicitly unit-valued gain scales.

| Run | Response hypothesis | Full-run realtime factor | Maximum lag |
| --- | --- | ---: | ---: |
| id02_auto | Existing ankle model | 0.99999885 | 4.04 ms |
| knee_candidate02_auto | Left-knee gain 0.723279, lag 4.8995 ms | 0.99999898 | 4.91 ms |
| knee_gain_only01_auto | Left-knee gain 0.725, no lag | 1.00000527 | 2.37 ms |

All complete playbacks had elastic support disabled, zero band force and no
reset contamination. The 1 ms physics step, 50 Hz policy rate, 500 Hz command
publication, 200 Hz truth logging, and 20 ms accumulated-lag rejection stayed
unchanged. DDS was isolated to `lo`, domain 188; no simulation process used a
real-robot interface. Simulated LowState CRC absence is recorded as such, not
misrepresented as hardware packet corruption.

Mean pelvis pitch 0.3-0.8 s after rise completion, range over three playbacks:

| Event | Existing ankle model | Gain and lag candidate | Final gain-only candidate | Real robot |
| --- | ---: | ---: | ---: | ---: |
| Second rise | -0.48 to 0.28 deg | 3.52 to 4.63 deg | 0.21 to 4.01 deg | 5.21 to 5.51 deg |
| Third rise | 0.82 to 1.10 deg | -1.44 to 1.55 deg | -0.78 to 0.96 deg | 4.52 to 6.89 deg |

Rise completion uses measured mean knee flexion crossing below 15 degrees
after each flexion peak. These orientations are not direct center-of-mass
measurements. Neither candidate consistently reproduces the late-rise real
lean. The earlier lag candidate brings the second rise closer, but this is
not enough to identify a hardware lag or to pass full-body validation.

The gain-only simulation still has net rearward root displacement of
0.079-0.142 m per playback. Net displacement cannot distinguish stepping,
sliding, and rolling by itself, and the hardware log has no verified absolute
root/foot position with which to count or locate the reported forward steps.

See the [final full-policy comparison](../../gear_sonic_deploy/logs/actuator_identification_20260907/policy_gain_only_final/measured_comparison.png),
[gain-only metrics](../../gear_sonic_deploy/logs/actuator_identification_20260907/policy_gain_only/rise_events.csv),
and [earlier gain/lag metrics](../../gear_sonic_deploy/logs/actuator_identification_20260907/policy_validation/rise_events.csv).
No default promotion follows from these results.

## Implementation and Excluded Runs

[MotorResponse](../../gear_sonic/utils/mujoco_sim/motor_response.py) is opt-in via
`H2_MOTOR_RESPONSE`. It acts after the official output-to-motor torque mapping
and before existing native clipping. Missing or empty configuration preserves
the previous path. Separate telemetry records original PD, mapped motor
request, candidate pre-cap effort, post-candidate request, candidate cap, and
native clipping. Gain/lag changes are not classified as clipping.

To retain realtime headroom, fixed-step scalar response coefficients are
cached and observation-only inverse torque mappings use direct square solves
instead of repeated pseudoinverses. The forward mapping is unchanged. Measured
single-motor response cost fell from about 12.2 to 2.0 us; the five-group
observation inverse fell from 45.4 to 7.7 us. Randomized response tests match
the prior vector formula bit-for-bit. A 500-step old/new comparison retains
bit-identical qpos, qvel and ctrl; tau_est differs only by floating-point
rounding. A total of 67 focused tests passed in the completed work.

Excluded development runs are retained for audit:

- `screen02`: invalid zero-pose linkage branch initialization caused warnings
  or resets. Its `INVALID.json` explicitly excludes it; `screen03` uses branch
  continuation and strict rejection instead.
- First complete-policy candidate attempt: accumulated lag exceeded 20 ms.
  Its incomplete data were excluded; the successful retry followed measured
  implementation optimization, not a relaxed timing threshold.
- Preliminary partial unloaded and intermediate audit directories are not
  substituted for the final completed `unloaded_gain_only` and `effort_audit`.

Offline replays initialize no DDS. The only active command publication in this
work was the isolated simulator validation. All validation sim, deploy, and
recorder processes were stopped after collection; no hardware command was sent.

## Reproduction

These commands are optional reproduction of completed work, not requested
user actions. Run from `/home/qingyul/workspace/SONIC-H2`, and choose unused
output directories because the scripts refuse to overwrite existing results.

Re-run the offline screening using the preserved measured dataset. This
analysis entry point uses `.venv_teleop` for its plotting-library dependency;
the unloaded and full-policy simulator entry points below use `.venv_sim`.

```bash
.venv_teleop/bin/python scripts/identify_h2_loaded_actuators.py \
  --stage screen \
  --dataset gear_sonic_deploy/logs/actuator_identification_20260907/loaded_dataset02/dataset.npz \
  --output test_logs/loaded_screen_reproduction
```

Re-run final gain-only initialization sensitivity and unloaded regression:

```bash
.venv_teleop/bin/python scripts/identify_h2_loaded_actuators.py \
  --stage nuisance \
  --dataset gear_sonic_deploy/logs/actuator_identification_20260907/loaded_dataset02/dataset.npz \
  --candidate gear_sonic_deploy/logs/actuator_identification_20260907/candidate_gain_only.json \
  --output test_logs/loaded_nuisance_reproduction

.venv_sim/bin/python scripts/validate_h2_motor_candidate_unloaded.py \
  --candidate gear_sonic_deploy/logs/actuator_identification_20260907/candidate_gain_only.json \
  --side both --cycles 2 \
  --output test_logs/unloaded_gain_only_reproduction
```

Optional simulator-only policy collection, with the existing identified ankle
model plus the candidate left-knee gain; add `--onscreen` for a local viewer:

```bash
.venv_sim/bin/python scripts/verify_h2_sim_realtime.py \
  --variant identified --repeats 3 \
  --motor-response-config gear_sonic_deploy/logs/actuator_identification_20260907/candidate_gain_only.json \
  --output test_logs/knee_gain_only_policy_reproduction
```

This harness fixes DDS to loopback domain 188, hashes the selected candidate
file, verifies unchanged model/reference files, checks complete playback and
realtime validity, and stops its own child processes. It does not accept a
real-robot network interface.

## Next Evidence Needed

The immediate result is a prioritized hypothesis, not a calibrated actuator
model: investigate loaded left-knee actuation and transmission/contact
interaction before further arbitrary ankle friction or lag inflation.
Separating effort gain from mechanical loss will require independent
output-torque information or controlled load variation; separating motor lag
from transport and estimator timing requires synchronized acquisition timing.
Root/foot tracking and support-force measurement would reduce the main loaded
replay nuisance uncertainty. Any new knee excitation should use an appropriate
supported, load-aware protocol rather than copying the unloaded ankle sweep.
