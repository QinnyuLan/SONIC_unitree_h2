# H2 Model Deployment Workflow Handoff

Recorded: 2026-09-07, Asia/Shanghai.
Source task: `新模型部署工作流`,
`codex://threads/019f641d-f595-7610-8814-7c5adb06d332`.
All 160 turns were retrieved; relevant final decisions were checked against
the current workspace. This document carries the workflow into the successor
task requested by the user. Historical test results below are not new tests.

Workspace: `/home/qingyul/workspace/SONIC-H2`.
The workspace contains extensive existing uncommitted work. Preserve it.
This handoff changes documentation only; it does not replace weights, start
controllers, or continue pending hardware experiments.

## Current Baseline

Latest 2026-09-18 update: sim2real now selects GPU3 stage2 step 5000 at
`experiments/gpu3_20260909_stage2_step005000/`, exported in the server's
`ablation_stage2_5k_30k_20260909/stage2_step_005000` bundle. The user requested
a hardware comparison after step 9450 also showed right ankle roll heating.
Current sim2real motion/gain settings are preserved. Sim2sim remains on step
9450. See the step 5000 experiment README for validation and source details.

2026-09-18 evening rollback: the configured sim2real and sim2sim launchers now
select `experiments/gpu3_20260907_stage2_step009450/` (GPU3 stage2 step 9450),
confirmed against `logs/h2_four_kp_20260908_225918/metadata.json`.
The sim2real configuration preserves current scales: ankle pitch 1.2/1.4,
ankle roll 0.7/0.7, knees 1.2/1.2, with `reference/example` motions.
Sim2sim retains nominal gains. The generic `exported/` directory still contains
step 70000; use the configured launchers for this rollback. The table below
describes the earlier step 70000 promotion.

Updated 2026-09-18: [stage2 step 70000 deployment and configuration](h2_stage2_70000_deployment.md).

| Item | Current value |
| --- | --- |
| Default profile | `h2_native`, non-v1.1 |
| Default checkpoint | gpu3 original stage2 step 70000, package `h2_stage2_070000_20260918` |
| Model directory | `sonic_h2_native_release/exported/` |
| Active encoder | `model_encoder_full.onnx`, `[1,1751] -> [1,64]` |
| Compact encoder | `model_encoder.onnx`, `[1,640] -> [1,64]`, H2 only |
| Shared decoder | `model_decoder.onnx`, `[1,994] -> [1,29]` |
| Observation YAML | `gear_sonic_deploy/policy/release/observation_config_h2_native.yaml` |
| Observation contract | `sonic_h2_native_v1` |
| Non-v1.1 orientation | `sonic_full_root_relative_v0` |
| Simulator preset | `h2_native_unitree_ros_loop` |
| Real network | `eno1`, DDS domain 0; historical workstation address `192.168.123.222/24` |
| Sim network | `lo`, DDS domain 0, or a matched isolated domain for automated checks |

Step 70000 replaces the default step 20000 as of 2026-09-18. The old default is
preserved in `exported/backup_before_stage2_step070000_20260918_181453/`.
`gear_sonic_deploy/config/h2_deploy.env` controls sim2real only; it currently
selects standard SONIC demos (`reference/example`) and per-joint Kp scales 1.4/1.6/1.3/1.3.
Sim2sim has independent settings in its launch script and uses nominal gains. See the update report for validation and commands.

Historical notes below describe the earlier handoff. The default step 20000 replaced step 25400 at the user's request. The previous
release is in `backup_before_default_step020000_20260906_hUxSps/`.
Temporary checkpoints remain separate:
`sonic_h2_native_release/experiments/gpu4_20260906_step001500/` and
`sonic_h2_native_release/experiments/m10_20260906_step020000/`.

Successor-task addition on 2026-09-07: temporary gpu4 step 7550 is in
`sonic_h2_native_release/experiments/gpu4_20260907_step007550/`.
Model checks, four-selector CPU inference, fresh TensorRT loading and supported
closed-chain sim2sim passed. See that experiment's README. It is not the default;
released-support motion stability and hardware behavior remain unvalidated.

Later addition on 2026-09-07: gpu2 `exported/latest` resolved to
`20260907_220000_step_012450`. This fixed export is preserved independently in
`sonic_h2_native_release/experiments/gpu2_20260907_step012450/`.
Hashes, four-selector CPU inference, fresh TensorRT loading, and supported
H2 reference smoke passed. Step 7550 remains available and step 20000 remains
the default. See the new experiment README for explicit launch paths.

Next temporary export: gpu3 stage 2 step 9450, preserved in
`sonic_h2_native_release/experiments/gpu3_20260907_stage2_step009450/`.
It keeps the four-selector 1751D deployment interface despite extra training
tokenizer/critic fields. Hashes, four-selector CPU inference, fresh TensorRT
loading and supported H2 reference smoke passed. Existing experiments and the
default remain unchanged; free-motion and hardware behavior are unvalidated.

Historical step 20000 SHA256 verified during the original handoff (current hashes
are in `exported/DEPLOY_SHA256SUMS`):

```text
80d16e85dcfb4b34a556569c6140d851227b12ec841b508b6cfdda649bf8be44  model_encoder.onnx
994addb6272e82a0a4b3edfe4dd73c56840600b097fa3656cb1f29ccf64adcb4  model_decoder.onnx
200d4cbb043a36c8b6e106c2e72059c0625f4a4c5d2ae1b74246025c68d810fd  model_encoder_full.onnx
```

v1.1 is independent: `sonic_h2_native_v1_1_release/exported/`, step 16150,
export `deployment_step_016150_cpu_fully_folded`, profile `h2_native_v1_1`,
orientation `sonic_robot_heading_v1`. It uses universal `model_encoder.onnx`
with selectors 0/1/2 and its matching decoder/YAML. Its 640D H2-only sidecar
cannot simply replace the universal encoder in the current v1.1 runtime.
Do not infer compatibility from equal dimensions or a repeated step number.

## Adding Or Updating Models

1. Resolve the requested profile, server, exact export directory, and whether
   this is a temporary experiment or a default replacement. A request for a
   temporary test means a separate experiment directory with explicit paths.
   Ordinary non-v1.1 updates must not overwrite v1.1 or G1 weights.
2. Retrieve the paired ONNX files and available configuration, export metadata,
   manifests, and server hashes into a staging directory. Use the server/path
   supplied for that update. Historical hosts include `qingyul@m10.precognition.team`,
   `qingyul@gpu4.precognition.team`, and `sky@120.79.230.50:6022` for v1.1.
   Credentials are intentionally omitted here. Historical m10 failures often
   came from the wrong intranet connection; a failed transfer is not an update.
3. Validate SHA256, ONNX graphs, tensor shapes, names, selector metadata,
   observation/orientation contracts, folding, PD/action scale, and joint order.
   Read new export metadata instead of assuming that filename implies semantics.
   Modern non-v1.1 full exports have all three files listed above.
4. Run finite CPU inference for each supported selector and encoder-to-decoder
   inference. Compare full selector 0 with the matching compact encoder where
   available. Distinguish synthetic inference checks from motion tracking tests.
5. For a default replacement, preserve the previous coherent model set,
   metadata, hashes, and active TensorRT caches in a timestamped backup. Promote
   the complete validated set while controllers are stopped. Never mix exports.
6. Rebuild affected TensorRT engines on this deployment machine. Active default
   caches are `encoder_model_encoder_full.trt` and `policy_model_decoder.trt`;
   cache names follow selected ONNX stems. Verify the actual deploy executable
   loads the intended files and accepts 1751D/994D observation layouts.
7. Run the existing deployment contract tests relevant to the change, then
   isolated loopback sim2sim. First establish loading/routing under support;
   separately test released-support full reference playback and real-time rate.
   State explicitly which level passed. Suspended smoke tests do not establish
   free-standing tracking or hardware stability.
8. Record source checkpoint/export, hashes, contract differences, backups,
   validation and launch commands in the release or experiment README/metadata.
   Only change deployment code when the new contract actually requires it.

Runtime invariants:

- One selected encoder and one shared H2 decoder at runtime. Decoder input is
  64D token plus 930D H2 history; output is 29 normalized H2 body actions.
- Preserve the complete 30D gravity history. Do not reinstate the legacy
  `obs[..., -29:] = last_action` patch.
- The current published model is `models/released`, trained by direct fine-tuning
  with native H2 inputs/outputs. It requires no LoRA or I/O-adapter merge.
  Runtime validation checks the interface, observation/orientation contracts
  and selector routing, without requiring an adapter-folding tag. Descriptions
  of folded adapters elsewhere in this historical report apply to earlier exports.
- Current PD/action scale comes from `unitree_rl_mjlab_h2`, reference commit
  `1425b15f73bd4095f0df53709d7c389c3eb9e790`. Historical nf10/nf16/nf20
  discussions do not replace the current fixed-gain contract.
- Keep policy 50 Hz, LowCmd publication 500 Hz, and physics 1000 Hz distinct.

## Reference And Planner Routing

| Input | Full selector | Data or behavior |
| --- | ---: | --- |
| H2 demo | 0 | `reference/example_h2/deploy`, 31D H2 q/qdot sidecars |
| VR three-point | 1 | Native teleop reference, G1 calibration by current preference |
| SMPL demo / POSE | 2 | `reference/example_smpl_h2/deploy`, SMPL joints/root |
| G1 demo | 3 | `reference/example`, native G1 29D q/qdot |
| Default planner | 3 | Native G1 planner future references |
| Explicit comparison planner | 0 | Only with `--planner-g1-to-h2` |

Reference directories must declare `reference_type` in
`h2_lora_reference_manifest.json`; see
[reference types](../../gear_sonic_deploy/reference/REFERENCE_MOTION_TYPES.md).
The legacy manifest filename does not make native H2 data a LoRA policy.
Do not silently convert native G1 demos or planner output to H2 reference.
The later user-requested `--planner-g1-to-h2` comparison applies the legacy
linear map to sampled planner q/qdot and restores selector 0 after VR3PT.
Without it, exiting VR3PT restores selector 3. VR3PT itself remains selector 1.

The standard H2 set has 8 clips, 5484 frames at 50 Hz. It was refreshed from
m10 `/mnt/nvme3/qingyul/seed/h2_soma_full_g1_smpl_source_ik_fixed_20260827/csv`;
120 Hz sources were resampled with quaternion Slerp. Native H2, G1 and paired
SMPL streams must retain their own semantics. Do not confuse generated semantic
sidecars with native G1 reference data. `reference/extreme_h2/` is separate;
`reference/h2_default_stand_only/` is synthetic diagnostic standing, not the
normal eight-demo collection.

## Sim2Sim Reference Workflow

Terminal 1:

```bash
cd /home/qingyul/workspace/SONIC-H2
export GEAR_SONIC_DOMAIN_ID=0
.venv_sim/bin/python gear_sonic/scripts/run_sim_loop.py \
  --wbc-version h2_native_unitree_ros_loop
```

Terminal 2:

```bash
cd /home/qingyul/workspace/SONIC-H2/gear_sonic_deploy
source scripts/setup_env.sh
export GEAR_SONIC_DOMAIN_ID=0
./deploy.sh \
  --robot-profile h2_native \
  --motion-data reference/example_h2/deploy \
  --no-planner --input-type keyboard --output-type zmq \
  --hand-backend none \
  sim
```

Confirm `y` at the existing deployment prompt; initialization can move joints.
Wait for `Init Done`, press `]` in deploy to enter policy control, then press
`9` in the MuJoCo window if support is still enabled. Verify
`ElasticBand enable: False`, settle, and press `t` in deploy to play.
Use `n/p` for motion selection, `r` to reset the clip, `i` to calibrate heading,
and `o` to stop. MuJoCo `9` is a toggle, not an unconditional disable command.
Do not toggle planner/encoder during a controlled reference comparison.

For G1 or SMPL tracking change the reference directory using the table above,
keeping the full encoder and shared H2 decoder. For keyboard planner testing,
replace `--no-planner` with
`--planner planner/target_vel/V2/planner_sonic.onnx` and retain keyboard input;
use the runtime keyboard help for planner mode/motion-set controls.

The preset uses official Unitree ROS `h2_description/H2_loop.xml`, including
physical knee, ankle and waist linkages. Current floor friction is
`[1.2, 0.005, 0.0001]`. Preserve official motor-side limits and passive terms.
The simulation computes output-coordinate PD and maps it through the current
closed-chain constraint Jacobian to motor torque; DDS remains native PR
output coordinates. Do not substitute an older open-chain asset unnoticed.

Real-time pacing now catches up against a monotonic deadline and rejects lag
over 20 ms. Earlier runs with roughly 0.65 real-time factor were invalid for
policy comparison. Check simulated versus wall duration; do not compensate by
arbitrarily changing physics dt, slowing references, or relaxing the lag limit.

## Sim2Real Reference Workflow

Use the same profile, reference and gains as the accepted simulation. Replace
the final `sim` with `eno1`; do not run a simulator for the hardware test.
Before each real launch, inspect current network and publisher state rather
than assuming a historical address or process report is still current:

```bash
cd /home/qingyul/workspace/SONIC-H2/gear_sonic_deploy
source scripts/setup_env.sh
export GEAR_SONIC_DOMAIN_ID=0
ip -br addr show eno1
./target/release/h2_state_reader eno1 --domain-id 0 --wait 5
```

The operator prepares support and the existing stop mechanism, confirms `y`,
waits for `Init Done`, presses `]`, settles, then `t`. No MuJoCo `9` step on
hardware. Keep CRC and the existing MotionSwitcher release path enabled.
There must be only one active body controller. `o` removes pose holding and
enters damping; it is not a hardware emergency stop. A read-only recorder
does not control or stop the robot.

Current independent startup gain options all default to 1.0:

| Option | Nominal gain |
| --- | --- |
| `--h2-waist-pitch-kp-scale` | Kp 150 |
| `--h2-waist-roll-kp-scale` | Kp 150 |
| `--h2-ankle-pitch-kp-scale` | Kp 40 on both ankles |
| `--h2-ankle-roll-kp-scale` | Kp 40 on both ankles |
| `--h2-ankle-pitch-kd-scale` | Kd 2 on both ankles |

Scaling applies once to actual LowCmd through initialization and control;
shutdown remains Kp=0/Kd=8. These are startup options, not online tuning.
The observed benefit of ankle Kp scale 1.2-1.4 is an experiment, not a new
default. Record actual q/dq/Kp/Kd/tau using `--enable-csv-logs --logs-dir ...`.

## VR Teleoperation And Inspire Hands

Pico must run XRoboToolkit with valid tracking. Current preference is the
original G1 calibration for both VR3PT and SMPL wrist references.
Terminal 1, hardware deploy:

```bash
cd /home/qingyul/workspace/SONIC-H2/gear_sonic_deploy
source scripts/setup_env.sh
export GEAR_SONIC_DOMAIN_ID=0
./deploy.sh \
  --robot-profile h2_native \
  --planner planner/target_vel/V2/planner_sonic.onnx \
  --input-type zmq_manager --output-type zmq \
  --motion-data reference/example_h2/deploy \
  --h2-vr-head-control --hand-backend inspire \
  eno1
```

Terminal 2:

```bash
cd /home/qingyul/workspace/SONIC-H2
.venv_teleop/bin/python gear_sonic/scripts/pico_manager_thread_server.py \
  --manager --vr3pt-robot-profile g1 --smpl-wrist-source g1
```

For simulation use terminal 1's `sim` endpoint and `--hand-backend none`,
plus the simulator terminal from the sim2sim section.

After deploy confirmation and `Init Done`, subsequent VR start/stop and mode
changes belong to the Pico manager. First ABXY in OFF calibrates and starts
PLANNER. For the current G1 calibration, stand upright, head forward, upper
arms down and forearms forward around 90 degrees; release both triggers.
Before pressing the left stick to enter VR3PT, align with the robot's current
arm posture; that transition recalibrates wrists. Left-stick press again
returns to ordinary planner. A+X selects full-body SMPL POSE. ABXY stops;
deploy `o` also stops. SLOW_WALK's current full-stick target is 0.45 m/s.

Do not combine `zmq_manager` with `--no-planner`: the manager first sends
PLANNER. Plain `zmq` subscribes to pose data and does not implement the same
ABXY command lifecycle. Feedback CLI options use underscores:
`--zmq_feedback_host localhost --zmq_feedback_port 5557`.
Port 5557 conflicts commonly indicate a second deploy. A manager's state
message alone does not prove the body controller entered CONTROL.

DFX/Inspire hand prerequisite on PC2:

```bash
ssh unitree@192.168.123.162 \
  'systemctl status h2-inspire-hands.service --no-pager'
cd /home/qingyul/workspace/SONIC-H2
.venv_teleop/bin/python gear_sonic_deploy/scripts/h2_inspire_hand_dds_demo.py \
  --network eno1 --monitor-seconds 5
```

The service should be active and both hands ONLINE. Only one hand command
sender may be active. Use `inspire`, not the launcher's legacy `dex3` default.
Triggers command whole-hand opening/closing, not per-finger tracking or
continuous proportional control. Deploy `c/x` decrease/increase maximum
closure by 0.1. Both hand states must be valid; after reconnect, release the
triggers to rearm. Lost feedback stops hand publication but does not open
the hands automatically. Release triggers and observe opening before stopping.

## Sim2Real Diagnostics And Outstanding Work

Use [squat comparison instructions](h2_squat_sim_real_diagnostics.md) and
`scripts/h2_squat_compare.py` for matched baseline/identified simulation,
read-only DDS recording, and hardware deployment. Use a fresh `--run` for
each experiment. The launcher selects current non-v1.1 weights, H2 squat,
selector 0, all five gain scales 1, no planner/hands, and clears inherited
`GEAR_SONIC_*` experimental overrides. Failed logs remain preserved.

The DDS reader startup race (`NoneType.take`) has been fixed by completing
reader construction before attaching callbacks. Simulation CRC failures
are recorded and do not justify disabling hardware CRC validation.

Latest conclusions from the source task:

- Analyze actual LowCmd target-to-measured q/dq, not encoder reference, when
  identifying actuators. Reference error includes policy decisions.
- Independent left/right ankle pitch response models exist; they are explicit
  candidates (`h2_native_unitree_ros_loop_ankle_sysid`), not baseline defaults.
  Dynamic internal state requires the matching runtime, not XML alone.
- Supported/unloaded data fits do not establish loaded squat behavior. Both
  ankle models still failed to reproduce persistent real post-rise forward tilt.
- A conditional left-knee effective torque gain of 0.725 improved held-out
  short windows but did not reproduce the complete real behavior. It is not
  measured hardware efficiency and must not be applied to real Kp or promoted
  to default. See [loaded identification report](h2_loaded_actuator_identification.md).
- Hardware logs lack reliable absolute body translation and foot load; tau_est
  is an uncalibrated drive estimate. Reconstructed foot-height difference of
  about 11 mm may also reflect zero/geometry/initial-state errors.
- Pending suggestion: audit all leg/waist LowCmd responses and geometry/IMU
  alignment, then build dedicated bilateral knee dynamic tests and loaded
  comparisons. Do not reuse the unloaded ankle script by renaming joints: its
  body-tilt stop was explicitly disabled for that limited ankle experiment.
  Knee support and force-sensing availability were still unresolved. The user
  has not requested execution of this pending experiment in the handoff turn.

## Primary Local Sources

- [H2 paper/demo sim2sim presentation mode](h2_paper_sim2sim.md)
- [Independent left/right ankle pitch and knee Kp scales, commands and validation](h2_independent_joint_kp_scales.md)
- [Release contract and validation](../../sonic_h2_native_release/exported/DEPLOYMENT_README.md)
- [v1.1 deployment notes](../../sonic_h2_native_v1_1_release/exported/DEPLOYMENT_NOTES.md)
- [Launcher](../../gear_sonic_deploy/deploy.sh)
- [Model contract validation](../../gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/h2_native_contract.hpp)
- [Policy control and routing](../../gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp)
- [Official closed-chain preset](../../gear_sonic/utils/mujoco_sim/wbc_configs/h2_native_sonic_unitree_ros_loop.yaml)
- [Pico manager](../../gear_sonic/scripts/pico_manager_thread_server.py)
- [Ankle response report](h2_ankle_pitch_identified_mujoco.md)

For future updates, refresh this document's baseline and the release metadata
after actual promotion. Prefer the latest explicit user decision and current
code/ONNX metadata over older source-task commands or claims.
