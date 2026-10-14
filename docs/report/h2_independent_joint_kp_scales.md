# Independent H2 Ankle Pitch And Knee Kp Scales

Added 2026-09-08 for non-v1.1 `h2_native` deployment. These startup options
multiply nominal Kp only, for the four named PR/output-coordinate joints.

| Option | DDS index | Nominal Kp | Nominal Kd |
| --- | ---: | ---: | ---: |
| `--h2-left-ankle-pitch-kp-scale` | 5 | 40 | 2 |
| `--h2-right-ankle-pitch-kp-scale` | 11 | 40 | 2 |
| `--h2-left-ankle-roll-kp-scale` | 4 | 40 | 2 |
| `--h2-right-ankle-roll-kp-scale` | 10 | 40 | 2 |
| `--h2-left-knee-pitch-kp-scale` | 3 | 200 | 4 |
| `--h2-right-knee-pitch-kp-scale` | 9 | 200 | 4 |

Knee pitch refers to `left_knee_joint` / `right_knee_joint` in the official
model; it does not select the motor-side closed-chain coordinate.

2026-09-18: sim2real configuration adds `LEFT_ANKLE_ROLL_KP=0.7` and
`RIGHT_ANKLE_ROLL_KP=0.7`, giving Kp=28 for each ankle roll while Kd stays 2.
Each explicit side overrides `--h2-ankle-roll-kp-scale` without compounding.
Raw deploy and sim2sim still default to nominal scale 1.0; older custom real
configs without these two keys also fall back to 1.0.

An explicitly supplied single-side ankle scale replaces the paired
`--h2-ankle-pitch-kp-scale` for that side, regardless of argument order. It is
not multiplied by the paired value. An omitted side inherits the paired
value (default 1). Knee scales default to 1. Explicit single-side scale 1
restores nominal Kp even when a paired ankle value is supplied.

Kp scaling happens once when constructing published LowCmd, including INIT,
WAIT_FOR_CONTROL and CONTROL. These options do not change Kd, action scale,
reference, q target or feedforward torque. Existing Kd options still work.
Shutdown remains Kp=0/Kd=8. Values must be finite and positive and fit in float
after scaling. CLI validation is not a physical safety bound.

Restart deploy to change scales. Startup logs print all four effective Kp/Kd
pairs. With CSV logs, metadata records the effective per-side scales and
`lowcmd_command.csv` records the gains actually sent.

## Hardware Command

2026-09-18 update: parameters now live in
[h2_deploy.env](../../gear_sonic_deploy/config/h2_deploy.env). The launcher reads
that sim2real-only configuration. See [current model and launch commands](h2_stage2_70000_deployment.md).
The explicit command below is the historical step 9450 example.

Editable launcher: [run_h2_sim2real_four_kp.sh](../../gear_sonic_deploy/scripts/run_h2_sim2real_four_kp.sh).
Edit the four scales, model, motion and network settings in the sim2real config, then run
`bash /home/qingyul/workspace/SONIC-H2/gear_sonic_deploy/scripts/run_h2_sim2real_four_kp.sh`
from any working directory. Changes take effect on the next launch.

The current experiment remains gpu3 stage 2 step 9450. The example explicitly
selects H2 squat; change `MOTION` to a different loaded H2 clip if desired.
All four scales initially use 1.0; no new tuned value has been established.

Stop any old controller/simulator and prepare the robot support before use.
Use a fresh terminal without inherited automatic-control environment overrides.

```bash
cd /home/qingyul/workspace/SONIC-H2/gear_sonic_deploy
source scripts/setup_env.sh
export GEAR_SONIC_DOMAIN_ID=0
unset GEAR_SONIC_AUTO_START_AFTER_SEC GEAR_SONIC_AUTO_PLAY_AFTER_SEC
unset GEAR_SONIC_AUTO_NEXT_AFTER_SEC GEAR_SONIC_AUTO_STOP_AFTER_SEC
unset GEAR_SONIC_SKIP_MOTION_SWITCHER

MODEL=../sonic_h2_native_release/experiments/gpu3_20260907_stage2_step009450
MOTION=squat_001__A359
LEFT_ANKLE_KP=1.0
RIGHT_ANKLE_KP=1.0
LEFT_KNEE_KP=1.0
RIGHT_KNEE_KP=1.0

GEAR_SONIC_INITIAL_MOTION_NAME="$MOTION" ./deploy.sh \
  --robot-profile h2_native \
  --encoder-file "$MODEL/model_encoder_full.onnx" \
  --decoder-file "$MODEL/model_decoder.onnx" \
  --obs-config policy/release/observation_config_h2_native.yaml \
  --motion-data reference/example_h2/deploy \
  --no-planner --input-type keyboard --output-type zmq \
  --hand-backend none \
  --h2-left-ankle-pitch-kp-scale "$LEFT_ANKLE_KP" \
  --h2-right-ankle-pitch-kp-scale "$RIGHT_ANKLE_KP" \
  --h2-left-knee-pitch-kp-scale "$LEFT_KNEE_KP" \
  --h2-right-knee-pitch-kp-scale "$RIGHT_KNEE_KP" \
  --enable-csv-logs \
  --logs-dir "logs/h2_four_kp_$(date +%Y%m%d_%H%M%S)" \
  eno1
```

Confirm `y` (initialization can move joints), wait for `Init Done`, press `]`
to enter policy control, settle, then `t` to play. No MuJoCo `9` step on real
hardware. `o` stops pose holding and enters damping; secure the robot first.
To test the same scale combination in simulation, use `sim` in place of
`eno1` and the normal `h2_native_unitree_ros_loop` simulator preset.

## Validation (2026-09-08)

- Deployment binary and unit-test target built successfully; all 14 H2Native
  tests passed. The four new CLI options rejected 24 invalid-value cases
  before DDS initialization.
- The actual launcher ran gpu3 stage 2 step 9450 through the complete H2 squat
  in the supported, headless official closed-chain simulator on loopback DDS
  domain 73. Both processes exited normally.
- All 9,011 active LowCmd rows were checked across all 31 joints in INIT,
  WAIT_FOR_CONTROL and CONTROL. Independent scales 1.1/1.2/0.9/1.3 produced
  left/right ankle Kp 44/48 and left/right knee Kp 180/260. Supplying paired
  ankle scale 1.5 afterward did not compound the explicit side values.
  All other Kp and every Kd matched nominal values; shutdown used Kp=0/Kd=8.
- Effective scales were recorded correctly in metadata. Real-time factor
  was 1.000007, maximum scheduler lag 1.50 ms. This supported simulation
  validates gain plumbing, not unsupported hardware stability or tuned gains.

Evidence: [validation.json](../../test_logs/h2_independent_kp_20260908/validation.json).
No hardware control was executed for this validation.
