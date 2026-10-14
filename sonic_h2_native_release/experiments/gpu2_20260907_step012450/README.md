# Temporary Release Step 12450

Source: `qingyul@gpu2.precognition.team`, retrieved on 2026-09-07.
The requested `exported/latest` resolved to the following fixed export:

`/home/qingyul/workspace/GR00T-WBC-H2TRAINBASE/logs_rl/sonic-h2-native-8xA6000-ori/20260906_183110/exported/20260907_220000_step_012450`

All files were copied from that fixed directory. No local symlink follows
remote `latest`. The source manifest and model configuration are preserved.
This is an independent non-v1.1 experiment. Default step 20000, temporary
gpu4 step 7550, v1.1, reference data and deployment parameters are retained.

## Contract

| File | Input | Output |
| --- | --- | --- |
| `model_encoder.onnx` | `[1,640]` | `[1,64]` |
| `model_encoder_full.onnx` | `[1,1751]` | `[1,64]` |
| `model_decoder.onnx` | `[1,994]` | `[1,29]` |

Full selectors: `0=H2`, `1=teleop`, `2=SMPL`, `3=G1`.
Observation contract: `sonic_h2_native_v1`.
Orientation contract: `sonic_full_root_relative_v0`.
The decoder has merged LoRA and folded H2 I/O adapters. Preserve the entire
gravity history and the existing native H2 PD/action scale.

The remote exporter reports exact frozen teleop/SMPL/G1 weight agreement with
its reference checkpoint. That source-checkpoint comparison is exporter
evidence, not a local recheck of training weights.

## Offline Validation

- Three SHA256 values match the export manifest and a fresh remote hash query.
- ONNX full checker, shapes, checkpoint IDs and deployment metadata pass.
- Five CPU encoder-to-decoder trials per selector produce finite outputs.
- Full selector 0 versus compact H2 maximum absolute error: 0.0.
- Config comparison with step 7550 is recorded in `local_model_validation.json`.
  Changes concern training, distillation batches, resume settings and paths;
  deployed observation, action scale and actuator settings are unchanged.
- Current deploy binary built successfully; 12 existing `H2Native*` tests pass.

## Deployment Smoke Result

Fresh full-encoder and decoder TensorRT engines were built locally; both CUDA
graphs captured successfully. The actual deploy executable loaded all eight H2
demos and completed the 399-frame `forward_lunge_R_001__A359_M` reference.
All 749 control samples of encoder/decoder input, action, q/dq and quaternion
are finite; selector stayed 0. Recorded ankle pitch gains are 40/2 and waist
roll/pitch gains are 150/3.

The official Unitree ROS closed-chain simulator ran headless on `lo`/domain 73
with elastic support enabled. Real-time factor was 0.999999; maximum lag was
1.311 ms under the unchanged 20 ms threshold, with no dropped diagnostic rows
or writer errors. Deploy and simulator exited with code 0; no test processes
remain. No real robot was connected. Released-support stability, planner
tracking and VR tracking have not been validated for this export.

Results are in `local_model_validation.json` and `local_sim_validation.json`;
raw logs and command provenance are in `smoke_supported/`. `run_smoke.py`
refuses to overwrite that directory; `analyze_smoke.py` checks the captured run.

Reproduce the offline checks from the workspace root:

```bash
uv run --no-project --with onnx --with onnxruntime --with pyyaml python \
  sonic_h2_native_release/experiments/gpu2_20260907_step012450/validate_models.py
```

## Window Sim2Sim

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
MODEL=../sonic_h2_native_release/experiments/gpu2_20260907_step012450
./deploy.sh \
  --robot-profile h2_native \
  --encoder-file "$MODEL/model_encoder_full.onnx" \
  --decoder-file "$MODEL/model_decoder.onnx" \
  --obs-config policy/release/observation_config_h2_native.yaml \
  --motion-data reference/example_h2/deploy \
  --no-planner --input-type keyboard --output-type zmq \
  --hand-backend none sim
```

Use fresh terminals without inherited experimental `GEAR_SONIC_*` settings.
Confirm `y`, wait for `Init Done`, and press `]` in deploy. In MuJoCo, press `9`
only if elastic support is enabled; verify it becomes disabled. Settle before
pressing `t` to play. `n/p` changes clips, `r` resets, and `o` stops deploy.
Only one controller may publish at a time.

For native G1 references change only `--motion-data` to `reference/example`;
the full encoder uses selector 3 and the same H2 decoder, without G1-to-H2
reference conversion. Prior teleoperation commands can select this experiment
by changing `MODEL` to the directory above, retaining `zmq_manager`, the planner,
G1 calibration and `--hand-backend inspire` for DFX five-finger hands.
