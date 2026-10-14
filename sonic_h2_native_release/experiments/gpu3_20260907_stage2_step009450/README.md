# Temporary Release: GPU3 Stage 2 Step 9450

Source: `qingyul@gpu3.precognition.team`, retrieved 2026-09-07.

`/home/qingyul/workspace/GR00T-WBC-H2TRAINBASE/logs_rl/sonic-h2-native-8xA6000-ori-2/20260906_204409/stage2/exported/stage2_step_009450`

This is an independent non-v1.1 experiment. Default step 20000, gpu4 step 7550,
gpu2 step 12450, v1.1 and reference datasets are retained. Remote export
manifest and configuration are preserved alongside the model files.

## Contract And Stage Differences

| File | Input | Output |
| --- | --- | --- |
| `model_encoder.onnx` | `[1,640]` | `[1,64]` |
| `model_encoder_full.onnx` | `[1,1751]` | `[1,64]` |
| `model_decoder.onnx` | `[1,994]` | `[1,29]` |

Full selectors remain `0=H2`, `1=teleop`, `2=SMPL`, `3=G1`.
Observation: `sonic_h2_native_v1`; orientation: `sonic_full_root_relative_v0`.
Decoder LoRA and H2 I/O adapters are folded. Keep the complete gravity history
and the existing native H2 actuator gains/action scale.

Compared with gpu2 step 12450, stage 2 config freezes the H2 encoder, disables
paired encoder distillation/auxiliary loss, adds training encoder-source
routing with H2/G1 weights 2:1, and uses a native critic input adapter.
Training tokenizer dimension grows from 2467 to 3108 with paired G1 fields.
These are training-side changes; the exported full encoder still accepts
1751D, and the deployment policy does not load the critic. Do not copy the
training tokenizer dimension into the runtime observation YAML.

The exporter reports frozen teleop/SMPL/G1 weights exactly matching its source
reference checkpoint. That training-weight comparison was not rerun locally.

## Offline Checks

- All three hashes match both the export manifest and a fresh remote SHA256 query.
- ONNX full checker, shapes, global/checkpoint IDs and runtime metadata passed.
- Five CPU encoder-to-decoder trials per selector were finite for all four routes.
- Full selector 0 versus H2-only maximum absolute difference: 0.0.
- Current deploy binary built successfully; 12 existing `H2Native*` tests passed.
- Complete config differences are in `local_model_validation.json`.

## Deployment Smoke Result

Fresh local full-encoder and decoder TensorRT engines loaded and both CUDA
graphs captured. The actual deploy executable loaded all eight H2 references
and completed the 399-frame `forward_lunge_R_001__A359_M` clip. Encoder input,
decoder input, action, q/dq and base quaternion were finite in all 749 control
samples; encoder selector remained 0. Logged ankle pitch gains were 40/2,
and waist roll/pitch gains were 150/3.

The official Unitree ROS H2 closed-chain simulator ran headless on loopback
domain 73 with elastic support enabled. Real-time factor was 0.999999, maximum
lag 1.528 ms under the unchanged 20 ms threshold, with no dropped diagnostic
rows or writer errors. Both processes exited with code 0 and were stopped.
No real hardware commands were sent. Released-support stability, G1 reference
tracking, planner tracking and VR behavior remain unvalidated for this export.

Recorded results: `local_sim_validation.json`; raw logs: `smoke_supported/`.
`run_smoke.py` records the isolated commands and refuses to overwrite the run;
`analyze_smoke.py` checks the saved signals and runtime validity.

Reproduce offline checks from `/home/qingyul/workspace/SONIC-H2`:

```bash
uv run --no-project --with onnx --with onnxruntime --with pyyaml python \
  sonic_h2_native_release/experiments/gpu3_20260907_stage2_step009450/validate_models.py
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
MODEL=../sonic_h2_native_release/experiments/gpu3_20260907_stage2_step009450
./deploy.sh \
  --robot-profile h2_native \
  --encoder-file "$MODEL/model_encoder_full.onnx" \
  --decoder-file "$MODEL/model_decoder.onnx" \
  --obs-config policy/release/observation_config_h2_native.yaml \
  --motion-data reference/example_h2/deploy \
  --no-planner --input-type keyboard --output-type zmq \
  --hand-backend none sim
```

Use fresh terminals without inherited experimental `GEAR_SONIC_*` overrides.
Confirm `y`, wait for `Init Done`, then press `]` in deploy. In MuJoCo press
`9` only if elastic support is enabled; verify it becomes disabled. After
settling, `t` plays, `n/p` selects, `r` resets and `o` stops. One controller
may publish at a time.

For native G1 reference testing change `--motion-data` to `reference/example`.
This uses selector 3 and the same H2 decoder, without G1-to-H2 projection.
Prior VR commands can select this experiment by changing `MODEL` above;
retain the planner, `zmq_manager`, G1 calibration and `--hand-backend inspire`
for the DFX five-finger hands. Hardware/VR behavior is not established by
offline or supported simulation checks.
