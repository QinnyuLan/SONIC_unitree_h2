# Temporary Release Step 7550

Downloaded and validated on 2026-09-07 from `qingyul@gpu4.precognition.team`:

`/home/qingyul/workspace/GR00T-WBC-H2TRAINBASE/logs_rl/sonic-h2-native-3xpro6000-ori/20260906_003658/exported/20260907_024551_step_007550`

This is an independent non-v1.1 experiment. Default step 20000, prior
experiments, v1.1 weights, deployment code, references and gains are unchanged.
The remote export manifest and model configuration are stored alongside ONNX.

| File | Input | Output |
| --- | --- | --- |
| `model_encoder.onnx` | `[1,640]` | `[1,64]` |
| `model_encoder_full.onnx` | `[1,1751]` | `[1,64]` |
| `model_decoder.onnx` | `[1,994]` | `[1,29]` |

Full selectors: `0=H2`, `1=teleop`, `2=SMPL`, `3=G1`.
The remote exporter reports frozen teleop/SMPL/G1 branches exactly matching
its reference checkpoint; this training-weight comparison was not repeated locally.
Observation: `sonic_h2_native_v1`; orientation: `sonic_full_root_relative_v0`.
Decoder LoRA and H2 I/O adapters are folded. Keep the full gravity history.

## Validation

- All three ONNX SHA256 values match both the downloaded manifest and a fresh
  remote `sha256sum` call; local manifest: `SHA256SUMS`.
- ONNX full checker, shape and metadata checks passed.
- Five CPU encoder-to-decoder trials for each of four selectors: finite.
- Full selector 0 versus compact H2 encoder: maximum absolute error 0.0.
- Config comparison with previously validated step 1500 found nine changes in
  training parallelism, batches, resume settings and paths. Deployment gains,
  action scale, observations and contracts are unchanged.
- Current deploy binary built successfully; 12 existing `H2Native*` tests passed.
- Fresh `encoder_model_encoder_full.trt` and `policy_model_decoder.trt` built
  locally and loaded by the actual deploy executable.
- Official Unitree ROS H2 closed-chain headless simulation, `lo`/domain 73,
  elastic support enabled. All eight H2 references loaded; the 399-frame
  `forward_lunge_R_001__A359_M` completed. 749 control samples were finite for
  encoder input, decoder input, action, q/dq and quaternion; selector stayed 0.
- Recorded ankle pitch Kp/Kd=40/2, waist roll/pitch Kp/Kd=150/3.
- Real-time factor 0.999995; max lag 1.471 ms under the unchanged 20 ms limit;
  no dropped diagnostic rows or writer errors.
- Both smoke launches exited normally and their simulators were stopped.
  The second launch explicitly skips the inapplicable MotionSwitcher wait.
- No real hardware commands were sent. This does not validate released-support
  motion stability, planner tracking or VR tracking.

Results: `local_model_validation.json`, `local_sim_validation.json`.
Logs: `smoke_supported_02/`. The initial smoke run is retained in
`smoke_supported/`; it also completed normally.
Offline checks can be reproduced with:

```bash
cd /home/qingyul/workspace/SONIC-H2
uv run --no-project --with onnx --with onnxruntime --with pyyaml python \
  sonic_h2_native_release/experiments/gpu4_20260907_step007550/validate_models.py
```

`run_smoke.py` records exact bounded loopback commands and refuses to overwrite
an existing run directory. `analyze_smoke.py` verifies recorded signals.

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
MODEL=../sonic_h2_native_release/experiments/gpu4_20260907_step007550
./deploy.sh \
  --robot-profile h2_native \
  --encoder-file "$MODEL/model_encoder_full.onnx" \
  --decoder-file "$MODEL/model_decoder.onnx" \
  --obs-config policy/release/observation_config_h2_native.yaml \
  --motion-data reference/example_h2/deploy \
  --no-planner --input-type keyboard --output-type zmq \
  --hand-backend none sim
```

Use a fresh terminal without inherited experimental `GEAR_SONIC_*` overrides.
Confirm `y`, wait for `Init Done`, press `]` in deploy. In the MuJoCo window,
press `9` if the elastic support is still enabled and check that it becomes
disabled. After settling, `t` plays, `n/p` changes the clip, `r` resets, and
`o` stops deploy. Run only one deploy controller at a time. The above paths
select step 7550 explicitly; ordinary default commands still use step 20000.
