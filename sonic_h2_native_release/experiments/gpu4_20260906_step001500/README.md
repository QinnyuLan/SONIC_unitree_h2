# Temporary Release Step 1500

This is a side-by-side, non-v1.1 release experiment. The default models in
`sonic_h2_native_release/exported` remain at step 25400. No deployment code,
default model paths, reference motions, or v1.1 models were changed.

Source host: `qingyul@gpu4.precognition.team`

Source export:
`/home/qingyul/workspace/GR00T-WBC-H2TRAINBASE/logs_rl/sonic-h2-native-3xpro6000-ori/20260906_003658/exported/20260906_163937_step_001500`

## Model Contract

| File | Input | Output | Route |
| --- | --- | --- | --- |
| model_encoder.onnx | 1 x 640 | 1 x 64 | H2-only |
| model_encoder_full.onnx | 1 x 1751 | 1 x 64 | 0=H2, 1=teleop, 2=SMPL, 3=G1 |
| model_decoder.onnx | 1 x 994 | 1 x 29 | H2, LoRA and I/O adapters folded |

Observation contract: `sonic_h2_native_v1`.
Reference orientation: `sonic_full_root_relative_v0`.
Keep the complete 30D gravity history; do not apply the legacy tail overwrite.

The saved config and remote actuator implementation use the existing
`unitree_rl_mjlab_h2` fixed gains and action scales. The 10 Hz config field does
not recompute gains for this profile. Thirteen selected deployment configuration
fields match the current step 25400 configuration, including the actuator asset
settings, history lengths, 50 Hz policy interval, action clipping, and contracts.
The local Unitree ROS closed-loop simulator keeps its 1 ms physics timestep and
floor friction `[1.2, 0.005, 0.0001]`.

## Validation

- All three file hashes match `deployment_manifest.json`.
- All three ONNX models pass full checker and deployment metadata checks.
- CPU inference: five samples per selector, all outputs finite.
- Full selector 0 versus H2-only encoder maximum absolute error: 0.0.
- Fresh local TensorRT caches built for full encoder and decoder.
- Headless Unitree ROS `H2_loop.xml` smoke test on `lo`, DDS domain 73.
- All eight H2 demos loaded; first forward-lunge reference played to completion.
- 599 control cycles, selector 0 throughout; encoder input, decoder input,
  action, joint position/velocity, and base quaternion logs are finite.
- Deploy exited normally with code 0. The test simulator was then stopped.
- Elastic support was enabled. This is not a free-motion stability validation
  and no real robot was connected.

Detailed records: `local_model_validation.json`, `local_sim_validation.json`.
Raw logs from this check: `/tmp/sonic_step1500_deploy.log`,
`/tmp/sonic_step1500_sim.log`, `/tmp/sonic_step1500_smoke/`.

## Sim2Sim: All Eight H2 References

Terminal 1:

```bash
cd /home/qingyul/workspace/SONIC-H2
source .venv_sim/bin/activate
export GEAR_SONIC_DOMAIN_ID=0
python gear_sonic/scripts/run_sim_loop.py --wbc-version h2_native_unitree_ros_loop
```

Terminal 2:

```bash
cd /home/qingyul/workspace/SONIC-H2/gear_sonic_deploy
source scripts/setup_env.sh
export GEAR_SONIC_DOMAIN_ID=0
MODEL=../sonic_h2_native_release/experiments/gpu4_20260906_step001500
./deploy.sh \
  --robot-profile h2_native \
  --encoder-file "$MODEL/model_encoder_full.onnx" \
  --decoder-file "$MODEL/model_decoder.onnx" \
  --obs-config policy/release/observation_config_h2_native.yaml \
  --motion-data reference/example_h2/deploy \
  --no-planner --input-type keyboard --output-type zmq \
  --hand-backend none sim
```

The reference manifest declares `h2`, so these motions select branch 0 without
G1-to-H2 projection. Confirm with Y, wait for `Init Done`, then press `]` in
terminal 2 to start policy control. Press `9` in the simulator window to release
the elastic support. In terminal 2, `t` plays, `n`/`p` selects the next/previous
motion, and `o` exits. Subsequent tests without these explicit model paths still
use the default step 25400.
