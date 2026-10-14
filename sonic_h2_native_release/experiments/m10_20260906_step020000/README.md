# Temporary Release Step 20000

Initially a side-by-side non-v1.1 release experiment. This model was subsequently
promoted to the default release at the user's request on 2026-09-06. The previous
step 25400 default is backed up under
`../../exported/backup_before_default_step020000_20260906_hUxSps/`.
The separate GPU4 step 1500 experiment and v1.1 models are unchanged.
Simulator settings and reference motions were not changed. Four optional Kp
scaling arguments are documented in `../../exported/DEPLOYMENT_README.md`.

Source host: `qingyul@m10.precognition.team`

Source export:
`/home/qingyul/workspace/GR00T-WholeBodyControl/logs_rl/sonic-h2-native-4xL40/manager/universal_token/all_modes/sonic_h2_native_e2e_release_g1_smpl_only_h2_native_release_g1smplonly_fullencoderlora_stdlearn_aggr_4xl40_100k_20260903-20260903_210325/exported/20260906_171609_step_020000`

## Contract and Validation

| Model | Input | Output | Route |
| --- | --- | --- | --- |
| model_encoder.onnx | 1 x 640 | 1 x 64 | H2-only |
| model_encoder_full.onnx | 1 x 1751 | 1 x 64 | 0=H2, 1=teleop, 2=SMPL, 3=G1 |
| model_decoder.onnx | 1 x 994 | 1 x 29 | H2, LoRA and I/O adapters folded |

Observation contract: `sonic_h2_native_v1`.
Reference orientation: `sonic_full_root_relative_v0`.
Keep all 30 gravity-history dimensions; do not enable the legacy tail overwrite.

The saved training configuration and remote actuator implementation use the
existing `unitree_rl_mjlab_h2` fixed PD gains and action scales. Thirteen selected
deployment configuration fields match default step 25400, including the actuator
asset settings, observation histories, 50 Hz policy interval, and action clipping.
The simulator retains official Unitree ROS `H2_loop.xml`, its local 1 ms physics
timestep, and floor friction `[1.2, 0.005, 0.0001]`.

- All three SHA256 hashes match `DEPLOY_SHA256SUMS`.
- All three ONNX models pass full checker and required deployment metadata checks.
- Five CPU encoder/decoder trials per selector: all outputs finite.
- Full selector 0 versus H2-only encoder maximum absolute error: 0.0.
- Fresh TensorRT caches built for full encoder and decoder.
- Isolated headless MuJoCo smoke test: `lo`, DDS domain 73, ZMQ output port 5567.
- All eight H2 demos loaded; first forward-lunge reference played to completion.
- 599 control cycles, selector 0 throughout; encoder input, decoder input,
  action, joint position/velocity, and base quaternion logs contain no NaN/Inf.
- Deploy exited normally with code 0; the owned test simulator was then stopped.
- Elastic support was enabled: this does not establish free-motion stability.
- No real robot was connected. Existing domain-0 user processes were left alone.

Detailed records: `local_model_validation.json`, `local_sim_validation.json`.
Raw logs: `/tmp/sonic_step20000_deploy.log`, `/tmp/sonic_step20000_sim.log`,
`/tmp/sonic_step20000_smoke/`.

## Sim2Sim: All Eight H2 References

Exit the previous domain-0 deploy process before starting another controller on
that domain. Restart the simulator for a fresh initial state.

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
MODEL=/home/qingyul/workspace/SONIC-H2/sonic_h2_native_release/experiments/m10_20260906_step020000
./deploy.sh \
  --robot-profile h2_native \
  --encoder-file "$MODEL/model_encoder_full.onnx" \
  --decoder-file "$MODEL/model_decoder.onnx" \
  --obs-config policy/release/observation_config_h2_native.yaml \
  --motion-data reference/example_h2/deploy \
  --no-planner --input-type keyboard --output-type zmq \
  --hand-backend none sim
```

The demo manifest declares reference type `h2`, selecting branch 0 without any
G1-to-H2 projection. Confirm with Y, wait for `Init Done`, and press `]` in
terminal 2 to start control. Press `9` in the simulator window to release elastic
support. In terminal 2, `t` plays, `n`/`p` selects next/previous motion, and `o`
exits. Commands omitting the explicit experiment model paths now also use step 20000.
