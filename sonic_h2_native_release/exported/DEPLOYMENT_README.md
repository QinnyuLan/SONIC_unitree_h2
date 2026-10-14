# H2-native Release: Stage 2 Step 70000

Promoted 2026-09-18 from gpu3 package `h2_stage2_070000_20260918.tar.gz`.

- Full encoder: `model_encoder_full.onnx`, 1751 → 64, selectors H2=0 / teleop=1 / SMPL=2 / G1=3.
- Compact H2 encoder: `model_encoder.onnx`, 640 → 64.
- Shared H2 decoder: `model_decoder.onnx`, 994 → 29; input/output adapters folded.
- Profile `h2_native`; observation contract `sonic_h2_native_v1`;
  orientation contract `sonic_full_root_relative_v0` (non-v1.1).
- Matching observation YAML: `gear_sonic_deploy/policy/release/observation_config_h2_native.yaml`.

[Current commands and editable configuration](../../docs/report/h2_stage2_70000_deployment.md).
[Source bundle and local validation](../versions/h2_stage2_070000_20260918/).

`deploy.sh --robot-profile h2_native` defaults to these models. The sim2real launcher selects an explicit version via
`gear_sonic_deploy/config/h2_deploy.env`; update that file to choose another version,
motion collection, initial clip, per-joint Kp scales, network or log directory.
Sim2sim settings are independent in `scripts/run_h2_sim2sim.sh` and use nominal gains.
Current real scales 1.4/1.6/1.3/1.3 preserve the user's prior setup; default raw deploy
without scale flags still uses nominal gains.

Archive and file hashes, ONNX full checks, all four CPU selector branches,
compiled runtime joint/q0/action-scale/PD contract and 14 H2Native tests passed.
Fresh TensorRT models completed supported loopback H2 lunge and G1 prayer runs;
actual LowCmd gains and finite inference logs were checked. This does not
establish unsupported stability. No hardware control executed during promotion.

Previous step 20000 top-level files including models, metadata and caches are
preserved in `backup_before_stage2_step070000_20260918_181453/`.
`local_deployment.json` records source, hashes, backup path and validation scope.
Verify current model files with `sha256sum -c DEPLOY_SHA256SUMS`.
