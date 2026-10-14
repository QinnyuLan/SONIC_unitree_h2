# GPU3 stage2 step 5000: hardware comparison

Retrieved 2026-09-18 from `qingyul@gpu3.precognition.team`:

```text
/home/qingyul/workspace/GR00T-WBC-H2TRAINBASE/logs_rl/sonic-h2-native-8xA6000-ori-2/20260906_204409/stage2/exported/ablation_stage2_5k_30k_20260909/stage2_step_005000
```

The user requested this earlier checkpoint after observing right ankle roll
heating with steps 70000 and 9450. This is an experimental comparison, not a
verified overheating fix.

Validation on this workstation:

- Three ONNX SHA256 hashes match the source deployment manifest.
- ONNX checker, metadata and input/output dimensions pass.
- CPU encoder-to-decoder inference is finite for all four selectors, five
  trials each. Compact H2 versus full selector 0 maximum error is 0.
- Model configuration has no differences from GPU3 stage2 step 9450.
- Fresh TensorRT generation and actual controller loading pass in a bounded
  loopback simulation using DDS domain 73, `reference/example`, nominal gains
  and elastic support. Both processes exit with code 0. This checks deployment
  and routing, not free-standing stability or hardware thermal behavior.

Evidence: `local_model_validation.json`, `SHA256SUMS`, and
`smoke_supported/process_result.json`, `smoke_supported/deploy.log`.
Reproduction scripts: `validate_models.py`, `run_smoke.py`.

`gear_sonic_deploy/config/h2_deploy.env` now selects this version for sim2real.
The motion collection is `reference/example`. Retained real Kp scales:
left/right ankle pitch 1.2/1.4, ankle roll 0.7/0.7, knees 1.2/1.2.
Kd values are unchanged. Sim2sim's independent launcher still selects step 9450.

Start the real controller with ankle recording (after stopping the old one):

```bash
bash /home/qingyul/workspace/SONIC-H2/gear_sonic_deploy/scripts/run_h2_sim2real_record_ankles.sh
```

This command uses the current real configuration and generates its own log
directory and ankle analysis after exit. No hardware test was started by the
deployment update.
