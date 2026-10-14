# Released H2 Models

These directly fine-tuned weights are shared by Sim2Sim, Sim2Real, and VR teleoperation. No adapter merge is required. Launchers use `MODEL=../models/released`, relative to `gear_sonic_deploy`.

All tensors are float32 with batch size 1.

| Model | Input | Output | Use |
|---|---|---|---|
| `model_encoder_full.onnx` | `obs_dict [1,1751]` | `encoded_tokens [1,64]` | Default: H2 / VR / SMPL / G1 |
| `model_encoder.onnx` | `obs_dict [1,640]` | `encoded_tokens [1,64]` | H2-only references with `observation_config_h2_compact.yaml` |
| `model_decoder.onnx` | `obs_dict [1,994]` | `action [1,29]` | Tokens and observation history to policy actions |

The full encoder selectors are 0 = H2, 1 = VR three-point input, 2 = SMPL, and 3 = G1. The controller maps the 29 policy actions to the H2 body; the head is handled separately. The observation and reference orientation contracts are checked by the runtime before execution.

The ONNX files retain their original bytes. Integrity hashes are recorded in [SHA256SUMS](SHA256SUMS). Run `python3 tools/verify_release.py` from the repository root to verify them. See [Reproducibility](../../release/REPRODUCIBILITY.md) for ONNX inference checks. TensorRT engines are generated on the target machine.
