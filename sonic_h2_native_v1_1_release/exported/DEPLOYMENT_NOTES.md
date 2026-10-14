# H2 SONIC v1.1 deployment model

- Checkpoint step: `16150`
- Export: `deployment_step_016150_cpu_fully_folded`
- Exported at: `2026-09-02T13:34:12`
- Encoder: `model_encoder.onnx`, `[1, 1751] -> [1, 64]`
- Decoder: `model_decoder.onnx`, `[1, 994] -> [1, 29]`
- Actor observation contract: `sonic_h2_native_v1`
- Reference orientation contract: `sonic_robot_heading_v1`
- Universal selector routes: `0=h2`, `1=teleop`, `2=smpl`
- Legacy actor-observation tail patch must remain disabled.

Source host: `sky@120.79.230.50:6022`

The `model_encoder_h2_native.onnx` and `model_decoder_h2_native.onnx` files are
the matching step 16150 H2-only pair (`[1,640] -> [1,64]`, `[1,994] -> [1,29]`).
The current `h2_native_v1_1` deploy profile uses the two default universal models
listed above; the 640-D sidecar remains an explicit interface and cannot replace
the required 1751-D selector input without a separate v1.1 H2-only runtime path.
