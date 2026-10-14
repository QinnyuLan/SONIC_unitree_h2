# Reference Motion Types

Every H2 reference-motion directory should contain
`h2_lora_reference_manifest.json`. Each entry under `motions` must declare
`reference_type`:

```json
{
  "motion_name": "walk_native_g1",
  "reference_type": "g1"
}
```

Supported types and universal encoder selectors:

| `reference_type` | Selector | Required data |
| --- | ---: | --- |
| `h2` | 0 | `h2_joint_pos.csv`, `h2_joint_vel.csv` (31D) |
| `teleop` | 1 | teleop/VR fields; normally streamed rather than preloaded |
| `smpl` | 2 | SMPL joints and root quaternion |
| `g1` | 3 | native G1 `joint_pos.csv`, `joint_vel.csv` (29D) |

The default `model_encoder_full.onnx` accepts all four selectors. Planner output is native G1 reference data and uses selector 3. VR three-point control uses selector 1. The compact `model_encoder.onnx` accepts only H2 references and requires `policy/release/observation_config_h2_compact.yaml`.
