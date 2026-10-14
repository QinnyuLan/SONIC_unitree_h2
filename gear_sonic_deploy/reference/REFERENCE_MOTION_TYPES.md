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

The non-v1.1 full release uses `model_encoder_full.onnx`. Planner output is
native G1 reference data and uses selector 3 by default; no G1-to-H2 reference
projection is performed in that mode. The H2-only `model_encoder.onnx` accepts only
`reference_type: h2` and cannot run planner/G1/SMPL references.

For an explicit non-v1.1 comparison, `--planner-g1-to-h2` projects every sampled
planner q/qdot frame through the legacy G1-to-H2 linear map and uses selector 0.
It requires the full encoder and an enabled planner. The stored planner trajectory
remains G1, and preloaded demo types are unchanged. VR 3-point control still uses
selector 1; leaving it restores selector 0 for this comparison. Omit the flag to
restore the default native G1 selector 3 workflow.
