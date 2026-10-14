# Reference Motions

The policy runtime creates its standing reference in memory and does not load this directory.

`example/` contains optional G1 CSV clips for the independent motion replay process. Pass one clip directory, not the collection root:

```bash
.venv_inputs/bin/python -m interface.replay \
  motions/example/walking_quip_360_R_002__A428
```

Replay reads `joint_pos.csv`, optional `joint_vel.csv`, and the first quaternion in `body_quat.csv`. CSV columns use native G1 semantic order; the default rate is 50 Hz. It resamples the clip to ten future frames at 0.1-second intervals. Existing metadata and provenance manifests are retained with the clips and are not startup requirements.

Offline text-to-motion can use an NPZ with `joint_pos`, `root_quat`, `fps` and optional `joint_vel`. See [replay instructions](../docs/DEPLOYMENT.md#motion-replay-and-offline-text-to-motion) and [the wire protocol](../docs/ARCHITECTURE.md).

The forward-kinematics fixture is stored separately under `src/runtime/tests/fixtures/motions/`. Live SMPL and VR references are sent through the teleoperation adapter; they are not preloaded motion-library modes.
