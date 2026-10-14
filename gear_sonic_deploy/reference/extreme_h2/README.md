# H2 extreme demo references

This bundle contains six 50 Hz H2-native references copied from the IK-fixed
20260827 dataset and exported from the exact paired robot/SMPL MotionLib cache
used by training.

Use this directory with the deploy controller:

```text
--motion-data reference/extreme_h2/deploy
```

Motion order:

1. `idle_one_foot_left_003__A043`
2. `idle_one_foot_right_002__A040`
3. `kneeling_start_003__A167_M`
4. `kneeling_stop_003__A167_M`
5. `jump_and_land_light_002__A001`
6. `one_leg_jumping_R_001__A361`

The last motion is intentionally ordered last because it is the highest-risk
reference in this set. Treat it as a simulation stress test, not a first-pass
real-robot demo.

Source layout:

- `h2/csv`: original H2 physical 31-DoF CSV data from m10
- `motionlib/robot`: paired 50 Hz robot MotionLib files used by training
- `motionlib/smpl`: paired 50 Hz SMPL MotionLib files used by training
- `deploy`: generated C++ deployment CSV bundles

See `SOURCE_MANIFEST.json` for hashes and measured motion statistics.
