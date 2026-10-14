# Deployment Assets

- `robots/h2/`: Unitree H2 MJCF, URDF and meshes shared by simulation and teleoperation kinematics.
- `robots/g1/`: G1 URDF and meshes used by the VR mapper.
- `human/`: Rest-joint data for SMPL pose conversion.

These are static resources. The simulator implementation lives in `src/sim/`; pose and robot kinematics code lives in `src/interface/`.

Robot-description licenses are collected in the repository-wide [license index](../licenses/README.md).
