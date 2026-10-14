# Repository Layout

This repository contains the runtime for H2 simulation, hardware deployment, and VR teleoperation.

| Path | Contents |
|---|---|
| `models/released/` | H2 ONNX models, interface manifest, and checksums |
| `gear_sonic/` | MuJoCo/DDS simulator, PICO input, pose mathematics, and runtime robot assets |
| `gear_sonic_deploy/` | C++ controller, TensorRT inference, H2 settings, launchers, and reference motions |
| `external_dependencies/` | Unitree Python SDK and XRoboToolkit binding sources/libraries |
| `scripts/` | Simulator launcher and native controller build |
| `tools/` | Offline release and ONNX checks |
| `requirements/` | Optional download/validation dependencies |
| `docs/` | Installation and operation guides |
| `release/` | Release inventory and validation records |
| `legal/` | Third-party license notices |

The retained robot assets are the H2 closed-loop MJCF/URDF and their mesh dependencies, plus G1 kinematics needed by the VR mapper. `reference/example` supplies the launchers' playback motions; `reference/bones_072925_test` is a small forward-kinematics regression fixture. Shared pose mathematics lives under `gear_sonic/utils/math`.

Training frameworks, MotionBricks, Isaac environments, VLA/data collection, historical checkpoints, experimental motion sets, and inherited documentation-site tooling are outside this repository's scope.

The planner is downloaded to `gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx` and ignored by Git. Python environments, native builds, TensorRT engines, logs, and recordings are generated locally. See [Planner](PLANNER.md) and [Reproducibility](../release/REPRODUCIBILITY.md).
