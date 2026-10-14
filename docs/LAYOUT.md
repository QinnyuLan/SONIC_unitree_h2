# Repository Layout

This repository contains the runtime for H2 simulation, hardware deployment, and VR teleoperation.

| Path | Contents |
|---|---|
| `runtime/` | C++ policy runtime and its CMake build entry |
| `runtime/include/` | Command handling, observations, policy execution, H2 parameters, head and hand control |
| `runtime/src/` | Runtime main loop and TensorRT C++ inference |
| `runtime/third_party/` | Unitree C++ SDK and DDS libraries |
| `runtime/tests/` | Runtime and numerical regression tests, test-only readers and fixtures |
| `gear_sonic/planner/` | Python planner context, resampling and blending |
| `gear_sonic/inference/` | TensorRT Python engine, cache and CUDA buffers |
| `gear_sonic/commands/` | Python command protocol and client |
| `gear_sonic/sources/` | Motion replay, planner service/requests, teleoperation and head/hand senders |
| `models/released/` | H2 ONNX models, interface manifest, and checksums |
| `models/planner/` | Local G1 planner weight and generated TensorRT cache |
| `motions/` | Optional G1 replay examples and format documentation |
| `config/` | H2 hardware, hand backend and teleoperation settings |
| `gear_sonic/` | MuJoCo/DDS simulator, PICO input, pose mathematics, and runtime robot assets |
| `external_dependencies/` | Unitree Python SDK and XRoboToolkit binding sources/libraries |
| `scripts/` | Independent runtime/planner/simulator launchers and native build |
| `tools/` | Offline release and ONNX checks |
| `requirements/` | Pinned Python deployment environments and optional download/validation dependencies |
| `docs/` | Installation and operation guides |
| `release/` | Release inventory and validation records |
| `legal/` | Third-party license notices |

The retained robot assets are the H2 closed-loop MJCF/URDF and their mesh dependencies, plus G1 kinematics needed by the VR mapper. `motions/example/` supplies optional replay clips; `runtime/tests/fixtures/` supplies numerical regression data. Shared Python pose mathematics lives under `gear_sonic/utils/math/`.

Build the native runtime with `bash scripts/build_controller.sh`. Its only CMake entry is `runtime/CMakeLists.txt`, and the executable is `runtime/build/bin/sonic_h2_runtime`. All C++ policy execution code and native dependencies live under `runtime/`; Python input services communicate through the command protocol. See the [runtime build guide](../runtime/README.md).

Training frameworks, MotionBricks, Isaac environments, VLA/data collection, historical checkpoints, experimental motion sets, and inherited documentation-site tooling are outside this repository's scope.

The planner is downloaded to `models/planner/planner_sonic.onnx` and ignored by Git. Python environments, native builds, TensorRT engines, logs, and recordings are generated locally. See [Planner](PLANNER.md) and [Reproducibility](../release/REPRODUCIBILITY.md).
