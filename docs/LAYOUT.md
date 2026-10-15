# Repository Layout

Source code is split into three components:

```text
src/
├── runtime/             # C++ SONIC policy execution and Unitree DDS output
├── interface/          # Online Python command producers and planner
│   ├── motion_replay/  # Motion loading, sampling and playback
│   ├── planner/
│   ├── teleop/
│   └── peripherals/
└── sim/                # Python MuJoCo simulator and DDS bridge

tools/
├── model_download/     # Download model assets
├── setup/              # Prepare build dependencies
├── validation/         # Check files, models and integration
└── robot_services/     # Robot-side peripheral services
```

| Path | Contents |
|---|---|
| `src/runtime/` | Encoder/decoder inference, observations, command arbitration, head/hand output, native SDK and CMake build |
| `src/interface/commands/` | Message schema, validation and Python command client |
| `src/interface/motion_replay/` | G1/H2/SMPL motion loading, reference sampling and playback |
| `src/interface/planner/` | Planner service, request client, TensorRT engine and trajectory processing |
| `src/interface/teleop/` | PICO input, SMPL/VR adapter, calibration, tracking transport and visualization |
| `src/interface/peripherals/` | Head/hand command client and optional Inspire serial tools |
| `src/interface/kinematics/` | G1/H2 kinematics used by teleoperation |
| `src/interface/math/` | Pose and rotation mathematics |
| `src/sim/` | Simulation loop, physics, motor response and Unitree DDS bridge |
| `src/sim/rendering/` | Viewer presentation and camera publishing |
| `src/sim/diagnostics/` | Simulation diagnostics and logging |
| `assets/` | Shared G1/H2 URDF, MJCF, meshes and human joint data |
| `models/sonic_h2_released/` | H2 ONNX models, interface manifest and checksums |
| `models/motion_planner_g1/` | Optional local G1 planner weight and TensorRT cache |
| `motions/` | G1, native H2 and SMPL reference motions |
| `config/` | Hardware, teleoperation and simulator settings |
| `scripts/` | Environment setup, native build and process launchers |
| `tests/` | Python interface and simulator regression tests |
| `src/runtime/tests/` | C++ regression tests and fixtures |
| `third_party/` | Pinned Unitree/XRoboToolkit source submodules and native XR SDK artifacts |
| `build/runtime/` | Generated native binaries and CMake files |
| `requirements/` | Dependency locks for each Python environment |
| `tools/model_download/` | Pinned model downloads and checksums |
| `tools/setup/` | Native dependency build preparation |
| `tools/validation/` | Release preflight, model and integration checks |
| `tools/robot_services/inspire_hands/` | Independent C++ Inspire hand service and robot systemd configuration |
| `docs/` | Installation, deployment, architecture and reproducibility guides |
| `licenses/` | Upstream software notices and robot-asset licenses |
| `outputs/validation/` | Local validation reports, ignored by Git |

`runtime` accepts commands from `interface` through ZeroMQ. `sim` exchanges robot state and motor targets with `runtime` through DDS. The simulator does not import the Python input services, and the input services do not import the simulator. Both read shared robot resources from `assets/`. See [tool categories and commands](../tools/README.md).

The root `pyproject.toml` installs `interface` and `sim` from `src/`. Use `./install.sh` with the environment profile for the process you need. The installer uses an editable installation, so resources resolve against this checkout even when a process starts from another working directory. Keep `assets/`, `config/` and `models/` with the checkout.

Build the C++ runtime with `bash scripts/build_controller.sh`; the executable is `build/runtime/bin/sonic_h2_runtime`. See the [runtime guide](../src/runtime/README.md), [interface guide](../src/interface/README.md), and [simulator guide](../src/sim/README.md).

The planner weight at `models/motion_planner_g1/planner_sonic.onnx`, environments, native build products, TensorRT engines and recordings are local files ignored by Git. See [planner setup](PLANNER.md) and [validation](REPRODUCIBILITY.md).
