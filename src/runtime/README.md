# H2 Policy Runtime

The C++ runtime executes the H2 encoder and decoder through TensorRT, receives commands from independent Python services, and publishes robot targets through Unitree DDS. It starts with the default standing reference. Head pitch/yaw and hand targets use separate command channels.

## Layout

| Path | Purpose |
|---|---|
| `CMakeLists.txt` | Native build entry |
| `include/` | Protocol, observations, policy, robot parameters, head and hand control |
| `src/main.cpp` | Startup, control loop, inference scheduling and DDS output |
| `src/inference/` | TensorRT engine loading, cache and CUDA execution |
| `cmake/` | TensorRT SDK discovery |
| `third_party/unitree_sdk2/` | Unitree C++ SDK and DDS dependencies |
| `tests/` | Offline tests, numerical helpers and fixtures |
| `../../build/runtime/` | Generated CMake files and binaries, ignored by Git |

## Build and run

Install the [native dependencies](../../docs/NATIVE_SETUP.md), then run from the repository root:

```bash
bash scripts/build_controller.sh
bash scripts/run_runtime.sh sim
```

The build produces `build/runtime/bin/sonic_h2_runtime`. Launch scripts supply the model directory and library paths. Hardware settings are in `config/h2_deploy.env`; teleoperation settings are in `config/h2_head_adjust.env`.

The equivalent CMake build is:

```bash
source scripts/setup_env.sh
cmake -S src/runtime -B build/runtime -DCMAKE_BUILD_TYPE=Release
cmake --build build/runtime --parallel 4
```

## Tests

Tests are optional and require GTest and zlib. From the repository root:

```bash
sudo apt install libgtest-dev zlib1g-dev
bash scripts/build_controller.sh -DBUILD_TESTING=ON
ctest --test-dir build/runtime --output-on-failure
```

CTest sets the working directory for model and fixture paths. CSV/NPZ readers, forward kinematics, and observation-YAML compatibility helpers are compiled only for regression tests.

See [architecture and messages](../../docs/ARCHITECTURE.md) for the Python input interface and [deployment](../../docs/DEPLOYMENT.md) for simulation, hardware and teleoperation commands.
