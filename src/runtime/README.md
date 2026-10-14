# H2 Policy Runtime

The C++ runtime executes the H2 encoder and decoder through TensorRT, receives commands from independent Python services, and publishes robot targets through Unitree DDS. It initializes the configured standing pose, holds with PD control, and waits for the operator to enable policy control. Head pitch/yaw and hand targets use separate command channels.

## Layout

| Path | Purpose |
|---|---|
| `CMakeLists.txt` | Native build entry |
| `include/` | Protocol, observations, policy, robot parameters, head and hand control |
| `src/unitree_audio_player.cpp` | Standalone PCM speaker tool for motion/audio replay, outside the policy loop |
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
bash scripts/run_sonic_h2_runtime.sh sim
```

After **[READY]**, press **]** or **s** in this terminal to enable policy. Press **o** to stop with damping; restart the runtime to resume. No Enter is needed. Early start keys are discarded. Keep support engaged during PD waiting. The standing reference is in `config/runtime/standing_pose.json`; use `--standing-pose PATH` to select another reference. Details are in the [deployment guide](../../docs/DEPLOYMENT.md#standing-pose-and-runtime-keys).

The build produces `build/runtime/bin/sonic_h2_runtime` and the independent `build/runtime/bin/unitree_audio_player`. See [speech/gesture replay](../../docs/TEXT2MOTION.md#unitree-built-in-speaker) for the audio utility. Launch scripts supply the model directory and library paths. Hardware settings are in `config/h2_deploy.env`; teleoperation settings are in `config/h2_head_adjust.env`.

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
