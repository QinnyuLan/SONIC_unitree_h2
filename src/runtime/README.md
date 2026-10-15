# H2 Policy Runtime

The C++ runtime executes the H2 encoder and decoder through TensorRT, receives commands from independent Python services, and publishes robot targets through Unitree DDS. It initializes the configured standing pose, holds with PD control, and waits for the operator to enable policy control. Head pitch/yaw and hand targets use separate command channels.

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
bash scripts/run_sonic_h2_runtime.sh sim
```

After **[READY]**, press **]** or **s** in this terminal to enable policy. Press **o** to stop with damping; restart the runtime to resume. No Enter is needed. Early start keys are discarded. Keep support engaged during PD waiting. The standing reference is in `config/runtime/standing_pose.json`; use `--standing-pose PATH` to select another reference. See [standing pose configuration](#standing-pose-configuration) below.

The build produces `build/runtime/bin/sonic_h2_runtime`. Launch scripts supply the model directory and library paths. Hardware settings are in `config/h2_deploy.env`; teleoperation settings are in `config/h2_head_adjust.env`.

The equivalent CMake build is:

```bash
source scripts/setup_env.sh
cmake -S src/runtime -B build/runtime -DCMAKE_BUILD_TYPE=Release
cmake --build build/runtime --parallel 4
```

## Standing pose configuration

The standing reference is configured in [`config/runtime/standing_pose.json`](../../config/runtime/standing_pose.json). Both legs use hip pitch **−0.10 rad**, knee **0.25 rad** (about **14°**), and ankle pitch **−0.15 rad**. The preset reduces knee flexion while retaining a slight bend. Both arms rest at the sides with a slight elbow bend: shoulder pitch **0.25 rad**, shoulder roll **+0.10/−0.10 rad** (left/right), and elbow **1.00 rad**. This shoulder/elbow combination was checked with the running policy to reduce torso lean as well as keep the hands at the sides. H2's elbow coordinate is offset: an elbow value of zero points the forearm forward. The preset is used during initialization, PD waiting, and policy standing when no body input is active. The simulator's reset pose matches this preset. Head commands remain independent.

To use another reference, pass `--standing-pose /absolute/path/to/standing_pose.json`. Angles are in radians. In `arm_joint_angles_rad`, `shoulder_roll` specifies the left shoulder angle; the right shoulder uses its negative. Omitting this section retains the model's nominal arm reference for compatibility with existing leg-only configurations. This changes the reference supplied to the policy, preserving the decoder's trained joint offsets and action scales. Validate adjusted references in simulation before hardware use.

## Tests

Tests are optional and require GTest and zlib. From the repository root:

```bash
sudo apt install libgtest-dev zlib1g-dev
bash scripts/build_controller.sh -DBUILD_TESTING=ON
ctest --test-dir build/runtime --output-on-failure
```

CTest sets the working directory for model and fixture paths. CSV/NPZ readers, forward kinematics, and observation-YAML compatibility helpers are compiled only for regression tests.

See [architecture and messages](../../docs/ARCHITECTURE.md) for the Python input interface and [deployment](../../docs/DEPLOYMENT.md) for simulation, hardware and teleoperation commands.
