# Reproducibility

The release contains the H2 runtime for MuJoCo, hardware deployment, and PICO VR teleoperation. The independent policy runtime loads `models/released`; producers connect through the documented command channels. The model interface is documented in the [model README](../models/released/README.md).

## Validated environment

Ubuntu x86_64, Python 3.10.20, CUDA 12.6, TensorRT 10.13.3.9, and CUDA Python bindings 12.9.4. Python runtime dependencies are pinned in `requirements/inputs.lock`, `requirements/planner.lock`, `requirements/sim.lock` and `requirements/teleop.lock`. The simulator and teleoperation environments were installed independently from empty environments.

## Offline checks

From the repository root:

```bash
python3 tools/verify_release.py
./install.sh sim2sim
.venv_sim/bin/python tools/verify_release.py --imports --sim
./install.sh teleop
.venv_teleop/bin/python tools/verify_release.py --teleop
```

The simulator check loads the H2 closed-loop assets and advances 100 physics steps. The teleoperation check loads the XRoboToolkit binding, VR mapper, and G1/H2 kinematics without initializing devices. Add `--require-planner` to require the separately downloaded planner.

Run the Python regression suites in their respective environments:

```bash
uv pip install --python .venv_sim/bin/python -r requirements/test.txt
.venv_sim/bin/python -m pytest gear_sonic/tests/sim gear_sonic/tests/commands -q
uv pip install --python .venv_teleop/bin/python -r requirements/test.txt
.venv_teleop/bin/python -m pytest gear_sonic/tests/teleop -q
```

Build and run the C++ tests:

```bash
sudo apt install libgtest-dev zlib1g-dev
export TensorRT_ROOT="$HOME/TensorRT"
bash scripts/build_controller.sh -DBUILD_TESTING=ON
ctest --test-dir runtime/build --output-on-failure
```

For ONNX graph, interface, checksum, and CPU inference checks:

```bash
uv venv .venv_model_validation --python 3.10
uv pip install --python .venv_model_validation/bin/python -r requirements/model_validation.txt
.venv_model_validation/bin/python tools/validate_released_models.py
```

## Deployment validation

These checks were repeated after consolidating the C++ build, inference wrapper, headers and Unitree SDK under `runtime/`. The runtime was compiled from an empty build directory; replay, planner, configuration and launcher paths were checked in the new layout. A clean checkout of the release files passed preflight, launcher previews, and CMake configuration with tests disabled; it contained neither local planner weights nor build products.

- 45 shared C++ regression tests and 12 runtime/protocol tests passed. These cover the unchanged model contract, native joint mappings, history order, sparse sampling, quaternion handling, ownership, expiry and separate head input.
- 27 simulator tests (53 additional subtests), 32 command/planner/replay/adapter tests and 34 teleoperation tests passed.
- The minimal `.venv_inputs` and dedicated `.venv_planner` installations passed. CSV replay validation and producer command-line parsing passed. Simulator/teleoperation preflight and all released model hashes passed.
- `tools/check_runtime.py --planner` ran the actual FP32 full encoder, decoder and G1 planner from an empty working directory using synthetic robot state and private IPC sockets. Startup standing, G1/VR/SMPL/token input, timeout-to-standing, producer takeover, head/hand independence, synthetic PICO packet conversion and planner inference all passed. DDS was not initialized by this check.
- A separate 10-second MuJoCo/runtime test used DDS domain 211 on loopback, with elastic support enabled. The runtime completed its initialization ramp and published 334 finite running-state frames in standing mode. This is a connection/control-loop smoke test, not a free-standing stability or motion-tracking benchmark.
- Sim, real and PICO launcher previews passed. Hardware control was not started; no real robot or PICO device was connected during validation.
- Published ONNX bytes are unchanged. The prior CPU and GPU model-comparison records remain applicable to those model files; they are not evidence of live hardware performance.

The planner service, context preparation and trajectory processing use Python; inference calls TensorRT directly. The native runtime no longer links ONNX Runtime. Its model contract is checked against the SHA256-bound release manifest and the actual TensorRT tensor interfaces. The offline ONNX validation tool also verifies that the manifest matches the embedded interfaces and metadata.

The Python planner matched the previous C++ implementation on initialization and one walking replan using the same engine: maximum joint-position error was below `6e-16`, joint-velocity error below `3e-14`, and root-quaternion error below `1e-10`. A fresh engine built through the Python API in about 38 seconds. Its maximum initialization differences from the existing cache were `2.49e-4` rad in joint position and `3.52e-3` rad/s in velocity; rebuilding TensorRT engines is not bitwise deterministic. Fifty repeated CUDA graph executions passed, with a median engine-call time of 2.86 ms on the validated RTX 4070 Ti SUPER. These measurements are not end-to-end control latency or hardware stability results.

Reproduce the model/transport integration test after building:

```bash
./install.sh planner
source scripts/setup_env.sh
.venv_planner/bin/python tools/check_runtime.py --planner
```

Machine-readable records: [deployment checks](deployment_validation.json), [shared C++ tests](cpp_tests.json), [runtime tests](runtime_tests.json), [MuJoCo connection test](runtime_sim_validation.json), [CPU model checks](model_validation.json), [GPU model checks](gpu_model_validation.json), and [planner checksum](planner_validation.json), and [Python/TensorRT planner comparison](python_planner_validation.json).

## Release inventory

[source_manifest.json](source_manifest.json) inventories the current tracked release files, excluding the manifest itself, local planner, environments, build products, and logs. Original source provenance and the published model hashes are retained. Models are unchanged by the deployment cleanup.
