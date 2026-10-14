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
.venv_sim/bin/python -m pytest tests/sim tests/interface/commands -q
uv pip install --python .venv_teleop/bin/python -r requirements/test.txt
.venv_teleop/bin/python -m pytest tests/interface/teleop -q
```

Build and run the C++ tests:

```bash
sudo apt install libgtest-dev zlib1g-dev
export TensorRT_ROOT="$HOME/TensorRT"
bash scripts/build_controller.sh -DBUILD_TESTING=ON
ctest --test-dir build/runtime --output-on-failure
```

For ONNX graph, interface, checksum, and CPU inference checks:

```bash
uv venv .venv_model_validation --python 3.10
uv pip install --python .venv_model_validation/bin/python -r requirements/model_validation.txt
.venv_model_validation/bin/python tools/validate_released_models.py
```

## Deployment validation

On 2026-10-15, source code was separated into `src/runtime/`, `src/interface/`, and `src/sim/`. All four Python environments were installed from the root project, and the previous package registration was removed. The C++ runtime was compiled from an empty build directory. Module entry points, independent imports, and G1/H2/human asset loading passed from outside the checkout. A clean checkout passed a fresh minimal Python installation, motion replay, offline simulation and native CMake configuration with tests disabled.

- 45 shared C++ regression tests and 12 runtime/protocol tests passed. These cover the unchanged model contract, native joint mappings, history order, sparse sampling, quaternion handling, ownership, expiry and separate head input.
- 27 simulator tests (53 additional subtests), 32 command/planner/replay/adapter tests and 34 teleoperation tests passed.
- The minimal `.venv_inputs` and dedicated `.venv_planner` installations passed. CSV replay validation and producer command-line parsing passed. Simulator/teleoperation preflight and all released model hashes passed.
- `tools/check_runtime.py --planner` ran the actual FP32 full encoder, decoder and G1 planner from an empty working directory using synthetic robot state and private IPC sockets. Startup standing, G1/VR/SMPL/token input, timeout-to-standing, producer takeover, head/hand independence, synthetic PICO packet conversion and planner inference all passed. DDS was not initialized by this check.
- The latest DDS closed-loop attempt did not complete: runtime encoder initialization and a separate system GPU query both became unresponsive. The simulator initialized its relocated assets and DDS bridge, then stopped on its realtime-lag guard. The previous 2026-10-14 check completed ten seconds on loopback DDS domain 211 with elastic support and 334 finite standing-state frames; that historical result is retained separately and does not validate the current closed loop. No free-standing stability or motion-tracking benchmark was performed.
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

## Third-party dependency validation

On 2026-10-15, the Unitree Python SDK and XRoboToolkit binding were replaced with pinned, unmodified upstream submodules under `third_party/`. The simulator's odometry extension moved to `src/sim/odometry.py`; its DDS type identifier, type information and serialized sample bytes match the previous implementation. The native XR SDK artifacts retain their previous SHA256 hashes.

Simulation and teleoperation installations, dependency checks, preflights and all 93 Python tests passed. A fresh clone fetched both pinned submodules from their official GitHub URLs, installed simulation and teleoperation into empty environments, and passed both preflights. The parent repository and both submodules remained clean after installation. A separate headless simulator published 120 finite LowState and 120 finite odometry messages through loopback DDS domain 217. This check did not start the policy runtime or exercise GPU inference. ARM64 artifact hashes were checked, but ARM64 execution and physical devices were not tested.

## Saving validation reports

Generated reports belong in `outputs/validation/`, which is ignored by Git. Record the tested revision alongside the results:

```bash
mkdir -p outputs/validation
git rev-parse HEAD > outputs/validation/commit.txt
git submodule status --recursive > outputs/validation/submodules.txt
```

After installing the test dependencies above, save Python test results:

```bash
.venv_sim/bin/python -m pytest tests/sim tests/interface/commands -q \
  --junitxml=outputs/validation/sim-and-commands.xml
.venv_teleop/bin/python -m pytest tests/interface/teleop -q \
  --junitxml=outputs/validation/teleoperation.xml
```

After building the native tests, save their JSON reports:

```bash
GTEST_OUTPUT="json:$PWD/outputs/validation/" \
  ctest --test-dir build/runtime --output-on-failure
```

After installing the model-validation environment, save the ONNX check report:

```bash
.venv_model_validation/bin/python tools/validate_released_models.py \
  --report outputs/validation/model_validation.json
```

Package these reports for attachment to the corresponding GitHub release:

```bash
tar -czf outputs/validation-reports.tar.gz -C outputs/validation .
```

Reports describe the environment and revision in which they were produced; rerun the relevant checks for a new release. Keep model interface manifests and checksums with the models under `models/released/`. Source provenance is documented in [Acknowledgments](ACKNOWLEDGMENTS.md); Git records the source files and pinned submodules.
