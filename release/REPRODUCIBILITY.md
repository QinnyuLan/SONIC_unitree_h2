# Reproducibility

The release contains the H2 runtime for MuJoCo, hardware deployment, and PICO VR teleoperation. All three launchers load `models/released`. The model interface is documented in the [model README](../models/released/README.md).

## Validated environment

Ubuntu x86_64, Python 3.10.20, CUDA 12.6, TensorRT 10.13.3.9, and ONNX Runtime C++ 1.16.3. Python runtime dependencies are pinned in `requirements/sim.lock` and `requirements/teleop.lock`. The simulator and teleoperation environments were installed independently from empty environments.

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
.venv_sim/bin/python -m pytest gear_sonic/tests/sim -q
uv pip install --python .venv_teleop/bin/python -r requirements/test.txt
.venv_teleop/bin/python -m pytest gear_sonic/tests/teleop -q
```

Build and run the C++ tests:

```bash
sudo apt install libgtest-dev
export TensorRT_ROOT="$HOME/TensorRT"
bash scripts/build_controller.sh -DBUILD_TESTING=ON
ctest --test-dir gear_sonic_deploy/build --output-on-failure
```

For ONNX graph, interface, checksum, and CPU inference checks:

```bash
uv venv .venv_model_validation --python 3.10
uv pip install --python .venv_model_validation/bin/python -r requirements/model_validation.txt
.venv_model_validation/bin/python tools/validate_released_models.py
```

## Results

- 46 C++ regression tests passed, including released model contracts, reference handling, head control, and Inspire hands.
- 27 simulation tests passed, with 53 additional subtests; 34 teleoperation tests passed.
- Headless H2 loading and physics stepping, XR binding loading, and G1/H2 kinematic loading passed in the separate environments and in an export containing only the files prepared for Git. Every exported file matched the inventory hash.
- All three controller launchers passed configuration validation with execution declined at the launch prompt. PICO argument parsing and all launcher command previews passed.
- All model hashes and ONNX graph/interface checks passed. Each of the four selectors completed five CPU encoder-to-decoder trials with finite outputs. Full/compact H2 encoder outputs matched exactly.
- The previously recorded TensorRT FP32 check for the unchanged ONNX models produced finite outputs for all selectors, with token error 0 and maximum action error below `5e-6` against CPU inference.

Machine-readable records: [deployment checks](deployment_validation.json), [C++ tests](cpp_tests.json), [CPU model checks](model_validation.json), [GPU model checks](gpu_model_validation.json), and [planner checksum](planner_validation.json).

These are offline regression and loading checks. They do not establish live VR tracking, real-time closed-loop tracking quality, or hardware stability on a new robot.

## Release inventory

[source_manifest.json](source_manifest.json) inventories the current tracked release files, excluding the manifest itself, local planner, environments, build products, and logs. Original source provenance and the published model hashes are retained. Models are unchanged by the deployment cleanup.
