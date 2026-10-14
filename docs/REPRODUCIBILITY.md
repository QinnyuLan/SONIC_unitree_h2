# Reproducibility

The release contains the H2 runtime for MuJoCo, hardware deployment, and PICO VR teleoperation. The independent policy runtime loads `models/sonic_h2_released`; producers connect through the documented command channels. The model interface is documented in the [model README](../models/sonic_h2_released/README.md).

## Validated environment

Ubuntu x86_64, Python 3.10.20, CUDA 12.6, TensorRT 10.13.3.9, and CUDA Python bindings 12.9.4. Python runtime dependencies are pinned in `requirements/inputs.lock`, `requirements/planner.lock`, `requirements/sim.lock`, `requirements/teleop.lock` and the optional `requirements/text2motion.lock`. The simulator and teleoperation environments were installed independently from empty environments.

## Offline checks

From the repository root:

```bash
python3 tools/validation/verify_release.py
./install.sh sim2sim
.venv_sim/bin/python tools/validation/verify_release.py --imports --sim
./install.sh teleop
.venv_teleop/bin/python tools/validation/verify_release.py --teleop
```

The simulator check loads the H2 closed-loop assets and advances 100 physics steps. The teleoperation check loads the XRoboToolkit binding, VR mapper, and G1/H2 kinematics without initializing devices. Add `--require-planner` to require the separately downloaded planner.

Run the Python regression suites in their respective environments:

```bash
uv pip install --python .venv_sim/bin/python -r requirements/test.txt
.venv_sim/bin/python -m pytest tests/sim tests/interface/commands tests/interface/motion_replay -q
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
.venv_model_validation/bin/python tools/validation/validate_released_models.py
```

## Deployment validation

On 2026-10-15, source code was separated into `src/runtime/`, `src/interface/`, and `src/sim/`. All four Python environments were installed from the root project, and the previous package registration was removed. The C++ runtime was compiled from an empty build directory. Module entry points, independent imports, and G1/H2/human asset loading passed from outside the checkout. A clean checkout passed a fresh minimal Python installation, motion replay, offline simulation and native CMake configuration with tests disabled.

- 45 shared C++ regression tests and 12 runtime/protocol tests passed. These cover the unchanged model contract, native joint mappings, history order, sparse sampling, quaternion handling, ownership, expiry and separate head input.
- 27 simulator tests (53 additional subtests), 32 command/planner/replay/adapter tests and 34 teleoperation tests passed.
- The minimal `.venv_inputs` and dedicated `.venv_planner` installations passed. CSV replay validation and producer command-line parsing passed. Simulator/teleoperation preflight and all released model hashes passed.
- `tools/validation/check_runtime.py --planner` ran the actual FP32 full encoder, decoder and G1 planner from an empty working directory using synthetic robot state and private IPC sockets. Startup standing, G1/VR/SMPL/token input, timeout-to-standing, producer takeover, head/hand independence, synthetic PICO packet conversion and planner inference all passed. DDS was not initialized by this check.
- The latest DDS closed-loop attempt did not complete: runtime encoder initialization and a separate system GPU query both became unresponsive. The simulator initialized its relocated assets and DDS bridge, then stopped on its realtime-lag guard. The previous 2026-10-14 check completed ten seconds on loopback DDS domain 211 with elastic support and 334 finite standing-state frames; that historical result is retained separately and does not validate the current closed loop. No free-standing stability or motion-tracking benchmark was performed.
- Sim, real and PICO launcher previews passed. Hardware control was not started; no real robot or PICO device was connected during validation.
- Published ONNX bytes are unchanged. The prior CPU and GPU model-comparison records remain applicable to those model files; they are not evidence of live hardware performance.

The planner service, context preparation and trajectory processing use Python; inference calls TensorRT directly. The native runtime no longer links ONNX Runtime. Its model contract is checked against the SHA256-bound release manifest and the actual TensorRT tensor interfaces. The offline ONNX validation tool also verifies that the manifest matches the embedded interfaces and metadata.

The Python planner matched the previous C++ implementation on initialization and one walking replan using the same engine: maximum joint-position error was below `6e-16`, joint-velocity error below `3e-14`, and root-quaternion error below `1e-10`. A fresh engine built through the Python API in about 38 seconds. Its maximum initialization differences from the existing cache were `2.49e-4` rad in joint position and `3.52e-3` rad/s in velocity; rebuilding TensorRT engines is not bitwise deterministic. Fifty repeated CUDA graph executions passed, with a median engine-call time of 2.86 ms on the validated RTX 4070 Ti SUPER. These measurements are not end-to-end control latency or hardware stability results.

Reproduce the model/transport integration test after building:

```bash
./install.sh planner
source scripts/setup_env.sh
.venv_planner/bin/python tools/validation/check_runtime.py --planner
```

## Typed motion replay validation

The four published motion collections contain 36 clips: 13 G1 basic, 7 G1 exploration, 8 native H2 and 8 SMPL. All 361 clip files match the original SONIC-H2 data byte for byte; collection manifests were replaced with portable typed manifests, and per-clip provenance JSON files were renamed to `reference.json`. The extreme H2 set is excluded.

After adding G1/H2/SMPL replay, 46 Python command/planner/replay/adapter tests and 59 C++ tests passed. The C++ runtime was rebuilt and release preflight passed. Tests cover type detection, conflicting metadata, H2 native joint permutation and head exclusion, SMPL root/wrist selection, velocity scaling, sample timing, malformed input, and all collection file hashes. Run the command suite and CTest using the instructions above.

An additional CPU-only comparison sent Python replay envelopes through the actual C++ command parser and observation builder for all 36 clips. Five cases per clip covered startup, intermediate frames, playback speed, packet ageing and the endpoint. All 180 comparisons against an independent reference calculated from the source CSVs passed; the maximum absolute observation difference was below `7.3e-7`. The report is saved locally under `outputs/validation/replay_observations.json`.

This change was not tested with GPU inference, closed-loop simulation or hardware. The existing closed-loop limitation above remains applicable; these checks establish data and observation consistency only.

## Third-party dependency validation

On 2026-10-15, the Unitree Python SDK and XRoboToolkit binding were replaced with pinned, unmodified upstream submodules under `third_party/`. The simulator's odometry extension moved to `src/sim/odometry.py`; its DDS type identifier, type information and serialized sample bytes match the previous implementation. The native XR SDK artifacts retain their previous SHA256 hashes.

Simulation and teleoperation installations, dependency checks, preflights and all 93 Python tests passed. A fresh clone fetched both pinned submodules from their official GitHub URLs, installed simulation and teleoperation into empty environments, and passed both preflights. The parent repository and both submodules remained clean after installation. A separate headless simulator published 120 finite LowState and 120 finite odometry messages through loopback DDS domain 217. This check did not start the policy runtime or exercise GPU inference. ARM64 artifact hashes were checked, but ARM64 execution and physical devices were not tested.

## Offline ECHO-G speech/gesture validation

The initial integration on 2026-10-15 used CPU PyTorch 2.12.1, Transformers 5.2.0, and the official ECHO-G implementation at `60b85fb932b124b1a2ddd0868f4917a73c383901` as a reference. The current implementation loads the same weights using local model and feature code; the upstream package and submodule have been removed. Dependency checks passed, and all 15 local model files matched their pinned SHA256 values.

Fresh Chinese and English text each completed Edge TTS with word timestamps, CPU audio/text feature extraction, and official EMA motion inference (eight Euler steps, guidance scale 1, seed 0). The Chinese example has 163 motion frames at 50 Hz and 3.264 seconds of speech; the English example has 177 frames and 3.528 seconds of speech. The three-file bundles are included under `motions/example_text2motion_replay/`. Cached-condition generation also passed. Generation commands are documented in [text-to-motion](TEXT2MOTION.md).

After this integration, all 62 Python command/interface tests and 59 C++ tests passed. The command suite includes the two shipped bundle checks, canonical joint ordering, the 6D rotation convention, SLERP across a yaw wrap, velocity resampling, file integrity, audio startup and failure cleanup. The native audio player built successfully, and release preflight passed.

The following check starts a mock voice service on loopback DDS and a private IPC body receiver; it does not start the policy runtime, contact hardware, or emit sound:

```bash
.venv_sim/bin/python tools/validation/check_audio_replay.py
```

Normal playback delivered 112896 PCM bytes in eight chunks, identical to the English WAV, after the first G1 command. A second run interrupted playback and verified that only the matching audio stream stopped and the body channel was released to `stand`. Both cases passed. The check writes its report and subprocess logs under `outputs/validation/`; choose an unused loopback DDS domain with `--domain` if needed.

The inference implementation was then reduced to the local CLI, feature extraction and model modules. The ECHO-G submodule/package, training YAML configuration, librosa, soundfile and their unused dependencies were removed. A synchronized environment contains 50 pinned dependencies plus this project (test tools are separate). No upstream source checkout is required to download models or generate motion.

On CPU with four PyTorch threads, both Chinese and English recordings were processed again from their WAVs and word-timestamp manifests. Audio features, text tokens, token times, pooled text and all exported motion arrays matched the original implementation bit for bit. EMA inference also matched for seed 7. The comparison reports are `outputs/validation/echo_minimal_model.json` and `echo_minimal_pipeline.json`. Cached-condition generation and the 15-file model check passed without the upstream package installed.

The 62 command/interface regressions and 11 additional inference-input tests passed. Run the latter in the optional generation environment:

```bash
uv pip install --python .venv_text2motion/bin/python -r requirements/test.txt
.venv_text2motion/bin/python -m pytest tests/tools/text2motion_offline/echo_g -q
```

These tests do not establish H2 firmware compatibility, speaker latency, motion quality, or closed-loop tracking. No GPU generation, robot motion or physical speaker playback was performed for this integration.

## Layout and tool-entry validation

Motion replay is packaged under `src/interface/motion_replay/`; offline ECHO-G generation is packaged separately under `tools/text2motion_offline/echo_g/`. Downloads, build preparation and validation scripts are grouped by purpose in `tools/`. All five installed Python environments were refreshed after this reorganization.

The 57 command/replay regressions and 16 offline generation tests passed. From outside the checkout, the installed replay and generation module entry points resolved correctly, the planner downloader verified the existing model at its original destination, and simulation/teleoperation preflights passed. The relocated native prerequisite check, XRoboToolkit preparation and ONNX CPU validation tools also passed. The runtime GPU test's command-line entry was checked without starting inference.

The new ECHO-G entry completed English WAV/text feature extraction and CPU motion generation; every exported motion array was identical to the published English example. The generated bundle loaded in the minimal replay environment. The relocated audio integration tool passed normal and interrupted playback against loopback mocks. These checks exercised no hardware or GPU inference.

Current regression commands:

```bash
.venv_sim/bin/python -m pytest tests/interface/commands tests/interface/motion_replay -q
.venv_text2motion/bin/python -m pytest tests/tools/text2motion_offline/echo_g -q
```

## Robot hand service

The H2 PC2 Inspire DDS-to-RS-485 service has an independent CMake build under `tools/robot_services/inspire_hands/`. It builds with the bundled Unitree SDK or an installed SDK selected through `UNITREE_SDK_ROOT`. Both configurations compiled on the development x86_64 host, and the staged installation resolved DDS libraries from its own `lib/` directory.

Three CTest suites cover ten tests: serial probing/backoff/reconnection, DDS-to-serial delivery with a read-only client default and no replay of disconnected commands, and systemd readiness/heartbeat with a deliberately blocked serial worker. A separate temporary systemd user service on loopback verified that missing USB ports do not restart the daemon, a paused main process triggers watchdog recovery, a killed process restarts, and explicit stop remains stopped. No physical serial port or robot was used; ARM64 compilation and PC2 boot/hardware acceptance remain device-side checks.

```bash
cmake -S tools/robot_services/inspire_hands -B build/inspire_hands -DBUILD_TESTING=ON
cmake --build build/inspire_hands -j2
ctest --test-dir build/inspire_hands --output-on-failure
python3 tools/validation/check_inspire_watchdog.py
```

The watchdog tool requires a running systemd user manager and writes `outputs/validation/inspire_watchdog.json`. Full installation, autostart and device acceptance steps are in the [robot hand service guide](INSPIRE_HANDS.md) / [中文](INSPIRE_HANDS_CN.md).

## Current simulation checks

Checked on 2026-10-15, starting from revision `e02d270`, with the simulator error-propagation fix applied:

- **Regression tests:** 85 Python simulation/command/replay tests, 53 subtests, 34 teleoperation tests, and 59 C++ controller/runtime tests passed. Replay checks cover all 36 bundled G1, H2 and SMPL clips.
- **Released policy:** hashes, ONNX interfaces and finite CPU inference passed for all four input selectors. Full/compact H2 encoder outputs matched exactly in the comparison cases.
- **MuJoCo and DDS:** a headless simulator with elastic support received synthetic PD commands on loopback domain 221. It completed 10.458 seconds of physics, published 10,458 finite LowState and odometry samples each, and followed a head-pitch command followed by a return to neutral. Realtime factor was 0.999994; maximum lag was 1.384 ms, below the existing 20 ms limit. This tests the simulator and DDS bridge, not the SONIC policy or unsupported balance.
- **Planner CPU diagnostic:** the real G1 ONNX weight produced finite trajectories for idle, slow walking and walking requests. The Python context mapping, trajectory merge and ten-frame sparse conversion were exercised with a temporary ONNX Runtime CPU adapter. The released planner implementation remains TensorRT-only; this diagnostic does not validate its GPU engine or closed-loop H2 tracking.
- **Error handling:** simulator exceptions propagate to a nonzero process exit instead of being swallowed.

**GPU closed-loop validation remains blocked on this host.** The earlier runtime process was still stuck at encoder loading with a pending SIGKILL; the earlier `nvidia-smi` process remained in an uninterruptible wait at `uvm_va_space_destroy`. No additional CUDA inference, GPU reset or host reboot was attempted. Unsupported standing, replay tracking, TensorRT planner control and the graphical viewer have **not** passed the current validation. Recover GPU access, then run both [three-terminal workflow](DEPLOYMENT.md#sim2sim) to complete those checks.

Local reports are `outputs/validation/simulation_demo_check.json`, `outputs/validation/sim_dds_current/result.json`, and `outputs/validation/planner_cpu_current.json`. No physical robot was connected.

## Three-terminal keyboard interfaces

The simulator, runtime and input interface have separate launch commands. The combined simulation demo launcher was removed. Motion replay's `--keyboard` entry browses the four typed collections; planner's `--keyboard` entry runs a responsive terminal panel with its own inference worker.

Twelve keyboard checks passed, alongside the existing simulation/command/replay regressions. Tests covered all four collections, manual start, pause/resume timing, switching, looping, input expiry and planner mode/speed/heading conversion. Three pseudo-terminal tests exercised the actual CLI processes and ZeroMQ messages: replay selection/pause/type changes/disconnection/exit, planner requests and worker shutdown, and cancellation during blocked engine startup. Terminal settings were restored and test workers were reaped. The planner tests replaced inference with a test double; runtime state was also synthetic. They validate keyboard interaction and service lifecycle, not TensorRT execution or physical tracking.

```bash
.venv_sim/bin/python -m pytest tests/interface/test_keyboard_controls.py tests/interface/test_keyboard_terminals.py -q
```

Minimal input and planner environment imports, direct clip inspection, launcher help and release preflight passed. The previously recorded GPU closed-loop limitation still applies.

## Runtime startup cancellation (2026-10-15)

The runtime was rebuilt and the existing 59 C++ tests passed. Three additional startup checks run the actual executable with its first CUDA call replaced by a test library: SIGINT and SIGTERM terminate blocked model initialization, and zero available CUDA devices produce a failure without attempting engine construction. These checks use private IPC sockets and `--dry-run`, access no GPU, and initialize no DDS. They are included in CTest when building with `-DBUILD_TESTING=ON`. Release preflight also passed.

The host GPU remained blocked during this check: the live runtime was waiting in `uvm_gpu_retain_by_uuid`, and an earlier GPU query was waiting in `uvm_va_space_destroy`. Termination signals remained pending. The startup signal fix does not unblock an uninterruptible kernel wait; actual TensorRT startup and simulation tracking must be retested after driver recovery. See [startup troubleshooting](DEPLOYMENT.md#runtime-startup-troubleshooting).

## Saving validation reports

Generated reports belong in `outputs/validation/`, which is ignored by Git. Record the tested revision alongside the results:

```bash
mkdir -p outputs/validation
git rev-parse HEAD > outputs/validation/commit.txt
git submodule status --recursive > outputs/validation/submodules.txt
```

After installing the test dependencies above, save Python test results:

```bash
.venv_sim/bin/python -m pytest tests/sim tests/interface/commands tests/interface/motion_replay -q \
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
.venv_model_validation/bin/python tools/validation/validate_released_models.py \
  --report outputs/validation/model_validation.json
```

Package these reports for attachment to the corresponding GitHub release:

```bash
tar -czf outputs/validation-reports.tar.gz -C outputs/validation .
```

Reports describe the environment and revision in which they were produced; rerun the relevant checks for a new release. Keep model interface manifests and checksums with the models under `models/sonic_h2_released/`. Source provenance is documented in [Acknowledgments](ACKNOWLEDGMENTS.md); Git records the source files and pinned submodules.
