# Reproducibility

The release contains the H2 runtime for MuJoCo, hardware deployment, and PICO VR teleoperation. The independent policy runtime loads `models/sonic_h2_released`; producers connect through the documented command channels. The model interface is documented in the [model README](../models/sonic_h2_released/README.md).

## Validated environment

Ubuntu x86_64, Python 3.10.20, CUDA 12.6, TensorRT 10.13.3.9, and CUDA Python bindings 12.9.4. Python runtime dependencies are pinned in `requirements/inputs.lock`, `requirements/planner.lock`, `requirements/sim.lock` and `requirements/teleop.lock`. The simulator and teleoperation environments were installed independently from empty environments.

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

**GPU closed-loop validation was blocked during this check.** The earlier runtime process was still stuck at encoder loading with a pending SIGKILL; the earlier `nvidia-smi` process remained in an uninterruptible wait at `uvm_va_space_destroy`. No additional CUDA inference, GPU reset or host reboot was attempted at that time. This check did not validate unsupported standing, replay tracking, TensorRT planner control or the graphical viewer. Later checks after driver recovery are recorded below.

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

The host GPU remained blocked during this check: the live runtime was waiting in `uvm_gpu_retain_by_uuid`, and an earlier GPU query was waiting in `uvm_va_space_destroy`. Termination signals remained pending. The startup signal fix does not unblock an uninterruptible kernel wait; actual TensorRT startup and simulation tracking must be retested after driver recovery. See [startup troubleshooting](TROUBLESHOOTING.md#gpu-startup).

## Manual policy activation and upright standing (2026-10-15)

After host GPU access recovered, the runtime was rebuilt with manual policy activation and the standing reference `hip_pitch=-0.10`, `knee=0.25`, `ankle_pitch=-0.15` rad. The model's trained decoder offsets and action scales were preserved. The launcher is now `scripts/run_sonic_h2_runtime.sh`.

- **Regression checks:** 64 C++ tests and three startup tests passed, including early-key rejection, both start keys, latched stop, terminal restoration and independent standing-reference normalization. The Python simulation/interface/replay suite passed 97 tests and 53 subtests. Release preflight and launcher checks passed.
- **Actual TensorRT integration:** `tools/validation/check_runtime.py --start-key s --planner` passed with synthetic robot state and private IPC sockets. It verified waiting at `ready`, rejecting a premature start, discarding queued body input, starting without Enter, all body input types, head/hand independence, planner inference, stopping on `o`, and terminal restoration. This check does not initialize DDS.
- **Unsupported standing:** actual C++ policy inference and the headless MuJoCo simulator communicated over loopback DDS domain 223. After initialization and a one-second manual wait, **]** enabled policy control; elastic support was released after three seconds of policy control. The robot completed 12.28 seconds without support. Mean left/right knee angles were 0.246/0.249 rad, minimum base height was 0.994 m, and maximum body tilt was 0.160 rad, including the support-release transient. Final-five-second mean pitch was approximately −0.5°. Maximum simulation lag was 2.70 ms, within the 20 ms limit.
- **Replay-to-standing transition:** a separate run sent six seconds of `example_h2_replay/walking_quip_360_R_002__A428`, then released replay and observed eight seconds of policy standing. It completed 16.36 seconds without support, with minimum base height 0.945 m and maximum tilt 0.163 rad. Maximum simulation lag was 1.91 ms. No fall or non-finite state occurred. This is a partial-clip transition check, not a tracking-accuracy benchmark for the full motion library.

Both DDS checks confirmed that **o** exited the runtime, restored terminal settings, and sent a final command with all position gains zero and damping 8. Detailed reports and local test harnesses are under `outputs/validation/standing_pose/`. These checks used no physical robot and no graphical viewer; hardware balance and tracking remain unvalidated.

## Arms at sides standing

The standing preset now uses shoulder pitch 0, mirrored shoulder roll ±0.10, and elbow 1.50 rad. Runtime initialization, idle policy references and simulator reset use the same pose. Static MuJoCo forward kinematics, with closed linkage constraints solved, placed each wrist 1.7 cm forward of the torso origin (previously 11.6 cm), with no arm contact penetration exceeding 1 mm.

The rebuilt runtime passed the 64 C++ tests and three startup checks. Actual TensorRT inference with headless MuJoCo over loopback DDS completed 12.26 seconds of unsupported standing, then a separate 16.33-second unsupported run including six seconds of H2 walking replay and eight seconds back in standing. Minimum base heights were 0.943/0.949 m; maximum body tilts were 0.192/0.186 rad. Both runs stayed within the 20 ms simulation lag limit, restored terminal settings and sent damping commands on **o**. During the final five seconds of each run, mean wrist positions were approximately 5–6 cm forward and 23 cm to either side of the torso origin, showing a small policy tracking offset from the reference. No fall or non-finite state occurred.

Local reports are `outputs/validation/standing_pose/closed_loop_044936/result.json`, `outputs/validation/standing_pose/replay_transition_045006/result.json`, and `outputs/validation/standing_pose/arms_at_sides.json`. These are simulation checks; this preset has not been tested on a physical robot.

## Torso posture with arms at the sides

The earlier arms-down preset passed pelvis-tilt checks but produced an approximately 8° forward torso lean. The standing reference now pairs shoulder pitch 0.25 rad with elbow 1.00 rad. The simulator reset pose uses the same values. The final change only adjusts these references; the model and controller are unchanged.

The rebuilt runtime passed 64 C++ tests, three startup checks and release preflight. A 30-second unsupported standing run used actual TensorRT inference and headless MuJoCo over loopback DDS. Posture measurements use the last five seconds with policy gains active and both feet in contact, excluding shutdown damping. The COM offset is the forward projection relative to the midpoint of the longitudinal sole bounds, computed from MuJoCo mass properties and foot collision meshes.

| Measurement | Previous preset | Revised preset |
|---|---:|---:|
| Mean torso pitch | 8.07° forward | 2.01° forward |
| Mean COM projection offset | 2.71 cm forward | 1.50 cm forward |
| Minimum front sole margin | 9.39 cm | 10.29 cm |
| Mean wrist position forward of torso | 5.0 / 5.3 cm | 2.8 / 2.9 cm |

A separate six-second H2 walking replay followed by eight seconds of standing also passed. Its final-five-second mean torso pitch was 0.87° forward, with a 1.92 cm forward COM offset. Both runs completed without falls, met the 20 ms simulation lag limit and sent damping commands on **o**. The additional posture checks required mean torso pitch within ±3.5°, mean COM offset within ±2.5 cm, front/rear sole margins above 5 cm and mean wrist forward reach below 6 cm.

Local comparison records are `outputs/validation/standing_pose/closed_loop_044936/posture.json`, `outputs/validation/standing_pose/closed_loop_050343/posture.json`, and `outputs/validation/standing_pose/replay_transition_050436/posture.json`. These measurements assess posture in simulation and do not establish hardware balance margins.

## Core deployment regression checks

The current deployment package passed 86 Python tests and 53 subtests covering simulation, command validation, all 36 G1/H2/SMPL reference clips and keyboard interfaces. Six additional CLI processes sent G1, H2 and SMPL references over private IPC, covering normal completion and interruption during looping; every process released its body channel to `stand`. These checks used synthetic references and no robot connection.

The native runtime configured and built successfully, and all three CTest suites passed. The minimal inputs environment installed successfully, and the built Python wheel contains the interface and simulator packages with the documented dependencies. Release preflight, shell syntax and changed documentation links passed. Local reports are under `outputs/validation/main_release_cleanup/`. This check did not repeat GPU inference or physical robot tests.

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
