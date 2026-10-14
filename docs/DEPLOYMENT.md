# Unitree H2 Deployment

[README](../README.md) · [中文](DEPLOYMENT_CN.md) · [Architecture](ARCHITECTURE.md)

Run commands from the repository root. The H2 runtime starts in a standing pose; motion replay, locomotion planning and teleoperation run as separate processes.

## Installation

For an existing checkout, run `git submodule update --init --recursive`. Upstream dependencies are pinned to the commits recorded by this repository; see [third-party setup](../third_party/README.md).

Validated on Ubuntu x86_64 with Python 3.10.20, CUDA 12.6, TensorRT 10.13.3.9. Follow [native setup](NATIVE_SETUP.md) for CUDA and TensorRT installation.

```bash
sudo apt install build-essential cmake pkg-config libzmq3-dev cppzmq-dev \
  libeigen3-dev nlohmann-json3-dev
export TensorRT_ROOT="$HOME/TensorRT"
bash scripts/build_controller.sh
python3 tools/validation/verify_release.py
```

The build produces `build/runtime/bin/sonic_h2_runtime`. It requires the released encoder, decoder and their `deployment_manifest.json` at startup. The planner runs independently as a Python service. TensorRT caches are generated locally on first use.

Install the Python environment needed for each process using [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
./install.sh inputs    # .venv_inputs: replay and command clients
./install.sh planner   # .venv_planner: TensorRT Python planner
./install.sh sim2sim   # .venv_sim: MuJoCo and DDS; includes replay dependencies
./install.sh teleop    # .venv_teleop: PICO, pose conversion and kinematics
./install.sh text2motion # .venv_text2motion: optional offline ECHO-G generator
```

Only install the environments you use. No Python training framework is needed for replay or simulation. The teleoperation environment uses CPU PyTorch for pose mathematics.

## Sim2Sim

Run three separate terminals from the repository root.

Terminal 1 — simulator:

```bash
bash scripts/run_h2_paper_sim.sh
```

Terminal 2 — SONIC policy runtime:

```bash
bash scripts/run_runtime.sh sim
```

The runtime transitions from the measured initial pose to the default pose over three seconds, then runs the standing policy while waiting for inputs. Both processes use loopback DDS domain 0; set the same `DDS_DOMAIN` in both terminals to change it. Once the standing policy is running, click the simulator window and press **9** to release the elastic support. Observe unsupported standing before sending motion commands.

Terminal 3 — interactive motion replay:

```bash
.venv_inputs/bin/python -m interface.motion_replay --keyboard
```

Press **1 / 2 / 3 / 4** to choose G1 / G1 exploration / H2 / SMPL. Use **Up/Down** to select a clip and **Enter** to play it. **Space** pauses/resumes; **n/p** switches to and plays the next/previous clip; **x** stops and rewinds. Selection alone does not start motion. Pausing, finishing or quitting releases the body channel so the runtime returns to standing. **q**, **Esc** or **Ctrl+C** closes the replay interface.

For planner control, keep terminals 1 and 2 running, exit replay, and replace terminal 3 with:

```bash
bash scripts/run_planner.sh --keyboard
```

This terminal runs the TensorRT planner worker and its keyboard panel. **1 / 2 / 3** selects slow walk / walk / run; **0** selects idle. **W/S** moves forward/backward, **A/D** strafes, **Q/E** changes heading, and **+/-** adjusts speed. **Space** returns to standing; **Esc** or **Ctrl+C** closes the panel and planner worker. Full controls, including all planner modes, are in the [interface guide](../src/interface/README.md#planner-keyboard-control).

Both input panels wait for fresh runtime state before sending motion. Replay does not automatically resume after a state interruption, and planner keyboard commands are cleared. Closing an input panel leaves the simulator and policy runtime running. The planner environment/model must be installed separately; first engine construction may take several minutes. See [planner setup](PLANNER.md) and [validation records](REPRODUCIBILITY.md).

## Motion replay and offline text-to-motion

For speech and accompanying gestures, use the [ECHO-G text-to-motion guide](TEXT2MOTION.md). It includes Chinese/English examples, offline generation, and local or Unitree speaker playback through `interface.motion_replay`. The included speech/gesture bundles need only the input environment.

Replay accepts **G1, H2 and SMPL** CSV clip directories or NPZ files, including offline text-to-motion output. Input type is detected from the collection manifest and clip metadata. The bundled collections are:

| Directory under `motions/` | Type | Clips |
|---|---|---:|
| `example_g1_replay` | G1 basic | 13 |
| `example_g1_explore_replay` | G1 exploration | 7 |
| `example_h2_replay` | Native H2 | 8 |
| `example_smpl_replay` | SMPL | 8 |

```bash
.venv_inputs/bin/python -m interface.motion_replay \
  motions/example_h2_replay/walking_quip_360_R_002__A428 --inspect
.venv_inputs/bin/python -m interface.motion_replay \
  motions/example_smpl_replay/walking_quip_360_R_002__A428
.venv_inputs/bin/python -m interface.motion_replay generated_motion.npz --inspect
```

`--inspect` validates the clip without connecting. `--type g1|h2|smpl` labels otherwise ambiguous external data and must agree with declared metadata. `--fps` overrides the recorded frame rate; CSV defaults to 50 Hz if unspecified. `--speed` adjusts playback and velocity together; `--loop` requires a continuous loop boundary.

G1 and H2 use ten samples at offsets `0.0, 0.1, …, 0.9` seconds. SMPL uses ten samples at `0.00, 0.02, …, 0.18` seconds. H2 reads the native 31D files, excludes head joints and sends native body policy angles. SMPL reads its own joints/root and semantic wrist angles. Head control remains independent.

See [motion formats and NPZ fields](../motions/README.md) for export conventions, and [the protocol](ARCHITECTURE.md#body-data) for transmitted fields. Optional ECHO-G generation runs separately and writes a motion/audio bundle consumed by the same replay entry point.

## Sim2Real

### Risks and staged validation

> [!WARNING]
> Real-robot testing can cause falls, collisions, equipment damage and injury. Before attempting any motion on hardware, verify stable tracking of that motion in simulation with the intended policy, reference motion and control settings. Successful simulation is a prerequisite, not a guarantee of stable hardware behavior.

Unitree H2 can exhibit a substantial simulation-to-reality gap. Pay particular attention to damping and mechanical backlash in the linkage-driven joints, especially the ankles, as well as IMU calibration and joint zero-position offsets. These differences can affect posture estimation and joint targets, alter the robot's posture and center-of-mass position, and reduce its balance margin.

Validate on hardware in the following order, advancing only after the current stage is consistently stable:

1. **Default standing:** establish stable balance in the default pose before sending motion commands.
2. **Low-difficulty motions:** verify stable tracking of slow motions with small amplitudes and limited balance demands.
3. **Progressively harder motions:** increase motion speed, amplitude and coordination demands gradually, confirming stable tracking at each step.

If standing or tracking becomes unstable, stop the test and resolve the mechanical, calibration or control issue before progressing. Passing a simpler motion does not validate a more demanding one; each candidate motion must first pass simulation.

### Before testing highly dynamic motions

- Reduce and characterize the remaining simulation-to-reality mismatch as far as practicable. Check linkage damping and backlash, particularly at the ankles, and verify IMU and joint zero-position calibration. Do not proceed with unresolved mechanical looseness or calibration errors.
- Use a clear, controlled test area with suitable footing and sufficient space for the full motion and a possible fall. Keep people outside the motion and fall zone, and have appropriate robot support and an accessible emergency stop ready.
- Fully charge the robot battery before testing and confirm normal power-system status.
- Remove dexterous hands and other fragile, nonessential attachments before testing. Disable their command backends; for the hands, set `HAND_BACKEND=none`. After removal, check that the simulation's mass and inertia settings represent the tested hardware, then repeat standing and low-difficulty validation.

These conditions apply in addition to the staged validation above. Resolve any unmet condition before attempting highly dynamic motions.

### Launch

Set the robot interface, DDS domain, hand backend and joint gain scales in [`h2_deploy.env`](../config/h2_deploy.env). Prepare robot support and emergency stop before enabling control.

```bash
bash scripts/run_runtime.sh real --print-command
bash scripts/run_runtime.sh real
```

The launcher prints the target interface and requires `RUN` before starting. The robot moves to its default standing pose automatically. Start replay or another producer separately using the same commands as simulation. Gain settings affect initialization and policy control; damping on shutdown is unchanged. `HAND_BACKEND=none` disables hand DDS, `inspire` connects to the robot's Inspire service, and `dex3` uses Unitree Dex3.

For Inspire hands, install the included `tools/robot_services/inspire_hands/` service on H2 PC2 first. Follow the [robot hand service guide](INSPIRE_HANDS.md) for serial mapping, autostart, watchdog setup and read-only checks. This service runs independently of the body policy.

## Planner

Download the optional shared G1 model:

```bash
uv venv .venv_download --python 3.10
uv pip install --python .venv_download/bin/python -r requirements/download.txt
.venv_download/bin/python tools/model_download/download_planner.py
```

The model is downloaded to `models/motion_planner_g1/planner_sonic.onnx`, ignored by Git. See [download details](PLANNER.md).

Install its environment with `./install.sh planner`. The installer uses the TensorRT Python wheel from `$TensorRT_ROOT/python`, matching the native SDK; set `TENSORRT_WHEEL` if the wheel is stored elsewhere.

For the three-terminal simulation workflow above, start the keyboard panel and inference worker together in terminal 3:

```bash
bash scripts/run_planner.sh --keyboard
```

For programmatic control, run `bash scripts/run_planner.sh` without `--keyboard`, then send requests from a separate client:

```bash
.venv_inputs/bin/python -m interface.planner.client \
  --mode 1 --speed 0.3 --movement 1 0 0
```

The Python planner owns TensorRT inference, trajectory blending and conversion to sparse G1 references. The runtime only receives those references. Stopping planner requests returns the runtime to standing. The service also publishes its output on port 5561; `--publish-only` disables direct body-channel forwarding for use by the teleoperation bridge. Keyboard worker logs are saved under `outputs/planner/`.

## VR teleoperation

Follow [PICO setup](VR_SETUP.md), then start the runtime in simulation or on the supported robot. In separate terminals:

```bash
# PICO tracking and calibration
bash scripts/run_h2_teleop_head_adjust.sh pico
# Translate tracking into body/head/hand messages
bash scripts/run_h2_teleop_head_adjust.sh bridge
# Optional: joystick locomotion and lower-body planning during VR control
bash scripts/run_h2_teleop_head_adjust.sh planner
```

The bridge sends SMPL references for the full-body pose mode and VR three-point references for the PICO VR mode. `bridge --pose-kind vr` uses wrist/head targets in pose mode too. Without a planner, VR uses a standing lower-body reference. SMPL packets contain ten samples at 20 ms intervals; the live tracking window introduces about 180 ms of reference buffering. Head pitch/yaw and hand targets use separate sockets and never enter the body policy.

Head settings belong to [`h2_head_adjust.env`](../config/h2_head_adjust.env); the hand backend belongs to `h2_deploy.env`. The PICO feedback port is **5559**. For multi-machine operation, explicitly bind the relevant runtime sockets to the network interface and pass matching endpoints to the bridge. Synchronize system clocks; commands carry wall-clock timestamps and expire.

## Independent head and hand commands

```bash
# Radians; this changes only the two head joints.
.venv_inputs/bin/python -m interface.peripherals.client head --pitch 0.3 --yaw -0.2
# Seven Dex3-compatible values; Inspire maps this representation to finger closure.
.venv_inputs/bin/python -m interface.peripherals.client hands --left '[0,0,0,0,0,0,0]'
```

Each process refreshes its own command until interrupted. Head targets are clamped and slew-limited; expiry returns the head to neutral. Expired hand input stops hand publishing without generating a new grip command. Inspire requires a released trigger before accepting closure. Body-source changes do not reset these independent channels.

## Validation

```bash
sudo apt install libgtest-dev zlib1g-dev
bash scripts/build_controller.sh -DBUILD_TESTING=ON
ctest --test-dir build/runtime --output-on-failure
source scripts/setup_env.sh
.venv_sim/bin/python tools/validation/check_runtime.py
./install.sh planner
.venv_planner/bin/python tools/validation/check_runtime.py --planner
```

The integration check runs the actual models with synthetic robot state and private IPC sockets. It does not initialize DDS or connect to hardware. It tests startup standing, all body input formats, expiry, source ownership and independent head/hand commands. See [validation records](REPRODUCIBILITY.md) for the scope of completed checks.
