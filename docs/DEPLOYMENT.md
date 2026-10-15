# Unitree H2 Deployment

[README](../README.md) · [中文](DEPLOYMENT_CN.md) · [Troubleshooting](TROUBLESHOOTING.md)

Run all commands from the repository root. For simulation, use three terminals: **simulator → policy runtime → input interface**. On hardware, use the runtime and input terminals only.

- [Prepare the environment](#prepare-the-environment)
- [Sim2Sim: motion replay or keyboard planner](#sim2sim)
- [Sim2Real: staged validation and operation](#sim2real)
- [Replay a specific motion](#motion-replay)
- [Planner setup and programmatic control](#planner)
- [VR teleoperation](#vr-teleoperation)
- [Independent head and hand commands](#independent-head-and-hand-commands)

## Prepare the environment

Complete [native setup](NATIVE_SETUP.md) for system packages, CUDA and TensorRT, and install [uv](https://docs.astral.sh/uv/getting-started/installation/). The validated environment is Ubuntu x86_64, Python 3.10.20, CUDA 12.6 and TensorRT 10.13.3.9.

Then build the runtime and install the replay and simulation environments:

```bash
git submodule update --init --recursive
export TensorRT_ROOT="$HOME/TensorRT"
export CUDAToolkit_ROOT=/usr/local/cuda-12.6
bash scripts/build_controller.sh
./install.sh inputs
./install.sh sim2sim
python3 tools/validation/verify_release.py
```

Set the two SDK paths to your actual installations and make them available in each launch terminal. The runtime executable is `build/runtime/bin/sonic_h2_runtime`; launchers select the H2 model in `models/sonic_h2_released/` automatically. Use the explicit Python paths below; environment activation is optional.

For planner control, also complete [planner setup](#planner). For VR, complete [VR setup](#vr-teleoperation). Installation errors and GPU startup issues are covered in [Troubleshooting](TROUBLESHOOTING.md).

## Sim2Sim

### Terminal 1: start the simulator

```bash
bash scripts/run_h2_paper_sim.sh
```

Keep the elastic support engaged while the runtime initializes.

### Terminal 2: start and enable the policy runtime

```bash
bash scripts/run_sonic_h2_runtime.sh sim
```

1. Wait for **[READY]**. After receiving robot state, the runtime takes three seconds to reach the configured standing pose, then holds it with PD control.
2. Focus **terminal 2** and press **]** or **s** to enable policy control. No Enter is needed. Keys pressed before **[READY]** are discarded.
3. Once the policy is running, focus the **simulator window** and press **9** to release the elastic support.
4. Confirm stable standing without support before sending motion commands.

PD waiting only positions the joints; policy balance control starts at step 2. The simulator and runtime default to loopback DDS domain 0. To use another domain, set the same `DDS_DOMAIN` in terminals 1 and 2 before launch.

### Terminal 3: choose one input

**Motion replay**

```bash
.venv_inputs/bin/python -m interface.motion_replay --keyboard
```

| Key in terminal 3 | Action |
|---|---|
| `1` / `2` / `3` / `4` | Select G1 / G1 exploration / H2 / SMPL |
| Up / Down | Select a clip without starting it |
| Enter | Play the selected clip from the beginning |
| Space | Pause / resume |
| `n` / `p` | Switch to and play the next / previous clip |
| `x` | Stop and rewind; return to standing |
| `q` / Esc / Ctrl+C | Exit replay |

Pause, completion and exit release the body input, so the policy returns to standing. See [the interface guide](../src/interface/README.md#keyboard-playback) for looping and additional controls.

**Keyboard planner**

Complete [planner setup](#planner) first. Exit replay, then run:

```bash
bash scripts/run_planner.sh --keyboard
```

| Key in terminal 3 | Action |
|---|---|
| `0` / `1` / `2` / `3` | Select idle / slow walk / walk / run |
| Hold `w` / `s` | Move forward / backward |
| Hold `a` / `d` | Strafe left / right |
| `q` / `e` | Adjust heading left / right |
| `+` / `-` | Increase / decrease speed |
| Enter | Activate / deactivate the selected mode |
| Space or `x` | Clear requests and return to standing |
| Esc / Ctrl+C | Exit the panel and planner worker |

Wait for both planner and runtime readiness before using movement keys. Mode and speed selection alone do not start a stopped planner. Begin with slow walking and a low speed. See [all planner controls](../src/interface/README.md#planner-keyboard-control).

### Switch inputs or finish the session

To change input, stop the current motion, exit terminal 3's input program, then start the other input. Terminals 1 and 2 stay running. After a runtime connection interruption, replay pauses and planner requests are cleared; a new operator action is needed to resume.

To end the session:

1. Return to standing with replay **x** or planner **Space**, then exit the input program.
2. Re-engage the simulator's elastic support with **9**.
3. In **terminal 2**, press **o** to stop with damping. This stop is latched: restart the runtime to control the robot again. **Ctrl+C** also exits the runtime.
4. Close the simulator last.

Damping does not maintain balance; stopping the runtime without support can cause a fall.

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

### 1. Configure hardware

Edit [`config/h2_deploy.env`](../config/h2_deploy.env) before launching:

| Setting | What to configure |
|---|---|
| `NETWORK_INTERFACE` | The workstation interface connected to H2 |
| `DDS_DOMAIN` | The domain used by the robot |
| `HAND_BACKEND` | `none`, `inspire` or `dex3` |
| Joint gain scales | The ankle and knee scales for the tested hardware configuration |

Gain scales apply during initialization and policy control. For Inspire hands, deploy the independent PC2 service using the [hand service guide](INSPIRE_HANDS.md), including serial mapping, autostart and watchdog configuration.

### 2. Terminal 1: start the runtime

Prepare robot support and an accessible emergency stop. Preview the command, then launch:

```bash
bash scripts/run_sonic_h2_runtime.sh real --print-command
bash scripts/run_sonic_h2_runtime.sh real
```

Check the printed network interface and DDS domain. Enter **y** and press Enter to start; an empty response or any other input cancels. The runtime moves the joints to the standing pose. Wait for **[READY]**, then press **]** or **s** in this terminal to enable policy control. Establish stable standing before sending motion.

### 3. Terminal 2: start the input

Use either the [motion replay or keyboard planner command above](#terminal-3-choose-one-input). No simulator is needed. Follow the staged validation order for every new motion.

To finish, stop and exit the input first, secure the robot with support, then press **o** in the runtime terminal. The runtime stops with damping and must be restarted before control can resume. This software stop does not replace the robot's emergency stop.

## Motion replay

For a specific clip, inspect it first, then replay it with the policy already running:

```bash
.venv_inputs/bin/python -m interface.motion_replay \
  motions/example_h2_replay/walking_quip_360_R_002__A428 --inspect
.venv_inputs/bin/python -m interface.motion_replay \
  motions/example_h2_replay/walking_quip_360_R_002__A428
```

`--inspect` validates files without connecting to the runtime. Replay accepts G1, native H2 and SMPL CSV directories or NPZ files. Use `--speed` to change playback speed, `--fps` to override the source frame rate, and `--type g1|h2|smpl` for external data without unambiguous type metadata. Use `--loop` only for clips with continuous boundaries.

The four bundled collections and input fields are documented in [motion formats](../motions/README.md). Sparse sampling and transmitted fields are described in [the command protocol](ARCHITECTURE.md#body-data).

## Planner

Set up the planner once before using its keyboard or programmatic interface:

```bash
uv venv .venv_download --python 3.10
uv pip install --python .venv_download/bin/python -r requirements/download.txt
.venv_download/bin/python tools/model_download/download_planner.py
./install.sh planner
python3 tools/validation/verify_release.py --require-planner
```

The model is stored at `models/motion_planner_g1/planner_sonic.onnx`. Source and checksum details are in the [planner guide](PLANNER.md). For interactive operation, use terminal 3 in [Sim2Sim](#sim2sim).

For programmatic control, start the service without the keyboard panel:

```bash
bash scripts/run_planner.sh
```

Then send requests from a separate terminal while the policy is running:

```bash
.venv_inputs/bin/python -m interface.planner.client \
  --mode 1 --speed 0.3 --movement 1 0 0
```

Stop the request client to return to standing. See [the interface guide](../src/interface/README.md#planner-and-teleoperation) for service options.

## VR teleoperation

Install the teleoperation environment and complete [PICO connection and calibration setup](VR_SETUP.md):

```bash
./install.sh teleop
```

Start and enable the runtime using the simulation or hardware procedure above. Run each of the following in a separate terminal:

```bash
# PICO tracking and calibration
bash scripts/run_h2_teleop_head_adjust.sh pico
```

```bash
# Send body, head and hand commands
bash scripts/run_h2_teleop_head_adjust.sh bridge
```

For joystick locomotion, also complete [planner setup](#planner), then run:

```bash
bash scripts/run_h2_teleop_head_adjust.sh planner
```

Without a planner, VR uses a standing lower-body reference. Head settings are in [`config/h2_head_adjust.env`](../config/h2_head_adjust.env); the hardware hand backend is in `config/h2_deploy.env`. Head and hand commands use independent channels. Multi-machine endpoint configuration is described in [the architecture guide](ARCHITECTURE.md).

## Independent head and hand commands

With the runtime running, start each needed client in its own terminal:

```bash
# Head pitch/yaw in radians
.venv_inputs/bin/python -m interface.peripherals.client head --pitch 0.3 --yaw -0.2
# Seven Dex3-compatible targets; Inspire maps them to finger closure
.venv_inputs/bin/python -m interface.peripherals.client hands --left '[0,0,0,0,0,0,0]'
```

Each client refreshes its targets until interrupted. Head expiry returns to neutral; hand expiry stops publishing without issuing a new grip command. Body-source changes do not reset these channels. Inspire requires a released trigger before accepting closure.

## Configuration and checks

- [Standing pose configuration](../src/runtime/README.md#standing-pose-configuration): reference angles and custom presets.
- [Troubleshooting](TROUBLESHOOTING.md): environment, build, GPU startup and connection issues.
- [Validation and reproducibility](REPRODUCIBILITY.md): offline tests, model checks and recorded validation scope.
