# Unitree H2 Deployment

[README](../README.md) · [中文](DEPLOYMENT_CN.md) · [Troubleshooting](TROUBLESHOOTING.md)

Run all commands from the repository root. For simulation, use three terminals: **simulator → policy runtime → input interface**. On hardware, use the runtime and input terminals only.

- [Prepare the environment](#prepare-the-environment)
- [Sim2Sim: motion replay or keyboard planner](#sim2sim)
- [How to test your own motions](#how-to-test-your-own-motions)
- [Planner](#planner)
- [Sim2Real: hardware startup](#sim2real)
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

#### Motion replay

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

##### How to test your own motions

Export a G1, native H2 or SMPL motion as a CSV clip directory or NPZ file following the [motion format specification](../motions/README.md). Replace the path below with your file or clip directory and inspect it first:

```bash
MOTION=/absolute/path/to/your_motion.npz
.venv_inputs/bin/python -m interface.motion_replay "$MOTION" --inspect
```

`--inspect` checks the data without connecting to the runtime. Once inspection passes, keep the simulator and policy running in terminals 1 and 2 and confirm stable standing. Exit any keyboard replay or planner process, then run in terminal 3:

```bash
.venv_inputs/bin/python -m interface.motion_replay "$MOTION"
```

This starts playback immediately. Completion or **Ctrl+C** releases the body input and returns the policy to standing. Check tracking in simulation before attempting the motion on hardware.

Use `--speed` to change playback speed, `--fps` to override the source frame rate, and `--type g1|h2|smpl` for data without unambiguous type metadata. Use `--loop` only for motions with continuous boundaries. Joint ordering, units and required fields are defined in the [motion format specification](../motions/README.md).

#### Planner

Install and download the planner once before its first use:

```bash
uv venv .venv_download --python 3.10
uv pip install --python .venv_download/bin/python -r requirements/download.txt
.venv_download/bin/python tools/model_download/download_planner.py
./install.sh planner
python3 tools/validation/verify_release.py --require-planner
```

The model is stored at `models/motion_planner_g1/planner_sonic.onnx`. Keep terminals 1 and 2 running, exit replay, then start the planner in terminal 3:

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

Wait for both planner and runtime readiness before using movement keys. Mode and speed selection alone do not start a stopped planner. Begin with slow walking and a low speed. See [all planner controls](../src/interface/README.md#planner-keyboard-control). Download details and programmatic requests are in the [planner guide](PLANNER.md).

### Switch inputs or finish the session

To change input, stop the current motion, exit terminal 3's input program, then start the other input. Terminals 1 and 2 stay running. After a runtime connection interruption, replay pauses and planner requests are cleared; a new operator action is needed to resume.

To end the session:

1. Return to standing with replay **x** or planner **Space**, then exit the input program.
2. Re-engage the simulator's elastic support with **9**.
3. In **terminal 2**, press **o** to stop with damping. This stop is latched: restart the runtime to control the robot again. **Ctrl+C** also exits the runtime.
4. Close the simulator last.

Damping does not maintain balance; stopping the runtime without support can cause a fall.

## Sim2Real

Before using hardware, read the [risk and staged-validation requirements](SIM2REAL_SAFETY.md). No simulator is needed on the hardware control host.

1. Set the robot-facing `NETWORK_INTERFACE`, `DDS_DOMAIN`, `HAND_BACKEND` and joint gain scales in [`config/h2_deploy.env`](../config/h2_deploy.env). If using Inspire hands, deploy the [PC2 hand service](INSPIRE_HANDS.md) first.
2. Prepare robot support and an accessible emergency stop. In **terminal 1**, preview the configuration, then launch:

```bash
bash scripts/run_sonic_h2_runtime.sh real --print-command
bash scripts/run_sonic_h2_runtime.sh real
```

3. Check the printed interface and DDS domain. Enter **y** and press Enter to begin; an empty response or any other input cancels.
4. Wait for **[READY]**, then press **]** or **s** in the runtime terminal to enable policy control.
5. Once standing is stable, open **terminal 2** and use the [input commands above](#terminal-3-choose-one-input).

Press **o** to stop with damping; support the robot before stopping. Restart the runtime to resume control.

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
