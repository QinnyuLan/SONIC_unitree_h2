# Unitree H2 Deployment

[README](../README.md) · [中文](DEPLOYMENT_CN.md) · [Architecture](ARCHITECTURE.md)

Run commands from the repository root. The H2 runtime starts in a standing pose; motion replay, locomotion planning and teleoperation run as separate processes.

## Installation

Validated on Ubuntu x86_64 with Python 3.10.20, CUDA 12.6, TensorRT 10.13.3.9 and ONNX Runtime C++ 1.16.3. Follow [native setup](NATIVE_SETUP.md) for CUDA, TensorRT and ONNX Runtime installation.

```bash
sudo apt install build-essential cmake pkg-config libzmq3-dev cppzmq-dev \
  libeigen3-dev nlohmann-json3-dev zlib1g-dev
export TensorRT_ROOT="$HOME/TensorRT"
bash scripts/build_controller.sh
python3 tools/verify_release.py
```

The executables are `gear_sonic_deploy/target/release/sonic_h2_runtime` and `sonic_g1_planner`. The runtime can be built on its own with `-DBUILD_PLANNER=OFF`. It requires only the released encoder and decoder at startup. TensorRT caches are generated locally on first use.

Install the Python environment needed for each process using [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
./install.sh inputs    # .venv_inputs: replay and command clients
./install.sh sim2sim   # .venv_sim: MuJoCo and DDS; includes replay dependencies
./install.sh teleop    # .venv_teleop: PICO, pose conversion and kinematics
```

Only install the environments you use. No Python training framework is needed for replay or simulation. The teleoperation environment uses CPU PyTorch for pose mathematics.

## Sim2Sim

Terminal 1 — simulator:

```bash
bash scripts/run_h2_paper_sim.sh
```

Terminal 2 — policy runtime:

```bash
bash scripts/run_runtime.sh sim
```

The runtime transitions from the measured initial pose to the default pose over three seconds, then runs the standing policy. No motion file or planner is loaded. Both processes use loopback DDS domain 0; set `DDS_DOMAIN` in both terminals to use another domain. The simulator starts with its elastic support enabled; release it using the simulator controls when ready.

Terminal 3 — choose a G1 motion clip:

```bash
.venv_sim/bin/python -m gear_sonic.sources.motion_replay \
  gear_sonic_deploy/reference/example/walking_quip_360_R_002__A428
```

Replay finishes by releasing the body channel. The runtime returns to standing. `Ctrl+C` in a producer releases its channel; `Ctrl+C` in the runtime ends control with damping.

## Motion replay and offline text-to-motion

Both use the same replay executable. Input is either one SONIC CSV clip directory or an NPZ containing:

| Key | Shape | Convention |
|---|---|---|
| `joint_pos` | `[N, 29]` | Absolute native G1 semantic joint positions, radians |
| `joint_vel` | `[N, 29]`, optional | rad/s; computed by finite differences if omitted |
| `root_quat` | `[N, 4]` | World-frame quaternion, wxyz |
| `fps` | scalar | Source sampling frequency |

CSV input uses `joint_pos.csv`, optional `joint_vel.csv`, and the first quaternion in `body_quat.csv`. CSV defaults to 50 Hz; specify `--fps` for other rates. Joint names and ordering are in [the protocol](ARCHITECTURE.md#joint-order).

```bash
.venv_inputs/bin/python -m gear_sonic.sources.motion_replay generated_motion.npz --inspect
.venv_inputs/bin/python -m gear_sonic.sources.motion_replay generated_motion.npz
```

`--speed` changes playback speed and scales velocity consistently. `--loop` requires a clip with a continuous boundary. All replay sources send ten samples at offsets `0.0, 0.1, …, 0.9` seconds. Offline text-to-motion output must be exported as a G1 motion before replay; this repository does not run a text-generation model.

## Sim2Real

Set the robot interface, DDS domain, hand backend and joint gain scales in [`h2_deploy.env`](../gear_sonic_deploy/config/h2_deploy.env). Prepare robot support and emergency stop before enabling control.

```bash
bash scripts/run_runtime.sh real --print-command
bash scripts/run_runtime.sh real
```

The launcher prints the target interface and requires `RUN` before starting. The robot moves to its default standing pose automatically. Start replay or another producer separately using the same commands as simulation. Gain settings affect initialization and policy control; damping on shutdown is unchanged. `HAND_BACKEND=none` disables hand DDS, `inspire` uses the existing Inspire service, and `dex3` uses Unitree Dex3.

## Planner

Download the optional shared G1 model:

```bash
uv venv .venv_download --python 3.10
uv pip install --python .venv_download/bin/python -r requirements/download.txt
.venv_download/bin/python download_from_hf.py --planner-only
```

The model stays at `gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx`, ignored by Git. See [download details](PLANNER.md).

With the runtime running, start the service in another terminal:

```bash
bash scripts/run_planner.sh
```

Send a request from a third terminal:

```bash
.venv_inputs/bin/python -m gear_sonic.sources.planner_command \
  --mode 1 --speed 0.3 --movement 1 0 0
```

The planner owns inference, trajectory blending and conversion to sparse G1 references. The runtime only receives those references. Stopping planner requests returns the runtime to standing. The service also publishes its output on port 5561; `--publish-only` disables direct body-channel forwarding for use by the teleoperation bridge.

## VR teleoperation

Follow [PICO setup](VR_SETUP.md), then start the runtime in simulation or on the supported robot. In separate terminals:

```bash
# PICO tracking and calibration
bash gear_sonic_deploy/scripts/run_h2_teleop_head_adjust.sh pico
# Translate tracking into body/head/hand messages
bash gear_sonic_deploy/scripts/run_h2_teleop_head_adjust.sh bridge
# Optional: joystick locomotion and lower-body planning during VR control
bash gear_sonic_deploy/scripts/run_h2_teleop_head_adjust.sh planner
```

The bridge sends SMPL references for the full-body pose mode and VR three-point references for the PICO VR mode. `bridge --pose-kind vr` uses wrist/head targets in pose mode too. Without a planner, VR uses a standing lower-body reference. SMPL packets contain ten samples at 20 ms intervals; the live tracking window introduces about 180 ms of reference buffering. Head pitch/yaw and hand targets use separate sockets and never enter the body policy.

Head settings belong to [`h2_head_adjust.env`](../gear_sonic_deploy/config/h2_head_adjust.env); the hand backend belongs to `h2_deploy.env`. The PICO feedback port is **5559**. For multi-machine operation, explicitly bind the relevant runtime sockets to the network interface and pass matching endpoints to the bridge. Synchronize system clocks; commands carry wall-clock timestamps and expire.

## Independent head and hand commands

```bash
# Radians; this changes only the two head joints.
.venv_inputs/bin/python -m gear_sonic.sources.peripherals head --pitch 0.3 --yaw -0.2
# Seven Dex3-compatible values; Inspire maps this representation to finger closure.
.venv_inputs/bin/python -m gear_sonic.sources.peripherals hands --left '[0,0,0,0,0,0,0]'
```

Each process refreshes its own command until interrupted. Head targets are clamped and slew-limited; expiry returns the head to neutral. Expired hand input stops hand publishing without generating a new grip command. Inspire requires a released trigger before accepting closure. Body-source changes do not reset these independent channels.

## Validation

```bash
sudo apt install libgtest-dev
bash scripts/build_controller.sh -DBUILD_TESTING=ON
ctest --test-dir gear_sonic_deploy/build --output-on-failure
source gear_sonic_deploy/scripts/setup_env.sh
.venv_sim/bin/python tools/check_runtime.py
.venv_sim/bin/python tools/check_runtime.py --planner
```

The integration check runs the actual models with synthetic robot state and private IPC sockets. It does not initialize DDS or connect to hardware. It tests startup standing, all body input formats, expiry, source ownership and independent head/hand commands. See [validation records](../release/REPRODUCIBILITY.md) for the scope of completed checks.
