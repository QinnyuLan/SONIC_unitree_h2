# Unitree H2 Deployment Guide

[README](../README.md) · [中文](DEPLOYMENT_CN.md)

This guide covers the released H2 controller in MuJoCo, on the robot, and with PICO VR input. Commands run from the repository root unless a different directory is shown.

## Environment

Use an Ubuntu workstation with an NVIDIA GPU. The Python environments use Python 3.10 and are created by [uv](https://docs.astral.sh/uv/getting-started/installation/).

| Component | Validation environment |
|---|---|
| Python | 3.10.20 |
| CUDA | 12.6 |
| TensorRT | 10.13.3.9 |
| ONNX Runtime C++ | 1.16.3 |

Python dependencies are defined in [`gear_sonic/pyproject.toml`](../gear_sonic/pyproject.toml). Native deployment dependencies and platform notes are listed in [requirements](../requirements/README.md) and the [deployment installation reference](source/getting_started/installation_deploy.md).

```bash
git clone https://github.com/QinnyuLan/SONIC_unitree_h2.git
cd SONIC_unitree_h2
python3 tools/verify_release.py
```

### Build the C++ controller

Install CUDA, TensorRT, and ONNX Runtime C++ before building. The verification script expects ONNX Runtime headers and libraries under `/opt/onnxruntime`. Point `TensorRT_ROOT` to the TensorRT installation, for example `$HOME/TensorRT`.

On Ubuntu, install the build dependencies:

```bash
sudo apt update
sudo apt install build-essential cmake pkg-config libzmq3-dev     libeigen3-dev libmsgpack-dev libgtest-dev zlib1g-dev
```

Then build the controller:

```bash
export TensorRT_ROOT="$HOME/TensorRT"
./verify_deploy.sh
cd gear_sonic_deploy
source scripts/setup_env.sh
export HAS_ROS2=0
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
cd ..
```

The three workflows below use this controller. TensorRT engines are built on the target machine from the ONNX files. Set `TensorRT_ROOT` in each terminal that runs the controller.

## Sim2Sim

Install the simulation environment:

```bash
./install.sh sim2sim
source .venv_sim/bin/activate
python3 tools/verify_release.py --imports
```

Start the simulator in terminal 1, from the repository root:

```bash
source .venv_sim/bin/activate
bash scripts/run_h2_paper_sim.sh
```

Start the controller in terminal 2, also from the repository root:

```bash
export TensorRT_ROOT="$HOME/TensorRT"
bash gear_sonic_deploy/scripts/run_h2_sim2sim.sh --print-command
bash gear_sonic_deploy/scripts/run_h2_sim2sim.sh
```

The simulator and controller use DDS domain `0`. The controller loads `models/released`, the `reference/example` motions, and nominal Kp/Kd settings. The simulation configuration is [`h2_native_sonic_unitree_ros_loop.yaml`](../gear_sonic/utils/mujoco_sim/wbc_configs/h2_native_sonic_unitree_ros_loop.yaml).

## Sim2Real

Configure the robot network interface, DDS domain, motion set, and joint gains in [`h2_deploy.env`](../gear_sonic_deploy/config/h2_deploy.env). The supplied interface name is `eno1`; change it to the interface connected to your robot. Confirm the emergency stop and support setup before enabling robot control.

```bash
export TensorRT_ROOT="$HOME/TensorRT"
./verify_deploy.sh
bash gear_sonic_deploy/scripts/run_h2_sim2real_four_kp.sh --print-command
bash gear_sonic_deploy/scripts/run_h2_sim2real_four_kp.sh
```

Ankle pitch, ankle roll, and knee gain scales are configured separately in the environment file. This launcher runs motion playback without the planner. Validate the simulation workflow before starting hardware experiments.

## Planner

VR deployment uses the shared G1 planner from [nvidia/GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC):

```bash
uv venv .venv_download --python 3.10
uv pip install --python .venv_download/bin/python -r requirements/download.txt
.venv_download/bin/python download_from_hf.py --planner-only
python3 tools/verify_release.py --require-planner
```

The downloader writes `gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx`, the path used by the VR launcher. It pins the upstream revision, verifies SHA256, and reuses an existing verified file. This downloaded weight is ignored by Git. See [planner download details](PLANNER.md) for the checksum, cache location, and manual download instructions.

## VR Teleoperation

Complete the controller build and planner download first. Install the Python environment, then set up the XRoboToolkit PC Service and pair the PICO headset using the [VR setup guide](source/getting_started/vr_teleop_setup.md).

```bash
./install.sh teleop
source .venv_teleop/bin/activate
```

Start the PICO input process in terminal 1, from the repository root:

```bash
bash gear_sonic_deploy/scripts/run_h2_teleop_head_adjust.sh pico --print-command
bash gear_sonic_deploy/scripts/run_h2_teleop_head_adjust.sh pico
```

Start the robot controller in terminal 2, from the repository root:

```bash
export TensorRT_ROOT="$HOME/TensorRT"
bash gear_sonic_deploy/scripts/run_h2_teleop_head_adjust.sh deploy --print-command
bash gear_sonic_deploy/scripts/run_h2_teleop_head_adjust.sh deploy
```

The `deploy` role controls the real robot. Network and gain settings come from [`h2_deploy.env`](../gear_sonic_deploy/config/h2_deploy.env); head and hand settings come from [`h2_head_adjust.env`](../gear_sonic_deploy/config/h2_head_adjust.env). The supplied configuration uses the Inspire hand backend and fixed head orientation.

## Models and Validation

All three launchers use `models/released`. These are directly fine-tuned ONNX weights and require no adapter merge. Model interfaces and checksums are documented in the [model README](../models/released/README.md).

```bash
# Check release files and model hashes without connecting to devices.
python3 tools/verify_release.py
# Require and validate the planner as well.
python3 tools/verify_release.py --require-planner
# After building, run the C++ tests from their expected working directory.
cd gear_sonic_deploy
./target/release/run_tests
```

The [reproducibility record](../release/REPRODUCIBILITY.md) documents CPU/GPU inference and C++ test results. These checks do not establish real-time tracking performance or hardware stability for a new setup. Optional assets and download exclusions are described in the [repository maintenance guide](GIT_WORKFLOW.md).
