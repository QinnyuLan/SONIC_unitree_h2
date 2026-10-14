# Unitree H2 部署指南

[中文首页](../README_CN.md) · [English](DEPLOYMENT.md)

本文介绍发布模型在 MuJoCo、H2 实机和 PICO VR 遥操作中的使用方法。除非另有说明，命令均在仓库根目录执行。

## 环境配置

使用配备 NVIDIA GPU 的 Ubuntu 工作站。Python 环境由 [uv](https://docs.astral.sh/uv/getting-started/installation/) 创建，版本为 Python 3.10。

| 组件 | 验证环境 |
|---|---|
| Python | 3.10.20 |
| CUDA | 12.6 |
| TensorRT | 10.13.3.9 |
| ONNX Runtime C++ | 1.16.3 |

Python 依赖定义在 [`gear_sonic/pyproject.toml`](../gear_sonic/pyproject.toml)。原生部署依赖和平台要求见[依赖说明](../requirements/README.md)及[部署安装参考](NATIVE_SETUP.md)。

```bash
git clone --depth 1 https://github.com/QinnyuLan/SONIC_unitree_h2.git
cd SONIC_unitree_h2
python3 tools/verify_release.py
```

### 编译 C++ 控制器

先安装 CUDA、TensorRT 和 ONNX Runtime C++。检查脚本默认从 `/opt/onnxruntime` 查找 ONNX Runtime 头文件与动态库；`TensorRT_ROOT` 应指向 TensorRT 的安装目录，例如 `$HOME/TensorRT`。

Ubuntu 编译依赖：

```bash
sudo apt update
sudo apt install build-essential cmake pkg-config libzmq3-dev \
  libeigen3-dev libmsgpack-dev libyaml-cpp-dev nlohmann-json3-dev cppzmq-dev zlib1g-dev
```

编译控制器：

```bash
export TensorRT_ROOT="$HOME/TensorRT"
./verify_deploy.sh
bash scripts/build_controller.sh
```

下述三个流程均使用该控制器。TensorRT engine 会在目标机器上根据 ONNX 模型生成。每个启动控制器的终端都需要设置 `TensorRT_ROOT`。

## Sim2Sim

安装仿真环境：

```bash
./install.sh sim2sim
source .venv_sim/bin/activate
python3 tools/verify_release.py --imports --sim
```

终端 1，在仓库根目录启动仿真：

```bash
source .venv_sim/bin/activate
bash scripts/run_h2_paper_sim.sh
```

终端 2，同样从仓库根目录启动控制器：

```bash
export TensorRT_ROOT="$HOME/TensorRT"
bash gear_sonic_deploy/scripts/run_h2_sim2sim.sh --print-command
bash gear_sonic_deploy/scripts/run_h2_sim2sim.sh
```

仿真和控制器均使用 DDS domain `0`。控制器加载 `models/released`、`reference/example` 动作集及 nominal Kp/Kd。仿真配置见 [`h2_native_sonic_unitree_ros_loop.yaml`](../gear_sonic/utils/mujoco_sim/wbc_configs/h2_native_sonic_unitree_ros_loop.yaml)。

## Sim2Real

在 [`h2_deploy.env`](../gear_sonic_deploy/config/h2_deploy.env) 中设置机器人网卡、DDS domain、动作集与关节增益。默认网卡名为 `eno1`，应修改为实际连接机器人的网卡。启动控制前确认急停和支撑措施。

```bash
export TensorRT_ROOT="$HOME/TensorRT"
./verify_deploy.sh
bash gear_sonic_deploy/scripts/run_h2_sim2real_four_kp.sh --print-command
bash gear_sonic_deploy/scripts/run_h2_sim2real_four_kp.sh
```

脚踝 pitch、脚踝 roll 和膝盖的增益缩放分别由配置文件控制。该入口使用动作回放，不需要 planner。实机实验前应先完成仿真验证。

<a id="planner"></a>

## Planner 下载

VR 部署使用 [nvidia/GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC) 提供的共享 G1 planner：

```bash
uv venv .venv_download --python 3.10
uv pip install --python .venv_download/bin/python -r requirements/download.txt
.venv_download/bin/python download_from_hf.py --planner-only
python3 tools/verify_release.py --require-planner
```

下载目标为 `gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx`，与 VR 启动脚本一致。脚本固定上游版本并校验 SHA256，已有正确文件时直接复用。该权重由 Git 忽略；校验值、缓存位置和手动下载方法见 [planner 下载说明](PLANNER.md)。

<a id="vr-teleoperation"></a>

## VR 遥操作

先完成控制器编译与 planner 下载，再安装 Python 环境，并按照 [VR 设备设置指南](VR_SETUP.md)启动 XRoboToolkit PC Service、配对 PICO 头显。

```bash
./install.sh teleop
source .venv_teleop/bin/activate
```

终端 1，在仓库根目录启动 PICO 输入进程：

```bash
bash gear_sonic_deploy/scripts/run_h2_teleop_head_adjust.sh pico --print-command
bash gear_sonic_deploy/scripts/run_h2_teleop_head_adjust.sh pico
```

终端 2，同样从仓库根目录启动机器人控制器：

```bash
export TensorRT_ROOT="$HOME/TensorRT"
bash gear_sonic_deploy/scripts/run_h2_teleop_head_adjust.sh deploy --print-command
bash gear_sonic_deploy/scripts/run_h2_teleop_head_adjust.sh deploy
```

`deploy` 角色控制实机。网卡与增益由 [`h2_deploy.env`](../gear_sonic_deploy/config/h2_deploy.env) 配置，头部与手部由 [`h2_head_adjust.env`](../gear_sonic_deploy/config/h2_head_adjust.env) 配置，默认使用 Inspire 灵巧手与固定头部朝向。

## 模型与验证

三个入口均使用 `models/released`。发布权重是直接微调后的 ONNX，无需合并适配器。接口与文件校验值见[模型说明](../models/released/README.md)。

```bash
# 文件和模型校验，不连接设备。
python3 tools/verify_release.py
# 同时要求 planner 存在且校验通过。
python3 tools/verify_release.py --require-planner
# 编译后，从测试所需的工作目录执行 C++ 测试。
sudo apt install libgtest-dev
bash scripts/build_controller.sh -DBUILD_TESTING=ON
ctest --test-dir gear_sonic_deploy/build --output-on-failure
```

CPU/GPU 推理及 C++ 测试结果见[复现记录](../release/REPRODUCIBILITY.md)。这些检查不替代新环境中的实时仿真和实机稳定性验证。可选资源与下载排除项见[仓库维护说明](LAYOUT.md)。
