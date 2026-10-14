# Unitree H2 部署

[README](../README_CN.md) · [English](DEPLOYMENT.md) · [架构与协议](ARCHITECTURE_CN.md)

所有命令在仓库根目录执行。先启动 H2 policy runtime，使机器人以默认姿态站立，再独立启动动作回放、planner 或遥操作发送端。

## 安装

验证环境为 Ubuntu x86_64、Python 3.10.20、CUDA 12.6、TensorRT 10.13.3.9。CUDA 和 TensorRT 安装见 [原生依赖](NATIVE_SETUP.md)。

```bash
sudo apt install build-essential cmake pkg-config libzmq3-dev cppzmq-dev \
  libeigen3-dev nlohmann-json3-dev
export TensorRT_ROOT="$HOME/TensorRT"
bash scripts/build_controller.sh
python3 tools/verify_release.py
```

编译生成 `runtime/build/bin/sonic_h2_runtime`。runtime 启动读取发布的 encoder / decoder 及其 `deployment_manifest.json`。planner 为独立 Python 服务。首次运行会在本机生成 TensorRT 缓存。

安装 [uv](https://docs.astral.sh/uv/getting-started/installation/) 后，按需选择 Python 环境：

```bash
./install.sh inputs    # .venv_inputs：回放及指令发送
./install.sh planner   # .venv_planner：TensorRT Python planner
./install.sh sim2sim   # .venv_sim：MuJoCo + DDS，也可执行回放
./install.sh teleop    # .venv_teleop：PICO、姿态转换与运动学
```

回放和仿真无需 PyTorch。遥操作使用 CPU PyTorch 完成姿态数学计算。

## Sim2Sim

终端 1 启动仿真：

```bash
bash scripts/run_h2_paper_sim.sh
```

终端 2 启动 runtime：

```bash
bash scripts/run_runtime.sh sim
```

runtime 用三秒从测量姿态过渡到默认站姿，随后运行站立策略，等待指令。两端默认使用 loopback、DDS domain 0；更换 domain 时，在两个终端设置相同的 `DDS_DOMAIN`。仿真初始启用弹性支撑，准备好后通过仿真窗口控制解除。

终端 3 发送一个 G1 动作：

```bash
.venv_sim/bin/python -m gear_sonic.sources.motion_replay \
  motions/example/walking_quip_360_R_002__A428
```

动作结束或发送端退出后回到站立。发送端的 `Ctrl+C` 释放其指令通道；runtime 的 `Ctrl+C` 结束控制并发送阻尼指令。

## 动作回放与离线 text2motion

两者统一使用 `motion_replay`。支持单个 SONIC CSV 动作目录，或者包含以下字段的 NPZ：

| 字段 | 形状 | 含义 |
|---|---|---|
| `joint_pos` | `[N,29]` | G1 semantic 顺序的绝对关节角，rad |
| `joint_vel` | `[N,29]`，可选 | rad/s；省略则通过差分计算 |
| `root_quat` | `[N,4]` | 世界坐标系根旋转，wxyz |
| `fps` | 标量 | 原动作帧率 |

CSV 读取 `joint_pos.csv`、可选的 `joint_vel.csv` 和 `body_quat.csv` 中的第一个四元数，默认 50 Hz，其他帧率用 `--fps` 指定。关节顺序见 [协议](ARCHITECTURE.md#joint-order)。

```bash
.venv_inputs/bin/python -m gear_sonic.sources.motion_replay generated_motion.npz --inspect
.venv_inputs/bin/python -m gear_sonic.sources.motion_replay generated_motion.npz
```

`--speed` 同时调整播放速度与关节速度。`--loop` 仅适用于首尾连续的动作。所有 G1 回放均发送未来 `0.0, 0.1, …, 0.9` 秒的稀疏 10 帧。离线 text2motion 结果需先导出为 G1 动作，本仓库不运行文本生成模型。

## Sim2Real

在 [`h2_deploy.env`](../config/h2_deploy.env) 设置机器人网卡、DDS domain、手部后端和增益。准备好支撑与急停后运行：

```bash
bash scripts/run_runtime.sh real --print-command
bash scripts/run_runtime.sh real
```

启动器显示目标网卡，输入 `RUN` 后开始控制，并自动过渡到默认站姿。回放等发送端仍在其他终端独立运行。`HAND_BACKEND=none` 禁用手部 DDS；`inspire` 对接已有 Inspire 服务；`dex3` 使用 Unitree Dex3。增益设置作用于初始化和 policy 控制，不改变退出时的阻尼。

## Planner

下载共享 G1 planner：

```bash
uv venv .venv_download --python 3.10
uv pip install --python .venv_download/bin/python -r requirements/download.txt
.venv_download/bin/python download_from_hf.py --planner-only
```

模型下载到 `models/planner/planner_sonic.onnx`，已被 Git 忽略，详见 [下载说明](PLANNER.md)。

执行 `./install.sh planner` 安装 planner 环境。安装脚本使用 `$TensorRT_ROOT/python` 中与 C++ SDK 配套的 TensorRT Python wheel；文件位于其他位置时设置 `TENSORRT_WHEEL`。

在 runtime 运行时，另开终端启动服务：

```bash
bash scripts/run_planner.sh
```

再开终端发送规划请求：

```bash
.venv_inputs/bin/python -m gear_sonic.sources.planner_command \
  --mode 1 --speed 0.3 --movement 1 0 0
```

Python planner 独立完成 TensorRT 推理、轨迹混合及稀疏参考输出；runtime 只接收结果。请求停止后回到站立。`--publish-only` 只在 5561 端口发布结果，供遥操作适配端组合下肢参考，不直接占用 runtime 身体通道。

## VR 遥操作

完成 [PICO 设备配置](VR_SETUP.md) 并启动仿真或实机 runtime 后，分别在独立终端执行：

```bash
# 设备读取与标定
bash scripts/run_h2_teleop_head_adjust.sh pico
# 转换为身体、头部和手部三路指令
bash scripts/run_h2_teleop_head_adjust.sh bridge
# 可选：摇杆行走和 VR 下肢规划
bash scripts/run_h2_teleop_head_adjust.sh planner
```

全身姿态模式发送 SMPL，PICO 的 VR 模式发送腕部/头部三点目标。`bridge --pose-kind vr` 可让姿态模式也使用三点目标。没有 planner 时，VR 使用站立下肢参考。SMPL 为间隔 20 ms 的 10 帧；实时输入窗口约引入 180 ms 缓冲延迟。

头部 pitch/yaw 和灵巧手走独立通道，不进入身体 policy。头部设置在 [`h2_head_adjust.env`](../config/h2_head_adjust.env)，手部后端在 `h2_deploy.env`。PICO 状态反馈端口改为 **5559**。跨机器时显式设置 runtime 的监听地址与发送端 endpoint，并同步系统时间。

## 独立头部和手部控制

```bash
.venv_inputs/bin/python -m gear_sonic.sources.peripherals head --pitch 0.3 --yaw -0.2
.venv_inputs/bin/python -m gear_sonic.sources.peripherals hands --left '[0,0,0,0,0,0,0]'
```

头部单位为 rad，指令经过关节限位和速度限制，超时缓慢回中。手部采用 7 维 Dex3 兼容目标；Inspire 将其映射为手指闭合度，须先松开扳机再允许闭合。手部超时停止发布，不生成新的开合动作。身体来源切换不会覆盖头手指令。

## 验证

```bash
sudo apt install libgtest-dev zlib1g-dev
bash scripts/build_controller.sh -DBUILD_TESTING=ON
ctest --test-dir runtime/build --output-on-failure
source scripts/setup_env.sh
.venv_sim/bin/python tools/check_runtime.py
./install.sh planner
.venv_planner/bin/python tools/check_runtime.py --planner
```

集成检查使用实际模型、合成机器人状态和私有 IPC 通道，不启动 DDS 或连接硬件，覆盖默认站立、四种身体输入、超时、来源切换以及头手独立性。已完成检查的范围见 [复现记录](../release/REPRODUCIBILITY.md)。
