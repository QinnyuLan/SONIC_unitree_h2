# Unitree H2 部署

[README](../README_CN.md) · [English](DEPLOYMENT.md) · [故障排查](TROUBLESHOOTING_CN.md)

所有命令均在仓库根目录执行。仿真使用三个终端：**仿真器 → policy runtime → 输入端**；实机只需 runtime 和输入端两个终端。

- [准备环境](#准备环境)
- [Sim2Sim：动作回放或键盘 planner](#sim2sim)
- [如何测试你自己的动作](#如何测试你自己的动作)
- [Planner](#planner)
- [Sim2Real：实机启动](#sim2real)
- [VR 遥操作](#vr-遥操作)
- [独立头部和手部控制](#独立头部和手部控制)

## 准备环境

先按[原生依赖说明](NATIVE_SETUP.md)安装系统依赖、CUDA 和 TensorRT，并安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)。已验证环境为 Ubuntu x86_64、Python 3.10.20、CUDA 12.6 和 TensorRT 10.13.3.9。

随后编译 runtime，安装回放和仿真环境：

```bash
git submodule update --init --recursive
export TensorRT_ROOT="$HOME/TensorRT"
export CUDAToolkit_ROOT=/usr/local/cuda-12.6
bash scripts/build_controller.sh
./install.sh inputs
./install.sh sim2sim
python3 tools/validation/verify_release.py
```

将两个 SDK 路径改为本机实际安装位置，并在各启动终端中设置。编译产物为 `build/runtime/bin/sonic_h2_runtime`；启动脚本自动选择 `models/sonic_h2_released/` 中的 H2 模型。以下命令使用各环境的 Python 完整相对路径，无需手动激活环境。

需要 planner 时，继续完成 [planner 安装](#planner)；需要 VR 时，完成 [VR 配置](#vr-遥操作)。安装报错、GPU 启动异常等问题见[故障排查](TROUBLESHOOTING_CN.md)。

## Sim2Sim

### 终端 1：启动仿真器

```bash
bash scripts/run_h2_paper_sim.sh
```

runtime 初始化期间保持弹力带支撑。

### 终端 2：启动并开启 policy runtime

```bash
bash scripts/run_sonic_h2_runtime.sh sim
```

1. 等待 **[READY]**。收到机器人状态后，runtime 用三秒过渡到配置的站姿，再通过 PD 保持姿态。
2. 切换到**终端 2**，按 **]** 或 **s** 开启 policy，无需回车。**[READY]** 前按下的启动键会被忽略。
3. policy 运行后，点击**仿真窗口**，按 **9** 解除弹力带。
4. 确认无支撑站立稳定后，再发送动作。

PD 等待阶段只保持关节姿态，第 2 步之后才启用 policy 平衡控制。仿真器与 runtime 默认使用 loopback DDS domain 0；如需更换，在终端 1、2 启动前设置相同的 `DDS_DOMAIN`。

### 终端 3：选择一种输入

#### 动作回放

```bash
.venv_inputs/bin/python -m interface.motion_replay --keyboard
```

| 终端 3 中的按键 | 操作 |
|---|---|
| `1` / `2` / `3` / `4` | 选择 G1 / G1 探索 / H2 / SMPL 集合 |
| 上 / 下方向键 | 选择动作，不自动播放 |
| Enter | 从头播放选中的动作 |
| 空格 | 暂停 / 继续 |
| `n` / `p` | 切换并播放下一条 / 上一条 |
| `x` | 停止并回到开头，机器人返回站立 |
| `q` / Esc / Ctrl+C | 退出回放 |

暂停、播放结束和退出都会释放身体输入，由 policy 返回站立。循环播放和其他按键见[接口说明](../src/interface/README.md#keyboard-playback)。

##### 如何测试你自己的动作

按[动作格式说明](../motions/README.md)将 G1、原生 H2 或 SMPL 动作导出为 CSV 动作目录或 NPZ 文件。将下面路径替换为自己的文件或动作目录，先检查数据：

```bash
MOTION=/absolute/path/to/your_motion.npz
.venv_inputs/bin/python -m interface.motion_replay "$MOTION" --inspect
```

`--inspect` 不连接 runtime。检查通过后，保持终端 1、2 的仿真器和 policy 运行，确认站立稳定。退出已有的键盘回放或 planner 程序，再在终端 3 执行：

```bash
.venv_inputs/bin/python -m interface.motion_replay "$MOTION"
```

命令执行后立即开始回放。动作结束或按 **Ctrl+C** 后释放身体输入，由 policy 返回站立。先在仿真中确认跟踪稳定，再用于实机。

`--speed` 调整播放速度，`--fps` 覆盖源帧率；缺少明确类型元数据时，可用 `--type g1|h2|smpl` 指定类型。`--loop` 仅用于首尾连续的动作。关节顺序、单位和必需字段见[动作格式说明](../motions/README.md)。

#### Planner

首次使用前，安装环境并下载模型：

```bash
uv venv .venv_download --python 3.10
uv pip install --python .venv_download/bin/python -r requirements/download.txt
.venv_download/bin/python tools/model_download/download_planner.py
./install.sh planner
python3 tools/validation/verify_release.py --require-planner
```

模型路径为 `models/motion_planner_g1/planner_sonic.onnx`。保持终端 1、2 运行，退出回放后，在终端 3 启动 planner：

```bash
bash scripts/run_planner.sh --keyboard
```

| 终端 3 中的按键 | 操作 |
|---|---|
| `0` / `1` / `2` / `3` | 选择静止 / 慢走 / 行走 / 跑步 |
| 按住 `w` / `s` | 前进 / 后退 |
| 按住 `a` / `d` | 左移 / 右移 |
| `q` / `e` | 向左 / 向右调整朝向 |
| `+` / `-` | 加速 / 减速 |
| Enter | 激活 / 取消激活所选模式 |
| 空格或 `x` | 清除请求，返回站立 |
| Esc / Ctrl+C | 退出面板和 planner 工作进程 |

等 planner 和 runtime 均就绪后再按移动键。仅选择模式或速度不会启动已停止的 planner。先以慢走模式和低速测试，完整按键见[planner 控制说明](../src/interface/README.md#planner-keyboard-control)，下载细节和程序化请求见 [planner 文档](PLANNER.md)。

### 切换输入与结束运行

切换输入时，先停止当前动作，退出终端 3 中的输入程序，再启动另一输入端。终端 1、2 保持运行。runtime 连接中断后，回放会暂停，planner 请求会清空；连接恢复后需要手动重新操作。

结束运行时：

1. 在回放中按 **x**，或在 planner 中按**空格**，使机器人回到站立，再退出输入端。
2. 在仿真窗口按 **9** 重新挂上弹力带。
3. 在**终端 2** 按 **o**，进入阻尼并锁定停止；恢复控制需重启 runtime。**Ctrl+C** 也会退出 runtime。
4. 最后关闭仿真器。

阻尼模式不维持平衡；没有支撑时停止 runtime，机器人可能倒下。

## Sim2Real

实机测试前先阅读[风险说明与分阶段验证要求](SIM2REAL_SAFETY_CN.md)。实机控制主机无需启动仿真器。

1. 在 [`config/h2_deploy.env`](../config/h2_deploy.env) 中配置连接机器人的 `NETWORK_INTERFACE`、`DDS_DOMAIN`、`HAND_BACKEND` 和关节增益比例。使用 Inspire 灵巧手时，先部署 [PC2 手部服务](INSPIRE_HANDS_CN.md)。
2. 准备好机器人支撑与可立即操作的急停。在**终端 1** 预览配置，再启动：

```bash
bash scripts/run_sonic_h2_runtime.sh real --print-command
bash scripts/run_sonic_h2_runtime.sh real
```

3. 核对输出的网卡和 DDS domain，输入 **y** 并回车开始控制；直接回车或输入其他内容则取消。
4. 等待 **[READY]**，在 runtime 终端按 **]** 或 **s** 开启 policy。
5. 确认站立稳定后，打开**终端 2**，使用[上面的输入端命令](#终端-3选择一种输入)。

按 **o** 进入阻尼并停止，停止前先支撑好机器人。恢复控制需重启 runtime。

## VR 遥操作

安装遥操作环境，并完成 [PICO 连接与标定配置](VR_SETUP.md)：

```bash
./install.sh teleop
```

按上面的仿真或实机流程启动 runtime 并开启 policy。以下进程各占一个终端：

```bash
# PICO 设备读取与标定
bash scripts/run_h2_teleop_head_adjust.sh pico
```

```bash
# 发送身体、头部和手部指令
bash scripts/run_h2_teleop_head_adjust.sh bridge
```

需要摇杆行走时，先完成 [planner 安装](#planner)，再启动：

```bash
bash scripts/run_h2_teleop_head_adjust.sh planner
```

没有 planner 时，VR 使用站立下肢参考。头部设置位于 [`config/h2_head_adjust.env`](../config/h2_head_adjust.env)，实机手部后端位于 `config/h2_deploy.env`。头手指令通过独立通道发送；跨机器 endpoint 配置见[架构说明](ARCHITECTURE.md)。

## 独立头部和手部控制

runtime 已运行时，按需在各自终端启动：

```bash
# 头部 pitch/yaw，单位 rad
.venv_inputs/bin/python -m interface.peripherals.client head --pitch 0.3 --yaw -0.2
# 7 维 Dex3 兼容目标，Inspire 将其映射为手指闭合度
.venv_inputs/bin/python -m interface.peripherals.client hands --left '[0,0,0,0,0,0,0]'
```

各客户端持续刷新目标，直到被中断。头部超时后回中；手部超时后停止发布，不生成新的开合指令。身体来源切换不重置头手通道。Inspire 接受闭合指令前，需先松开扳机。

## 配置与检查

- [站姿配置](../src/runtime/README.md#standing-pose-configuration)：参考角度与自定义站姿。
- [故障排查](TROUBLESHOOTING_CN.md)：环境、编译、GPU 启动和连接异常。
- [验证与复现](REPRODUCIBILITY.md)：离线测试、模型检查及已完成验证的范围。
