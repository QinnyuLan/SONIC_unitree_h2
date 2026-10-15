# Unitree H2 部署

[README](../README_CN.md) · [English](DEPLOYMENT.md) · [架构与协议](ARCHITECTURE_CN.md)

所有命令在仓库根目录执行。先启动 H2 policy runtime，使机器人以默认姿态站立，再独立启动动作回放、planner 或遥操作发送端。

## 安装

如果此前没有递归克隆，先运行 `git submodule update --init --recursive`。上游依赖固定在仓库记录的提交，详见[第三方依赖](../third_party/README.md)。

验证环境为 Ubuntu x86_64、Python 3.10.20、CUDA 12.6、TensorRT 10.13.3.9。CUDA 和 TensorRT 安装见 [原生依赖](NATIVE_SETUP.md)。

```bash
sudo apt install build-essential cmake pkg-config libzmq3-dev cppzmq-dev \
  libeigen3-dev nlohmann-json3-dev
export TensorRT_ROOT="$HOME/TensorRT"
bash scripts/build_controller.sh
python3 tools/validation/verify_release.py
```

编译生成 `build/runtime/bin/sonic_h2_runtime`。runtime 启动读取发布的 encoder / decoder 及其 `deployment_manifest.json`。planner 为独立 Python 服务。首次运行会在本机生成 TensorRT 缓存。

安装 [uv](https://docs.astral.sh/uv/getting-started/installation/) 后，按需选择 Python 环境：

```bash
./install.sh inputs    # .venv_inputs：回放及指令发送
./install.sh planner   # .venv_planner：TensorRT Python planner
./install.sh sim2sim   # .venv_sim：MuJoCo + DDS，也可执行回放
./install.sh teleop    # .venv_teleop：PICO、姿态转换与运动学
./install.sh text2motion # .venv_text2motion：可选 ECHO-G 离线生成器
```

回放和仿真无需 PyTorch。遥操作使用 CPU PyTorch 完成姿态数学计算。

## Sim2Sim

分别打开三个终端，均进入仓库根目录。

终端 1：启动仿真器。

```bash
bash scripts/run_h2_paper_sim.sh
```

终端 2：启动 SONIC policy runtime。

```bash
bash scripts/run_sonic_h2_runtime.sh sim
```

runtime 用三秒从测量姿态过渡到配置的站姿，显示 **[READY]** 后通过 PD 保持姿态，policy 尚未启用。在**终端 2** 按 **]** 或 **s** 开启 policy，无需回车；初始化结束前按下的启动键会被忽略，需要在就绪后重新按键。按 **o** 停止 policy 并进入阻尼模式，停止后需重新启动 runtime，不能直接按启动键恢复。两端默认使用 loopback、DDS domain 0；更换 domain 时，在两个终端设置相同的 `DDS_DOMAIN`。站立策略开始运行后，点击仿真窗口并按 **9** 解除弹力带，观察无支撑站立，再发送动作指令。

终端 3：启动交互式 motion replay。

```bash
.venv_inputs/bin/python -m interface.motion_replay --keyboard
```

按 **1 / 2 / 3 / 4** 选择 G1 / G1 探索 / H2 / SMPL 集合，用 **上下方向键** 选择动作，按 **Enter** 播放。**空格** 暂停/继续；**n/p** 切换并播放下一条/上一条；**x** 停止并回到动作开头。仅选择动作不会自动播放。暂停、播放结束或退出时释放身体指令，runtime 回到站立。**q**、**Esc** 或 **Ctrl+C** 退出回放界面。

使用 planner 时，保持终端 1、2 运行，退出 replay，将终端 3 改为：

```bash
bash scripts/run_planner.sh --keyboard
```

该终端同时运行 TensorRT planner 工作进程和键盘面板。**1 / 2 / 3** 选择慢走 / 行走 / 跑步，**0** 选择静止；**W/S** 前进/后退，**A/D** 左右平移，**Q/E** 转向，**+/-** 调整速度。**空格** 停止并返回站立，**Esc** 或 **Ctrl+C** 退出面板及 planner 工作进程。完整按键及全部模式切换见[接口说明](../src/interface/README.md#planner-keyboard-control)。

两个输入面板均等待 runtime 的新鲜状态后才发送动作。状态中断后，回放会暂停且不会自动恢复，planner 会清除键盘指令。退出输入端后，仿真器和 runtime 继续运行。planner 需要单独安装环境、下载权重，首次构建引擎可能需要数分钟。安装与测试范围见 [planner 配置](PLANNER.md)和[验证记录](REPRODUCIBILITY.md)。

### 站姿配置与 runtime 按键

初始化以及 **[READY]** 阶段保持弹力带支撑。此时 PD 只保持关节姿态，按 **]** 或 **s** 后才开启 policy 平衡控制。按键在 runtime 所在终端生效，无需回车。**o** 将位置增益设为零、阻尼设为 8，停止控制并要求重启 runtime；无支撑的机器人可能在停止后倒下。

站姿参考位于 [`config/runtime/standing_pose.json`](../config/runtime/standing_pose.json)。双腿统一采用：髋俯仰 **−0.10 rad**、膝关节 **0.25 rad**（约 **14°**）、踝俯仰 **−0.15 rad**，减小屈膝幅度并保留轻微弯曲。双臂位于身体两侧，肘部保留自然弯曲：肩俯仰 **0.25 rad**、左右肩侧摆分别为 **+0.10/−0.10 rad**、肘关节 **1.00 rad**。这组肩肘参考经过 policy 闭环验证，在保持双手位于身体两侧的同时减小躯干前倾。H2 的肘关节坐标带有几何偏置，零角度对应前臂朝前。该参考同时用于初始化、PD 等待，以及没有身体输入时的 policy 站立；仿真器的初始姿态与此预设一致。头部仍独立接收指令。

可以通过 `--standing-pose /absolute/path/to/standing_pose.json` 指定其他参考，角度单位为 rad。`arm_joint_angles_rad` 中的 `shoulder_roll` 为左肩角度，右肩取相反数；省略该配置段时，手臂沿用模型的默认参考，以兼容原有仅配置腿部的文件。调整作用于输入 policy 的参考姿态，decoder 训练时的关节基准角与动作缩放保持原定义。修改参考后，先验证仿真中的站立和动作跟踪，再用于实机。

### Runtime 启动故障排查

启动日志会分别显示 CUDA 设备初始化、TensorRT 缓存复用或引擎构建，以及 CUDA graph 捕获。缓存缺失或不兼容时，构建引擎可能需要数分钟；兼容的缓存会直接复用。两个模型全部就绪后才启动 DDS 控制。模型初始化阶段按 **Ctrl+C** 直接终止进程；初始化完成后则通过控制循环执行正常退出。

如果长期停在 `Initializing CUDA device ...`，可在另一终端查看进程，不需要再次发起 GPU 查询：

```bash
ps -C sonic_h2_runtime,nvidia-smi -o pid,stat,etime,wchan:40,args
```

如果进程持续处于 `D` 状态，等待位置为 `uvm_gpu_retain_by_uuid`、`uvm_va_space_destroy` 等，说明正在 NVIDIA 驱动内部等待。信号可能要等内核调用返回后才能处理；不可中断等待中的进程，即使执行 `kill -9` 也无法立即结束。这种情况并非 TensorRT 引擎编译，不要反复启动 runtime 或 `nvidia-smi`。

如果驱动持续阻塞，先保存其他工作，再安排重启主机。GPU reset 要求设备没有活跃客户端，且不保证能恢复已卡住的 GPU，参见 [NVIDIA 的恢复条件](https://docs.nvidia.com/deploy/nvidia-smi/index.html#r-gpu-reset)。恢复后，先确认 `nvidia-smi` 能正常返回，再重新启动三个终端。删除模型缓存不能解决驱动等待问题。

## 动作回放与离线 text2motion

需要语音与配套手势时，参见 [ECHO-G text2motion 指南](TEXT2MOTION_CN.md)：包含中英文示例、离线生成，以及通过 `interface.motion_replay` 同时播放动作与本机或 Unitree 扬声器音频。仓库内的语音动作示例只需安装 inputs 环境即可回放。

统一使用 `interface.motion_replay`，自动识别 **G1、H2、SMPL** 动作。输入可以是单个 CSV 动作目录或 NPZ；离线 text2motion 结果也通过此入口发送。

| `motions/` 下的目录 | 输入类型 | 数量 |
|---|---|---:|
| `example_g1_replay` | G1 基础动作 | 13 |
| `example_g1_explore_replay` | G1 探索动作 | 7 |
| `example_h2_replay` | H2 原生动作 | 8 |
| `example_smpl_replay` | SMPL 动作 | 8 |

```bash
.venv_inputs/bin/python -m interface.motion_replay \
  motions/example_h2_replay/walking_quip_360_R_002__A428 --inspect
.venv_inputs/bin/python -m interface.motion_replay \
  motions/example_smpl_replay/walking_quip_360_R_002__A428
.venv_inputs/bin/python -m interface.motion_replay generated_motion.npz --inspect
```

`--inspect` 只检查数据，不连接 runtime。类型优先读取集合的 `manifest.json` 和动作元数据；未标注且存在歧义的外部数据使用 `--type g1|h2|smpl`，不能覆盖已有的不同类型声明。`--fps` 覆盖原始帧率；CSV 未记录帧率时默认 50 Hz。`--speed` 同时调整播放速度与关节速度，`--loop` 仅适用于首尾连续的动作。

G1、H2 发送未来 `0.0, 0.1, …, 0.9` 秒的 10 帧；SMPL 发送 `0.00, 0.02, …, 0.18` 秒的 10 帧，以匹配模型输入。H2 读取原生 31 维文件，排除头部并重排为 29 维原生身体关节，不使用其附带的 G1 映射文件。SMPL 使用独立的 SMPL 根旋转、关节点及腕部角度。头部仍通过独立通道控制。

CSV 文件、NPZ 字段与坐标约定见[动作格式说明](../motions/README.md)，发送协议见[架构文档](ARCHITECTURE_CN.md)。可选的 ECHO-G 生成进程独立运行，输出语音动作包，仍通过同一个 replay 入口播放。

## Sim2Real

### 风险说明与分阶段验证

> [!WARNING]
> 实机测试存在跌倒、碰撞、设备损坏和人员受伤风险。任何动作在真实机器人上测试前，都应使用计划部署的 policy、参考动作和控制配置，在仿真中确认能够稳定跟踪。仿真通过是实机测试的前提，不代表实机一定能够稳定执行。

Unitree H2 可能存在较大的仿真到实机差异。应重点检查连杆传动关节，尤其是脚踝的阻尼特性与机械间隙（结构虚位），以及 IMU 标定误差和关节零位偏差。这些差异可能影响姿态估计与关节控制目标，进而改变机器人姿态和重心位置，降低平衡裕度。

实机验证应遵循以下顺序，只有当前阶段能够持续稳定运行，才进入下一阶段：

1. **默认姿态稳定站立：** 先确认机器人能够在默认姿态下保持平衡，再发送动作指令。
2. **低难度动作稳定跟踪：** 从低速、小幅度、平衡要求较低的动作开始，确认跟踪过程稳定。
3. **逐步提高动作难度：** 逐级增加速度、幅度与协调要求，每个阶段均确认稳定跟踪后再继续。

若出现站立或跟踪不稳定，应停止测试，排查并解决机械、标定或控制问题后再继续。简单动作通过不代表更高难度动作已经得到验证；每个待测动作都必须先完成仿真验证。

### 大动态动作测试前提

- 尽可能减小并评估剩余的仿真到实机差异。重点检查脚踝等连杆关节的阻尼与机械间隙，核对 IMU 和关节零位标定；存在尚未解决的机械松旷或标定异常时，不应开展大动态测试。
- 确保场地清空、地面条件适宜，留出覆盖完整动作及可能跌倒范围的空间；人员应处于运动与跌倒区域之外，并准备合适的机器人支撑和可立即操作的急停。
- 测试前将机器人电池充满，并确认供电系统状态正常。
- 拆卸灵巧手及其他非必要、易损的附属机构，并禁用其控制后端；手部设置为 `HAND_BACKEND=none`。拆卸后核对仿真中的质量与惯量设置是否符合实际测试配置，并重新验证站立和低难度动作。

上述条件与分阶段验证要求同时适用。任何条件尚未满足时，应先完成整改，再进行大动态动作测试。

### 启动

在 [`h2_deploy.env`](../config/h2_deploy.env) 设置机器人网卡、DDS domain、手部后端和增益。准备好支撑与急停后运行：

```bash
bash scripts/run_sonic_h2_runtime.sh real --print-command
bash scripts/run_sonic_h2_runtime.sh real
```

启动器显示目标网卡，输入 `y`（也接受 `Y`）并回车后开始控制，并自动过渡到配置的站姿；直接回车或输入其他内容则取消。显示 **[READY]** 后，在 runtime 终端按 **]** 或 **s** 开启 policy；按 **o** 进入阻尼并锁定停止，恢复时需重启 runtime。回放等发送端仍在其他终端独立运行。`HAND_BACKEND=none` 禁用手部 DDS；`inspire` 对接机器人端 Inspire 服务；`dex3` 使用 Unitree Dex3。增益设置作用于初始化和 policy 控制，不改变退出时的阻尼。

使用 Inspire 灵巧手时，先在 H2 PC2 部署仓库中的 `tools/robot_services/inspire_hands/` 服务。串口映射、开机自启动、看门狗和只读验收见[机器人端手部服务指南](INSPIRE_HANDS_CN.md)。该服务独立于身体 policy 运行。

## Planner

下载共享 G1 planner：

```bash
uv venv .venv_download --python 3.10
uv pip install --python .venv_download/bin/python -r requirements/download.txt
.venv_download/bin/python tools/model_download/download_planner.py
```

模型下载到 `models/motion_planner_g1/planner_sonic.onnx`，已被 Git 忽略，详见 [下载说明](PLANNER.md)。

执行 `./install.sh planner` 安装 planner 环境。安装脚本使用 `$TensorRT_ROOT/python` 中与 C++ SDK 配套的 TensorRT Python wheel；文件位于其他位置时设置 `TENSORRT_WHEEL`。

上面的三终端流程中，在终端 3 同时启动键盘面板和推理工作进程：

```bash
bash scripts/run_planner.sh --keyboard
```

需要程序化控制时，使用不带 `--keyboard` 的 `bash scripts/run_planner.sh` 启动服务，再从独立客户端发送规划请求：

```bash
.venv_inputs/bin/python -m interface.planner.client \
  --mode 1 --speed 0.3 --movement 1 0 0
```

Python planner 独立完成 TensorRT 推理、轨迹混合及稀疏参考输出；runtime 只接收结果。请求停止后回到站立。`--publish-only` 只在 5561 端口发布结果，供遥操作适配端组合下肢参考，不直接占用 runtime 身体通道。键盘模式的工作进程日志保存在 `outputs/planner/`。

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
.venv_inputs/bin/python -m interface.peripherals.client head --pitch 0.3 --yaw -0.2
.venv_inputs/bin/python -m interface.peripherals.client hands --left '[0,0,0,0,0,0,0]'
```

头部单位为 rad，指令经过关节限位和速度限制，超时缓慢回中。手部采用 7 维 Dex3 兼容目标；Inspire 将其映射为手指闭合度，须先松开扳机再允许闭合。手部超时停止发布，不生成新的开合动作。身体来源切换不会覆盖头手指令。

## 验证

```bash
sudo apt install libgtest-dev zlib1g-dev
bash scripts/build_controller.sh -DBUILD_TESTING=ON
ctest --test-dir build/runtime --output-on-failure
source scripts/setup_env.sh
.venv_sim/bin/python tools/validation/check_runtime.py
./install.sh planner
.venv_planner/bin/python tools/validation/check_runtime.py --planner
```

集成检查使用实际模型、合成机器人状态和私有 IPC 通道，不启动 DDS 或连接硬件，覆盖默认站立、四种身体输入、超时、来源切换以及头手独立性。已完成检查的范围见 [复现记录](REPRODUCIBILITY.md)。
