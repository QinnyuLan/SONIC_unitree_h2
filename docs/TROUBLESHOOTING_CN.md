# 部署故障排查

[部署指南](DEPLOYMENT_CN.md) · [English](TROUBLESHOOTING.md) · [原生依赖](NATIVE_SETUP.md)

安装或启动报错时使用本页。所有命令均在仓库根目录执行；正常操作顺序见[部署指南](DEPLOYMENT_CN.md)。

| 现象 | 对应章节 |
|---|---|
| 找不到 Python 模块、`uv` 或 TensorRT wheel | [Python 环境](#python-环境) |
| CMake 编译报错或缺少动态库 | [原生编译与动态库](#原生编译与动态库) |
| 模型缺失或校验失败 | [模型与引擎缓存](#模型与引擎缓存) |
| 启动缓慢、CUDA 初始化卡住、Ctrl+C 无效 | [GPU 启动](#gpu-启动) |
| runtime 或输入面板一直等待，指令不生效 | [连接与操作](#连接与操作) |
| MuJoCo 无法打开窗口或因延迟退出 | [仿真器](#仿真器) |

## Python 环境

每个进程使用对应环境中的解释器：

| 进程 | Python 路径 | 安装或修复 |
|---|---|---|
| 回放及指令客户端 | `.venv_inputs/bin/python` | `./install.sh inputs` |
| MuJoCo 仿真器 | `.venv_sim/bin/python` | `./install.sh sim2sim` |
| Planner | `.venv_planner/bin/python` | `./install.sh planner` |
| PICO 与遥操作适配端 | `.venv_teleop/bin/python` | `./install.sh teleop` |

出现 `ModuleNotFoundError` 时，重新运行对应安装命令，并使用该环境的解释器启动。向其他系统或 Conda 环境安装包不会修复这些环境。缺少 `uv` 时，先完成[依赖安装说明](../requirements/README.md)。

以下检查不启动控制：

```bash
uv pip check --python .venv_inputs/bin/python
.venv_inputs/bin/python -m interface.motion_replay --help
```

如果安装提示找不到 TensorRT Python wheel，将 `TensorRT_ROOT` 指向解压后的 SDK。安装脚本在 `$TensorRT_ROOT/python` 下寻找 Python 3.10 wheel；文件位于其他位置时，设置 `TENSORRT_WHEEL=/absolute/path/to/the-matching-wheel.whl`。Python 绑定与原生 SDK 的版本应一致，详见[原生依赖说明](NATIVE_SETUP.md#tensorrt-python-planner)。

如果缺少 Unitree 或 XRoboToolkit 源码，初始化仓库固定的子模块后，重新运行对应安装命令：

```bash
git submodule update --init --recursive
```

## 原生编译与动态库

CMake 找不到 CUDA、TensorRT、ZeroMQ、Eigen 或 JSON 头文件时，先补齐[原生依赖](NATIVE_SETUP.md)。在当前终端设置实际 SDK 路径：

```bash
export TensorRT_ROOT="$HOME/TensorRT"
export CUDAToolkit_ROOT=/usr/local/cuda-12.6
source scripts/setup_env.sh
bash tools/validation/check_native.sh
bash scripts/build_controller.sh
```

并行编译过程中进程被系统终止时，可用 `BUILD_JOBS=2 bash scripts/build_controller.sh` 降低并行度。

启动时出现 `libnvinfer.so: cannot open shared object file` 等错误，可加载同一套环境后检查动态库解析结果：

```bash
source scripts/setup_env.sh
ldd build/runtime/bin/sonic_h2_runtime
```

根据实际安装位置修复标为 `not found` 的条目。启动脚本会自动加载 `scripts/setup_env.sh`；直接运行二进制或 Python planner 时也需要相同环境。更换原生 SDK 后重新编译。

## 模型与引擎缓存

检查模型文件、manifest 和哈希：

```bash
python3 tools/validation/verify_release.py
# 同时要求可选 planner 模型存在：
python3 tools/validation/verify_release.py --require-planner
```

H2 runtime 从 `models/sonic_h2_released/` 读取 encoder、decoder 和 `deployment_manifest.json`。模型检查失败时，应恢复配套的发布文件，不要修改记录的哈希以跳过检查。

Planner 权重需要单独下载，路径为 `models/motion_planner_g1/planner_sonic.onnx`。按[下载说明](PLANNER.md)获取；把 Hugging Face 网页保存成 ONNX 文件会导致校验失败。

TensorRT 缓存在模型目录中生成，兼容时直接复用；缺失或不兼容时会在本机构建，首次可能需要数分钟。先看日志再处理缓存，CUDA 驱动等待不能通过删除缓存解决。手动处理某个进程的缓存前应停止该进程，并保留 ONNX 和 manifest 文件。

## GPU 启动

### 先区分启动阶段

Runtime 日志分别显示 CUDA 设备初始化、TensorRT 缓存复用或引擎构建，以及 CUDA graph 捕获。两个 policy 模型就绪后才启动 DDS 控制。Planner 键盘模式的工作进程日志位于 `outputs/planner/keyboard-*.log`。

| 最后显示的阶段 | 检查方向 |
|---|---|
| CUDA 设备初始化 | 长时间不返回时，按下文检查进程状态 |
| TensorRT 缓存缺失或不兼容，正在构建 | 等待首次构建完成，观察日志 |
| CUDA 内存分配或引擎报错 | 根据错误检查 SDK 路径和可用 GPU 资源 |
| 模型已加载，但没有 **[READY]** | 按[连接与操作](#连接与操作)检查 DDS 状态接收 |

模型初始化阶段按 **Ctrl+C** 直接终止 runtime；初始化完成后，通过控制循环执行正常退出。Planner 键盘模式按 **Esc** 或 **Ctrl+C** 退出面板及其工作进程。

### CUDA 初始化长期不返回

如果一直停在 `Initializing CUDA device ...`，先从另一终端查看进程，不要立即再次发起 GPU 查询：

```bash
ps -C sonic_h2_runtime,nvidia-smi -o pid,stat,etime,wchan:40,args
```

持续处于 `D` 状态，且等待位置为 `uvm_gpu_retain_by_uuid`、`uvm_va_space_destroy` 等，说明进程在 NVIDIA 驱动内部等待。信号可能要等内核调用返回后才能处理；不可中断等待中的进程即使收到 `kill -9`，也无法立即退出。这与 TensorRT 引擎构建不同，此时不要反复启动 runtime 或 `nvidia-smi`。

驱动持续阻塞时，先保存其他工作，再安排重启主机。GPU reset 要求设备没有活跃客户端，且不保证能恢复已卡住的 GPU，参见 [NVIDIA 的恢复条件](https://docs.nvidia.com/deploy/nvidia-smi/index.html#r-gpu-reset)。恢复后，先确认 `nvidia-smi` 正常返回，再重新启动部署流程。删除模型缓存不能解决驱动等待问题。

## 连接与操作

### Runtime 一直未显示 [READY]

先启动仿真器。仿真器和 runtime 的 DDS domain 必须一致，默认均使用 loopback 和 domain 0；需要修改时，在两个终端启动前设置相同的 `DDS_DOMAIN`。

实机检查 `config/h2_deploy.env` 中的 `NETWORK_INTERFACE` 和 `DDS_DOMAIN` 是否与机器人连接一致。以下命令查看网卡并预览启动参数，不启动控制：

```bash
ip -brief address
bash scripts/run_sonic_h2_runtime.sh real --print-command
```

仿真保持文档中的 loopback 配置，不要将仿真和实机控制放在同一个机器人网络的 DDS 域中运行。

### 输入端等待或动作未启动

- **Policy 还在等待：** **[READY]** 表示 PD 保持。在此提示之后，去 runtime 终端按 **]** 或 **s**；过早按键会被忽略。
- **回放只选中动作：** 选集合或动作不会自动播放，在回放终端按 Enter。
- **Planner 只选中模式：** 等工作进程就绪后按移动键或 Enter；仅修改模式和速度不激活请求。
- **身体通道被其他输入端占用：** 先停止并退出之前的回放、planner 或遥操作进程，再启动新的输入端。
- **Runtime 状态曾中断：** 回放保持暂停，planner 请求已清空；恢复连接后需要手动继续。
- **已按 o 停止 runtime：** 该停止会锁定，须重启 runtime，等待 **[READY]** 后重新开启 policy。

### 端口被占用或收不到状态

用 `ss -ltnp` 查看监听进程。旧 runtime 或 planner 可能仍占用端口，应关闭属于上一会话的进程；不要对同一机器人同时运行两个控制器。

Runtime 默认监听 `127.0.0.1`：身体 **5558**、状态 **5559**、头部 **5562**、手部 **5563**；planner 请求和输出分别为 **5560**、**5561**。跨机器运行时，需要显式设置 runtime 监听地址和对应客户端 endpoint，并同步系统时间，因为指令按时间戳过期。详见[端口与协议](ARCHITECTURE.md)。

## 仿真器

MuJoCo 导入或资源加载失败时，先运行离线检查：

```bash
.venv_sim/bin/python tools/validation/verify_release.py --imports --sim
```

该检查加载资源并执行物理步，不启动 DDS 或图形窗口。若检查通过但窗口无法打开，检查图形会话和[原生依赖](NATIVE_SETUP.md)中的 OpenGL 包；远程终端需要可用的显示环境才能打开 viewer。

如果仿真器因实时延迟保护退出，查看报告的延迟和主机负载。先完成首次引擎构建、关闭多余负载，再重新启动仿真器与 runtime。离线检查不能证明实时运行性能，不应通过放宽保护阈值认定控制测试通过。

PICO 连接和标定问题见 [VR 配置](VR_SETUP.md)。机器人端手部服务日志、串口映射与看门狗恢复见[手部服务指南](INSPIRE_HANDS_CN.md)。
