# SONIC_unitree_h2

这是 SONIC-H2 的可搬迁发布版本，包含 H2 的 sim2sim、sim2real 和 PICO VR 三点遥操作代码、动作参考、机器人模型、部署源码、ONNX 模型和依赖安装脚本。发布目录不依赖上级工作区的源码路径；所有启动脚本都以本目录为根目录解析相对路径。

## 目录与工作流

| 目录 | 内容 |
|---|---|
| `gear_sonic/` | Python WBC、MuJoCo、DDS 桥接、VR/SMPL 遥操作 |
| `gear_sonic_deploy/` | C++ TensorRT/ONNX Runtime 部署端、H2 配置、动作和 planner |
| `models/released/` | 正式发布模型（sim2sim、sim2real、VR 共用） |
| `sonic_h2_native_release/` | 历史模型与实验验证记录 |
| `external_dependencies/` | Unitree SDK、XRoboToolkit Python binding |
| `install_scripts/` | 原始环境安装脚本 |
| `docs/` | 上手、遥操作、部署和实验报告 |
| `release/` | 发布版本、源文件清单和复现记录 |
| `requirements/` | 依赖及平台前置条件 |

## 最短复现路径

### 1. 克隆与检查

本工程使用普通 Git，默认分支为 `main`，不需要 Git LFS。正式 H2 模型随仓库提供；共享 G1 planner 在本地保留供测试，但不提交到 Git，新克隆的工程需按下文单独下载。

```bash
git clone git@github.com:QinnyuLan/SONIC_unitree_h2.git
cd SONIC_unitree_h2
python3 tools/verify_release.py
```

本地工作路径为 `~/workspace/SONIC_unitree_h2`，与 `SONIC-H2` 平级。独立仓库的日常同步和资产范围见 [Git 管理说明](docs/GIT_WORKFLOW.md)。

### 2. sim2sim（推荐先做）

```bash
./install.sh sim2sim
source .venv_sim/bin/activate
python3 tools/verify_release.py --imports
# 终端 1：启动 MuJoCo 仿真窗口
bash scripts/run_h2_paper_sim.sh
```

另开一个终端，从发布目录启动控制端：

```bash
cd SONIC_unitree_h2/gear_sonic_deploy
./scripts/run_h2_sim2sim.sh --print-command
./scripts/run_h2_sim2sim.sh
```

两个终端共同完成本机 MuJoCo/DDS 回环。控制端使用 `models/released` 下的 H2 Native ONNX 模型、`reference/example` 动作集和 nominal Kp/Kd；仿真窗口与控制端的 DDS domain 均为 0。

### 3. sim2real（仅在确认机器人安全后）

先安装并检查本机 CUDA/TensorRT/ONNX Runtime：

```bash
export TensorRT_ROOT="$HOME/TensorRT"
./verify_deploy.sh
cd gear_sonic_deploy
./scripts/run_h2_sim2real_four_kp.sh --print-command
./scripts/run_h2_sim2real_four_kp.sh
```

当前实机配置在 `gear_sonic_deploy/config/h2_deploy.env`：`released` 模型、`reference/example` 动作集、DDS domain 0、网卡 `eno1`，脚踝 pitch/roll 和膝盖的增益缩放分别由配置文件控制。运行前必须核对网卡、DDS domain、急停、动作集合和增益。

### 4. PICO VR 三点遥操作

先从 SONIC 的 Hugging Face 仓库下载 G1 planner（本地已有正确文件时直接跳过）：

```bash
uv venv .venv_download --python 3.10
uv pip install --python .venv_download/bin/python -r requirements/download.txt
.venv_download/bin/python download_from_hf.py --planner-only
python3 tools/verify_release.py --require-planner
```

文件固定放在 `gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx`，与启动脚本使用的路径相同，并由 `.gitignore` 忽略。下载来源、校验值和手动下载方法见 [Planner 下载说明](docs/PLANNER.md)。

桌面端安装遥操作环境并确保 XRoboToolkit PC Service 已运行：

```bash
./install.sh teleop
source .venv_teleop/bin/activate
cd gear_sonic_deploy
./scripts/run_h2_teleop_head_adjust.sh pico --print-command
./scripts/run_h2_teleop_head_adjust.sh pico
```

另一个终端启动部署端：

```bash
cd SONIC_unitree_h2/gear_sonic_deploy
./scripts/run_h2_teleop_head_adjust.sh deploy --print-command
./scripts/run_h2_teleop_head_adjust.sh deploy
```

VR 默认使用 `released` 模型、SONIC planner、`HAND_BACKEND=inspire` 和固定头部角度。头部和手部设置见 `gear_sonic_deploy/config/h2_head_adjust.env`。PICO 配对、校准和网络要求见 [`docs/source/getting_started/vr_teleop_setup.md`](docs/source/getting_started/vr_teleop_setup.md)。

## 发布入口

| 流程 | 启动脚本 | 模型 | 参考/输入 |
|---|---|---|---|
| sim2sim | `run_h2_sim2sim.sh` | `models/released` | MuJoCo + `reference/example` |
| sim2real | `run_h2_sim2real_four_kp.sh` | `models/released` | H2 实机 DDS + `reference/example` |
| VR 遥操作部署端 | `run_h2_teleop_head_adjust.sh deploy` | `models/released` | H2 planner + ZMQ |
| VR 遥操作采集端 | `run_h2_teleop_head_adjust.sh pico` | — | XRoboToolkit/PICO + ZMQ |

正式发布模型的校验值见 [`models/released/SHA256SUMS`](models/released/SHA256SUMS)，输入输出及验证结果见 [模型说明](models/released/README.md) 和 [`release/model_validation.json`](release/model_validation.json)。这是直接微调模型，无需 LoRA 合并。运行时校验输入输出维度、模型角色、观测/朝向契约与 selector，不再要求适配器已合并。发布入口和文档统一使用 `released`，不展示训练轮次或实验名称；ONNX 文件保持原始字节和 SHA256。源提交和排除项记录在 [`release/source_manifest.json`](release/source_manifest.json)。历史实验报告保留在 `docs/report/`；报告中出现的旧路径或旧轮次不覆盖上表的实际启动脚本配置。

## 常见检查

```bash
# 不连接设备的文件/配置检查
python3 tools/verify_release.py

# 检查部署端的本机原生依赖
export TensorRT_ROOT="$HOME/TensorRT"
./verify_deploy.sh

# 只打印命令，不启动程序
cd gear_sonic_deploy
./scripts/run_h2_sim2sim.sh --print-command
./scripts/run_h2_sim2real_four_kp.sh --print-command
./scripts/run_h2_teleop_head_adjust.sh deploy --print-command
```

## 发布边界

发布目录排除了 `.venv_*`、运行日志、构建目录、历史备份、测试视频和已生成的 TensorRT engine 缓存，以避免把机器特定状态带入部署包。正式 H2 ONNX、MuJoCo/H2 机器人资源、动作参考和源代码随 Git 提供。planner 保留本地副本并单独下载；超过 GitHub 单文件限制的历史模型与可选训练/人体可视化资产保留本地，忽略范围见 [Git 管理说明](docs/GIT_WORKFLOW.md)。目标机第一次部署需要重新编译 C++ 程序并生成与其 TensorRT 版本匹配的 engine。

真实机器人控制属于现场操作：先在 sim2sim 完成加载和回环检查，再由有权限的操作员确认急停、网络和增益后执行 sim2real。
