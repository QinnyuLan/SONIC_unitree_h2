# G1 planner 下载与本地测试

独立 planner 服务使用 SONIC 的共享 G1 kinematic planner。正式 H2 控制器位于 `models/sonic_h2_released/`，本下载操作仅补齐 planner。

## 本地路径与 Git

下载脚本和 planner 服务使用相同路径；也可将已有权重放入此位置：

```text
models/motion_planner_g1/planner_sonic.onnx
```

此文件已加入根目录 `.gitignore`，不会随普通 `git add` / `git push` 上传。新克隆的仓库不含该权重。文件大小为 773,952,989 字节（约 738 MiB）。runtime、离线动作回放和原地 SMPL/VR 遥操作不依赖 planner；摇杆行走需要启动独立 planner 服务。

## 推荐下载方法

在工程根目录运行；`uv` 安装方法见 [依赖说明](../requirements/README.md)。

```bash
uv venv .venv_download --python 3.10
uv pip install --python .venv_download/bin/python -r requirements/download.txt
.venv_download/bin/python tools/model_download/download_planner.py
python3 tools/validation/verify_release.py --require-planner
```

也可以在已有 Python 环境中执行 `python -m pip install -r requirements/download.txt`，然后执行 `python tools/model_download/download_planner.py`。默认目标路径以仓库根目录为基准，由脚本位置解析，执行命令时的当前目录不影响它。

若本地文件的 SHA256 已匹配，脚本直接复用，不访问网络，也不要求安装下载依赖。需要下载时，脚本使用 Hugging Face 缓存，先写临时文件，通过校验后才替换目标文件；下载或校验失败不会覆盖原文件。缓存通常在 `~/.cache/huggingface`，可通过 `HF_HOME` 指定其他磁盘。首次下载为缓存和目标副本预留至少 1.6 GB 空间。

## 固定来源与校验

官方仓库：[nvidia/GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC)。文件：[planner_sonic.onnx](https://huggingface.co/nvidia/GEAR-SONIC/blob/72935a37d3860204617ebaaf6c855a1c5aff7b8c/planner_sonic.onnx)。

```text
revision: 72935a37d3860204617ebaaf6c855a1c5aff7b8c
SHA256:   39b553e197f62f077975ba38512bc04781a3fc37c2af7c6756e04629f760edea
```

下载脚本固定 revision 和 SHA256，确保复现同一份 planner。上述 SHA256 已与本机保留的权重及 Hugging Face 文件信息核对一致。

如果需要手动获取，可下载上述链接中的原始文件到同一路径，再执行 `python3 tools/validation/verify_release.py --require-planner`。不要将网页保存为 `.onnx`。

## 检查与运行

```bash
# 确认文件被 Git 忽略
git check-ignore models/motion_planner_g1/planner_sonic.onnx
# 校验发布模型与 planner，不连接机器人
python3 tools/validation/verify_release.py --require-planner
# 安装 TensorRT Python planner 环境
./install.sh planner
# 独立 planner 服务
bash scripts/run_planner.sh
```

不带 `--require-planner` 的检查允许新克隆仓库缺少 planner，并给出下载提示；若文件已经存在但校验不符，检查仍会失败。完整 VR 环境、设备连接和启动步骤见 [主 README](../README.md)。

## Python 服务与调用接口

服务、上下文构造、30 Hz → 50 Hz 重采样、轨迹混合和稀疏 10 帧发送均由 Python 完成。模型通过 TensorRT Python API 推理，CUDA 缓冲区持续复用，使用 CUDA Graph 执行。policy runtime 继续使用 C++ TensorRT。

```bash
# 与启动脚本等价；需要先配置原生库路径
source scripts/setup_env.sh
.venv_planner/bin/python -m interface.planner.service
```

程序调用入口为 `interface.planner.motion.G1Planner`：`initialize(h2_joints)` 接收 31 维 H2 DDS 关节状态，`update(trajectory, cursor, command)` 生成后续轨迹。`Trajectory.sparse(cursor)` 输出与 replay 相同的 G1 稀疏参考。指令字段和端口见 [架构文档](ARCHITECTURE_CN.md)。

运行实际模型与通信集成检查：

```bash
source scripts/setup_env.sh
.venv_planner/bin/python tools/validation/check_runtime.py --planner
```

此检查使用合成机器人状态，不连接机器人或 PICO。
