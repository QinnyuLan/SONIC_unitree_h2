# 依赖文件说明

发布工程通过 `install.sh` 创建 Python 3.10 环境，并从 `gear_sonic/pyproject.toml` 安装四种功能集：

| 环境 | 用途 | 安装命令 |
|---|---|---|
| `sim2sim` | MuJoCo、DDS 回环、WBC 仿真 | `./install.sh sim2sim` |
| `teleop` | PICO/XRoboToolkit、ZMQ、VR 三点遥操作 | `./install.sh teleop` |
| `data_collection` | 相机、LeRobot 数据采集 | `./install.sh data_collection` |
| `inference` | Isaac-GR00T 推理客户端 | `./install.sh inference` |

`gear_sonic_deploy` 是 C++/TensorRT 运行时，不能用 pip 完成。实机或 C++ 控制端还需要：

- Ubuntu 22.04/24.04，CMake、GCC、ZeroMQ 开发包；
- NVIDIA CUDA 与匹配的驱动；
- TensorRT 10.x，使用前设置 `TensorRT_ROOT`；
- ONNX Runtime C++，默认由 `verify_deploy.sh` 检查 `/opt/onnxruntime`；
- 真实机器人需要 Unitree H2 网络、DDS 和 `eno1` 网卡；VR 需要 XRoboToolkit PC Service 与 PICO 设备。

当前开发机记录：Python 3.10.20、CUDA 12.6、TensorRT 10.13.3.9、ONNX Runtime 1.16.3。TensorRT engine 缓存未放入发布目录，会在目标机器上按实际 TensorRT 版本重新生成。

## 下载工具

环境安装使用 [uv](https://docs.astral.sh/uv/getting-started/installation/)。共享 G1 planner 的下载依赖固定在 `requirements/download.txt`，单独安装到 `.venv_download`；安装与下载命令见 [Planner 下载说明](../docs/PLANNER.md)。已有完整 planner 文件时，无需安装此依赖。
