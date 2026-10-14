# SONIC_unitree_h2

论文 [TRACE: Efficient Transfer Learning for Neural Whole-Body Control Across Humanoid Embodiments](https://arxiv.org/abs/2610.17999) 的 Unitree H2 部署官方实现。

[项目主页](https://bfm-trace.github.io/) · [arXiv 论文](https://arxiv.org/abs/2610.17999) · [在线演示](https://bfm-trace.github.io/demo/index.html) · [English](README.md)

## 项目概览

policy runtime 启动后以默认姿态站立，接收独立 motion replay、planner 和遥操作进程的指令。仿真与实机共用同一 runtime；头部 pitch/yaw 和灵巧手分别使用独立通道。输入端与 planner 统一使用 Python，实时 policy runtime 使用 C++，模型推理统一采用 TensorRT。

| 工作流 | 功能 | 使用指南 |
|---|---|---|
| Sim2Sim | MuJoCo 仿真与 C++ 控制器通过 DDS 通信 | [仿真运行](docs/DEPLOYMENT_CN.md#sim2sim) |
| Sim2Real | Unitree H2 实机全身动作跟踪 | [实机部署](docs/DEPLOYMENT_CN.md#sim2real) |
| VR 遥操作 | 基于 PICO 与 XRoboToolkit 的头部、双手三点遥操作 | [遥操作](docs/DEPLOYMENT_CN.md#vr-遥操作) |

源码分为 [`src/runtime/`](src/runtime/README.md)（C++ policy）、[`src/interface/`](src/interface/README.md)（Python 上游推理与指令发送）、[`src/sim/`](src/sim/README.md)（仿真器）。

## 快速开始

```bash
git clone --depth 1 --recurse-submodules https://github.com/QinnyuLan/SONIC_unitree_h2.git
cd SONIC_unitree_h2
python3 tools/verify_release.py
```

按照[部署指南](docs/DEPLOYMENT_CN.md)安装依赖、编译控制器并运行仿真。H2 模型位于 [`models/sonic_h2_released/`](models/sonic_h2_released/)，需要行走规划时，可另行下载并运行 [SONIC planner](docs/DEPLOYMENT_CN.md#planner)。

## 文档

- [架构与指令协议](docs/ARCHITECTURE_CN.md)：独立 runtime、稀疏参考、头部和手部通道。
- [工程结构](docs/LAYOUT.md)：C++ runtime、Python 服务、模型、配置与动作示例。
- [安装与部署](docs/DEPLOYMENT_CN.md)：环境配置、Sim2Sim、Sim2Real 和 VR 遥操作。
- [发布模型说明](models/sonic_h2_released/README.md)：输入输出接口与文件校验。
- [复现记录](docs/REPRODUCIBILITY.md)：模型验证、推理检查与测试结果。
- [文档索引](docs/README.md)：配置、故障排查与开发说明。

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@misc{lan2026traceefficienttransferlearning,
  title={TRACE: Efficient Transfer Learning for Neural Whole-Body Control Across Humanoid Embodiments},
  author={Qingyu Lan and Zhihao Tian and Zifan Wang and Junwei Liang},
  year={2026},
  eprint={2610.17999},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2610.17999},
}
```

引用元数据同时提供于 [`CITATION.cff`](CITATION.cff) 和 [`references.bib`](docs/references.bib)。

## 致谢

- [SONIC / GR00T Whole-Body Control](https://github.com/NVlabs/GR00T-WholeBodyControl)：预训练控制器、动作表征与部署框架。
- [MuJoCo](https://github.com/google-deepmind/mujoco)：物理仿真与可视化。
- [Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2)：机器人 DDS 通信。
- [XRoboToolkit](https://github.com/XR-Robotics/XRoboToolkit-PC-Service)：PICO VR 设备与三点跟踪接口。

完整依赖与来源见[上游依赖与参考工作](docs/ACKNOWLEDGMENTS.md)。

## 许可证

源代码采用 Apache-2.0 许可证，基于 SONIC 的模型权重遵循 NVIDIA Open Model License。第三方组件保留各自的许可证，详见 [LICENSE](LICENSE) 与[第三方声明](licenses/README.md)。

Licensed by NVIDIA Corporation under the NVIDIA Open Model License.
