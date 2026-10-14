# TRACE: Efficient Transfer Learning for Neural Whole-Body Control Across Humanoid Embodiments

**Qingyu Lan · Zhihao Tian · Zifan Wang · Junwei Liang**

[项目主页](https://bfm-trace.github.io/) · [arXiv 论文](https://arxiv.org/abs/2610.17999) · [在线演示](https://bfm-trace.github.io/demo/index.html) · [English](README.md)

TRACE 研究预训练人形机器人全身控制器的跨本体迁移。本仓库提供项目的 Unitree H2 部署代码与模型，基于 [SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl) 开发。

[![TRACE 方法概览](https://bfm-trace.github.io/assets/method-overview.webp)](https://bfm-trace.github.io/)

## 项目概览

本项目支持 Unitree H2 的动作跟踪与 VR 遥操作，仿真和实机部署共用同一套控制器。

| 工作流 | 功能 | 使用指南 |
|---|---|---|
| Sim2Sim | MuJoCo 仿真与 C++ 控制器通过 DDS 通信 | [仿真运行](docs/DEPLOYMENT_CN.md#sim2sim) |
| Sim2Real | Unitree H2 实机全身动作跟踪 | [实机部署](docs/DEPLOYMENT_CN.md#sim2real) |
| VR 遥操作 | 基于 PICO 与 XRoboToolkit 的头部、双手三点遥操作 | [遥操作](docs/DEPLOYMENT_CN.md#vr-teleoperation) |

## 快速开始

```bash
git clone https://github.com/QinnyuLan/SONIC_unitree_h2.git
cd SONIC_unitree_h2
python3 tools/verify_release.py
```

按照[部署指南](docs/DEPLOYMENT_CN.md)安装依赖、编译控制器并运行仿真。H2 模型位于 [`models/released/`](models/released/)，VR 遥操作还需下载 [SONIC planner](docs/DEPLOYMENT_CN.md#planner)。

## 文档

- [安装与部署](docs/DEPLOYMENT_CN.md)：环境配置、Sim2Sim、Sim2Real 和 VR 遥操作。
- [发布模型说明](models/released/README.md)：输入输出接口与文件校验。
- [复现记录](release/REPRODUCIBILITY.md)：模型验证、推理检查与测试结果。
- [文档索引](docs/README.md)：配置、故障排查与开发说明。

## 引用

如果本工作对您的研究有帮助，请引用 TRACE 及其基础工作 SONIC：

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

@article{luo2025sonic,
    title={SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control},
    author={Luo, Zhengyi and Yuan, Ye and Wang, Tingwu and Li, Chenran and Chen, Sirui and Casta\~neda, Fernando and Cao, Zi-Ang and Li, Jiefeng and Minor, David and Ben, Qingwei and Da, Xingye and Ding, Runyu and Hogg, Cyrus and Song, Lina and Lim, Edy and Jeong, Eugene and He, Tairan and Xue, Haoru and Xiao, Wenli and Wang, Zi and Yuen, Simon and Kautz, Jan and Chang, Yan and Iqbal, Umar and Fan, Linxi and Zhu, Yuke},
    journal={arXiv preprint arXiv:2511.07820},
    year={2025}
}
```

引用元数据同时提供于 [`CITATION.cff`](CITATION.cff) 和 [`references.bib`](docs/references.bib)。

## 致谢

本项目基于 [SONIC / GR00T Whole-Body Control](https://github.com/NVlabs/GR00T-WholeBodyControl) 的预训练控制器、动作表征与部署框架，并使用 [MuJoCo](https://github.com/google-deepmind/mujoco)、[Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2) 和 [XRoboToolkit](https://github.com/XR-Robotics/XRoboToolkit-PC-Service) 提供的仿真、机器人通信与 VR 接口。感谢这些开源项目的贡献者，详细来源见[上游依赖与参考工作](docs/ACKNOWLEDGMENTS.md)。

## 许可证

源代码采用 Apache-2.0 许可证，基于 SONIC 的模型权重遵循 NVIDIA Open Model License。第三方组件保留各自的许可证，详见 [LICENSE](LICENSE) 与[第三方声明](legal/)。

Licensed by NVIDIA Corporation under the NVIDIA Open Model License.
