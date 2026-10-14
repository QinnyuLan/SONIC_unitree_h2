# Upstream Dependencies and References

[English README](../README.md) · [中文首页](../README_CN.md)

## Research

- **TRACE: Efficient Transfer Learning for Neural Whole-Body Control Across Humanoid Embodiments.** Qingyu Lan, Zhihao Tian, Zifan Wang, and Junwei Liang, 2026. [Paper](https://arxiv.org/abs/2610.17999) · [Project](https://bfm-trace.github.io/).
- **SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control.** Zhengyi Luo et al., 2025. TRACE builds on SONIC's pretrained motion representation and whole-body controller. [Paper](https://arxiv.org/abs/2511.07820) · [Code](https://github.com/NVlabs/GR00T-WholeBodyControl) · [Models](https://huggingface.co/nvidia/GEAR-SONIC).

The project citation is provided in [references.bib](references.bib) and [CITATION.cff](../CITATION.cff).

## Software and Assets

| Project | Use in this repository | Source |
|---|---|---|
| SONIC / GR00T Whole-Body Control | Controller architecture, motion interfaces, C++ deployment, and the shared G1 planner | [Repository](https://github.com/NVlabs/GR00T-WholeBodyControl) |
| MuJoCo | H2 physics simulation and visualization | [Repository](https://github.com/google-deepmind/mujoco) |
| Unitree SDK2 and Python SDK | DDS communication with the robot and simulator | [C++ SDK](https://github.com/unitreerobotics/unitree_sdk2) · [Python SDK](https://github.com/unitreerobotics/unitree_sdk2_python) |
| Unitree robot descriptions | Robot models and meshes | [MuJoCo assets](https://github.com/unitreerobotics/unitree_mujoco) · [ROS descriptions](https://github.com/unitreerobotics/unitree_ros) |
| XRoboToolkit PC Service | PICO device communication and head/controller tracking | [Repository](https://github.com/XR-Robotics/XRoboToolkit-PC-Service) |
| NVIDIA TensorRT | GPU inference for the exported ONNX models | [Project](https://developer.nvidia.com/tensorrt) |
| ONNX Runtime | ONNX inference and model validation | [Repository](https://github.com/microsoft/onnxruntime) |

Source provenance is recorded in [source_manifest.json](../release/source_manifest.json). Vendored libraries and assets retain their original copyright and license notices; see [LICENSE](../LICENSE) and [legal/](../legal/).
