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
| ECHO-G | Official PyTorch speech-conditioned G1 gesture inference and feature extraction | [Repository](https://github.com/NJU-SOARLAB/ECHO-G) · [Models](https://huggingface.co/gaopusen/ECHO-G) |
| Qwen3.5-4B and wav2vec2 XLSR-53 English | ECHO-G text and audio feature encoders, pinned by its upstream manifest | [Qwen](https://huggingface.co/Qwen/Qwen3.5-4B) · [wav2vec2](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english) |
| edge-tts | Optional speech synthesis with word timestamps | [Repository](https://github.com/rany2/edge-tts) |
| MuJoCo | H2 physics simulation and visualization | [Repository](https://github.com/google-deepmind/mujoco) |
| Unitree SDK2 and Python SDK | DDS communication with the robot and simulator | [C++ SDK](https://github.com/unitreerobotics/unitree_sdk2) · [Python SDK](https://github.com/unitreerobotics/unitree_sdk2_python) |
| Unitree robot descriptions | Robot models and meshes | [MuJoCo assets](https://github.com/unitreerobotics/unitree_mujoco) · [ROS descriptions](https://github.com/unitreerobotics/unitree_ros) |
| XRoboToolkit PC Service | PICO device communication and head/controller tracking | [Repository](https://github.com/XR-Robotics/XRoboToolkit-PC-Service) |
| XRoboToolkit Python binding | Python interface to the PC Service SDK | [Repository](https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind) |
| NVIDIA TensorRT | GPU inference for the exported ONNX models | [Project](https://developer.nvidia.com/tensorrt) |
| NVIDIA CUDA Python | CUDA memory, streams and graphs for the Python planner | [Repository](https://github.com/NVIDIA/cuda-python) |
| ONNX Runtime | Optional offline model validation | [Repository](https://github.com/microsoft/onnxruntime) |

## Source provenance

The initial deployment extraction used the `SONIC-H2` checkout at commit `4716be4bcc5f8c61c5f26e946905036f78eee478`, together with its local H2 deployment changes. Subsequent changes are tracked in this repository's Git history. The source SDK revisions are pinned by [.gitmodules](../.gitmodules) and the submodule commits; native XR artifacts have a separate [checksum manifest](../third_party/xrobotoolkit_sdk/manifest.json).

Software and robot assets retain their original copyright and license notices. See the root [LICENSE](../LICENSE) and the consolidated [license index](../licenses/README.md).
