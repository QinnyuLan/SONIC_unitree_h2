# SONIC_unitree_h2

Official Unitree H2 deployment implementation for [TRACE: Efficient Transfer Learning for Neural Whole-Body Control Across Humanoid Embodiments](https://arxiv.org/abs/2610.17999).

[Project Page](https://bfm-trace.github.io/) · [arXiv](https://arxiv.org/abs/2610.17999) · [Interactive Demo](https://bfm-trace.github.io/demo/index.html) · [中文](README_CN.md)

## Overview

The policy runtime starts in the default standing pose and accepts commands from independent motion replay, planner, and teleoperation processes. Simulation and hardware use the same runtime. Head pitch/yaw and dexterous hands have separate command channels. Input services and the planner use Python; the real-time policy runtime uses C++. All model inference uses TensorRT.

| Workflow | Description | Guide |
|---|---|---|
| Sim2Sim | MuJoCo simulation with the C++ controller and DDS communication | [Simulation](docs/DEPLOYMENT.md#sim2sim) |
| Sim2Real | Whole-body motion tracking on Unitree H2 | [Hardware deployment](docs/DEPLOYMENT.md#sim2real) |
| VR teleoperation | PICO head-and-hand tracking with XRoboToolkit | [Teleoperation](docs/DEPLOYMENT.md#vr-teleoperation) |

Source code is organized as [`src/runtime/`](src/runtime/README.md) (C++ policy), [`src/interface/`](src/interface/README.md) (Python command producers), and [`src/sim/`](src/sim/README.md) (simulator).

## Getting Started

```bash
git clone --depth 1 --recurse-submodules https://github.com/QinnyuLan/SONIC_unitree_h2.git
cd SONIC_unitree_h2
python3 tools/verify_release.py
```

Follow the [deployment guide](docs/DEPLOYMENT.md) to install dependencies, build the controller, and run simulation. H2 checkpoints are provided in [`models/sonic_h2_released/`](models/sonic_h2_released/). The [SONIC planner](docs/DEPLOYMENT.md#planner) is optional and runs as a separate service.

## Documentation

- [Architecture and command protocol](docs/ARCHITECTURE.md): process boundaries, sparse references, and independent head/hand control.
- [Repository layout](docs/LAYOUT.md): C++ runtime, Python services, models, configuration and motions.
- [Installation and deployment](docs/DEPLOYMENT.md): environment setup, Sim2Sim, Sim2Real, and VR teleoperation.
- [Released model](models/sonic_h2_released/README.md): input/output interfaces and checksums.
- [Reproducibility](docs/REPRODUCIBILITY.md): model validation, inference checks, and test results.
- [Documentation index](docs/README.md): configuration, troubleshooting, and development notes.

## Citation

If you use this project in your research, please cite:

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

Bibliographic metadata is also available in [`CITATION.cff`](CITATION.cff) and [`references.bib`](docs/references.bib).

## Acknowledgments

- [SONIC / GR00T Whole-Body Control](https://github.com/NVlabs/GR00T-WholeBodyControl): pretrained controller, motion representation, and deployment framework.
- [MuJoCo](https://github.com/google-deepmind/mujoco): physics simulation and visualization.
- [Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2): robot DDS communication.
- [XRoboToolkit](https://github.com/XR-Robotics/XRoboToolkit-PC-Service): PICO VR device and head-and-hand tracking interfaces.

See [upstream dependencies and references](docs/ACKNOWLEDGMENTS.md) for the full dependency list.

## License

Source code is licensed under Apache-2.0. SONIC-derived model weights are covered by the NVIDIA Open Model License. Third-party components retain their respective licenses; see [LICENSE](LICENSE) and [third-party notices](licenses/README.md).

Licensed by NVIDIA Corporation under the NVIDIA Open Model License.
