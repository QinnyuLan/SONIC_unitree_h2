# SONIC_unitree_h2

[![Project Page](https://img.shields.io/badge/Project-Page-4A90E2?logo=googlechrome&logoColor=white)](https://bfm-trace.github.io/)
[![arXiv Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.17999)
[![Interactive Demo](https://img.shields.io/badge/Interactive-Demo-2ea44f?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI%2BPHBhdGggZmlsbD0id2hpdGUiIGQ9Ik03IDN2MThsMTUtOXoiLz48L3N2Zz4%3D)](https://bfm-trace.github.io/demo/index.html)
[![中文](https://img.shields.io/badge/Language-%E4%B8%AD%E6%96%87-8A2BE2?logo=googletranslate&logoColor=white)](README_CN.md)

Official Unitree H2 deployment implementation for [TRACE: Efficient Transfer Learning for Neural Whole-Body Control Across Humanoid Embodiments](https://arxiv.org/abs/2610.17999).

## Overview

![SONIC deployment on Unitree H2 with TRACE and multiple motion and command interfaces](docs/images/deployment_overview.png)

TRACE transfers the pretrained SONIC whole-body policy to Unitree H2 with low additional training cost. This deployment provides unified command interfaces for G1, H2, and SMPL reference motions, supporting motion replay, motion planning, offline text-to-motion, and VR teleoperation. A direct token interface is reserved for future VLA/VLN integration.

## Workflow

| Workflow | Description | Guide |
|---|---|---|
| Sim2Sim | MuJoCo simulation with the C++ controller and DDS communication | [Simulation](docs/DEPLOYMENT.md#sim2sim) |
| Sim2Real | Whole-body motion tracking on Unitree H2 | [Hardware deployment](docs/DEPLOYMENT.md#sim2real) |
| Text-to-motion | Offline speech and G1 gesture generation with ECHO-G, presented through motion replay | [Text-to-motion](docs/TEXT2MOTION.md) |
| VR teleoperation | PICO head-and-hand tracking with XRoboToolkit | [Teleoperation](docs/DEPLOYMENT.md#vr-teleoperation) |

Source code is organized as [`src/runtime/`](src/runtime/README.md) (C++ policy), [`src/interface/`](src/interface/README.md) (Python command producers), and [`src/sim/`](src/sim/README.md) (simulator). Offline generation, asset preparation and validation tools are organized under [`tools/`](tools/README.md).

## Getting Started

The runtime initializes a standing pose, waits for **]** or **s** to enable the policy, then accepts commands from independent motion replay, planner, and teleoperation processes. Press **o** to stop with damping. Simulation and hardware use the same runtime. Head pitch/yaw and dexterous hands have separate command channels. Input services and the planner use Python; the real-time policy runtime uses C++. Policy and planner inference use TensorRT. The optional ECHO-G generator prepares speech and G1 gestures offline for motion replay.

```bash
git clone --depth 1 --recurse-submodules https://github.com/QinnyuLan/SONIC_unitree_h2.git
cd SONIC_unitree_h2
python3 tools/validation/verify_release.py
```

Follow the [deployment guide](docs/DEPLOYMENT.md) to install dependencies, build the controller, and run simulation. H2 checkpoints are provided in [`models/sonic_h2_released/`](models/sonic_h2_released/). The [SONIC planner](docs/DEPLOYMENT.md#planner) is optional and runs as a separate service.

The [Chinese and English H2 introduction demos](docs/TEXT2MOTION.md#replay-the-included-examples) include speech and ECHO-G gestures ready for motion replay. No ECHO-G setup or generation models are needed to play them.

## Documentation

- [Architecture and command protocol](docs/ARCHITECTURE.md): process boundaries, sparse references, and independent head/hand control.
- [Repository layout](docs/LAYOUT.md): C++ runtime, Python services, models, configuration and motions.
- [Installation and deployment](docs/DEPLOYMENT.md): environment setup, Sim2Sim, Sim2Real, and VR teleoperation.
- [Offline text-to-motion](docs/TEXT2MOTION.md): Chinese/English speech, G1 gestures, and audio playback.
- [Robot hand service](docs/INSPIRE_HANDS.md): Inspire DDS/RS-485 deployment, autostart and watchdog.
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
- [ECHO-G](https://github.com/NJU-SOARLAB/ECHO-G): speech-conditioned G1 gesture generation.
- [MuJoCo](https://github.com/google-deepmind/mujoco): physics simulation and visualization.
- [Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2): robot DDS communication.
- [DFX Inspire Service](https://github.com/unitreerobotics/dfx_inspire_service): Inspire hand DDS service and serial protocol implementation.
- [XRoboToolkit](https://github.com/XR-Robotics/XRoboToolkit-PC-Service): PICO VR device and head-and-hand tracking interfaces.

See [upstream dependencies and references](docs/ACKNOWLEDGMENTS.md) for the full dependency list.

## License

Unless otherwise noted, source code is licensed under Apache-2.0. Model weights and third-party components are subject to their respective licenses; see [LICENSE](LICENSE) and [licenses and notices](licenses/README.md).
