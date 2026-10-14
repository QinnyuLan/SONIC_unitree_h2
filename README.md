# TRACE: Efficient Transfer Learning for Neural Whole-Body Control Across Humanoid Embodiments

**Qingyu Lan · Zhihao Tian · Zifan Wang · Junwei Liang**

[Project Page](https://bfm-trace.github.io/) · [arXiv](https://arxiv.org/abs/2610.17999) · [Interactive Demo](https://bfm-trace.github.io/demo/index.html) · [中文](README_CN.md)

TRACE transfers pretrained humanoid whole-body control across embodiments. This repository provides the Unitree H2 deployment code and checkpoints for the project, built on [SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl).

[![TRACE framework](https://bfm-trace.github.io/assets/method-overview.webp)](https://bfm-trace.github.io/)

## Overview

The release supports motion tracking and VR teleoperation on Unitree H2, with a shared controller for simulation and hardware deployment.

| Workflow | Description | Guide |
|---|---|---|
| Sim2Sim | MuJoCo simulation with the C++ controller and DDS communication | [Simulation](docs/DEPLOYMENT.md#sim2sim) |
| Sim2Real | Whole-body motion tracking on Unitree H2 | [Hardware deployment](docs/DEPLOYMENT.md#sim2real) |
| VR teleoperation | PICO head-and-hand tracking with XRoboToolkit | [Teleoperation](docs/DEPLOYMENT.md#vr-teleoperation) |

## Getting Started

```bash
git clone https://github.com/QinnyuLan/SONIC_unitree_h2.git
cd SONIC_unitree_h2
python3 tools/verify_release.py
```

Follow the [deployment guide](docs/DEPLOYMENT.md) to install dependencies, build the controller, and run simulation. H2 checkpoints are provided in [`models/released/`](models/released/). VR teleoperation also requires the [SONIC planner](docs/DEPLOYMENT.md#planner).

## Documentation

- [Installation and deployment](docs/DEPLOYMENT.md): environment setup, Sim2Sim, Sim2Real, and VR teleoperation.
- [Released model](models/released/README.md): input/output interfaces and checksums.
- [Reproducibility](release/REPRODUCIBILITY.md): model validation, inference checks, and test results.
- [Documentation index](docs/README.md): configuration, troubleshooting, and development notes.

## Citation

If you use this work in your research, please cite TRACE and the underlying SONIC work:

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

Bibliographic metadata is also available in [`CITATION.cff`](CITATION.cff) and [`references.bib`](docs/references.bib).

## Acknowledgments

This project builds on [SONIC / GR00T Whole-Body Control](https://github.com/NVlabs/GR00T-WholeBodyControl) for the pretrained controller, motion representation, and deployment framework. We also thank the developers of [MuJoCo](https://github.com/google-deepmind/mujoco), [Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2), and [XRoboToolkit](https://github.com/XR-Robotics/XRoboToolkit-PC-Service) for the simulation, robot communication, and VR interfaces used in this release. See [upstream dependencies and references](docs/ACKNOWLEDGMENTS.md) for details.

## License

Source code is licensed under Apache-2.0. SONIC-derived model weights are covered by the NVIDIA Open Model License. Third-party components retain their respective licenses; see [LICENSE](LICENSE) and [third-party notices](legal/).

Licensed by NVIDIA Corporation under the NVIDIA Open Model License.
