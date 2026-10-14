# Licenses and Notices

The repository's source-code and model-weight terms are in the root [LICENSE](../LICENSE).

This directory collects the notices distributed with SONIC and the imported robot assets. Their original text is preserved.

| Source | Documents | Applies to |
|---|---|---|
| SONIC / GR00T Whole-Body Control | [License](sonic/LICENSE.txt), [DCO](sonic/DCO.txt), [software notices](sonic/THIRD_PARTY_SOFTWARE_NOTICES.txt), [software and asset notices](sonic/THIRD_PARTY_ASSET_NOTICES.txt) | Materials inherited from SONIC |
| Unitree MuJoCo | [License](unitree_mujoco/LICENSE) | Imported MuJoCo robot assets |
| Unitree ROS | [License](unitree_ros/LICENSE) | Imported robot descriptions and meshes |

The SONIC notices describe the upstream distribution; the deployment dependencies used here are listed in [Acknowledgments](../docs/ACKNOWLEDGMENTS.md).

The released H2 weights are derived from SONIC, whose [upstream model card](https://huggingface.co/nvidia/GEAR-SONIC) specifies the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). The required model attribution is retained in [sonic/NOTICE](sonic/NOTICE). Other models and components follow their own terms, as listed below.

Dependency license files and notices are retained at:

- ECHO-G: [source license](echo_g/LICENSE) (PolyForm Noncommercial), [data/weight terms](echo_g/LICENSE-DATA-WEIGHTS) (CC BY-NC 4.0), and [notice](echo_g/NOTICE). The adapted model and feature code retain their upstream terms; see [source provenance](echo_g/README.md). Optional weights are downloaded separately.
- [Unitree C++ SDK](../src/runtime/third_party/unitree_sdk2/LICENSE).
- Inspire hand service: [BSD 3-Clause license](inspire_hands/LICENSE) and [source notice](inspire_hands/NOTICE).
- [Unitree Python SDK](../third_party/unitree_sdk2_python/LICENSE).
- [XRoboToolkit Python binding](../third_party/xrobotoolkit_pybind/LICENSE).
- [Bundled XRoboToolkit SDK notice](../third_party/xrobotoolkit_sdk/LICENSE).

The source submodules must be initialized for their local license links to resolve; see [third-party setup](../third_party/README.md).
