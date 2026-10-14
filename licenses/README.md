# Licenses and Notices

The repository's source-code and model-weight terms are in the root [LICENSE](../LICENSE).

This directory collects the notices distributed with SONIC and the imported robot assets. Their original text is preserved.

| Source | Documents | Applies to |
|---|---|---|
| SONIC / GR00T Whole-Body Control | [License](sonic/LICENSE.txt), [DCO](sonic/DCO.txt), [software notices](sonic/THIRD_PARTY_SOFTWARE_NOTICES.txt), [software and asset notices](sonic/THIRD_PARTY_ASSET_NOTICES.txt) | Materials inherited from SONIC |
| Unitree MuJoCo | [License](unitree_mujoco/LICENSE) | Imported MuJoCo robot assets |
| Unitree ROS | [License](unitree_ros/LICENSE) | Imported robot descriptions and meshes |

The SONIC notices describe the upstream distribution; the deployment dependencies used here are listed in [Acknowledgments](../docs/ACKNOWLEDGMENTS.md).

Dependencies retain the license files shipped with their source trees:

- ECHO-G: [source license](../third_party/echo_g/LICENSE) (PolyForm Noncommercial), [data/weight terms](../third_party/echo_g/LICENSE-DATA-WEIGHTS) (CC BY-NC 4.0), and [notice](../third_party/echo_g/NOTICE). These terms apply to the upstream component and its weights; the optional weights are downloaded separately.
- [Unitree C++ SDK](../src/runtime/third_party/unitree_sdk2/LICENSE).
- [Unitree Python SDK](../third_party/unitree_sdk2_python/LICENSE).
- [XRoboToolkit Python binding](../third_party/xrobotoolkit_pybind/LICENSE).
- [Bundled XRoboToolkit SDK notice](../third_party/xrobotoolkit_sdk/LICENSE).

The source submodules must be initialized for their local license links to resolve; see [third-party setup](../third_party/README.md).
