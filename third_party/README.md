# Third-party Dependencies

The external Python sources are unmodified Git submodules pinned to these commits:

| Path | Upstream | Commit |
|---|---|---|
| `unitree_sdk2_python/` | [Unitree Python SDK](https://github.com/unitreerobotics/unitree_sdk2_python) | [`892693e`](https://github.com/unitreerobotics/unitree_sdk2_python/commit/892693efe035ae3db7a62ed429ebd0bc15f88ca7) |
| `xrobotoolkit_pybind/` | [XRoboToolkit Python binding](https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind) | [`c64ccf6`](https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind/commit/c64ccf6acd577a333e03b66fafe8efeeceb511b1) |

Clone the release with its dependencies:

```bash
git clone --recurse-submodules https://github.com/QinnyuLan/SONIC_unitree_h2.git
```

For an existing checkout:

```bash
git submodule update --init --recursive
```

`./install.sh sim2sim` and `./install.sh teleop` also initialize their required submodule. They use the commit recorded by the parent repository; they do not follow the upstream branch tip.

## Local integration

The simulator's `OdoState_` extension lives in [`src/sim/odometry.py`](../src/sim/odometry.py). Its DDS type name, type identifier and serialized field layout match the previous SDK extension. The published LowCmd/LowState definitions remain upstream code. Python 3.10 and CycloneDDS 0.10.2 are pinned by this project's environment.

The XRoboToolkit source, setup code, binding and common headers match the pinned upstream commit. The additional native SDK libraries and ARM64 headers are retained under [`xrobotoolkit_sdk/`](xrobotoolkit_sdk/README.md). During teleoperation installation, `tools/setup/prepare_xrobotoolkit.py` verifies their checksums and prepares a copy of the binding sources plus the platform's SDK in `build/third_party/xrobotoolkit_pybind/`. Compilation and Python installation use that generated directory, leaving the source submodule untouched.

ECHO-G generation uses a minimal local inference implementation under `tools/text2motion_offline/echo_g/`, with separately downloaded weights. Its source and weight notices are retained in [licenses/echo_g/](../licenses/echo_g/README.md); it is not a source submodule. See [text-to-motion setup](../docs/TEXT2MOTION.md).

The C++ runtime's Unitree SDK remains under `src/runtime/third_party/`; it is managed with the native runtime. The SDK libraries and robot assets retain their original licenses; see the [license index](../licenses/README.md).

To update a submodule, select and test a specific upstream commit, then commit the updated Git link in this repository. Changes specific to this deployment belong in `src/`, rather than in the submodule working trees.
