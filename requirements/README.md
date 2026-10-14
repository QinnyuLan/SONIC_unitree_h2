# Dependencies

| Environment | Purpose | Install |
|---|---|---|
| `.venv_inputs` | Motion replay, planner requests, head/hand clients | `./install.sh inputs` |
| `.venv_planner` | G1 planner, TensorRT Python API and CUDA bindings | `./install.sh planner` |
| `.venv_sim` | MuJoCo simulation and Unitree DDS | `./install.sh sim2sim` |
| `.venv_teleop` | PICO input, CPU pose mathematics, robot kinematics, visualization | `./install.sh teleop` |
| `.venv_download` | Optional planner download | `requirements/download.txt` |
| `.venv_model_validation` | Optional ONNX validation | `requirements/model_validation.txt` |

Direct dependencies are declared in [gear_sonic/pyproject.toml](../gear_sonic/pyproject.toml). The installer also applies [inputs.lock](inputs.lock), [planner.lock](planner.lock), [sim.lock](sim.lock), or [teleop.lock](teleop.lock) to pin transitive dependencies for Python 3.10 on Linux. Installation requires [uv](https://docs.astral.sh/uv/getting-started/installation/), Python 3.10, and a C++ compiler for the XRoboToolkit binding. The teleoperation environment uses CPU PyTorch; planner and policy inference use TensorRT. The planner environment installs the TensorRT Python wheel from the matching native SDK and does not install PyTorch. Simulation does not install PyTorch or Pinocchio.

The native controller requires CUDA, TensorRT, ZeroMQ, Eigen, JSON headers, and zlib. GTest and pytest are needed only for regression tests. See the [deployment guide](../docs/DEPLOYMENT.md) for installation and build commands.

The included Unitree SDK and XRoboToolkit binding supply DDS/device communication. The XRoboToolkit PC Service is installed separately on the workstation; see [VR setup](../docs/VR_SETUP.md).
