# Dependencies

| Environment | Purpose | Install |
|---|---|---|
| `.venv_inputs` | Motion replay, planner requests, head/hand clients | `./install.sh inputs` |
| `.venv_sim` | MuJoCo simulation and Unitree DDS | `./install.sh sim2sim` |
| `.venv_teleop` | PICO input, CPU pose mathematics, robot kinematics, visualization | `./install.sh teleop` |
| `.venv_download` | Optional planner download | `requirements/download.txt` |
| `.venv_model_validation` | Optional ONNX validation | `requirements/model_validation.txt` |

Direct dependencies are declared in [gear_sonic/pyproject.toml](../gear_sonic/pyproject.toml). The installer also applies [inputs.lock](inputs.lock), [sim.lock](sim.lock), or [teleop.lock](teleop.lock) to pin transitive dependencies for Python 3.10 on Linux. Installation requires [uv](https://docs.astral.sh/uv/getting-started/installation/), Python 3.10, and a C++ compiler for the XRoboToolkit binding. The teleoperation environment uses CPU PyTorch; ONNX inference runs in the C++ controller. Simulation does not install PyTorch or Pinocchio.

The native controller requires CUDA, TensorRT, ONNX Runtime C++, ZeroMQ, Eigen, JSON headers, and zlib. GTest and pytest are needed only for regression tests. See the [deployment guide](../docs/DEPLOYMENT.md) for installation and build commands.

The included Unitree SDK and XRoboToolkit binding supply DDS/device communication. The XRoboToolkit PC Service is installed separately on the workstation; see [VR setup](../docs/VR_SETUP.md).
