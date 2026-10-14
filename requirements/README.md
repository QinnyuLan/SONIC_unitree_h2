# Dependencies

| Environment | Purpose | Install |
|---|---|---|
| `.venv_inputs` | Motion replay, planner requests, head/hand clients | `./install.sh inputs` |
| `.venv_planner` | G1 planner, TensorRT Python API and CUDA bindings | `./install.sh planner` |
| `.venv_sim` | MuJoCo simulation and Unitree DDS | `./install.sh sim2sim` |
| `.venv_teleop` | PICO input, CPU pose mathematics, robot kinematics, visualization | `./install.sh teleop` |
| `.venv_text2motion` | Optional ECHO-G generation, text/audio encoders and TTS | `./install.sh text2motion` |
| `.venv_download` | Optional planner download | `requirements/download.txt` |
| `.venv_model_validation` | Optional ONNX validation | `requirements/model_validation.txt` |

Direct dependencies are declared in the root [pyproject.toml](../pyproject.toml). The installer installs the `interface` and `sim` packages from `src/` in editable mode; assets and configuration remain in this checkout. The installer also applies [inputs.lock](inputs.lock), [planner.lock](planner.lock), [sim.lock](sim.lock), [teleop.lock](teleop.lock), or [text2motion.lock](text2motion.lock) to pin transitive dependencies for Python 3.10 on Linux. Installation requires [uv](https://docs.astral.sh/uv/getting-started/installation/), Python 3.10, and a C++ compiler for the XRoboToolkit binding. The teleoperation and optional text-to-motion environments use CPU PyTorch; planner and policy inference use TensorRT. The planner environment installs the TensorRT Python wheel from the matching native SDK and does not install PyTorch. Simulation does not install PyTorch or Pinocchio.

The native runtime requires CUDA, TensorRT, ZeroMQ, Eigen, and JSON headers. GTest and zlib are needed only for the C++ regression tests; pytest is needed for the Python tests. The Unitree C++ SDK is included under `src/runtime/third_party/`. See the [deployment guide](../docs/DEPLOYMENT.md) for installation and build commands.

The pinned submodules under `third_party/` supply the Unitree Python SDK and XRoboToolkit binding. Installation initializes the required source submodule automatically; the XR binding is built under `build/third_party/`. See [dependency setup](../third_party/README.md). The XRoboToolkit PC Service is installed separately on the workstation; see [VR setup](../docs/VR_SETUP.md).

The ECHO-G environment loads weights through the local inference implementation; it does not install the upstream package. The pinned environment contains 50 dependencies plus this project. It reads PCM WAV with the standard library, without librosa, soundfile or their audio-analysis dependencies. Install `curl` for model downloads, `ffmpeg` for the optional Edge TTS adapter and `alsa-utils` for local sound playback. Generation models are downloaded separately; neither replay nor the runtime loads them. See [text-to-motion](../docs/TEXT2MOTION.md).
