# Deployment Troubleshooting

[Deployment guide](DEPLOYMENT.md) · [中文](TROUBLESHOOTING_CN.md) · [Native setup](NATIVE_SETUP.md)

Use this guide when installation or a launch step fails. Run commands from the repository root. For normal operation, follow the [deployment guide](DEPLOYMENT.md).

| Symptom | Start here |
|---|---|
| Missing Python modules, `uv`, or TensorRT wheel | [Python environments](#python-environments) |
| CMake errors or missing shared libraries | [Native build and libraries](#native-build-and-libraries) |
| Missing or invalid model files | [Models and engine caches](#models-and-engine-caches) |
| Slow startup, CUDA initialization stuck, Ctrl+C ineffective | [GPU startup](#gpu-startup) |
| Runtime or input panel keeps waiting; commands have no effect | [Connections and controls](#connections-and-controls) |
| MuJoCo window fails or simulation stops on lag | [Simulator](#simulator) |

## Python environments

Use the interpreter belonging to the process you are launching:

| Process | Interpreter | Install or repair |
|---|---|---|
| Replay and command clients | `.venv_inputs/bin/python` | `./install.sh inputs` |
| MuJoCo simulator | `.venv_sim/bin/python` | `./install.sh sim2sim` |
| Planner | `.venv_planner/bin/python` | `./install.sh planner` |
| PICO and teleoperation bridge | `.venv_teleop/bin/python` | `./install.sh teleop` |

For `ModuleNotFoundError`, rerun the matching installer and launch with that environment's interpreter. Installing into a different system or Conda environment will not repair these environments. If `uv` is missing, follow the [dependency setup instructions](../requirements/README.md).

To check an environment without launching control:

```bash
uv pip check --python .venv_inputs/bin/python
.venv_inputs/bin/python -m interface.motion_replay --help
```

If installation reports a missing TensorRT Python wheel, set `TensorRT_ROOT` to the extracted SDK. The installer expects its Python 3.10 wheel under `$TensorRT_ROOT/python`; use `TENSORRT_WHEEL=/absolute/path/to/the-matching-wheel.whl` if it is stored elsewhere. Keep the Python binding and native SDK at the same version. See [native setup](NATIVE_SETUP.md#tensorrt-python-planner).

For missing Unitree or XRoboToolkit sources, initialize the pinned submodules and rerun the relevant installer:

```bash
git submodule update --init --recursive
```

## Native build and libraries

If CMake cannot locate CUDA, TensorRT, ZeroMQ, Eigen or JSON headers, complete [native setup](NATIVE_SETUP.md). Set the actual SDK paths in the current terminal:

```bash
export TensorRT_ROOT="$HOME/TensorRT"
export CUDAToolkit_ROOT=/usr/local/cuda-12.6
source scripts/setup_env.sh
bash tools/validation/check_native.sh
bash scripts/build_controller.sh
```

If compilation is killed while using several jobs, reduce concurrency with `BUILD_JOBS=2 bash scripts/build_controller.sh`.

For startup errors such as `libnvinfer.so: cannot open shared object file`, inspect library resolution after sourcing the same environment:

```bash
source scripts/setup_env.sh
ldd build/runtime/bin/sonic_h2_runtime
```

Resolve entries marked `not found` against the installed SDK paths. The launch scripts source `scripts/setup_env.sh` automatically; a directly launched executable or Python planner needs the same environment. Rebuild after changing native SDK installations.

## Models and engine caches

Check model presence, manifests and hashes:

```bash
python3 tools/validation/verify_release.py
# Also require the optional planner model:
python3 tools/validation/verify_release.py --require-planner
```

The H2 runtime expects the encoder, decoder and `deployment_manifest.json` under `models/sonic_h2_released/`. A failed model check should be resolved by restoring the matching release files, without editing their recorded hashes.

The planner model is downloaded separately to `models/motion_planner_g1/planner_sonic.onnx`. Use the [planner download procedure](PLANNER.md); saving the Hugging Face HTML page as an ONNX file will fail validation.

TensorRT caches are generated beside the models and are reused when compatible. Missing or incompatible caches trigger a local build. This can take several minutes on first launch. Inspect the startup log before changing caches; a CUDA driver wait is not resolved by deleting them. Stop the affected process before managing its cache, and preserve the ONNX files and manifests.

## GPU startup

### Identify the startup stage

Runtime logs distinguish CUDA device initialization, TensorRT cache reuse or engine construction, and CUDA graph capture. DDS control starts after both policy models are ready. The planner keyboard worker saves its output under `outputs/planner/keyboard-*.log`.

| Last reported stage | What to check |
|---|---|
| CUDA device initialization | If it remains stuck, inspect process state as below |
| Missing/incompatible TensorRT cache; building engine | Allow the first build to complete and follow its logs |
| CUDA allocation or engine error | Read the reported error; check SDK paths and available GPU resources |
| Models ready, but no **[READY]** | Check DDS state delivery in [Connections and controls](#connections-and-controls) |

During model initialization, **Ctrl+C** terminates the runtime directly. Once initialization finishes, it requests an orderly control shutdown. In planner keyboard mode, **Esc** or **Ctrl+C** closes the panel and its worker.

### CUDA initialization does not return

If startup remains at `Initializing CUDA device ...`, inspect processes from another terminal before starting another GPU query:

```bash
ps -C sonic_h2_runtime,nvidia-smi -o pid,stat,etime,wchan:40,args
```

A persistent `D` state with a wait location such as `uvm_gpu_retain_by_uuid` or `uvm_va_space_destroy` indicates a wait inside the NVIDIA driver. Signals may remain pending until the kernel call returns; even `kill -9` cannot force immediate termination during an uninterruptible wait. This differs from TensorRT engine construction. Avoid repeatedly launching runtimes or `nvidia-smi` while this condition persists.

If the driver remains blocked, save other work and arrange a host reboot. GPU reset requires the GPU to have no active clients and is not guaranteed to recover a hung GPU; see [NVIDIA's recovery requirements](https://docs.nvidia.com/deploy/nvidia-smi/index.html#r-gpu-reset). After recovery, confirm that `nvidia-smi` returns normally, then restart the deployment workflow. Clearing model caches does not resolve a driver wait.

## Connections and controls

### Runtime never reaches [READY]

Start the simulator first. Simulator and runtime must use the same DDS domain; the simulation launchers use loopback and domain 0 by default. For a different simulation domain, set the same `DDS_DOMAIN` in both terminals before starting them.

For hardware, check `NETWORK_INTERFACE` and `DDS_DOMAIN` in `config/h2_deploy.env` against the robot connection. Preview the launch configuration without starting control:

```bash
ip -brief address
bash scripts/run_sonic_h2_runtime.sh real --print-command
```

Do not run simulation and hardware control on a shared robot-facing DDS network. Keep the documented loopback configuration for simulation.

### Input is waiting or motion does not start

- **Policy is waiting:** **[READY]** means PD hold. Press **]** or **s** in the runtime terminal after that message; an early keypress is discarded.
- **Replay is selected but stopped:** selecting a collection or clip does not start playback. Press Enter in the replay terminal.
- **Planner mode is selected but inactive:** wait for the worker to be ready, then use a movement key or Enter. Mode/speed selection alone does not activate requests.
- **Another producer owns the body input:** stop and exit the previous replay, planner or teleoperation process before starting another.
- **Runtime state was interrupted:** replay stays paused and planner requests are cleared. Restore the connection, then resume manually.
- **Runtime was stopped with o:** restart it, wait for **[READY]**, and enable policy again. This stop is latched.

### Address already in use or no state received

Inspect listening processes with `ss -ltnp`. A previous runtime or planner may still own the port. Close the process belonging to the previous session; do not start a second controller on the same robot.

Default runtime endpoints are body **5558**, state **5559**, head **5562** and hands **5563**, bound to `127.0.0.1`. Planner requests and output use **5560** and **5561**. Across machines, set explicit runtime bind addresses and matching client endpoints, and synchronize system clocks because commands expire by timestamp. See [ports and protocol](ARCHITECTURE.md).

## Simulator

For MuJoCo import or asset errors, run the offline simulator check:

```bash
.venv_sim/bin/python tools/validation/verify_release.py --imports --sim
```

This loads assets and steps physics without DDS or a graphical window. If it passes but the viewer fails, check the graphical session and OpenGL packages in [native setup](NATIVE_SETUP.md). A remote terminal needs access to a working display to open the viewer.

If the simulator exits on its realtime-lag guard, inspect the reported lag and host load. Finish any first-time engine build, close unnecessary workloads, then restart the simulator and runtime. Offline checks do not establish real-time performance; do not relax the guard to claim a successful control test.

For PICO connection/calibration issues, follow [VR setup](VR_SETUP.md). For robot-side hand service logs, serial mapping and watchdog recovery, use the [hand service guide](INSPIRE_HANDS.md).
