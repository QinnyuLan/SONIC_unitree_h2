# Native Controller Setup

[Deployment guide](DEPLOYMENT.md) · [中文部署指南](DEPLOYMENT_CN.md)

The validated workstation uses Ubuntu x86_64, CUDA 12.6, TensorRT 10.13.3.9. The Python simulator and the C++ controller run in separate processes.

## System packages

The build requires CMake 3.18 or newer, a C++20 compiler, CUDA 12 and TensorRT 10.

```bash
sudo apt update
sudo apt install build-essential cmake pkg-config libzmq3-dev cppzmq-dev \
  libeigen3-dev nlohmann-json3-dev \
  libgl1 libglib2.0-0
```

## CUDA and TensorRT

Install the NVIDIA driver and CUDA toolkit using the [CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/). Extract the matching TensorRT 10.13.3.9 Linux archive from [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt/download) and set:

```bash
export CUDAToolkit_ROOT=/usr/local/cuda-12.6
export TensorRT_ROOT="$HOME/TensorRT"
```

`TensorRT_ROOT` must contain `include/NvInfer.h` and `lib/libnvinfer.so`. TensorRT engines are generated locally and are not portable between arbitrary TensorRT/GPU combinations.

## TensorRT Python planner

Install the planner environment after extracting the native SDK:

```bash
./install.sh planner
bash scripts/run_planner.sh --help
```

The installer uses the Python 3.10 wheel under `$TensorRT_ROOT/python` and pins CUDA Python bindings in `requirements/planner.lock`. For a separately stored wheel, set `TENSORRT_WHEEL=/absolute/path/to/tensorrt-10.13.3.9-cp310-none-linux_x86_64.whl`. Keep the Python and C++ TensorRT libraries at the same version.

The planner uses the [TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/python-api-docs.html), persistent pinned buffers and CUDA graphs. Neither the planner nor the native runtime requires ONNX Runtime. The optional offline model-validation environment uses it to compare model outputs.

## Build

From the repository root:

```bash
source scripts/setup_env.sh
./verify_deploy.sh
bash scripts/build_controller.sh
```

The build produces `build/runtime/bin/sonic_h2_runtime`; the planner is a Python module. Build and launch scripts do not install system packages or edit shell startup files.

For offline C++ tests, additionally install `libgtest-dev zlib1g-dev`, build with `-DBUILD_TESTING=ON`, then run `ctest --test-dir build/runtime --output-on-failure`.
