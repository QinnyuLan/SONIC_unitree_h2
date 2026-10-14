# Native Controller Setup

[Deployment guide](DEPLOYMENT.md) · [中文部署指南](DEPLOYMENT_CN.md)

The validated workstation uses Ubuntu x86_64, CUDA 12.6, TensorRT 10.13.3.9, and ONNX Runtime 1.16.3. The Python simulator and the C++ controller run in separate processes.

## System packages

```bash
sudo apt update
sudo apt install build-essential cmake pkg-config libzmq3-dev cppzmq-dev \
  libeigen3-dev nlohmann-json3-dev zlib1g-dev \
  libgl1 libglib2.0-0
```

## CUDA and TensorRT

Install the NVIDIA driver and CUDA toolkit using the [CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/). Extract the matching TensorRT 10.13.3.9 Linux archive from [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt/download) and set:

```bash
export CUDAToolkit_ROOT=/usr/local/cuda-12.6
export TensorRT_ROOT="$HOME/TensorRT"
```

`TensorRT_ROOT` must contain `include/NvInfer.h` and `lib/libnvinfer.so`. TensorRT engines are generated locally and are not portable between arbitrary TensorRT/GPU combinations.

## ONNX Runtime C++

For the validated x86_64 setup:

```bash
curl -fLO https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-1.16.3.tgz
sudo mkdir -p /opt/onnxruntime
sudo cp -a onnxruntime-linux-x64-1.16.3/include onnxruntime-linux-x64-1.16.3/lib /opt/onnxruntime/
```

The C++ runtime uses this library for model metadata and inference support. Installing the Python `onnxruntime` wheel does not provide these C++ development files.

## Build

From the repository root:

```bash
./verify_deploy.sh
bash scripts/build_controller.sh
```

The build produces `gear_sonic_deploy/target/release/sonic_h2_runtime` and `sonic_g1_planner`. Use `-DBUILD_PLANNER=OFF` to build only the policy runtime. Build and launch scripts do not install system packages or edit shell startup files.

For offline C++ tests, additionally install `libgtest-dev`, build with `-DBUILD_TESTING=ON`, then run `ctest --test-dir gear_sonic_deploy/build --output-on-failure`.
