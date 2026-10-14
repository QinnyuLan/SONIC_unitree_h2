#!/usr/bin/env bash
export TensorRT_ROOT="${TensorRT_ROOT:-$HOME/TensorRT}"
export CUDAToolkit_ROOT="${CUDAToolkit_ROOT:-/usr/local/cuda}"
export HAS_ROS2=0
export PATH="$CUDAToolkit_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$TensorRT_ROOT/lib:$CUDAToolkit_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
