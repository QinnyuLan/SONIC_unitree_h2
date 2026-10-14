"""TensorRT 10 inference with persistent buffers and a captured CUDA graph.

The cache format matches the C++ runtime: a 64-byte compatibility digest followed
by the serialized engine. No PyTorch or ONNX Runtime is needed.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cuda


def checked(result):
    error, *values = result
    if int(error):
        raise RuntimeError(f"CUDA operation failed: {error}")
    return values[0] if len(values) == 1 else values


def cache_digest(model: Path, device: int) -> str:
    digest = hashlib.sha256()
    with model.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    key = digest.hexdigest()
    gpu = checked(cuda.cudaGetDeviceProperties(device)).name.decode()
    major, minor, patch, build = map(int, trt.__version__.split("."))
    for part in (gpu, "0", f"{major * 10000 + minor * 100 + patch}.{build}"):
        key = hashlib.sha256((key + part).encode()).hexdigest()
    return key


class TensorRTEngine:
    """Single-threaded FP32 engine. Each instance owns its stream and buffers."""

    def __init__(self, model: str | Path, *, prefix="planner_", device=0):
        if not trt.__version__.startswith("10."):
            raise RuntimeError("This deployment requires TensorRT 10.x")
        self.device = device
        self.stream = self.graph = self.graph_exec = None
        self.buffers = {}
        self.context = self.engine = self.runtime = None
        self.logger = trt.Logger(trt.Logger.WARNING)
        checked(cuda.cudaSetDevice(device))
        model = Path(model).resolve(strict=True)
        self.cache = model.with_name(prefix + model.stem + ".trt")
        key = cache_digest(model, device).encode()
        try:
            self.runtime = trt.Runtime(self.logger)
            if self.cache.exists():
                with self.cache.open("rb") as stream:
                    if stream.read(64) == key:
                        self.engine = self.runtime.deserialize_cuda_engine(stream.read())
            if self.engine is None:
                self.engine = self.runtime.deserialize_cuda_engine(self._build(model, key))
            if self.engine is None:
                raise RuntimeError("TensorRT could not load the engine")
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("TensorRT could not create an execution context")
            self.stream = checked(cuda.cudaStreamCreate())
            self.inputs, self.outputs = {}, {}
            for index in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(index)
                shape = tuple(self.engine.get_tensor_shape(name))
                if any(size <= 0 for size in shape):
                    raise ValueError(f"Deployment requires static tensor shapes: {name} {shape}")
                if self.engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
                    raise ValueError(f"Expected a device tensor: {name}")
                dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
                size = int(np.prod(shape)) * dtype.itemsize
                host = checked(cuda.cudaHostAlloc(size, 0))
                self.buffers[name] = [host, None, None]
                pointer = checked(cuda.cudaMalloc(size))
                storage = (ctypes.c_byte * size).from_address(host)
                array = np.frombuffer(storage, dtype=dtype).reshape(shape)
                array.fill(0)
                self.buffers[name] = [host, pointer, array]
                if not self.context.set_tensor_address(name, pointer):
                    raise RuntimeError(f"Could not bind {name}")
                tensors = self.inputs if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else self.outputs
                tensors[name] = (shape, dtype)
        except BaseException:
            self.close()
            raise

    def _build(self, model, key):
        print(f"Building TensorRT engine: {model.name}", flush=True)
        builder = trt.Builder(self.logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, self.logger)
        if not parser.parse_from_file(str(model)):
            raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        config = builder.create_builder_config()
        # Match the native FP32 builder: FP16 is disabled, default TF32 is retained.
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT engine build failed")
        payload = bytes(serialized)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.cache.parent, prefix=".trt-", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(key)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.cache)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return payload

    def infer(self, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.context is None:
            raise RuntimeError("TensorRT engine is closed")
        if values.keys() != self.inputs.keys():
            raise ValueError(f"Expected inputs {sorted(self.inputs)}, got {sorted(values)}")
        # Validate the complete request before submitting any asynchronous copies.
        for name, (shape, dtype) in self.inputs.items():
            value = np.asarray(values[name])
            if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
                raise ValueError(f"{name} requires finite {dtype} with shape {shape}")
        for name in self.inputs:
            host, pointer, array = self.buffers[name]
            np.copyto(array, values[name])
            checked(cuda.cudaMemcpyAsync(pointer, host, array.nbytes,
                cuda.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream))
        if self.graph_exec is None:
            if not self.context.execute_async_v3(stream_handle=int(self.stream)):
                raise RuntimeError("TensorRT inference failed")
        else:
            checked(cuda.cudaGraphLaunch(self.graph_exec, self.stream))
        for name in self.outputs:
            host, pointer, array = self.buffers[name]
            checked(cuda.cudaMemcpyAsync(host, pointer, array.nbytes,
                cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream))
        checked(cuda.cudaStreamSynchronize(self.stream))
        result = {name: self.buffers[name][2].copy() for name in self.outputs}
        if self.graph_exec is None:
            checked(cuda.cudaStreamBeginCapture(self.stream, cuda.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal))
            success = self.context.execute_async_v3(stream_handle=int(self.stream))
            self.graph = checked(cuda.cudaStreamEndCapture(self.stream))
            if not success:
                raise RuntimeError("TensorRT CUDA graph capture failed")
            self.graph_exec = checked(cuda.cudaGraphInstantiate(self.graph, 0))
        return result

    def close(self):
        if self.stream is not None:
            cuda.cudaStreamSynchronize(self.stream)
        if self.graph_exec is not None:
            cuda.cudaGraphExecDestroy(self.graph_exec)
            self.graph_exec = None
        if self.graph is not None:
            cuda.cudaGraphDestroy(self.graph)
            self.graph = None
        self.context = self.engine = self.runtime = None
        for host, pointer, _ in self.buffers.values():
            if pointer is not None:
                cuda.cudaFree(pointer)
            cuda.cudaFreeHost(host)
        self.buffers.clear()
        if self.stream is not None:
            cuda.cudaStreamDestroy(self.stream)
            self.stream = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
