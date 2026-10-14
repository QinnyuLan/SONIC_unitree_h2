#!/usr/bin/env python3
"""Check released ONNX interfaces and finite CPU inference, without DDS or CUDA."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
MODELS = {"model_encoder_full": (1751, 64), "model_encoder": (640, 64), "model_decoder": (994, 29)}
CONTRACT_KEYS = (
    "model_role", "policy_interface", "observation_contract",
    "reference_orientation_contract", "selector_contract", "selector_encoder_names",
    "selector_slot_names", "input_layout_encoder_names", "input_features",
    "tokenizer_feature_aliases", "shared_robot_reference_slot",
)


def interface(model):
    def describe(values):
        return [{"name": value.name, "dtype": value.type.tensor_type.elem_type,
                 "shape": [dim.dim_value for dim in value.type.tensor_type.shape.dim]}
                for value in values]
    return {"inputs": describe(model.graph.input), "outputs": describe(model.graph.output)}


def validate(model_dir: Path, comparisons: list[Path]) -> dict:
    manifest = json.loads((model_dir / "deployment_manifest.json").read_text())
    report = {"model": "released", "models": {}, "previous_interface_matches": {}}
    sessions = {}
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    for name, (input_dim, output_dim) in MODELS.items():
        path = model_dir / f"{name}.onnx"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != manifest["model_sha256"][name]:
            raise ValueError(f"SHA256 mismatch: {name}")
        model = onnx.load(path)
        onnx.checker.check_model(model, full_check=True)
        signature = interface(model)
        expected = {
            "inputs": [{"name": "obs_dict", "dtype": onnx.TensorProto.FLOAT, "shape": [1, input_dim]}],
            "outputs": [{"name": "action" if name == "model_decoder" else "encoded_tokens",
                         "dtype": onnx.TensorProto.FLOAT, "shape": [1, output_dim]}],
        }
        if signature != expected:
            raise ValueError(f"Unexpected input/output interface: {name}")
        metadata = {item.key: item.value for item in model.metadata_props}
        required = {
            "model_role": "decoder" if name == "model_decoder" else "encoder",
            "policy_interface": "universal" if name == "model_encoder_full" else "h2_native",
            "observation_contract": "sonic_h2_native_v1",
            "reference_orientation_contract": "sonic_full_root_relative_v0",
        }
        if name == "model_encoder_full":
            required["selector_contract"] = "sonic_h2_native_full_v1"
            if json.loads(metadata.get("selector_encoder_names", "[]")) != ["h2", "teleop", "smpl", "g1"]:
                raise ValueError("Unexpected selector routing")
        for key, value in required.items():
            if metadata.get(key) != value:
                raise ValueError(f"{name}: incompatible {key}")
        recorded = manifest["onnx_check"][path.name]
        for kind in ("inputs", "outputs"):
            if recorded[kind] != [[item["name"], *item["shape"]] for item in signature[kind]]:
                raise ValueError(f"{name}: manifest {kind} do not match ONNX")
        if recorded["input_dtype"] != "float32" or recorded["output_dtype"] != "float32":
            raise ValueError(f"{name}: manifest tensor type mismatch")
        if not required.keys() <= recorded["metadata"].keys() or any(
                metadata.get(key) != value for key, value in recorded["metadata"].items()):
            raise ValueError(f"{name}: manifest metadata do not match ONNX")
        for index, previous_dir in enumerate(comparisons, start=1):
            previous = onnx.load(previous_dir / f"{name}.onnx")
            previous_metadata = {item.key: item.value for item in previous.metadata_props}
            matches = signature == interface(previous) and all(
                metadata.get(key) == previous_metadata.get(key) for key in CONTRACT_KEYS)
            report["previous_interface_matches"].setdefault(f"baseline_{index}", {})[name] = matches
            if not matches:
                raise ValueError(f"Interface differs from baseline {index}: {name}")
        sessions[name] = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        report["models"][name] = {"sha256": digest, **signature,
                                  "onnx_checker": "passed", "manifest_check": "passed", "contract": required}

    def infer(name, values):
        result = sessions[name].run(None, {"obs_dict": values})[0]
        if not np.isfinite(result).all():
            raise ValueError(f"Non-finite inference: {name}")
        return result

    rng = np.random.default_rng(42)
    max_error = 0.0
    for selector in range(4):
        for _ in range(5):
            values = rng.normal(0, 0.1, (1, 1751)).astype(np.float32)
            values[:, :4] = 0
            values[:, 0] = selector
            tokens = infer("model_encoder_full", values)
            history = rng.normal(0, 0.1, (1, 930)).astype(np.float32)
            infer("model_decoder", np.concatenate([tokens, history], axis=1))
            if selector == 0:
                compact = np.concatenate([values[:, 4:584], values[:, 590:650]], axis=1)
                other = infer("model_encoder", compact)
                np.testing.assert_allclose(tokens, other, atol=1e-6, rtol=1e-6)
                max_error = max(max_error, float(np.abs(tokens - other).max()))
    report.update(status="passed", cpu_trials_per_selector=5, cpu_finite_selectors=[0, 1, 2, 3],
                  full_vs_compact_max_abs_error=max_error, onnx_version=onnx.__version__,
                  onnxruntime_version=ort.__version__)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/sonic_h2_released")
    parser.add_argument("--compare", type=Path, action="append", default=[])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = validate(args.model_dir, args.compare)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("released: SHA256, ONNX, I/O contracts and all four CPU inference routes PASS")
    print(f"Full vs compact encoder maximum error: {report['full_vs_compact_max_abs_error']}")
    if args.compare:
        print(f"Input/output and runtime contracts match all {len(args.compare)} baselines")


if __name__ == "__main__":
    main()
