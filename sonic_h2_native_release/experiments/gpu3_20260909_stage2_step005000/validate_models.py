"""Offline checks for this export; no DDS or robot control."""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import yaml

HERE = Path(__file__).resolve().parent
manifest = json.loads((HERE / "deployment_manifest.json").read_text())
assert manifest["global_step"] == 5000
assert manifest["training_stage"] == 2
assert manifest["h2_native_io_folding"]["folded"] is True
assert manifest["frozen_encoder_validation"]["exact_match"] is True
report = {"step": 5000, "training_stage": 2, "models": {}}
sessions = {}
options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
for name, dims in {
    "model_encoder": (640, 64),
    "model_encoder_full": (1751, 64),
    "model_decoder": (994, 29),
}.items():
    path = HERE / (name + ".onnx")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == manifest["model_sha256"][name], name
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    metadata = {p.key: p.value for p in model.metadata_props}
    assert metadata["global_step"] == "5000"
    assert metadata["checkpoint_sha256"] == manifest["checkpoint_sha256"]
    assert metadata["observation_contract"] == "sonic_h2_native_v1"
    assert metadata["reference_orientation_contract"] == "sonic_full_root_relative_v0"
    if name == "model_decoder":
        assert metadata["h2_native_io_folded"] == "true"
    else:
        assert metadata.get("h2_native_io_folded", "false") == "false"
    if name == "model_encoder_full":
        assert metadata["selector_contract"] == "sonic_h2_native_full_v1"
        assert json.loads(metadata["selector_encoder_names"]) == ["h2", "teleop", "smpl", "g1"]
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape == [1, dims[0]]
    assert session.get_outputs()[0].shape == [1, dims[1]]
    sessions[name] = session
    report["models"][name] = dict(sha256=digest, metadata=metadata,
        input_shape=[1, dims[0]], output_shape=[1, dims[1]], onnx_checker="passed")

def infer(name, x):
    session = sessions[name]
    y = session.run(None, {session.get_inputs()[0].name: x})[0]
    assert np.isfinite(y).all()
    return y

rng = np.random.default_rng(5000)
max_error = 0.0
for selector in range(4):
    for _ in range(5):
        x = rng.normal(0, .1, (1, 1751)).astype(np.float32)
        x[:, :4] = 0
        x[:, 0] = selector
        tokens = infer("model_encoder_full", x)
        proprio = rng.normal(0, .1, (1, 930)).astype(np.float32)
        infer("model_decoder", np.concatenate([tokens, proprio], axis=1))
        if selector == 0:
            # Universal layout: selector 4, q/dq 580, current orientation 6,
            # future orientation 60. Compact H2 omits selector/current orientation.
            compact = np.concatenate([x[:, 4:584], x[:, 590:650]], axis=1)
            single = infer("model_encoder", compact)
            np.testing.assert_allclose(tokens, single, atol=1e-6, rtol=1e-6)
            max_error = max(max_error, float(np.abs(tokens - single).max()))
old = yaml.safe_load((HERE.parent / "gpu3_20260907_stage2_step009450/model_config.yaml").read_text())
new = yaml.safe_load((HERE / "model_config.yaml").read_text())
changes = []

def compare(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(a.keys() | b.keys()):
            compare(a.get(key), b.get(key), path + "." + key)
    elif a != b:
        changes.append(dict(path=path.lstrip("."), previous=a, current=b))

compare(old, new)
report.update(cpu_trials_per_selector=5, cpu_finite_selectors=[0, 1, 2, 3],
    full0_vs_h2_only_max_abs_error=max_error, config_changes_since_gpu3_step9450=changes)
(HERE / "local_model_validation.json").write_text(json.dumps(report, indent=2) + "\n")
(HERE / "SHA256SUMS").write_text("".join(
    f"{value['sha256']}  {name}.onnx\n" for name, value in report["models"].items()))
print(json.dumps({"step": 5000, "models_checked": 3, "finite_selectors": [0, 1, 2, 3],
    "full0_vs_h2_only_max_abs_error": max_error, "config_changes": changes}, indent=2))
