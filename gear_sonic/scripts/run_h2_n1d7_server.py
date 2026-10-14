#!/usr/bin/env python3
"""Serve the exported H2 N1.7 bundle using its isolated Python 3.12 runtime."""

import argparse
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party/isaac_gr00t_n1d7"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/h2_n1d7/latest")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5550)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    model = args.model.resolve(strict=True)
    manifest = json.loads((model / "deployment_manifest.json").read_text())
    if manifest["embodiment_tag"] != "UNITREE_G1_SONIC" or manifest["action_width"] != 76:
        raise ValueError("Not an exported H2 N1.7 bundle")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    # Both saved configs refer to ../backbone. No training-machine absolute path.
    os.chdir(model)
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.policy.server_client import PolicyServer
    import torch

    start = time.monotonic()
    policy = Gr00tPolicy("UNITREE_G1_SONIC", str(model), device=args.device)
    print(
        json.dumps(
            {
                "event": "model_loaded",
                "model": str(model),
                "step": manifest["training_step"],
                "load_seconds": time.monotonic() - start,
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(),
            }
        ),
        flush=True,
    )
    with PolicyServer(policy, host=args.host, port=args.port) as server:
        server.register_endpoint("h2_deployment_info", lambda: manifest, requires_input=False)
        server.run()


if __name__ == "__main__":
    main()
