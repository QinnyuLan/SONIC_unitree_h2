# G1 Planner

The optional Python planner service loads `models/motion_planner_g1/planner_sonic.onnx`. The downloader and service use this same default path. The weight and generated TensorRT engines are ignored by Git.

From the repository root:

```bash
uv venv .venv_download --python 3.10
uv pip install --python .venv_download/bin/python -r requirements/download.txt
.venv_download/bin/python tools/model_download/download_planner.py
python3 tools/validation/verify_release.py --require-planner
```

The downloader retrieves the pinned SONIC weight from [nvidia/GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC) and verifies its SHA256. It reuses a matching local file without downloading it again.

See [planner setup](../../docs/PLANNER.md) for the source revision and checksum, and [deployment](../../docs/DEPLOYMENT.md#planner) for installation and launch commands.
