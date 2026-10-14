# Deployment Tools

Tools prepare assets, generate offline content and validate the deployment. Online command producers live under [`src/interface/`](../src/interface/README.md).

| Directory | Purpose |
|---|---|
| `model_download/` | Download and verify the shared G1 planner |
| `setup/` | Prepare native dependency build sources |
| `validation/` | Release checks, model validation and integration checks |
| `text2motion_offline/echo_g/` | Generate speech and G1 motion before playback |

Run script commands from the repository root, or use absolute script paths from another directory. Each script resolves repository resources from its own location. The installed `text2motion_offline` package also supports module execution outside the checkout.

## Asset and environment preparation

```bash
.venv_download/bin/python tools/model_download/download_planner.py
.venv_teleop/bin/python tools/setup/prepare_xrobotoolkit.py
```

The planner destination remains `models/motion_planner_g1/planner_sonic.onnx`; see [planner download](../docs/PLANNER.md). Teleoperation installation calls the XRoboToolkit preparation tool automatically and writes build sources under `build/third_party/`.

## Offline text-to-motion

```bash
./install.sh text2motion
.venv_text2motion/bin/python -m text2motion_offline.echo_g download
.venv_text2motion/bin/python -m text2motion_offline.echo_g generate \
  --text "Hello. Welcome to our lab." --language en \
  --output outputs/text2motion/greeting_en
```

The output is a portable motion/audio bundle. Send it through the independent replay process:

```bash
.venv_inputs/bin/python -m interface.motion_replay outputs/text2motion/greeting_en
```

See [text-to-motion](../docs/TEXT2MOTION.md) / [中文](../docs/TEXT2MOTION_CN.md) for dependencies, existing speech input and speaker setup.

## Validation

| Tool | Environment and scope |
|---|---|
| `validation/verify_release.py` | Standard Python: files and hashes; optional simulation/teleoperation imports |
| `validation/check_native.sh` | Shell: native compiler, headers and source prerequisites |
| `validation/validate_released_models.py` | `.venv_model_validation`: ONNX contracts and CPU inference |
| `validation/check_runtime.py` | `.venv_planner`: actual TensorRT policy inference over private IPC; `--planner` also exercises the planner |
| `validation/check_audio_replay.py` | `.venv_sim`: mock speaker service on loopback DDS, replay completion and interruption |

```bash
python3 tools/validation/verify_release.py
source scripts/setup_env.sh
bash tools/validation/check_native.sh
```

Dependency setup and the full validation commands are in [reproducibility](../docs/REPRODUCIBILITY.md). Generated reports belong in `outputs/validation/`.
