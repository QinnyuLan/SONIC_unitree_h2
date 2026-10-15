# Deployment Tools

Tools prepare assets, install robot support services and validate the deployment. Online command producers live under [`src/interface/`](../src/interface/README.md).

| Directory | Purpose |
|---|---|
| `model_download/` | Download and verify the shared G1 planner |
| `setup/` | Prepare native dependency build sources |
| `validation/` | Release checks, model validation and integration checks |
| `robot_services/inspire_hands/` | H2 PC2 Inspire DDS-to-RS-485 service, client and systemd configuration |

Run script commands from the repository root, or use absolute script paths from another directory. Each script resolves repository resources from its own location.

## Asset and environment preparation

```bash
.venv_download/bin/python tools/model_download/download_planner.py
.venv_teleop/bin/python tools/setup/prepare_xrobotoolkit.py
```

The planner destination remains `models/motion_planner_g1/planner_sonic.onnx`; see [planner download](../docs/PLANNER.md). Teleoperation installation calls the XRoboToolkit preparation tool automatically and writes build sources under `build/third_party/`.

## Robot services

Build the Inspire hand service independently of the policy runtime:

```bash
cmake -S tools/robot_services/inspire_hands -B build/inspire_hands
cmake --build build/inspire_hands -j2
```

The service is installed on the robot computer connected to the hand serial ports. The DDS client reads state by default; motion tests require an explicit position option. Source transfer, serial mapping, systemd installation, autostart, watchdog recovery and rollback are documented in [robot deployment](../docs/INSPIRE_HANDS.md) / [中文](../docs/INSPIRE_HANDS_CN.md).

## Validation

| Tool | Environment and scope |
|---|---|
| `validation/verify_release.py` | Standard Python: files and hashes; optional simulation/teleoperation imports |
| `validation/check_native.sh` | Shell: native compiler, headers and source prerequisites |
| `validation/validate_released_models.py` | `.venv_model_validation`: ONNX contracts and CPU inference |
| `validation/check_runtime.py` | `.venv_planner`: actual TensorRT policy inference over private IPC; `--planner` also exercises the planner |
| `validation/check_inspire_watchdog.py` | Python 3.10+ and systemd user manager: temporary loopback service, process hang/crash recovery and explicit stop |

```bash
python3 tools/validation/verify_release.py
source scripts/setup_env.sh
bash tools/validation/check_native.sh
```

Dependency setup and the full validation commands are in [reproducibility](../docs/REPRODUCIBILITY.md). Generated reports belong in `outputs/validation/`.
