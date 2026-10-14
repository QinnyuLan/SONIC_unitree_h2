# Documentation

[English README](../README.md) · [中文首页](../README_CN.md)

## Deployment

| Topic | Documentation |
|---|---|
| Environment setup and controller build | [English](DEPLOYMENT.md#environment) · [中文](DEPLOYMENT_CN.md#环境配置) |
| MuJoCo simulation | [English](DEPLOYMENT.md#sim2sim) · [中文](DEPLOYMENT_CN.md#sim2sim) |
| H2 hardware deployment | [English](DEPLOYMENT.md#sim2real) · [中文](DEPLOYMENT_CN.md#sim2real) |
| PICO VR teleoperation | [English](DEPLOYMENT.md#vr-teleoperation) · [中文](DEPLOYMENT_CN.md#vr-teleoperation) |
| Planner download and checksum | [Planner](PLANNER.md) |
| Device pairing and calibration | [VR setup](source/getting_started/vr_teleop_setup.md) |
| Model interfaces and hashes | [Released model](../models/released/README.md) |
| Validation and reproduction | [Release record](../release/REPRODUCIBILITY.md) |
| Common errors | [Troubleshooting](source/user_guide/troubleshooting.md) |

## Research and Development

- [Upstream dependencies and acknowledgments](ACKNOWLEDGMENTS.md).
- [BibTeX references](references.bib) and [citation metadata](../CITATION.cff).
- [Repository maintenance and optional assets](GIT_WORKFLOW.md).
- [Experiment reports](report/) and [source provenance](../release/source_manifest.json).
- [Original upstream README](../README.upstream.md).

The `source/` tree contains the extended documentation inherited from SONIC, including G1 and training workflows. Use the H2 deployment guides above for this release's launch commands and configuration.

## Build the Extended Documentation

From the repository root:

```bash
python -m pip install -r docs/requirements.txt
sphinx-build -b html docs/source docs/build/html
python -m http.server 8000 --directory docs/build/html
```

The Sphinx site is built from `source/`; the Markdown guides linked above are available directly in the repository. The GitHub Pages workflow in `.github/workflows/docs.yml` is triggered manually. The TRACE project page is [bfm-trace.github.io](https://bfm-trace.github.io/).
