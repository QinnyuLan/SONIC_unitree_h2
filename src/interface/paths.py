"""Resources in the SONIC_unitree_h2 checkout (editable installation)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
PLANNER_MODEL = ROOT / "models/planner/planner_sonic.onnx"
