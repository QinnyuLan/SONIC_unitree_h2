#!/usr/bin/env python3
"""Entrypoint for the read-only H2 + Inspire multimodal recorder."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'external_dependencies/unitree_sdk2_python'))
from gear_sonic.data.h2_collection.collector import main
if __name__ == '__main__':
    raise SystemExit(main())
