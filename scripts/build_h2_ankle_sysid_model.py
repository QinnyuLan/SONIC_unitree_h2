#!/usr/bin/env python3
"""Build a separate official H2 derivative with embedded dynamic-friction data."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from gear_sonic.utils.mujoco_sim.ankle_response import BristleParameters, MODEL_TEXT_KEY

SOURCE = ROOT/"gear_sonic/data/assets/robot_description/unitree_ros/h2_description/H2_loop.xml"


def build(left, right, output):
    output.mkdir(parents=True, exist_ok=False)
    fits = {side: json.loads(path.read_text()) for side, path in (("left", left), ("right", right))}
    parameters = {side: BristleParameters(**fit["parameters"]).__dict__ for side, fit in fits.items()}
    tree = ET.parse(SOURCE)
    root = tree.getroot()
    root.set("model", "h2_official_ankle_sysid_candidate")
    root.find("compiler").set("meshdir", os.path.relpath(SOURCE.parent, output))
    custom = root.find("custom")
    if custom is None:
        custom = ET.SubElement(root, "custom")
    ET.SubElement(custom, "text", name=MODEL_TEXT_KEY, data=json.dumps(parameters, separators=(",", ":")))
    root.insert(0, ET.Comment("Dynamic friction requires SONIC-H2 ankle_response runtime; generic MuJoCo ignores custom text."))
    xml = output/"H2_loop_ankle_sysid.xml"
    tree.write(xml, encoding="utf-8", xml_declaration=True)
    metadata = dict(source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    parameters=parameters, fits={"left": str(left.resolve()), "right": str(right.resolve())},
                    validation_scope="unloaded ready pose, Kp=40/Kd=2, recorded sweeps; not loaded whole-body validated",
                    runtime="gear_sonic.utils.mujoco_sim.ankle_response", native_asset_parameters="unchanged")
    (output/"identification.json").write_text(json.dumps(metadata, indent=2)+"\n")
    overlay = dict(ROBOT_SCENE=os.path.relpath(xml, ROOT))
    (output/"sim_overlay.yaml").write_text(yaml.safe_dump(overlay, sort_keys=False))
    print(xml)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--left-fit", type=Path, required=True)
    p.add_argument("--right-fit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    build(a.left_fit, a.right_fit, a.output)
