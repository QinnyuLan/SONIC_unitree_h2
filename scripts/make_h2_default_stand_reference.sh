#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-gear_sonic_deploy/reference/h2_default_stand_only}"
STAND_FRAMES="${STAND_FRAMES:-1000}"
LOG_DIR="${LOG_DIR:-/tmp/h2_default_stand_real_test}"
PYTHON_BIN="${PYTHON_BIN:-}"
POLICY_HEADER="gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp"

cd "$REPO_ROOT"

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  else
    echo "Error: neither python3 nor python was found in PATH." >&2
    exit 1
  fi
fi

"$PYTHON_BIN" - "$OUTPUT_DIR" "$STAND_FRAMES" "$POLICY_HEADER" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
expected_frames = int(sys.argv[2])
policy_header = Path(sys.argv[3])
motion_name = "h2_default_stand"
fps = 50
root_pos = [0.0, 0.0, 1.04]
root_quat_wxyz = [1.0, 0.0, 0.0, 0.0]
body_to_h2_mujoco = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9,
    17, 24, 5, 11, 18, 25, 4, 10, 19, 26,
    20, 27, 21, 28, 22, 29, 23, 30,
]
target_to_source_hip = [
    [0.8660245537757874, 0.0, -0.08700508624315262, 0.9847439527511597],
    [0.8660245537757874, 0.0, 0.08700508624315262, 0.9847439527511597],
]


def parse_h2_default_angles(path: Path) -> list[float]:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"const\s+std::array<[^>]+>\s+h2_default_angles\s*=\s*\{(?P<body>.*?)\};",
        text,
        flags=re.DOTALL,
    )
    if not match:
        raise SystemExit(f"Could not find h2_default_angles in {path}")
    body = re.sub(r"//.*", "", match.group("body"))
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.DOTALL)
    values = [float(token.strip()) for token in body.split(",") if token.strip()]
    if len(values) != 31:
        raise SystemExit(f"Expected 31 h2_default_angles, got {len(values)} from {path}")
    return values


def h2_physical_to_g1_semantic(h2_angles: list[float]) -> list[float]:
    values = [h2_angles[index] for index in body_to_h2_mujoco]

    def apply_pair(pitch_index: int, roll_index: int, matrix: list[float]) -> None:
        pitch = values[pitch_index]
        roll = values[roll_index]
        values[pitch_index] = matrix[0] * pitch + matrix[1] * roll
        values[roll_index] = matrix[2] * pitch + matrix[3] * roll

    apply_pair(0, 3, target_to_source_hip[0])
    apply_pair(1, 4, target_to_source_hip[1])
    return values


def headers(prefix: str, width: int) -> list[str]:
    return [f"{prefix}_{i}" for i in range(width)]


def headers_3d(prefix: str, count: int) -> list[str]:
    return [f"{prefix}_{i}_{axis}" for i in range(count) for axis in ("x", "y", "z")]


def headers_quat(prefix: str, count: int) -> list[str]:
    return [f"{prefix}_{i}_{axis}" for i in range(count) for axis in ("w", "x", "y", "z")]


def write_csv(path: Path, header: list[str], row: list[float], frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        formatted = [f"{float(value):.9f}" for value in row]
        for _ in range(frames):
            writer.writerow(formatted)


h2_default = parse_h2_default_angles(policy_header)
joint_pos = h2_physical_to_g1_semantic(h2_default)
joint_vel = [0.0] * len(joint_pos)
h2_joint_vel = [0.0] * len(h2_default)
motion_dir = output_dir / motion_name
motion_dir.mkdir(parents=True, exist_ok=True)

write_csv(motion_dir / "joint_pos.csv", headers("joint", 29), joint_pos, expected_frames)
write_csv(motion_dir / "joint_vel.csv", headers("joint_vel", 29), joint_vel, expected_frames)
write_csv(motion_dir / "h2_joint_pos.csv", headers("h2_joint", 31), h2_default, expected_frames)
write_csv(motion_dir / "h2_joint_vel.csv", headers("h2_joint_vel", 31), h2_joint_vel, expected_frames)
write_csv(motion_dir / "body_pos.csv", headers_3d("body", 1), root_pos, expected_frames)
write_csv(motion_dir / "body_quat.csv", headers_quat("body", 1), root_quat_wxyz, expected_frames)
write_csv(motion_dir / "body_lin_vel.csv", headers_3d("body_lin_vel", 1), [0.0, 0.0, 0.0], expected_frames)
write_csv(motion_dir / "body_ang_vel.csv", headers_3d("body_ang_vel", 1), [0.0, 0.0, 0.0], expected_frames)

info = {
    "motion_name": motion_name,
    "source_key": motion_name,
    "source_robot_pkl": None,
    "source_smpl_pkl": None,
    "fps": fps,
    "frames": expected_frames,
    "joint_space": "g1_semantic_29d_from_h2_physical_headless",
    "source_joint_space": "h2_physical_31d_mujoco_default_pose",
    "joint_pos_shape": [expected_frames, 29],
    "h2_joint_pos_shape": [expected_frames, 31],
    "body_quat_format": "wxyz",
    "body_count": 1,
    "smpl": None,
    "h2_default_angles": h2_default,
    "root_pos": root_pos,
    "root_quat_wxyz": root_quat_wxyz,
    "note": "Synthetic H2LoRA nominal default stand generated from deploy policy_parameters.hpp.",
}
(motion_dir / "h2_lora_reference.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
(motion_dir / "metadata.txt").write_text(
    "\n".join(
        [
            f"Metadata for: {motion_name}",
            "=" * 30,
            "",
            "Body part indexes:",
            "[0]",
            "",
            f"Total timesteps: {expected_frames}",
            f"FPS: {fps}",
            "Deploy source: h2_lora",
            "Reference joint space: g1_semantic_29d_from_h2_physical_headless",
            "Synthetic reference: true",
            f"Source deploy header: {policy_header}",
            "",
        ]
    ),
    encoding="utf-8",
)

manifest = {
    "schema": "h2_lora_deploy_reference_v1",
    "description": "Synthetic H2 default-stand reference for H2LoRA real deploy diagnostics.",
    "motion_root": None,
    "smpl_root": None,
    "output_dir": str(output_dir),
    "fps": fps,
    "num_motions": 1,
    "total_frames": expected_frames,
    "motions": [info],
}
(output_dir / "h2_lora_reference_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
(output_dir / "motion_summary.txt").write_text(
    "\n".join(
        [
            "H2LoRA Deploy Reference Summary",
            "=" * 36,
            f"Schema: {manifest['schema']}",
            "Motions: 1",
            f"Total frames: {expected_frames}",
            "",
            f"- {motion_name}: {expected_frames} frames, source=synthetic_h2_default_stand",
            "",
        ]
    ),
    encoding="utf-8",
)

manifest_path = output_dir / "h2_lora_reference_manifest.json"

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("schema") != "h2_lora_deploy_reference_v1":
    raise SystemExit(f"Unexpected manifest schema: {manifest.get('schema')!r}")
if manifest.get("num_motions") != 1:
    raise SystemExit(f"Expected one motion, got {manifest.get('num_motions')}")
if manifest.get("total_frames") != expected_frames:
    raise SystemExit(f"Expected {expected_frames} frames, got {manifest.get('total_frames')}")

required = [
    "joint_pos.csv",
    "joint_vel.csv",
    "h2_joint_pos.csv",
    "h2_joint_vel.csv",
    "body_pos.csv",
    "body_quat.csv",
    "body_lin_vel.csv",
    "body_ang_vel.csv",
    "metadata.txt",
    "h2_lora_reference.json",
]
missing = [name for name in required if not (motion_dir / name).is_file()]
if missing:
    raise SystemExit(f"Missing files under {motion_dir}: {missing}")

with (motion_dir / "h2_joint_pos.csv").open(newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    first = [float(x) for x in next(reader)]

if len(header) != 31 or len(first) != 31:
    raise SystemExit(f"h2_joint_pos.csv should contain 31 columns, got header={len(header)}, row={len(first)}")

left_hip_pitch = first[0]
left_knee = first[3]
left_ankle_roll = first[4]
left_ankle_pitch = first[5]
right_hip_pitch = first[6]
right_knee = first[9]
right_ankle_roll = first[10]
right_ankle_pitch = first[11]

print("")
print("Default-stand reference check:")
print(f"  output: {output_dir}")
print(f"  motion: {motion_dir.name}")
print(f"  frames: {expected_frames} at 50 Hz")
print(f"  source: {policy_header}")
print("  first h2_joint_pos lower-body values:")
print(f"    left:  hip_pitch={left_hip_pitch:+.6f}, knee={left_knee:+.6f}, ankle_roll={left_ankle_roll:+.6f}, ankle_pitch={left_ankle_pitch:+.6f}")
print(f"    right: hip_pitch={right_hip_pitch:+.6f}, knee={right_knee:+.6f}, ankle_roll={right_ankle_roll:+.6f}, ankle_pitch={right_ankle_pitch:+.6f}")
PY

cat <<EOF

Reference is ready.

Run the real deploy test with:

  cd "$REPO_ROOT/gear_sonic_deploy"
  ./deploy.sh --robot h2_lora --motion-data reference/h2_default_stand_only --enable-csv-logs --logs-dir "$LOG_DIR" real

Suggested test:
  1. Start deploy and enter CONTROL as usual.
  2. Do not press play at first; keep it holding h2_default_stand for 20-40 seconds.
  3. Stop safely, then send me the log directory:
     $LOG_DIR
EOF
