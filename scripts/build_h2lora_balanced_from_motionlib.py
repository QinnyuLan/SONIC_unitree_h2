#!/usr/bin/env python3
"""Build a balanced H2LoRA MotionLib by resampling an existing MotionLib.

This intentionally does not redo H2 retargeting or SMPL-X FK.  It treats the
input MotionLib as the verified data contract and writes a new paired
robot/SMPL MotionLib with renamed duplicate keys when a source clip is
oversampled.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil

import joblib
import numpy as np
from tqdm import tqdm


LOCOMOTION_RE = re.compile(
    r"(walk|jog|run|hurry|forward|backward|sideway|leftward|rightward|"
    r"locomotion|turn_walk|turn_jog|walk_ff|jog_ff|run_ff|ff_loop|"
    r"step_forward|step_backward|step_side)",
    re.IGNORECASE,
)
DYNAMIC_RE = re.compile(
    r"(dance|jump|kick|punch|throw|catch|basketball|football|tennis|boxing|"
    r"squat|lunge|exercise|climb|crawl|bend|stoop|rotate|swing|wave)",
    re.IGNORECASE,
)
STATIC_RE = re.compile(
    r"(idle|stand|standing|sit|sitting|lie|lying|static|pose|tpose|a_pose)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MotionInfo:
    key: str
    robot_path: Path
    smpl_path: Path
    rel_parent: Path
    frames: int
    fps: int
    root_xy_disp: float
    bucket: str
    smpl_schema_version: int | None
    smpl_joints_source: str
    smpl_root_mode: str


@dataclass(frozen=True)
class Selection:
    source: MotionInfo
    output_key: str
    copy_index: int


def _parse_quotas(text: str) -> dict[str, float]:
    quotas: dict[str, float] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid quota item: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if key not in {"locomotion", "dynamic", "general", "static"}:
            raise ValueError(f"Unknown quota bucket: {key}")
        quotas[key] = float(value)
    if not quotas:
        raise ValueError("No quotas were provided")
    total = sum(quotas.values())
    if total <= 0:
        raise ValueError("Quota sum must be positive")
    return {key: value / total for key, value in quotas.items()}


def _load_single(path: Path) -> tuple[str, dict]:
    data = joblib.load(path)
    if not isinstance(data, dict) or len(data) != 1:
        raise ValueError(f"Expected one-motion pkl at {path}, got {type(data).__name__}")
    key, entry = next(iter(data.items()))
    return str(key), entry


def _classify_motion(key: str, frames: int, fps: int, root_xy_disp: float) -> str:
    duration = max(frames / max(float(fps), 1.0), 1e-6)
    mean_xy_speed = root_xy_disp / duration
    if LOCOMOTION_RE.search(key) or root_xy_disp >= 2.0 or mean_xy_speed >= 0.35:
        return "locomotion"
    if DYNAMIC_RE.search(key) or root_xy_disp >= 0.5 or mean_xy_speed >= 0.12:
        return "dynamic"
    if STATIC_RE.search(key) or root_xy_disp < 0.2:
        return "static"
    return "general"


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _validate_source_pair(
    source_key: str,
    robot_entry: dict,
    smpl_entry: dict,
    robot_path: Path,
    smpl_path: Path,
    args: argparse.Namespace,
) -> tuple[int | None, str, str]:
    robot_fps = int(robot_entry.get("fps", -1))
    smpl_fps = int(smpl_entry.get("fps", -1))
    required_fps = int(args.require_fps)
    if required_fps > 0 and robot_fps != required_fps:
        raise ValueError(f"{robot_path} fps={robot_fps}, expected {required_fps}")
    if smpl_fps != robot_fps:
        raise ValueError(f"Robot/SMPL fps mismatch for {source_key}: robot={robot_fps}, smpl={smpl_fps}")

    root = np.asarray(robot_entry.get("root_trans_offset"), dtype=np.float32)
    smpl_joints = np.asarray(smpl_entry.get("smpl_joints"), dtype=np.float32)
    pose_aa = np.asarray(smpl_entry.get("pose_aa"), dtype=np.float32)
    if root.ndim != 2 or root.shape[-1] != 3:
        raise ValueError(f"{robot_path} root_trans_offset has invalid shape {root.shape}")
    if smpl_joints.ndim != 3 or smpl_joints.shape[1:] != (24, 3):
        raise ValueError(f"{smpl_path} smpl_joints has invalid shape {smpl_joints.shape}")
    if pose_aa.ndim != 2 or pose_aa.shape[-1] != 72:
        raise ValueError(f"{smpl_path} pose_aa has invalid shape {pose_aa.shape}")
    if smpl_joints.shape[0] != root.shape[0] or pose_aa.shape[0] != root.shape[0]:
        raise ValueError(
            f"Robot/SMPL frame mismatch for {source_key}: "
            f"robot={root.shape[0]}, smpl_joints={smpl_joints.shape[0]}, pose_aa={pose_aa.shape[0]}"
        )

    schema_value = smpl_entry.get("schema_version")
    try:
        schema_version = int(schema_value) if schema_value is not None else None
    except Exception:
        schema_version = None
    required_schema = int(args.require_smpl_schema_version)
    if required_schema > 0 and schema_version != required_schema:
        raise ValueError(f"{smpl_path} schema_version={schema_value!r}, expected {required_schema}")

    smpl_joints_source = str(smpl_entry.get("smpl_joints_source", ""))
    required_source = str(args.require_smpl_source)
    if required_source and smpl_joints_source != required_source:
        raise ValueError(f"{smpl_path} smpl_joints_source={smpl_joints_source!r}, expected {required_source!r}")

    smpl_root_mode = str(smpl_entry.get("smpl_root_mode", ""))
    required_root_mode = str(args.require_smpl_root_mode)
    if required_root_mode and smpl_root_mode != required_root_mode:
        raise ValueError(f"{smpl_path} smpl_root_mode={smpl_root_mode!r}, expected {required_root_mode!r}")
    return schema_version, smpl_joints_source, smpl_root_mode


def _index_motionlib(source_root: Path, args: argparse.Namespace) -> list[MotionInfo]:
    robot_root = source_root / "robot"
    smpl_root = source_root / "smpl"
    if not robot_root.exists():
        raise FileNotFoundError(robot_root)
    if not smpl_root.exists():
        raise FileNotFoundError(smpl_root)

    robot_paths = sorted(robot_root.rglob("*.pkl"))
    by_key: dict[str, MotionInfo] = {}
    duplicates: list[str] = []
    for robot_path in tqdm(robot_paths, desc="Indexing source MotionLib", unit="motion"):
        source_key, robot_entry = _load_single(robot_path)
        if source_key in by_key:
            duplicates.append(source_key)
            continue
        rel = robot_path.relative_to(robot_root)
        smpl_path = smpl_root / rel
        if not smpl_path.exists():
            raise FileNotFoundError(f"Missing paired SMPL pkl for {robot_path}: {smpl_path}")
        smpl_key, smpl_entry = _load_single(smpl_path)
        if smpl_key != source_key:
            raise ValueError(f"Robot/SMPL key mismatch: {robot_path}={source_key}, {smpl_path}={smpl_key}")

        schema_version, smpl_joints_source, smpl_root_mode = _validate_source_pair(
            source_key,
            robot_entry,
            smpl_entry,
            robot_path,
            smpl_path,
            args,
        )
        root = np.asarray(robot_entry["root_trans_offset"], dtype=np.float32)
        frames = int(root.shape[0])
        if frames <= 0:
            continue
        fps = int(robot_entry.get("fps", 50))
        root_xy_disp = float(np.linalg.norm(root[-1, :2] - root[0, :2]))
        bucket = _classify_motion(source_key, frames, fps, root_xy_disp)
        by_key[source_key] = MotionInfo(
            key=source_key,
            robot_path=robot_path,
            smpl_path=smpl_path,
            rel_parent=rel.parent,
            frames=frames,
            fps=fps,
            root_xy_disp=root_xy_disp,
            bucket=bucket,
            smpl_schema_version=schema_version,
            smpl_joints_source=smpl_joints_source,
            smpl_root_mode=smpl_root_mode,
        )
    if duplicates:
        sample = ", ".join(sorted(set(duplicates))[:10])
        raise ValueError(f"Duplicate source motion keys are not supported: {sample}")
    return list(by_key.values())


def _sample_bucket(
    rng: np.random.Generator,
    motions: list[MotionInfo],
    target_frames: int,
    already_used: dict[str, int],
) -> list[Selection]:
    if target_frames <= 0 or not motions:
        return []
    selected: list[Selection] = []
    total = 0
    pool = list(motions)
    while total < target_frames:
        order = rng.permutation(len(pool))
        for idx in order:
            motion = pool[int(idx)]
            copy_index = already_used.get(motion.key, 0)
            output_key = motion.key if copy_index == 0 else f"{motion.key}__bal{copy_index:03d}"
            already_used[motion.key] = copy_index + 1
            selected.append(Selection(motion, output_key, copy_index))
            total += motion.frames
            if total >= target_frames:
                break
    return selected


def _trim_to_target(
    rng: np.random.Generator,
    selected: list[Selection],
    target_frames: int,
    required_keys: set[str],
) -> list[Selection]:
    total = sum(item.source.frames for item in selected)
    if total <= target_frames:
        return selected

    # Remove non-required occurrences while keeping the final count at or just
    # above target_frames.  The small overshoot is a whole-clip artifact.
    indices = list(range(len(selected)))
    rng.shuffle(indices)
    remove: set[int] = set()
    for idx in indices:
        item = selected[idx]
        if item.output_key in required_keys:
            continue
        frames = item.source.frames
        if total - frames >= target_frames:
            remove.add(idx)
            total -= frames
    return [item for idx, item in enumerate(selected) if idx not in remove]


def _rename_entry(entry: dict, source_key: str, output_key: str, copy_index: int, bucket: str) -> dict:
    cloned = dict(entry)
    cloned["balanced_source_key"] = source_key
    cloned["balanced_output_key"] = output_key
    cloned["balanced_copy_index"] = int(copy_index)
    cloned["balanced_bucket"] = bucket
    return cloned


def _write_selection(selection: Selection, robot_out: Path, smpl_out: Path, source_root: Path) -> None:
    source = selection.source
    rel_parent = Path(source.bucket) / source.rel_parent
    robot_path = robot_out / rel_parent / f"{selection.output_key}.pkl"
    smpl_path = smpl_out / rel_parent / f"{selection.output_key}.pkl"
    robot_path.parent.mkdir(parents=True, exist_ok=True)
    smpl_path.parent.mkdir(parents=True, exist_ok=True)

    robot_key, robot_entry = _load_single(source.robot_path)
    smpl_key, smpl_entry = _load_single(source.smpl_path)
    if robot_key != source.key or smpl_key != source.key:
        raise ValueError(f"Unexpected key while writing {source.key}")
    robot_entry = _rename_entry(robot_entry, source.key, selection.output_key, selection.copy_index, source.bucket)
    smpl_entry = _rename_entry(smpl_entry, source.key, selection.output_key, selection.copy_index, source.bucket)
    robot_entry["balanced_source_robot_pkl"] = str(source.robot_path.relative_to(source_root / "robot"))
    smpl_entry["balanced_source_smpl_pkl"] = str(source.smpl_path.relative_to(source_root / "smpl"))
    joblib.dump({selection.output_key: robot_entry}, robot_path, compress=True)
    joblib.dump({selection.output_key: smpl_entry}, smpl_path, compress=True)


def _bucket_summary(motions: list[MotionInfo]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for motion in motions:
        bucket = summary.setdefault(motion.bucket, {"motions": 0, "frames": 0, "xy_disp_sum": 0.0})
        bucket["motions"] += 1
        bucket["frames"] += motion.frames
        bucket["xy_disp_sum"] += motion.root_xy_disp
    for values in summary.values():
        values["mean_xy_disp"] = values["xy_disp_sum"] / max(values["motions"], 1)
        values.pop("xy_disp_sum", None)
    return dict(sorted(summary.items()))


def _selection_summary(selected: list[Selection]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for item in selected:
        bucket = summary.setdefault(item.source.bucket, {"motions": 0, "frames": 0, "unique_sources": 0})
        bucket["motions"] += 1
        bucket["frames"] += item.source.frames
    for bucket_name, values in summary.items():
        values["unique_sources"] = len({item.source.key for item in selected if item.source.bucket == bucket_name})
    return dict(sorted(summary.items()))


def _write_manifest(
    output_root: Path,
    source_root: Path,
    motions: list[MotionInfo],
    selected: list[Selection],
    quotas: dict[str, float],
    args: argparse.Namespace,
) -> None:
    selected_frames = int(sum(item.source.frames for item in selected))
    unique_sources = {item.source.key for item in selected}
    duplicate_occurrences = sum(1 for item in selected if item.copy_index > 0)
    source_manifest = _read_json(source_root / "manifest.json")
    manifest = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "robot_motion_file": str(output_root / "robot"),
        "smpl_motion_file": str(output_root / "smpl"),
        "target_frames": int(args.target_frames),
        "selected_total_frames": selected_frames,
        "num_output_motions": len(selected),
        "num_unique_source_motions": len(unique_sources),
        "duplicate_occurrences": duplicate_occurrences,
        "random_seed": int(args.random_seed),
        "bucket_quotas": quotas,
        "required_keys": args.required_keys,
        "required_fps": int(args.require_fps),
        "required_smpl_schema_version": int(args.require_smpl_schema_version),
        "required_smpl_source": str(args.require_smpl_source),
        "required_smpl_root_mode": str(args.require_smpl_root_mode),
        "source_manifest_smpl_schema_version": source_manifest.get("smpl_schema_version"),
        "source_manifest_smpl_mode": source_manifest.get("smpl_mode"),
        "source_manifest_smpl_root_mode": source_manifest.get("smpl_root_mode"),
        "source_manifest_selected_total_frames": source_manifest.get("selected_total_frames"),
        "source_bucket_summary": _bucket_summary(motions),
        "selected_bucket_summary": _selection_summary(selected),
        "selected_examples": [
            {
                "output_key": item.output_key,
                "source_key": item.source.key,
                "copy_index": item.copy_index,
                "bucket": item.source.bucket,
                "frames": item.source.frames,
                "root_xy_disp": item.source.root_xy_disp,
            }
            for item in selected[:200]
        ],
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/mnt/data/seed/h2lora_motion_lib/seed_5m_50hz_schema6"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-frames", type=int, default=5_000_000)
    parser.add_argument("--random-seed", type=int, default=20260603)
    parser.add_argument(
        "--bucket-quotas",
        default="locomotion=0.60,dynamic=0.20,general=0.15,static=0.05",
        help="Comma-separated frame quotas. Missing/nonexistent buckets are redistributed.",
    )
    parser.add_argument(
        "--required-keys",
        default=(
            "walk_forward_loop_002__A036,"
            "jog_forward_loop_003__A029_M,"
            "kozak_dance_002__A354,"
            "high_jump_R_002__A350_M"
        ),
        help="Comma-separated source keys that must be present once with their original key names.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-fps", type=int, default=50, help="0 disables source fps validation.")
    parser.add_argument(
        "--require-smpl-schema-version",
        type=int,
        default=6,
        help="0 disables SMPL schema validation.",
    )
    parser.add_argument(
        "--require-smpl-source",
        default="soma_bvh_fk_name_mapped_to_smpl24",
        help="Empty string disables SMPL joint-source validation.",
    )
    parser.add_argument(
        "--require-smpl-root-mode",
        default="soma-bvh-compatible",
        help="Empty string disables SMPL root-mode validation.",
    )
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} already exists; pass --overwrite to rebuild")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    quotas = _parse_quotas(args.bucket_quotas)
    motions = _index_motionlib(source_root, args)
    by_key = {motion.key: motion for motion in motions}
    by_bucket: dict[str, list[MotionInfo]] = {}
    for motion in motions:
        by_bucket.setdefault(motion.bucket, []).append(motion)
    available_quota_total = sum(quotas.get(bucket, 0.0) for bucket in by_bucket)
    if available_quota_total <= 0:
        raise ValueError("No quota bucket matches available motions")
    quotas = {bucket: quotas.get(bucket, 0.0) / available_quota_total for bucket in by_bucket}

    required_keys = [key.strip() for key in args.required_keys.split(",") if key.strip()]
    missing_required = [key for key in required_keys if key not in by_key]
    if missing_required:
        raise KeyError(f"Required keys not found in source MotionLib: {missing_required}")

    rng = np.random.default_rng(args.random_seed)
    used_counts: dict[str, int] = {}
    selected: list[Selection] = []
    for key in required_keys:
        motion = by_key[key]
        selected.append(Selection(motion, key, 0))
        used_counts[key] = 1

    already_frames_by_bucket: dict[str, int] = {}
    for item in selected:
        already_frames_by_bucket[item.source.bucket] = (
            already_frames_by_bucket.get(item.source.bucket, 0) + item.source.frames
        )

    for bucket, motions_in_bucket in sorted(by_bucket.items()):
        target = int(round(args.target_frames * quotas.get(bucket, 0.0)))
        target = max(0, target - already_frames_by_bucket.get(bucket, 0))
        selected.extend(_sample_bucket(rng, motions_in_bucket, target, used_counts))

    selected = _trim_to_target(rng, selected, args.target_frames, set(required_keys))
    rng.shuffle(selected)
    selected.sort(key=lambda item: (0 if item.output_key in required_keys else 1, item.source.bucket, item.output_key))

    robot_out = output_root / "robot"
    smpl_out = output_root / "smpl"
    for item in tqdm(selected, desc="Writing balanced MotionLib", unit="motion"):
        _write_selection(item, robot_out, smpl_out, source_root)

    _write_manifest(output_root, source_root, motions, selected, quotas, args)
    manifest = json.loads((output_root / "manifest.json").read_text())
    print(json.dumps(manifest["selected_bucket_summary"], indent=2))
    print(
        f"Done: output={output_root}, motions={manifest['num_output_motions']}, "
        f"unique_sources={manifest['num_unique_source_motions']}, "
        f"frames={manifest['selected_total_frames']}, duplicates={manifest['duplicate_occurrences']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
