#!/usr/bin/env python3
"""Render paired native G1/H2 deploy CSV references without a policy or physics rollout."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
G1_ASSET = ROOT / "gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml"
H2_ASSET = ROOT / "gear_sonic/data/assets/robot_description/unitree_ros/h2_description/h2_mujoco.xml"
MANIFEST = "h2_lora_reference_manifest.json"


def literal_constant(path: Path, name: str) -> tuple[str, ...]:
    """Reuse joint-order constants without importing training/teleop dependencies."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return tuple(ast.literal_eval(node.value))
    raise ValueError(f"Missing literal constant {name} in {path}")


G1_JOINT_NAMES = literal_constant(
    ROOT / "gear_sonic/envs/env_utils/joint_utils.py", "G1_ISAACLab_ORDER"
)
H2_JOINT_NAMES = literal_constant(
    ROOT / "gear_sonic/data/robot_model/instantiation/h2.py", "H2_JOINT_NAMES_LOCAL"
)


def read_csv(path: Path, width: int | None = None) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if not values.size or not np.isfinite(values).all():
        raise ValueError(f"Empty or non-finite reference: {path}")
    if width is not None and values.shape[1] != width:
        raise ValueError(f"{path}: expected {width} columns, got {values.shape[1]}")
    return values


@dataclass
class Reference:
    joints: np.ndarray
    root_pos: np.ndarray
    root_quat: np.ndarray
    fps: float

    def __post_init__(self):
        n = len(self.joints)
        if n == 0 or self.root_pos.shape != (n, 3) or self.root_quat.shape != (n, 4):
            raise ValueError("Joint/root reference frame counts or widths do not match")
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Reference FPS must be positive")
        if not all(np.isfinite(x).all() for x in (self.joints, self.root_pos, self.root_quat)):
            raise ValueError("Reference contains NaN/Inf")
        norms = np.linalg.norm(self.root_quat, axis=1)
        if np.any(norms < 1e-8):
            raise ValueError("Reference contains zero-length root quaternions")
        self.root_quat = self.root_quat / norms[:, None]

    @classmethod
    def load(cls, directory: Path, kind: str, fps: float):
        filename, width = ("joint_pos.csv", 29) if kind == "g1" else ("h2_joint_pos.csv", 31)
        positions = read_csv(directory / "body_pos.csv")
        quaternions = read_csv(directory / "body_quat.csv")
        if positions.shape[1] % 3 or quaternions.shape[1] % 4:
            raise ValueError(f"Invalid body CSV layout in {directory}")
        return cls(
            read_csv(directory / filename, width), positions[:, :3], quaternions[:, :4], fps
        )

    def aligned(self):
        rotations = Rotation.from_quat(self.root_quat[:, [1, 2, 3, 0]])
        forward = rotations[0].as_matrix()[:, 0]
        yaw = math.atan2(forward[1], forward[0])
        align = Rotation.from_euler("z", -yaw)
        origin = self.root_pos[0].copy()
        origin[2] = 0.0
        return Reference(
            self.joints.copy(), align.apply(self.root_pos - origin),
            (align * rotations).as_quat()[:, [3, 0, 1, 2]], self.fps,
        )

    def sample(self, times: np.ndarray):
        source_times = np.arange(len(self.joints)) / self.fps
        times = np.clip(times, 0, source_times[-1])
        linear = lambda x: np.column_stack([
            np.interp(times, source_times, x[:, i]) for i in range(x.shape[1])
        ])
        if len(source_times) == 1:
            quaternions = np.repeat(self.root_quat, len(times), axis=0)
        else:
            quaternions = Slerp(
                source_times, Rotation.from_quat(self.root_quat[:, [1, 2, 3, 0]])
            )(times).as_quat()[:, [3, 0, 1, 2]]
        return linear(self.joints), linear(self.root_pos), quaternions


def render_model_xml(path: Path) -> str:
    tree = ET.parse(path).getroot()
    compiler = tree.find("compiler")
    mesh_dir = (path.parent / compiler.get("meshdir", ".")).resolve()
    compiler.set("meshdir", str(mesh_dir))
    visual = ET.SubElement(tree, "visual")
    ET.SubElement(visual, "headlight", ambient="0.25 0.25 0.25", diffuse="0.5 0.5 0.5", specular="0.1 0.1 0.1")
    ET.SubElement(visual, "map", znear="0.01", zfar="30")
    ET.SubElement(visual, "global", fovy="40")
    asset = tree.find("asset")
    ET.SubElement(asset, "texture", name="comparison_floor", type="2d", builtin="checker",
                  width="512", height="512", rgb1="0.62 0.66 0.68", rgb2="0.76 0.79 0.81")
    ET.SubElement(asset, "material", name="comparison_floor", texture="comparison_floor",
                  texrepeat="2 2", texuniform="true", reflectance="0")
    ET.SubElement(asset, "texture", type="skybox", builtin="flat", width="256", height="1536",
                  rgb1="0.94 0.96 0.98", rgb2="0.94 0.96 0.98")
    world = tree.find("worldbody")
    ET.SubElement(world, "geom", name="comparison_floor", type="plane", size="0 0 0.05",
                  material="comparison_floor", group="2", contype="0", conaffinity="0")
    ET.SubElement(world, "light", pos="2 -2 5", dir="-0.3 0.3 -1", directional="true",
                  diffuse="0.4 0.4 0.4", castshadow="true")
    return ET.tostring(tree, encoding="unicode")


def joint_qpos_addresses(model, names: tuple[str, ...]) -> np.ndarray:
    import mujoco

    indices = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    if any(i < 0 for i in indices):
        raise ValueError("Robot asset does not contain every native reference joint")
    if any(model.jnt_type[i] != mujoco.mjtJoint.mjJNT_HINGE for i in indices):
        raise ValueError("Reference joint must have a single angular coordinate")
    return model.jnt_qposadr[indices].copy()


class RobotRenderer:
    def __init__(self, path: Path, names: tuple[str, ...], width: int, height: int):
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_string(render_model_xml(path))
        self.data = mujoco.MjData(self.model)
        self.addresses = joint_qpos_addresses(self.model, names)
        free = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) != 1:
            raise ValueError("Expected one floating root")
        self.root_adr = self.model.jnt_qposadr[free[0]]
        self.model.vis.global_.offwidth = width
        self.model.vis.global_.offheight = height
        self.model.vis.quality.offsamples = 4
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = 4.2
        self.camera.azimuth = -145
        self.camera.elevation = -12
        self.options = mujoco.MjvOption()
        self.options.geomgroup[:] = 0
        self.options.geomgroup[1:3] = 1
        self.options.sitegroup[:] = 0

    def pose(self, joints, position, quaternion):
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[self.root_adr:self.root_adr + 3] = position
        self.data.qpos[self.root_adr + 3:self.root_adr + 7] = quaternion
        self.data.qpos[self.addresses] = joints
        self.mj.mj_forward(self.model, self.data)
        self.camera.lookat[:] = [position[0], position[1], 0.95]
        self.renderer.update_scene(self.data, self.camera, scene_option=self.options)

    def image(self):
        return self.renderer.render().copy()

    def framing(self):
        self.renderer.enable_segmentation_rendering()
        try:
            segment = self.renderer.render()
        finally:
            self.renderer.disable_segmentation_rendering()
        ids = segment[:, :, 0]
        valid = (segment[:, :, 1] == int(self.mj.mjtObj.mjOBJ_GEOM)) & (ids >= 0)
        mask = np.zeros(ids.shape, dtype=bool)
        mask[valid] = self.model.geom_bodyid[ids[valid]] > 0
        ys, xs = np.where(mask)
        if len(xs) < 100:
            raise ValueError("Rendered robot is blank or too small")
        bounds = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        h, w = mask.shape
        if min(bounds[0], bounds[1], w - 1 - bounds[2], h - 1 - bounds[3]) < 8:
            raise ValueError(f"Robot touches viewport edge: {bounds}, viewport={w}x{h}")
        return {"robot_pixels": len(xs), "bounds": bounds}

    def close(self):
        self.renderer.close()


class VideoWriter:
    def __init__(self, path: Path, width: int, height: int, fps: int):
        self.process = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-preset", "fast", "-crf", "20", "-threads", "2",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
        ], stdin=subprocess.PIPE)

    def append(self, image):
        self.process.stdin.write(np.asarray(image, dtype=np.uint8).tobytes())

    def close(self):
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError("ffmpeg failed to encode reference video")


def compose(g1, h2, name, time_s, duration, heights, index, count):
    from PIL import Image, ImageDraw, ImageFont

    width, panel_height = g1.shape[1], g1.shape[0]
    image = Image.new("RGB", (2 * width, panel_height + 120), "#f7f9fa")
    image.paste(Image.fromarray(g1), (0, 80))
    image.paste(Image.fromarray(h2), (width, 80))
    draw = ImageDraw.Draw(image)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 20)
    title = ImageFont.truetype(font_path.replace(".ttf", "-Bold.ttf"), 28)
    for x, label, color, height in ((0, "G1", "#197d89", heights[0]), (width, "H2", "#bd5c37", heights[1])):
        draw.rectangle((x, 0, x + width, 5), fill=color)
        draw.text((x + 22, 13), label, font=title, fill=color)
        draw.text((x + 90, 19), f"Root height {height:.3f} m", font=font, fill="#48545b")
        draw.text((x + 22, 49), name, font=font, fill="#263238")
    draw.line((width, 0, width, panel_height + 120), fill="#b6c3c8", width=2)
    footer = panel_height + 89
    draw.text((22, footer), f"{index + 1:02d}/{count:02d}  |  {time_s:05.2f} / {duration:05.2f} s", font=font, fill="#263238")
    draw.text((2 * width - 340, footer), "NATIVE REFERENCE  |  1x", font=font, fill="#48545b")
    return image


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-root", type=Path, default=ROOT / "gear_sonic_deploy/reference/example")
    parser.add_argument("--h2-root", type=Path, default=ROOT / "gear_sonic_deploy/reference/example_h2/deploy")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "visualizations/g1_h2_reference_comparison")
    parser.add_argument("--motion", action="append", help="Render only named paired clips")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=0, help="Preview cap per clip; 0 renders full duration")
    args = parser.parse_args()
    if args.fps <= 0 or args.max_frames < 0:
        parser.error("FPS must be positive; max-frames must be nonnegative")
    os.environ.setdefault("MUJOCO_GL", "egl")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.h2_root / MANIFEST).read_text())
    pairs = [entry for entry in manifest["motions"]
             if (args.g1_root / entry["motion_name"]).is_dir()]
    if args.motion:
        missing = set(args.motion) - {e["motion_name"] for e in pairs}
        if missing:
            raise ValueError(f"Missing paired motions: {sorted(missing)}")
        pairs = [e for e in pairs if e["motion_name"] in args.motion]
    if not pairs:
        raise ValueError("No matching G1/H2 references")
    g1_manifest = json.loads((args.g1_root / MANIFEST).read_text())
    width, panel_height = 720, 780
    g1_renderer = RobotRenderer(G1_ASSET, G1_JOINT_NAMES, width, panel_height)
    h2_renderer = RobotRenderer(H2_ASSET, H2_JOINT_NAMES, width, panel_height)
    report = {"kind": "native_reference_fk_no_policy_no_physics", "video_fps": args.fps,
              "alignment": "independent fixed initial XY/yaw alignment; no size scaling or time warping; preserve root Z",
              "g1_asset": str(G1_ASSET), "h2_asset": str(H2_ASSET),
              "g1_root": str(args.g1_root.resolve()), "h2_root": str(args.h2_root.resolve()),
              "h2_asset_note": "Unitree ROS H2.urdf converted to open-chain MJCF for native joint FK; not a closed-loop dynamics rollout",
              "clips": []}
    try:
        for clip_index, entry in enumerate(pairs):
            name = entry["motion_name"]
            g1_path, h2_path = args.g1_root / name, args.h2_root / name
            g1 = Reference.load(g1_path, "g1", float(g1_manifest["fps"])).aligned()
            h2 = Reference.load(h2_path, "h2", float(entry.get("fps", manifest["fps"]))).aligned()
            duration = max(len(g1.joints) / g1.fps, len(h2.joints) / h2.fps)
            n = math.ceil(duration * args.fps)
            if args.max_frames:
                n = min(n, args.max_frames)
            times = np.arange(n) / args.fps
            g1_samples, h2_samples = g1.sample(times), h2.sample(times)
            path = output / f"{name}.mp4"
            writer = VideoWriter(path, 2 * width, panel_height + 120, args.fps)
            checks = []
            previous = None
            motion_delta = []
            print(f"[{clip_index + 1}/{len(pairs)}] {name}: G1={len(g1.joints)} H2={len(h2.joints)}, video={n} frames", flush=True)
            try:
                for i, time_s in enumerate(times):
                    g1_renderer.pose(*(x[i] for x in g1_samples))
                    h2_renderer.pose(*(x[i] for x in h2_samples))
                    check = i in {0, n // 4, n // 2, 3 * n // 4, n - 1}
                    if check:
                        checks.append({"frame": i, "g1": g1_renderer.framing(), "h2": h2_renderer.framing()})
                    g1_image, h2_image = g1_renderer.image(), h2_renderer.image()
                    if check:
                        if previous is not None:
                            motion_delta.append(float(np.mean(np.abs(g1_image.astype(float) - previous))))
                        previous = g1_image.astype(float)
                    frame = compose(g1_image, h2_image, name, time_s, duration,
                                    (g1_samples[1][i, 2], h2_samples[1][i, 2]), clip_index, len(pairs))
                    if i == n // 2:
                        frame.save(output / f"{name}.png")
                    writer.append(frame)
            finally:
                writer.close()
            record = {"name": name, "video": str(path), "video_frames": n,
                      "g1_frames": len(g1.joints), "h2_frames": len(h2.joints),
                      "g1_fps": g1.fps, "h2_fps": h2.fps,
                      "shorter_clip_endpoint": "hold final pose, no time scaling",
                      "framing_checks": checks, "sampled_image_motion_deltas": motion_delta,
                      "csv_sha256": {str(p.resolve()): sha256(p) for p in (
                          g1_path / "joint_pos.csv", g1_path / "body_pos.csv", g1_path / "body_quat.csv",
                          h2_path / "h2_joint_pos.csv", h2_path / "body_pos.csv", h2_path / "body_quat.csv")}}
            report["clips"].append(record)
            (output / "comparison_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"  Wrote {path}", flush=True)
    finally:
        g1_renderer.close()
        h2_renderer.close()
    playlist = output / "concat.txt"
    playlist.write_text("".join(f"file {Path(c['video']).name}\n" for c in report["clips"]))
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
                    "-safe", "0", "-i", str(playlist), "-c", "copy", "-movflags", "+faststart",
                    str(output / "all_paired_references.mp4")], check=True)
    print(f"Combined video: {output / 'all_paired_references.mp4'}", flush=True)


if __name__ == "__main__":
    main()
