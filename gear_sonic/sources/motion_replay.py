"""Stream an offline G1 motion (including text-to-motion output) to the runtime."""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from gear_sonic.commands import RuntimeClient, g1_reference
from gear_sonic.commands.protocol import G1_OFFSETS


class Motion:
    def __init__(self, joint_pos, root_quat, fps: float, joint_vel=None):
        self.q = np.asarray(joint_pos, dtype=np.float64)
        self.root = np.asarray(root_quat, dtype=np.float64)
        self.fps = float(fps)
        if self.q.ndim != 2 or self.q.shape[1] != 29 or len(self.q) < 2:
            raise ValueError("joint_pos requires at least two frames of 29 native G1 semantic joints")
        if self.root.shape != (len(self.q), 4):
            raise ValueError("root_quat must be [frames, 4], wxyz")
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be positive")
        self.dq = np.gradient(self.q, 1 / self.fps, axis=0) if joint_vel is None else np.asarray(joint_vel, dtype=np.float64)
        if self.dq.shape != self.q.shape or not all(np.isfinite(x).all() for x in (self.q, self.root, self.dq)):
            raise ValueError("motion contains invalid shapes or non-finite data")
        if np.any(np.linalg.norm(self.root, axis=1) < 1e-6):
            raise ValueError("motion contains a zero root quaternion")
        self.times = np.arange(len(self.q)) / self.fps
        self.duration = self.times[-1]
        self.rotation = Slerp(self.times, Rotation.from_quat(self.root[:, [1, 2, 3, 0]]))

    @classmethod
    def load(cls, path: Path, fps: float | None = None):
        if path.is_dir():
            def csv(name):
                file = path / name
                try:
                    return np.loadtxt(file, delimiter=",", ndmin=2)
                except ValueError:
                    return np.loadtxt(file, delimiter=",", skiprows=1, ndmin=2)
            q = csv("joint_pos.csv")
            dq = csv("joint_vel.csv") if (path / "joint_vel.csv").exists() else None
            root = csv("body_quat.csv")[:, :4]
            return cls(q, root, fps or 50, dq)
        with np.load(path, allow_pickle=False) as data:
            return cls(data["joint_pos"], data["root_quat"],
                       fps if fps is not None else float(data["fps"]),
                       data["joint_vel"] if "joint_vel" in data else None)

    def sample(self, elapsed: float, speed: float = 1):
        if not np.isfinite(elapsed) or not np.isfinite(speed) or speed <= 0:
            raise ValueError("elapsed must be finite and speed positive")
        requested = elapsed + G1_OFFSETS * speed
        times = np.clip(requested, 0, self.duration)
        q = np.column_stack([np.interp(times, self.times, col) for col in self.q.T])
        dq = np.column_stack([np.interp(times, self.times, col) for col in self.dq.T]) * speed
        dq[requested >= self.duration] = 0
        root = self.rotation(times).as_quat()[:, [3, 0, 1, 2]]
        return g1_reference(q, dq, root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", type=Path, help="SONIC CSV clip directory or NPZ")
    parser.add_argument("--fps", type=float, help="CSV default 50; overrides NPZ fps")
    parser.add_argument("--speed", type=float, default=1)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5558")
    parser.add_argument("--takeover", action="store_true")
    parser.add_argument("--loop", action="store_true", help="clip must have a continuous loop boundary")
    parser.add_argument("--inspect", action="store_true", help="validate and print metadata without connecting")
    args = parser.parse_args()
    motion = Motion.load(args.motion, args.fps)
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be positive")
    if args.inspect:
        print(f"{len(motion.q)} frames, {motion.fps:g} Hz, {motion.duration:.2f} s; G1 10 x 29")
        return
    with RuntimeClient("motion_replay", args.endpoint, takeover=args.takeover) as client:
        started = time.monotonic()
        try:
            while True:
                elapsed = (time.monotonic() - started) * args.speed
                if elapsed >= motion.duration:
                    if not args.loop:
                        break
                    elapsed %= motion.duration
                client.send("g1", motion.sample(elapsed, args.speed))
                time.sleep(.02)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
