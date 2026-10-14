"""Interactive collection selection and motion playback; the runtime is external."""
import json
from pathlib import Path
import time

from interface.commands import RuntimeClient
from interface.commands.status import RuntimeStatus
from interface.paths import ROOT
from interface.terminal import Terminal
from .motion import Motion

COLLECTIONS = ("example_g1_replay", "example_g1_explore_replay", "example_h2_replay", "example_smpl_replay")


class ReplayControl:
    def __init__(self, client, root=ROOT / "motions", speed=1., loop=False):
        self.client, self.root, self.speed, self.loop = client, Path(root), speed, loop
        self.collection = None
        self.clips = []
        self.index = 0
        self.motion = None
        self.playing = False
        self.offset = 0.
        self.started = 0.
        self.notice = "Choose a collection with 1 / 2 / 3 / 4"

    def position(self, now):
        return self.offset + (now - self.started) * self.speed if self.playing else self.offset

    def stop(self, *, rewind=True, now=None):
        if self.playing:
            self.offset = self.position(time.monotonic() if now is None else now)
            self.client.stand()
        self.playing = False
        if rewind:
            self.offset = 0.

    def select_collection(self, index):
        self.stop()
        self.collection = COLLECTIONS[index]
        self.clips, self.motion, self.index = [], None, 0
        directory = self.root / self.collection
        manifest = json.loads((directory / "manifest.json").read_text())
        self.clips = [directory / item["motion_name"] for item in manifest["motions"]]
        if not self.clips:
            raise ValueError("Collection is empty")
        self.select(0)

    def select(self, index):
        self.stop()
        self.index = index % len(self.clips)
        self.motion = None
        self.motion = Motion.load(self.clips[self.index])
        self.notice = "Selected. Enter: play from start; Space: play / pause"

    def play(self, now, *, restart=False):
        if self.motion is None:
            return
        if restart:
            self.offset = 0.
        elif self.playing:
            return
        self.started, self.playing = now, True
        self.notice = "Playing"

    def key(self, key, now, ready):
        if key in ("1", "2", "3", "4"):
            self.select_collection(int(key) - 1)
        elif key in ("up", "k", "down", "j", "n", "p") and self.clips:
            self.select(self.index + (-1 if key in ("up", "k", "p") else 1))
            if key in ("n", "p") and ready:
                self.play(now, restart=True)
        elif key in ("\n", "\r", "r", " "):
            if key == " " and self.playing:
                self.stop(rewind=False, now=now)
                self.notice = "Paused; runtime returns to standing. Space: resume; Enter: restart"
            elif ready:
                self.play(now, restart=key != " ")
            else:
                self.notice = "Wait for the external runtime to finish initialization"
        elif key == "x":
            self.stop()
            self.notice = "Stopped; runtime returns to standing"
        elif key == "l":
            self.loop = not self.loop

    def tick(self, now, ready):
        if not self.playing:
            return
        if not ready:
            self.stop(rewind=False, now=now)
            self.notice = "Runtime state lost; playback paused"
            return
        elapsed = self.position(now)
        if elapsed >= self.motion.duration:
            if self.loop:
                self.offset = elapsed % self.motion.duration
                self.started = now
                elapsed = self.offset
            else:
                self.stop()
                self.notice = "Finished; runtime returns to standing"
                return
        if not self.client.send(self.motion.reference_type, self.motion.sample(elapsed, self.speed)):
            self.stop(rewind=False, now=now)
            self.notice = "Command connection lost; playback paused"


def run(args):
    status = RuntimeStatus(args.state)
    try:
        with RuntimeClient("motion_replay", args.endpoint, takeover=args.takeover) as client, Terminal() as terminal:
            control = ReplayControl(client, args.motions_root, args.speed, args.loop)
            while True:
                now = time.monotonic()
                status.poll()
                for key in terminal.keys():
                    if key in ("q", "\x1b", "\x03"):
                        return
                    try:
                        control.key(key, now, status.ready)
                    except (ValueError, KeyError, OSError) as error:
                        control.notice = str(error)
                control.tick(now, status.ready)
                lines = ["SONIC_unitree_h2 | Motion replay", status.label,
                         "1 G1   2 G1 explore   3 H2   4 SMPL",
                         "Up/Down (j/k): select   Enter: play/restart   Space: pause/resume",
                         "n/p: next/previous + play   x: stop   l: loop   q/Esc: quit",
                         f"Collection: {control.collection or '-'} | Loop: {'on' if control.loop else 'off'}",
                         control.notice]
                if control.motion:
                    lines.append(f"{control.motion.reference_type.upper()} | {control.position(now):.2f} / {control.motion.duration:.2f} s")
                visible = max(1, terminal.rows - len(lines) - 1)
                start = max(0, min(control.index - visible // 2, len(control.clips) - visible))
                lines.extend(f"{'>' if i == control.index else ' '} {i+1:2}. {control.clips[i].name}"
                             for i in range(start, min(len(control.clips), start + visible)))
                terminal.draw(lines)
                time.sleep(max(0, now + .02 - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        status.close()
