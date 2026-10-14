"""Keyboard requests with a dedicated TensorRT planner worker in the same terminal."""
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from interface.commands import RuntimeClient
from interface.commands.protocol import planner_command
from interface.commands.status import RuntimeStatus
from interface.paths import ROOT
from interface.terminal import Terminal

MODE_NAMES = (
    "Idle", "Slow walk", "Walk", "Run", "Squat", "Kneel (two legs)", "Kneel",
    "Lie face down", "Crawl", "Boxing idle", "Boxing walk", "Left punch", "Right punch",
    "Random punch", "Elbow crawl", "Left hook", "Right hook", "Forward jump",
    "Stealth walk", "Injured walk", "Ledge walk", "Object carrying", "Stealth walk 2",
    "Happy dance walk", "Zombie walk", "Gun walk", "Scare walk",
)


class PlannerControl:
    def __init__(self, speed=.3):
        self.mode, self.speed, self.heading = 1, speed, 0.
        self.enabled = False
        self.direction = (0., 0.)
        self.move_until = 0.

    def stop(self):
        self.enabled = False
        self.direction = (0., 0.)
        self.move_until = 0.

    def key(self, key, now):
        if isinstance(key, str) and key in tuple(str(i) for i in range(10)):
            self.mode = int(key)
        elif key in ("[", "]"):
            self.mode = (self.mode + (1 if key == "]" else -1)) % len(MODE_NAMES)
        elif key in ("+", "=", "-", "_"):
            self.speed = round(min(7.5, max(0., self.speed + (.1 if key in ("+", "=") else -.1))), 2)
        elif key in (" ", "x"):
            self.stop()
        elif key in ("\n", "\r"):
            if self.enabled:
                self.stop()
            else:
                self.enabled = True
        elif key in ("w", "s", "a", "d"):
            self.direction = {"w": (1., 0.), "s": (-1., 0.), "a": (0., 1.), "d": (0., -1.)}[key]
            self.move_until, self.enabled = now + .35, True
        elif key in ("q", "e"):
            self.heading += math.radians(10) * (1 if key == "q" else -1)
            self.heading = math.atan2(math.sin(self.heading), math.cos(self.heading))
            self.enabled = True

    def request(self, now):
        if not self.enabled:
            return None
        forward, left = self.direction if now < self.move_until else (0., 0.)
        c, s = math.cos(self.heading), math.sin(self.heading)
        movement = [c * forward - s * left, s * forward + c * left, 0.]
        moving = bool(forward or left)
        mode = 0 if self.mode in (1, 2, 3) and not moving else self.mode
        return planner_command(mode=mode, movement=movement, facing=[c, s, 0.],
                               speed=self.speed if moving and mode else 0.)


def stop_worker(process):
    if process.poll() is None:
        for sig, timeout in ((signal.SIGTERM, 3), (signal.SIGKILL, 2)):
            try:
                os.killpg(process.pid, sig)
                process.wait(timeout=timeout)
                return
            except ProcessLookupError:
                return
            except subprocess.TimeoutExpired:
                pass
        print(f"Planner PID {process.pid} did not exit after SIGKILL; GPU access may be blocked.", file=sys.stderr)


def control_loop(args, *, worker=None, log_path=None):
    """With no worker, this is a keyboard client for an already running service."""
    endpoint = args.input_bind.replace("tcp://0.0.0.0:", "tcp://127.0.0.1:").replace("tcp://*:", "tcp://127.0.0.1:")
    status = RuntimeStatus(args.state)
    try:
        with RuntimeClient("planner_keyboard", endpoint, ttl_ms=500) as client, Terminal() as terminal:
            control = PlannerControl()
            active = False
            prepared = worker is None
            started = time.monotonic()
            notice = "Choose a mode; hold WASD to move, or Enter to activate the selected mode"
            while True:
                now = time.monotonic()
                if worker is not None:
                    if worker.poll() is not None:
                        raise RuntimeError(f"Planner worker exited ({worker.returncode}); see {log_path}")
                    if not prepared:
                        prepared = "Planner ready;" in log_path.read_text(errors="replace")
                        if not prepared and now - started > args.startup_timeout:
                            raise RuntimeError(f"Planner startup timed out; see {log_path}")
                status.poll()
                ready = prepared and status.ready
                for key in terminal.keys():
                    if key in ("\x1b", "\x03"):
                        return
                    # Mode/speed selection is available while engines initialize.
                    if not ready and key in ("w", "a", "s", "d", "q", "e", "\n", "\r"):
                        notice = "Waiting for the planner and external policy runtime"
                        continue
                    control.key(key, now)
                if not ready:
                    control.stop()
                request = control.request(now)
                if request is not None:
                    active = client.send("planner", request)
                    if not active:
                        control.stop()
                        notice = "Planner connection unavailable; press a control key to retry"
                    else:
                        notice = "Requests active; Space: release body control and return to standing"
                elif active:
                    client.stand()
                    active = False
                    notice = "Stopped; runtime returns to standing"
                terminal.draw([
                    "SONIC_unitree_h2 | G1 planner keyboard control", status.label,
                    "Planner: ready" if prepared else "Planner: preparing TensorRT engine (Esc cancels)",
                    "0-9: mode   [ / ]: previous/next mode (0-26)   +/-: speed (0.1 m/s)",
                    "Hold W/S: forward/back   A/D: strafe   Q/E: turn left/right (10 deg)",
                    "Enter: activate/deactivate mode   Space/x: stand   Esc/Ctrl+C: quit",
                    f"Mode: {control.mode} {MODE_NAMES[control.mode]} | Speed: {control.speed:.1f} m/s | Heading: {math.degrees(control.heading):.0f} deg",
                    f"Movement: {request['movement'] if request else [0, 0, 0]}",
                    "Direction expires 0.35 s after the last key repeat; keep this terminal focused.",
                    notice,
                    f"Planner log: {log_path}" if log_path else "Connected to external planner service",
                    "",
                    *["  ".join(f"{i}: {MODE_NAMES[i]}" for i in range(start, min(start + 3, len(MODE_NAMES))))
                      for start in range(0, len(MODE_NAMES), 3)],
                ])
                time.sleep(max(0, now + .05 - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        status.close()


def run(args):
    # Keep inference in its own process so a slow engine build cannot block keys.
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise ValueError("Keyboard control requires an interactive terminal")
    directory = ROOT / "outputs/planner"
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", prefix="keyboard-", suffix=".log", dir=directory,
                                     delete=False) as log:
        command = [sys.executable, "-m", "interface.planner.service"]
        for name in ("model", "device", "input_bind", "output_bind", "runtime", "state"):
            command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
        if args.publish_only:
            command.append("--publish-only")
        worker = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                  env=dict(os.environ, PYTHONUNBUFFERED="1"), start_new_session=True)
        try:
            control_loop(args, worker=worker, log_path=Path(log.name))
        finally:
            previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                stop_worker(worker)
            finally:
                signal.signal(signal.SIGINT, previous)
    print(f"Planner log: {log.name}")
