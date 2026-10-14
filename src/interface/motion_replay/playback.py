"""Portable motion/audio bundles and optional local or Unitree audio playback."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import selectors
import shlex
import shutil
import subprocess
import time
import wave


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wav_duration(path):
    with wave.open(str(path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) != (16000, 1, 2, "NONE"):
            raise ValueError("Audio must be PCM16, 16 kHz, mono WAV")
        frames = wav.getnframes()
        if frames == 0 or len(wav.readframes(frames)) != frames * 2:
            raise ValueError("Empty or truncated WAV")
        return frames / 16000


@dataclass(frozen=True)
class ReplayBundle:
    motion: Path
    audio: Path | None
    metadata: dict

    @classmethod
    def load(cls, directory):
        directory = Path(directory).resolve()
        data = json.loads((directory / "replay.json").read_text())
        if data.get("schema") != "sonic.replay.v1":
            raise ValueError("Unsupported replay bundle schema")

        def asset(key):
            spec = data[key]
            relative = Path(spec["file"])
            path = (directory / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(directory) or not path.is_file():
                raise ValueError(f"Invalid bundle {key} path")
            if sha256(path) != spec["sha256"]:
                raise ValueError(f"Bundle {key} checksum mismatch")
            return path

        return cls(asset("motion"), asset("audio") if "audio" in data else None, data)

    @staticmethod
    def write(directory, *, motion="motion.npz", audio="speech.wav", metadata=None):
        directory = Path(directory)
        data = {"schema": "sonic.replay.v1", "metadata": metadata or {}}
        for key, filename in (("motion", motion), ("audio", audio)):
            if filename is not None:
                data[key] = {"file": filename, "sha256": sha256(directory / filename)}
        # Publish the manifest last; incomplete generation is not a replay bundle.
        temp = directory / ".replay.json.tmp"
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        temp.replace(directory / "replay.json")
        return ReplayBundle.load(directory)


class AudioPlayer:
    """An argv-only player; optional READY/PLAY handshake for the native adapter."""

    def __init__(self, audio, command="aplay -q", *, ready=False):
        self.audio = Path(audio)
        self.duration = wav_duration(audio)
        self.argv = shlex.split(command)
        if not self.argv or shutil.which(self.argv[0]) is None:
            raise ValueError("Audio player not found; install alsa-utils or specify --audio-command")
        self.ready, self.process = ready, None

    def prepare(self, timeout=15):
        if not self.ready:
            return
        self.process = subprocess.Popen([*self.argv, str(self.audio)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        deadline, pending = time.monotonic() + timeout, b""
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while time.monotonic() < deadline:
                self.check()
                if self.process.poll() is not None:
                    raise RuntimeError("Audio player exited before READY")
                if not selector.select(timeout=.1):
                    continue
                chunk = os.read(self.process.stdout.fileno(), 4096)
                if not chunk:
                    raise RuntimeError("Audio player closed its readiness pipe")
                pending += chunk
                if len(pending) > 65536:
                    raise RuntimeError("Audio readiness output exceeded limit")
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    if line.strip() == b"READY":
                        return
        raise TimeoutError("Audio player did not become READY")

    def start(self):
        if self.ready:
            self.check()
            self.process.stdin.write(b"PLAY\n")
            self.process.stdin.flush()
            self.process.stdin.close()
        else:
            self.process = subprocess.Popen([*self.argv, str(self.audio)])

    def check(self):
        if self.process is not None and self.process.poll() not in (None, 0):
            raise RuntimeError(f"Audio player exited with status {self.process.returncode}")

    def finish(self):
        if self.process is not None:
            self.process.wait(timeout=5)
            self.check()

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
            for stream in (self.process.stdin, self.process.stdout):
                if stream is not None and not stream.closed:
                    stream.close()
