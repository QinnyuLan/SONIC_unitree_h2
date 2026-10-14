#!/usr/bin/env python3
"""Supported, unloaded H2 ankle pitch identification; preview unless --execute.

No policy, planner, coordinate adaptation, or automatic MotionSwitcher release.
All logged joint quantities are absolute, native H2 PR coordinates in DDS order.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import queue
import select
import signal
import socket
import sys
import termios
import threading
import time
import tty
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "gear_sonic/data/assets/robot_description/unitree_ros/h2_description/H2_loop.xml"
NAMES = tuple(
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_roll", "ankle_pitch")
) + ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint") + tuple(
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                  "wrist_roll", "wrist_pitch", "wrist_yaw")
) + ("head_pitch_joint", "head_yaw_joint")
KP = (200, 200, 200, 200, 40, 40) * 2 + (150, 150, 150) + (40, 40, 40, 40, 40, 20, 20) * 2 + (40, 40)
KD = (4, 4, 4, 4, 2, 2) * 2 + (3, 3, 3) + (2, 2, 2, 2, 2, 1, 1) * 2 + (2, 2)
PUBLISH_HZ = 500
TARGET_HZ = 50
STATE_TIMEOUT = 0.05
TICK_TIMEOUT = 0.05
LOOP_TIMEOUT = 0.02
GAIN_RAMP = 3.0
HOLD_SECONDS = 3.0
WAIT_TIMEOUT = 60.0
DAMPING_KD = 8.0
MAX_RUN_SECONDS = 900.0


class SafetyStop(RuntimeError):
    pass


def bounded(low, high):
    def parse(text):
        value = float(text)
        if not math.isfinite(value) or not low <= value <= high:
            raise argparse.ArgumentTypeError(f"must be finite and in [{low}, {high}]")
        return value
    return parse


def frequencies(text):
    try:
        values = tuple(bounded(0.1, 3.0)(s.strip()) for s in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if len(values) > 12 or any(b <= a for a, b in zip(values, values[1:])):
        raise argparse.ArgumentTypeError("use at most 12 strictly increasing frequencies")
    return values


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", choices=("real", "sim"), default="real")
    p.add_argument("--interface", default="eno1")
    p.add_argument("--domain", type=int, default=0)
    p.add_argument("--side", choices=("left", "right"), required=True)
    p.add_argument("--amplitude-deg", type=bounded(0.1, 2.0), default=0.5)
    p.add_argument("--frequencies", type=frequencies, default=(0.25, 0.5, 1.0, 1.5, 2.0))
    p.add_argument("--kp-scale", type=bounded(0.25, 1.5), default=1.0)
    p.add_argument("--kd-scale", type=bounded(0.5, 2.0), default=1.0)
    p.add_argument("--cycles", type=int, default=5, help="constant-amplitude measurement cycles (3-12)")
    p.add_argument("--execute", action="store_true", help="enable manual arming and LowCmd publication")
    p.add_argument("--confirm-supported", action="store_true", help="H2 body/legs secured, feet clear, safe to lose holding torque")
    p.add_argument("--auto", action="store_true", help="simulation only: arm and advance automatically")
    p.add_argument("--monitor-seconds", type=bounded(1, 60), help="read-only recording; no LowCmd writer")
    p.add_argument("--output", type=Path, help="new log directory; refuses an existing path")
    a = p.parse_args(argv)
    if not 0 <= a.domain <= 232 or not 3 <= a.cycles <= 12:
        p.error("domain must be 0-232; cycles must be 3-12")
    if a.execute and a.monitor_seconds:
        p.error("--execute and --monitor-seconds are mutually exclusive")
    if a.execute and not a.confirm_supported:
        p.error("--execute requires --confirm-supported (not a substitute for physical support)")
    if a.target == "sim" and (a.interface != "lo" or a.domain == 0):
        p.error("simulation requires --interface lo and a nonzero --domain")
    if a.target == "real" and a.interface == "lo":
        p.error("real hardware cannot use lo")
    if a.auto and (a.target != "sim" or not a.execute):
        p.error("--auto is only allowed for explicit simulation execution")
    # Bounds include the two-cycle raised-cosine envelope derivatives.
    amp = math.radians(a.amplitude_deg)
    omega = 2 * math.pi * max(a.frequencies)
    if amp * omega * 1.125 > 0.2 or amp * omega**2 * 1.282 > 3.0:
        p.error("amplitude/frequency combination exceeds phase-1 target speed/acceleration caps")
    if sum((a.cycles + 4) / f + HOLD_SECONDS for f in a.frequencies) > 600:
        p.error("planned excitation exceeds 600 seconds; split into smaller experiments")
    return a


def limits_from_asset():
    joints = {j.get("name"): j for j in ET.parse(ASSET).iter("joint") if j.get("range")}
    return tuple(tuple(map(float, joints[n].get("range").split())) for n in NAMES)


def wave(t, frequency, amplitude, cycles):
    """Two-cycle cosine ramps around an integer-cycle, constant-amplitude window."""
    ramp = 2.0 / frequency
    duration = (cycles + 4) / frequency
    if t <= 0 or t >= duration:
        return 0.0, False
    envelope = 1.0
    if t < ramp:
        envelope = 0.5 * (1 - math.cos(math.pi * t / ramp))
    elif t > duration - ramp:
        envelope = 0.5 * (1 - math.cos(math.pi * (duration - t) / ramp))
    return amplitude * envelope * math.sin(2 * math.pi * frequency * t), ramp <= t < duration - ramp


def gains(index, kp_scale, kd_scale, ramp=1.0, damping=False):
    if damping:
        return [0.0] * 31, [DAMPING_KD] * 31
    kp, kd = list(KP), list(KD)
    kp[index] *= kp_scale
    kd[index] *= kd_scale
    return [v * ramp for v in kp], kd


@dataclass(frozen=True)
class State:
    receive_ns: int
    source_ns: int
    tick: int
    mode_pr: int
    mode_machine: int
    q: tuple
    dq: tuple
    tau: tuple
    temperature: tuple
    errors: tuple
    quaternion: tuple
    gyro: tuple
    accel: tuple
    remote_keys: int
    crc_ok: bool

    @classmethod
    def from_message(cls, msg, received, crc_ok):
        motors = msg.motor_state[:31]
        return cls(
            received, int(getattr(getattr(msg, "sample_info", None), "source_timestamp", 0)),
            int(msg.tick), int(msg.mode_pr), int(msg.mode_machine),
            tuple(float(m.q) for m in motors), tuple(float(m.dq) for m in motors),
            tuple(float(m.tau_est) for m in motors),
            tuple(v for m in motors for v in m.temperature), tuple(int(m.motorstate) for m in motors),
            tuple(msg.imu_state.quaternion), tuple(msg.imu_state.gyroscope),
            tuple(msg.imu_state.accelerometer),
            int(msg.wireless_remote[2]) | (int(msg.wireless_remote[3]) << 8), crc_ok,
        )


def check_state(s, now_ns, limits, base=None, target=None, index=5, initial_quat=None, mode=None):
    if s is None or (now_ns - s.receive_ns) / 1e9 > STATE_TIMEOUT:
        raise SafetyStop("LowState missing/stale (>50 ms)")
    numbers = s.q + s.dq + s.tau + s.temperature + s.quaternion + s.gyro + s.accel
    if not all(math.isfinite(x) for x in numbers):
        raise SafetyStop("non-finite LowState")
    if len(s.q) != 31 or s.mode_pr != 0 or (mode is not None and s.mode_machine != mode):
        raise SafetyStop("unexpected joint count, PR mode, or mode_machine change")
    if any(s.errors) or max(s.temperature) >= 65:
        raise SafetyStop("motor fault or temperature >=65 C")
    if s.remote_keys & (1 << 3):
        raise SafetyStop("Unitree remote SELECT pressed")
    norm = math.sqrt(sum(x*x for x in s.quaternion))
    if not 0.9 < norm < 1.1:
        raise SafetyStop("invalid IMU quaternion")
    for i, (q, dq, (lo, hi)) in enumerate(zip(s.q, s.dq, limits)):
        if not lo - 0.02 <= q <= hi + 0.02 or abs(dq) > (1.0 if i == index else 3.0):
            raise SafetyStop(f"joint limit/speed violation: DDS {i} {NAMES[i]}")
        if base is not None and abs(q - base[i]) > (0.10 if i == index else 0.15):
            raise SafetyStop(f"joint moved too far from captured pose: DDS {i}")
    if target is not None and abs(s.q[index] - target[index]) > 0.08:
        raise SafetyStop("ankle tracking error >0.08 rad")
    if abs(s.tau[index]) > 12 or abs(s.tau[index-1]) > 8:
        raise SafetyStop("ankle estimated torque exceeds pitch 12 / roll 8 Nm")
    if initial_quat is not None:
        norm0 = math.sqrt(sum(x*x for x in initial_quat))
        dot = min(1.0, abs(sum(a*b for a, b in zip(initial_quat, s.quaternion))) / (norm * norm0))
        if 2 * math.acos(dot) > math.radians(10):
            raise SafetyStop("body orientation changed by >10 degrees")


STATE_FIELDS = ["receive_monotonic_ns", "dds_source_timestamp_ns", "tick", "mode_pr", "mode_machine", "crc_ok", "remote_keys"]
for prefix, count in (("q", 31), ("dq", 31), ("tau_est", 31), ("temperature", 62), ("motorstate", 31), ("imu_quat_wxyz", 4), ("imu_gyro", 3), ("imu_accel", 3)):
    STATE_FIELDS += [f"{prefix}_{i}" for i in range(count)]
CMD_FIELDS = ["send_begin_monotonic_ns", "send_end_monotonic_ns", "sequence", "phase", "frequency_hz", "segment_time_s", "measure", "state_receive_ns", "state_tick", "mode_machine"]
for prefix in ("q_des", "dq_des", "kp", "kd", "tau_ff"):
    CMD_FIELDS += [f"{prefix}_{i}" for i in range(31)]


class Recorder:
    """Bounded, asynchronous file writes; any loss aborts an active experiment."""

    def __init__(self, path):
        self.path = path
        path.mkdir(parents=True, exist_ok=False)
        self.queue = queue.Queue(maxsize=8192)
        self.error = None
        self.counts = {"states": 0, "commands": 0, "events": 0}
        self.thread = threading.Thread(target=self._write, name="sysid-log", daemon=True)
        self.thread.start()

    def put(self, kind, row):
        try:
            self.queue.put_nowait((kind, row))
        except queue.Full:
            self.error = "log queue overflow; data incomplete"

    def event(self, name, **data):
        self.put("events", dict(monotonic_ns=time.monotonic_ns(), event=name, **data))

    def _write(self):
        try:
            with (self.path / "states.csv").open("x", newline="") as sf, \
                 (self.path / "commands.csv").open("x", newline="") as cf, \
                 (self.path / "events.jsonl").open("x") as ef:
                sw, cw = csv.writer(sf), csv.writer(cf)
                sw.writerow(STATE_FIELDS)
                cw.writerow(CMD_FIELDS)
                while True:
                    item = self.queue.get()
                    if item is None:
                        break
                    kind, row = item
                    if kind == "events":
                        ef.write(json.dumps(row, allow_nan=False) + "\n")
                    else:
                        (sw if kind == "states" else cw).writerow(row)
                    self.counts[kind] += 1
        except Exception as exc:
            self.error = f"log write failed: {exc}"

    def close(self):
        while self.thread.is_alive():
            try:
                self.queue.put(None, timeout=0.1)
                break
            except queue.Full:
                pass
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            self.error = "log flush did not finish"


class Transport:
    """SDK message/CRC types on CycloneDDS, preserving raw receive samples."""

    def __init__(self, args, recorder, *, domain_initialized=False):
        sys.path.insert(0, str(ROOT / "external_dependencies/unitree_sdk2_python"))
        from cyclonedds.core import Listener
        from cyclonedds.domain import Domain, DomainParticipant
        from cyclonedds.qos import Qos, Policy
        from cyclonedds.sub import DataReader
        from cyclonedds.topic import Topic
        from unitree_sdk2py.core.channel_config import ChannelConfigHasInterface
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        self.args, self.recorder = args, recorder
        self.crc = CRC()
        self.latest = None
        self.error = None
        self.state_writers = self.command_writers = 0
        self.last_tick = None
        self.tick_changed_ns = 0
        self.last_state_ns = 0
        self.state_count = 0
        self.max_state_gap_ms = 0.0
        self.writer = None
        self.own_writer_created = False
        self.lock = threading.Lock()
        self.callback_lock = threading.Lock()
        self.closed = False
        config = ChannelConfigHasInterface.replace("$__IF_NAME__$", args.interface)
        # The range tool can first configure this domain through the SDK for RPC.
        self.domain = None if domain_initialized else Domain(args.domain, config)
        self.participant = DomainParticipant(args.domain)
        self.command_topic = Topic(self.participant, "rt/lowcmd", LowCmd_)
        self.state_topic = Topic(self.participant, "rt/lowstate", LowState_)
        qos = Qos(Policy.History.KeepLast(1024), Policy.Reliability.BestEffort)
        self.state_reader = DataReader(self.participant, self.state_topic, qos=qos, listener=Listener(
            on_data_available=self._states, on_subscription_matched=self._state_match,
            on_sample_lost=self._lost, on_sample_rejected=self._lost))
        self.command_reader = DataReader(self.participant, self.command_topic, qos=qos, listener=Listener(
            on_data_available=lambda r: r.take(1024), on_subscription_matched=self._command_match))

    def _lost(self, reader, status):
        self.error = "DDS reports lost/rejected LowState samples"

    def _state_match(self, reader, status):
        self.state_writers = status.current_count
        if self.state_writers > 1:
            self.error = "multiple LowState publishers; ambiguous robot source"

    def _command_match(self, reader, status):
        self.command_writers = status.current_count
        if self.own_writer_created and self.command_writers > 1:
            self.error = "another LowCmd publisher appeared; stop all other controllers"

    def _states(self, reader):
        with self.callback_lock:
            if not self.closed:
                self._read_states(reader)

    def _read_states(self, reader):
        try:
            for msg in reader.take(1024):
                if not getattr(getattr(msg, "sample_info", None), "valid_data", True):
                    continue
                received = time.monotonic_ns()
                ok = self.crc.Crc(msg) == msg.crc
                s = State.from_message(msg, received, ok)
                self.recorder.put("states", [s.receive_ns, s.source_ns, s.tick, s.mode_pr, s.mode_machine,
                    int(s.crc_ok), s.remote_keys, *s.q, *s.dq, *s.tau, *s.temperature, *s.errors,
                    *s.quaternion, *s.gyro, *s.accel])
                if not ok and self.args.target == "real":
                    self.error = "LowState CRC mismatch"
                    continue
                with self.lock:
                    if self.last_tick != s.tick:
                        self.tick_changed_ns = received
                        self.last_tick = s.tick
                    if self.last_state_ns:
                        self.max_state_gap_ms = max(self.max_state_gap_ms, (received-self.last_state_ns)/1e6)
                    self.last_state_ns = received
                    self.state_count += 1
                    self.latest = s
        except Exception as exc:
            self.error = f"LowState callback failed: {exc}"

    def get(self):
        with self.lock:
            return self.latest

    def healthy(self, *, before_release=False):
        if self.error or self.recorder.error:
            raise SafetyStop(self.error or self.recorder.error)
        if self.args.target == "real" and (time.monotonic_ns()-self.tick_changed_ns)/1e9 > TICK_TIMEOUT:
            raise SafetyStop("device tick did not advance for 50 ms")
        if self.state_writers != 1:
            raise SafetyStop("requires exactly one LowState publisher")
        if before_release and self.own_writer_created:
            raise SafetyStop("cannot relax publisher checks after LowCmd writer creation")
        expected = 1 if self.own_writer_created else 0
        if not before_release and self.command_writers != expected:
            raise SafetyStop(f"unexpected LowCmd publisher count: {self.command_writers}, expected {expected}")

    def create_writer(self):
        from cyclonedds.pub import DataWriter
        from cyclonedds.qos import Qos, Policy
        from cyclonedds.util import duration
        self.healthy()
        self.own_writer_created = True
        self.writer = DataWriter(self.participant, self.command_topic, qos=Qos(
            Policy.History.KeepLast(1), Policy.Reliability.Reliable(max_blocking_time=duration(milliseconds=5))))
        deadline = time.monotonic() + 2
        while self.command_writers != 1 and time.monotonic() < deadline:
            if self.error:
                raise SafetyStop(self.error)
            time.sleep(0.01)
        self.healthy()

    def send(self, target, kp, kd, mode, phase, frequency, segment_t, measure, sequence):
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        cmd = unitree_hg_msg_dds__LowCmd_()
        cmd.mode_pr, cmd.mode_machine = 0, mode
        for i, m in enumerate(cmd.motor_cmd):
            m.mode = int(i < 31)
            m.q, m.dq, m.tau = (target[i] if i < 31 else 0.0), 0.0, 0.0
            m.kp, m.kd = (kp[i], kd[i]) if i < 31 else (0.0, 0.0)
        cmd.crc = self.crc.Crc(cmd)
        s = self.get()
        begin = time.monotonic_ns()
        self.writer.write(cmd)
        end = time.monotonic_ns()
        self.recorder.put("commands", [begin, end, sequence, phase, frequency, segment_t, int(measure),
            s.receive_ns if s else 0, s.tick if s else 0, mode,
            *target, *([0.0]*31), *kp, *kd, *([0.0]*31)])
        return (end-begin)/1e9


@contextmanager
def keyboard(enabled):
    old = termios.tcgetattr(sys.stdin) if enabled else None
    try:
        if enabled:
            tty.setcbreak(sys.stdin.fileno())
        yield lambda: sys.stdin.read(1) if enabled and select.select([sys.stdin], [], [], 0)[0] else None
    finally:
        if old is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)


def save_json(path, data):
    with path.open("w") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")


def run(args):
    index = NAMES.index(f"{args.side}_ankle_pitch_joint")
    amp = math.radians(args.amplitude_deg)
    limits = limits_from_asset()
    print(f"H2 PR / {NAMES[index]} / DDS {index}; amplitude +/-{args.amplitude_deg} deg")
    print(f"Ankle Kp={KP[index]*args.kp_scale:g}, Kd={KD[index]*args.kd_scale:g}; dq_des=0, tau_ff=0")
    print(f"Frequencies {args.frequencies} Hz; publish {PUBLISH_HZ} Hz / target {TARGET_HZ} Hz")
    print("Other 30 joints: capture measured pose, hold with release nominal gains (3 s Kp ramp).")
    if not args.execute and not args.monitor_seconds:
        print("PREVIEW ONLY: no DDS connection or motor commands. Add --execute --confirm-supported to arm.")
        return 0
    if args.execute and not args.auto and not sys.stdin.isatty():
        raise SafetyStop("interactive terminal required; real hardware has no automatic arming")
    if args.interface not in dict((name, i) for i, name in socket.if_nameindex()):
        raise SafetyStop(f"interface not found: {args.interface}")
    output = args.output or ROOT / "gear_sonic_deploy/logs" / (
        f"ankle_sysid_{args.target}_{args.side}_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    rec = Recorder(output)
    metadata = dict(schema="h2_ankle_sysid_v1", args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        joints_dds=NAMES, active_index=index, publish_hz=PUBLISH_HZ, target_hz=TARGET_HZ,
        asset=str(ASSET), asset_sha256=hashlib.sha256(ASSET.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        nominal_kp=KP, nominal_kd=KD, started_wall_time=datetime.now().isoformat(),
        coordinate_system="absolute H2 PR, DDS order, rad/rad_s; tau_est semantics require hardware verification",
        simulator_crc_bypass=args.target == "sim", status="initializing")
    save_json(output / "metadata.json", metadata)
    transport = None
    active = False
    reason = "completed"
    code = 0
    stop = threading.Event()
    previous_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.signal(sig, lambda *_: stop.set())
    sequence = 0
    base, mode = None, None
    try:
        transport = Transport(args, rec)
        print(f"Logs: {output}")
        deadline = time.monotonic() + 10
        while transport.get() is None:
            if stop.is_set() or time.monotonic() >= deadline or transport.error:
                raise SafetyStop(transport.error or "no usable LowState within 10 seconds / interrupted")
            time.sleep(0.01)
        if args.monitor_seconds:
            end = time.monotonic() + args.monitor_seconds
            while time.monotonic() < end and not stop.is_set():
                s = transport.get()
                print(f"q={s.q[index]:+.4f} rad dq={s.dq[index]:+.4f} tau_est={s.tau[index]:+.3f} "
                      f"PR={s.mode_pr} mode={s.mode_machine} CRC={s.crc_ok} "
                      f"state_age_ms={(time.monotonic_ns()-s.receive_ns)/1e6:.1f}")
                if transport.error or rec.error:
                    raise SafetyStop(transport.error or rec.error)
                if (time.monotonic_ns()-s.receive_ns)/1e9 > STATE_TIMEOUT:
                    raise SafetyStop("LowState stale during monitor")
                time.sleep(0.5)
            reason = "monitor_only"
        else:
            # Discovery grace plus a continuous stationary window, before creating a writer.
            end = time.monotonic() + 1.0
            while time.monotonic() < end:
                s = transport.get()
                transport.healthy()
                check_state(s, time.monotonic_ns(), limits, index=index)
                if max(abs(v) for v in s.dq) > 0.15 or stop.is_set():
                    raise SafetyStop("pre-arm requires stationary, supported robot / interrupted")
                time.sleep(0.01)
            print("Stop deploy/VR/other LowCmd writers and the robot high-level controller first.")
            print("Body AND legs must be secured; both feet clear; support must remain safe when all Kp becomes zero.")
            print("This tool cannot detect the support fixture or verify H2 hardware identity.")
            if not args.auto:
                phrase = f"ARM H2 {args.side.upper()}"
                print(f"Type exactly {phrase!r} then Enter (still no motor commands):", flush=True)
                end = time.monotonic() + 120
                while not select.select([sys.stdin], [], [], 0.1)[0]:
                    if stop.is_set() or time.monotonic() > end:
                        raise SafetyStop("arming cancelled/timed out")
                if sys.stdin.readline().strip() != phrase or stop.is_set():
                    raise SafetyStop("arming cancelled")
            transport.healthy()
            s = transport.get()
            check_state(s, time.monotonic_ns(), limits, index=index)
            if max(abs(v) for v in s.dq) > 0.15:
                raise SafetyStop("robot must be stationary at arming")
            base, mode, initial_quat = s.q, s.mode_machine, s.quaternion
            lo, hi = limits[index]
            if not lo + 0.05 < base[index] - amp < base[index] + amp < hi - 0.05:
                raise SafetyStop("captured ankle pose lacks joint-limit margin")
            metadata.update(base_q=base, mode_machine=mode, initial_quaternion=initial_quat, status="armed")
            save_json(output / "metadata.json", metadata)
            transport.create_writer()
            rec.event("armed", base_q=base)
            # Waits never block the publisher. Every frequency needs a new keypress.
            phase, segment, frequency_index = "gain_ramp", 0.0, 0
            started = phase_start = time.monotonic()
            deadline = started
            last_send = started
            last_target_slot = None
            target = list(base)
            active = True
            print("Armed: captured-pose hold. n starts each frequency; q / Esc / Ctrl+C aborts to damping.")
            with keyboard(not args.auto) as key:
                while True:
                    now = time.monotonic()
                    if stop.is_set():
                        raise SafetyStop("signal stop requested")
                    pressed = key()
                    if pressed in ("q", "\x1b", "\x03", ""):
                        raise SafetyStop("operator stop / terminal closed")
                    if now - last_send > LOOP_TIMEOUT:
                        raise SafetyStop("command loop stalled for >20 ms")
                    if now - started > MAX_RUN_SECONDS:
                        raise SafetyStop("maximum experiment duration reached")
                    transport.healthy()
                    s = transport.get()
                    check_state(s, time.monotonic_ns(), limits, base, target, index, initial_quat, mode)
                    segment = now - phase_start
                    if phase == "gain_ramp" and segment >= GAIN_RAMP:
                        phase, phase_start = "settle", now
                    elif phase == "settle" and segment >= HOLD_SECONDS:
                        phase, phase_start = "waiting", now
                        print(f"Ready: {args.frequencies[frequency_index]:g} Hz. Inspect fixture; press n (60 s timeout).")
                    elif phase == "waiting":
                        if segment > WAIT_TIMEOUT:
                            raise SafetyStop("frequency confirmation timed out")
                        if args.auto or pressed == "n":
                            phase, phase_start = "excitation", now
                            rec.event("frequency_start", frequency_hz=args.frequencies[frequency_index])
                            print(f"Excitation {args.frequencies[frequency_index]:g} Hz", flush=True)
                    elif phase == "excitation" and segment >= (args.cycles + 4) / args.frequencies[frequency_index]:
                        phase, phase_start = "return_hold", now
                        rec.event("frequency_complete", frequency_hz=args.frequencies[frequency_index])
                    elif phase == "return_hold" and segment >= HOLD_SECONDS:
                        frequency_index += 1
                        if frequency_index == len(args.frequencies):
                            break
                        phase, phase_start = "waiting", now
                        print(f"Ready: {args.frequencies[frequency_index]:g} Hz. Inspect last response; press n (60 s timeout).")
                    segment = now - phase_start
                    frequency = args.frequencies[frequency_index]
                    slot = (phase, frequency_index, int(segment * TARGET_HZ))
                    measure = False
                    if slot != last_target_slot:
                        target = list(base)
                        if phase == "excitation":
                            delta, _ = wave(slot[2] / TARGET_HZ, frequency, amp, args.cycles)
                            target[index] += delta
                        last_target_slot = slot
                    if phase == "excitation":
                        _, measure = wave(segment, frequency, amp, args.cycles)
                    kp, kd = gains(index, args.kp_scale, args.kd_scale,
                        ramp=min(1, segment / GAIN_RAMP) if phase == "gain_ramp" else 1)
                    predicted_tau = kp[index] * (target[index]-s.q[index]) - kd[index]*s.dq[index]
                    if abs(predicted_tau) > 10:
                        raise SafetyStop("estimated commanded ankle PD torque >10 Nm")
                    elapsed = transport.send(target, kp, kd, mode, phase, frequency, segment, measure, sequence)
                    sequence += 1
                    last_send = time.monotonic()
                    if elapsed > LOOP_TIMEOUT:
                        raise SafetyStop("DDS write blocked for >20 ms")
                    deadline += 1 / PUBLISH_HZ
                    if deadline < last_send:
                        deadline = last_send
                    stop.wait(max(0, deadline-time.monotonic()))
    except (Exception, KeyboardInterrupt) as exc:
        reason, code = str(exc) or "interrupted", 1
        print(f"STOP: {reason}", file=sys.stderr)
    finally:
        if active and transport is not None:
            # Best effort only. Do not fight another writer or a changed firmware mode.
            rec.event("damping_start", reason=reason)
            end = time.monotonic() + 1.0
            while time.monotonic() < end:
                s = transport.get()
                if transport.command_writers > 1 or s is None or s.mode_machine != mode:
                    rec.event("damping_skipped", reason="publisher conflict or firmware mode changed")
                    break
                try:
                    kp, kd = gains(index, 1, 1, damping=True)
                    transport.send([0.0]*31, kp, kd, mode, "damping", 0, 0, False, sequence)
                    sequence += 1
                except Exception as exc:
                    rec.event("damping_failed", error=str(exc))
                    break
                time.sleep(1 / PUBLISH_HZ)
        rec.event("finished", reason=reason, exit_code=code)
        # Remove readers before closing the log queue: no callbacks after its sentinel.
        if transport is not None:
            with transport.callback_lock:
                transport.closed = True
            transport.state_reader = None
            transport.command_reader = None
            transport.writer = None
        rec.close()
        if rec.error:
            reason, code = rec.error, 1
        metadata.update(status=reason, exit_code=code, log_counts=rec.counts,
            max_state_receive_gap_ms=transport.max_state_gap_ms if transport else None,
            ended_wall_time=datetime.now().isoformat())
        save_json(output / "metadata.json", metadata)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        print(f"Finished: {reason}. Logs: {output}")
    return code


def main(argv=None):
    args = parse_args(argv)
    try:
        return run(args)
    except (SafetyStop, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
