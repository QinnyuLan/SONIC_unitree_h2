#!/usr/bin/env python3
"""Deploy-ready H2 pose, then unloaded ankle range or dynamic-response tests.

Default is a no-DDS preview. Body support must allow the full ready-pose motion;
feet and ankle linkages must remain unconstrained. This is not a balance policy.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import threading
import time

import yaml

from h2_ankle_sysid import (
    ASSET, ROOT, NAMES, KP, KD, Recorder, SafetyStop, Transport, bounded, check_state,
    gains, keyboard, limits_from_asset, save_json, TARGET_HZ, PUBLISH_HZ, LOOP_TIMEOUT,
    frequencies, wave,
)

PROFILE = ROOT / "gear_sonic/utils/mujoco_sim/wbc_configs/h2_native_sonic_unitree_ros_loop.yaml"
MARGIN = 0.05
WAIT_SECONDS = 120
HOLD_ERROR_LIMIT = 0.15
READY_ERROR_LIMITS = tuple(0.10 if i in (4, 5, 10, 11) else HOLD_ERROR_LIMIT for i in range(31))
READY_SPEED_LIMIT = 0.15
READY_STABLE_SECONDS = 1.0
READY_SETTLE_SECONDS = 3.0
READY_TIMEOUT_SECONDS = 10.0
DYNAMIC_SPEED_LIMIT = 0.4
DYNAMIC_ACCEL_LIMIT = 3.0
DYNAMIC_AMPLITUDE_LIMIT_DEG = 25.0
MAX_DYNAMIC_SPEED_LIMIT = 0.8
MAX_DYNAMIC_ACCEL_LIMIT = 8.0
DDS_TO_LOCAL = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 15, 16)


def ready_pose():
    config = yaml.safe_load(PROFILE.read_text())
    q = tuple(float(config["DEFAULT_DOF_ANGLES"][i]) for i in DDS_TO_LOCAL)
    kp = tuple(float(config["MOTOR_KP"][i]) for i in DDS_TO_LOCAL)
    kd = tuple(float(config["MOTOR_KD"][i]) for i in DDS_TO_LOCAL)
    if kp != KP or kd != KD or not all(math.isfinite(v) for v in q):
        raise ValueError("release profile changed; reconcile ready pose / PD contract before running")
    return q


def read_sweep_plan(path):
    data = json.loads(path.read_text())
    required = {"schema", "bands"}
    optional = {"target_speed_limit_rad_s", "target_accel_limit_rad_s2"}
    if (not isinstance(data, dict) or not required <= set(data) or set(data)-required-optional
            or data["schema"] != "h2_ankle_sweep_plan_v1"):
        raise ValueError("sweep plan requires schema h2_ankle_sweep_plan_v1 and bands")
    if not isinstance(data["bands"], list) or not 1 <= len(data["bands"]) <= 8:
        raise ValueError("sweep plan requires 1-8 amplitude bands")
    bands, seen = [], set()
    for band in data["bands"]:
        if not isinstance(band, dict) or set(band) != {"amplitude_deg", "frequencies_hz"}:
            raise ValueError("each band requires amplitude_deg and frequencies_hz")
        amplitude, fs = band["amplitude_deg"], band["frequencies_hz"]
        if type(amplitude) not in (int, float) or not math.isfinite(amplitude) or not .1 <= amplitude <= DYNAMIC_AMPLITUDE_LIMIT_DEG:
            raise ValueError("sweep amplitude must be finite, 0.1-25 deg")
        if amplitude in seen:
            raise ValueError("sweep amplitude bands must be unique")
        if not isinstance(fs, list) or not fs or any(type(f) not in (int, float) for f in fs):
            raise ValueError("frequencies_hz must be a nonempty numeric list")
        fs = frequencies(",".join(map(str, fs)))
        bands.append(dict(amplitude_deg=float(amplitude), frequencies_hz=fs))
        seen.add(amplitude)
    limits = {}
    for key, default, maximum in (("target_speed_limit_rad_s", DYNAMIC_SPEED_LIMIT, MAX_DYNAMIC_SPEED_LIMIT),
                                   ("target_accel_limit_rad_s2", DYNAMIC_ACCEL_LIMIT, MAX_DYNAMIC_ACCEL_LIMIT)):
        value = data.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError(f"{key} must be finite and in (0, {maximum}]")
        limits[key] = float(value)
    return bands, limits


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", choices=("real", "sim"), default="real")
    p.add_argument("--interface", default="eno1")
    p.add_argument("--domain", type=int, default=0)
    p.add_argument("--experiment", choices=("range", "dynamic"), default="range")
    p.add_argument("--side", choices=("left", "right", "both"))
    ranges = p.add_mutually_exclusive_group()
    ranges.add_argument("--full-range", action="store_true", help="explicitly request official pitch limits minus 0.05 rad margin")
    ranges.add_argument("--range-deg", nargs=2, type=bounded(-65, 35), metavar=("LOW", "HIGH"),
                        help="absolute pitch angles, must include the ready angle")
    p.add_argument("--range-step-deg", type=bounded(5, 20),
                   help="maximum expansion from ready per manually confirmed stage")
    p.add_argument("--speed-deg-s", type=bounded(1, 10))
    p.add_argument("--amplitude-deg", type=bounded(.1, DYNAMIC_AMPLITUDE_LIMIT_DEG), help="dynamic sine peak amplitude, default 3 deg")
    p.add_argument("--frequencies", type=frequencies, help="dynamic increasing frequencies in Hz, comma separated")
    p.add_argument("--sweep-plan", type=Path, help="explicit JSON amplitude/frequency bands, advanced automatically")
    p.add_argument("--cycles", type=int, help="dynamic measurement cycles per frequency, default 6 (3-12)")
    p.add_argument("--kp-scale", type=bounded(.25, 1.5), default=1.)
    p.add_argument("--kd-scale", type=bounded(.5, 2), default=1.)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--confirm-supported", action="store_true")
    p.add_argument("--auto", action="store_true", help="private loopback simulation only")
    p.add_argument("--output", type=Path)
    a = p.parse_args(argv)
    if not 0 <= a.domain <= 232:
        p.error("domain must be 0-232")
    if a.target == "sim" and (a.interface != "lo" or a.domain == 0):
        p.error("simulation requires lo and a private nonzero domain")
    if a.target == "real" and a.interface == "lo":
        p.error("real hardware cannot use lo")
    if a.auto and (a.target != "sim" or not a.execute):
        p.error("--auto is only for explicit simulation execution")
    if a.execute and not a.confirm_supported:
        p.error("--execute requires --confirm-supported")
    a.side = a.side or ("left" if a.experiment == "dynamic" else "both")
    if a.experiment == "dynamic":
        if a.side == "both":
            p.error("dynamic identification uses one ankle per run: --side left or right")
        if a.full_range or a.range_deg or a.range_step_deg is not None or a.speed_deg_s is not None:
            p.error("range options cannot be combined with --experiment dynamic")
        a.cycles = 6 if a.cycles is None else a.cycles
        if not 3 <= a.cycles <= 12:
            p.error("cycles must be 3-12")
        if a.sweep_plan:
            if a.amplitude_deg is not None or a.frequencies is not None:
                p.error("--sweep-plan cannot be combined with --amplitude-deg or --frequencies")
            try:
                a.sweep_bands, caps = read_sweep_plan(a.sweep_plan)
            except (OSError, ValueError, argparse.ArgumentTypeError) as exc:
                p.error(str(exc))
        else:
            a.amplitude_deg = 3. if a.amplitude_deg is None else a.amplitude_deg
            a.frequencies = a.frequencies or (.25, .5, .75, 1.)
            a.sweep_bands = [dict(amplitude_deg=a.amplitude_deg, frequencies_hz=a.frequencies)]
            caps = dict(target_speed_limit_rad_s=DYNAMIC_SPEED_LIMIT, target_accel_limit_rad_s2=DYNAMIC_ACCEL_LIMIT)
        a.target_speed_limit_rad_s = caps["target_speed_limit_rad_s"]
        a.target_accel_limit_rad_s2 = caps["target_accel_limit_rad_s2"]
        for band in a.sweep_bands:
            amplitude, omega = math.radians(band["amplitude_deg"]), 2*math.pi*max(band["frequencies_hz"])
            if amplitude*omega*1.125>a.target_speed_limit_rad_s or amplitude*omega**2*1.282>a.target_accel_limit_rad_s2:
                p.error(f"amplitude/frequency exceeds dynamic target caps "
                        f"({a.target_speed_limit_rad_s:g} rad/s, {a.target_accel_limit_rad_s2:g} rad/s^2) "
                        f"at +/-{band['amplitude_deg']:g} deg, {omega/(2*math.pi):g} Hz; lower this band's frequency")
        if sum((a.cycles+4)/f+3 for b in a.sweep_bands for f in b["frequencies_hz"])>600:
            p.error("dynamic excitation exceeds 600 seconds; split into separate runs")
    else:
        if any(v is not None for v in (a.amplitude_deg, a.frequencies, a.cycles, a.sweep_plan)):
            p.error("sine options require --experiment dynamic")
        a.range_step_deg = 15 if a.range_step_deg is None else a.range_step_deg
        a.speed_deg_s = 5 if a.speed_deg_s is None else a.speed_deg_s
    return a


def smooth_move(a, b, elapsed, duration):
    t = min(1., max(0., elapsed/duration))
    blend = t*t*t*(10-15*t+6*t*t)
    return tuple(x+(y-x)*blend for x, y in zip(a, b))


def move_duration(a, b, speed, acceleration):
    distance = max(abs(y-x) for x, y in zip(a, b))
    # Exact peak derivative bounds of the quintic time scaling.
    return max(2., 1.875*distance/speed, math.sqrt(5.774*distance/acceleration))


@dataclass(frozen=True)
class Stage:
    side: str
    index: int
    level: int
    lower: float
    upper: float

    @property
    def name(self):
        return f"range_{self.side}_{self.level}"


@dataclass(frozen=True)
class FrequencyStage(Stage):
    frequency_hz: float
    amplitude_deg: float

    @property
    def name(self):
        return f"dynamic_{self.side}_{self.amplitude_deg:g}deg_{self.frequency_hz:g}Hz"


def plan(args, ready, limits):
    if args.experiment == "dynamic":
        i = NAMES.index(f"{args.side}_ankle_pitch_joint")
        result = []
        for band in args.sweep_bands:
            amplitude = math.radians(band["amplitude_deg"])
            lower, upper = ready[i]-amplitude, ready[i]+amplitude
            if not limits[i][0]+MARGIN < lower < upper < limits[i][1]-MARGIN:
                raise ValueError("dynamic excitation lacks joint-limit margin")
            for frequency in band["frequencies_hz"]:
                result.append(FrequencyStage(args.side, i, len(result)+1, lower, upper,
                                             frequency, band["amplitude_deg"]))
        return result
    result = []
    for side in (("left", "right") if args.side == "both" else (args.side,)):
        i = NAMES.index(f"{side}_ankle_pitch_joint")
        lo, hi = (limits[i][0]+MARGIN, limits[i][1]-MARGIN)
        if args.range_deg:
            lower, upper = map(math.radians, args.range_deg)
        elif args.full_range:
            lower, upper = lo, hi
        else:
            lower, upper = ready[i]-math.radians(15), ready[i]+math.radians(15)
        if not lo-1e-10 <= lower < ready[i] < upper <= hi+1e-10:
            raise ValueError(f"range must include ready {math.degrees(ready[i]):.3f} deg and stay inside "
                             f"[{math.degrees(lo):.3f}, {math.degrees(hi):.3f}] deg")
        span = max(ready[i]-lower, upper-ready[i])
        step = math.radians(args.range_step_deg)
        for level in range(1, math.ceil(span/step)+1):
            radius = min(span, level*step)
            result.append(Stage(side, i, level, max(lower, ready[i]-radius), min(upper, ready[i]+radius)))
    return result


def stage_segments(stage, ready, speed):
    start = ready
    for label, angle in (("down", stage.lower), ("up", stage.upper), ("return", ready[stage.index])):
        end = list(ready)
        end[stage.index] = angle
        end = tuple(end)
        yield label, start, end, move_duration(start, end, speed, math.radians(10))
        yield label+"_hold", end, end, 2.0
        start = end


def trial_segments(stage, ready, args):
    if isinstance(stage, FrequencyStage):
        return [("excitation", ready, ready, (args.cycles+4)/stage.frequency_hz),
                ("return_hold", ready, ready, 3.)]
    return list(stage_segments(stage, ready, math.radians(args.speed_deg_s)))


def trial_target(stage, segment, sample_t, args):
    label, a, b, duration = segment
    if isinstance(stage, FrequencyStage) and label == "excitation":
        delta, measure = wave(sample_t, stage.frequency_hz, math.radians(stage.amplitude_deg), args.cycles)
        target = list(a)
        target[stage.index] += delta
        return tuple(target), measure
    return smooth_move(a, b, sample_t, duration), not isinstance(stage, FrequencyStage)


def command_phase(stage, segment_label, args):
    return stage.name+"_"+segment_label if args.sweep_plan else segment_label


def check_tracking(s, target, ready, index, preparation=False):
    for i in range(31):
        tolerance = 0.20 if preparation else (0.12 if i == index else HOLD_ERROR_LIMIT)
        if abs(s.q[i]-target[i]) > tolerance:
            raise SafetyStop(f"DDS {i} {NAMES[i]} target tracking error "
                             f"{math.degrees(s.q[i]-target[i]):+.2f} deg exceeds {math.degrees(tolerance):.2f}")
        if not preparation and i != index and target[i] != ready[i]:
            raise SafetyStop("internal error: a non-test target left the ready pose")


def reached_ready(s, ready):
    return (all(abs(s.q[i]-ready[i]) < READY_ERROR_LIMITS[i] for i in range(31))
            and all(abs(v) < READY_SPEED_LIMIT for v in s.dq))


def ready_blockers(s, ready):
    blockers = []
    for i in range(31):
        for kind, value, limit in (("position", s.q[i]-ready[i], READY_ERROR_LIMITS[i]),
                                   ("speed", s.dq[i], READY_SPEED_LIMIT)):
            if not math.isfinite(value) or abs(value) >= limit:
                blockers.append(dict(index=i, joint=NAMES[i], kind=kind,
                                     value=value, limit=limit))
    return blockers


def ready_failure_summary(blockers):
    if not blockers:
        return f"within limits now, but not stable for {READY_STABLE_SECONDS:g} s continuously"
    parts = []
    for b in sorted(blockers, key=lambda b: abs(b["value"])/b["limit"], reverse=True)[:3]:
        unit = "deg" if b["kind"] == "position" else "deg/s"
        parts.append(f"DDS {b['index']} {b['joint']} {b['kind']} "
                     f"{math.degrees(b['value']):+.2f} {unit} (limit {math.degrees(b['limit']):.2f})")
    if len(blockers)>3:
        parts.append(f"{len(blockers)-3} more violations")
    return "; ".join(parts)


def create_motion_switcher(args):
    sys.path.insert(0, str(ROOT / "external_dependencies/unitree_sdk2_python"))
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

    ChannelFactoryInitialize(args.domain, args.interface)
    client = MotionSwitcherClient()
    client.SetTimeout(2.0)
    client.Init()
    return client


def wait_for_start(stop, check, automatic=False):
    print("Secure the body and free both feet. y releases high-level control and starts preparation; "
          "q/Esc/Ctrl+C cancels. No motor commands yet.", flush=True)
    check()
    if automatic:
        return
    deadline = time.monotonic()+WAIT_SECONDS
    with keyboard(True) as key:
        while True:
            if stop.is_set() or time.monotonic()>deadline:
                raise SafetyStop("start timed out / interrupted")
            check()
            pressed = key()
            if pressed in ("q", "\x1b", "\x03", ""):
                raise SafetyStop("start cancelled")
            if pressed == "y":
                check()
                return
            stop.wait(.02)


def read_motion_mode(client, rec):
    code, result = client.CheckMode()
    rec.event("motion_switcher_check", code=code, result=result)
    if code != 0:
        raise SafetyStop(f"MotionSwitcher.CheckMode failed: code={code}; no control takeover")
    if not isinstance(result, dict) or not isinstance(result.get("name"), str):
        raise SafetyStop("MotionSwitcher.CheckMode returned an invalid mode; no control takeover")
    return result["name"]


def release_high_level(client, tr, rec, stop, check, *, authorized=False):
    if not authorized:
        raise SafetyStop("control release requires the operator start key")
    if tr.own_writer_created:
        raise SafetyStop("release must happen before creating the LowCmd writer")
    check()
    name = read_motion_mode(client, rec)
    print(f"MotionSwitcher mode: {name!r}; LowCmd publishers: {tr.command_writers}")
    rec.event("motion_switcher_release_authorized", initial_mode=name)
    # Bounded version of deploy's CheckMode / ReleaseMode loop. No LowCmd yet.
    for attempt in range(1, 4):
        check()
        name = read_motion_mode(client, rec)
        check()
        if not name:
            break
        code, _ = client.ReleaseMode()
        rec.event("motion_switcher_release", attempt=attempt, code=code)
        if code != 0:
            raise SafetyStop(f"MotionSwitcher.ReleaseMode failed: code={code}; no control takeover")
        deadline = time.monotonic()+2
        while time.monotonic()<deadline:
            check()
            stop.wait(.02)
    check()
    if read_motion_mode(client, rec):
        raise SafetyStop("MotionSwitcher mode still active after release attempts; no control takeover")
    print("High-level mode released. Waiting for LowCmd publishers to leave...", flush=True)
    deadline, clear_since = time.monotonic()+10, None
    while True:
        check()
        now = time.monotonic()
        clear_since = (now if clear_since is None else clear_since) if tr.command_writers == 0 else None
        if clear_since is not None and now-clear_since>=1:
            break
        if now>=deadline:
            raise SafetyStop(f"mode released but LowCmd publishers={tr.command_writers}, expected 0; "
                             "stop other controllers; no control takeover")
        stop.wait(.02)
    if read_motion_mode(client, rec):
        raise SafetyStop("high-level mode reactivated during takeover; no control takeover")
    check()
    tr.healthy()
    rec.event("motion_switcher_released", command_writers=tr.command_writers)
    print("Control release verified: mode empty, LowCmd publishers=0.", flush=True)


def run(args):
    dynamic = args.experiment == "dynamic"
    ready, limits = ready_pose(), limits_from_asset()
    stages = plan(args, ready, limits)
    for stage in stages:
        if dynamic:
            print(f"{stage.name}: center {math.degrees(ready[stage.index]):+.2f} deg, "
                  f"sine +/-{stage.amplitude_deg:g} deg")
        else:
            print(f"{stage.name}: {math.degrees(stage.lower):+.2f} -> {math.degrees(stage.upper):+.2f} "
                  f"-> ready {math.degrees(ready[stage.index]):+.2f} deg")
    if dynamic:
        duration = sum((args.cycles+4)/s.frequency_hz+3 for s in stages)
        print(f"Dynamic sine: {len(stages)} trials, {duration:.1f} s; 2 ramp-in + {args.cycles} measurement + "
              "2 ramp-out cycles per frequency. All listed amplitudes/frequencies advance after one n.")
        print(f"Planned target caps: {args.target_speed_limit_rad_s:g} rad/s, "
              f"{args.target_accel_limit_rad_s2:g} rad/s^2. Measured-speed/error/torque guards are unchanged.")
        print("Body IMU orientation stop: DISABLED for supported dynamic tests. "
              "IMU logging and data-validity checks remain enabled.")
    else:
        print(f"Max scan speed {args.speed_deg_s:g} deg/s")
    print(f"Kp={40*args.kp_scale:g}, Kd={2*args.kd_scale:g}")
    print("Same release ready angles/PD as deploy; 50 Hz targets / 500 Hz publication, dq_des=tau_ff=0.")
    print(f"Ready check: ankle pitch/roll <{math.degrees(READY_ERROR_LIMITS[5]):.2f} deg error, "
          f"other joints <{math.degrees(HOLD_ERROR_LIMIT):.2f} deg; "
          f"all speeds <{math.degrees(READY_SPEED_LIMIT):.2f} deg/s, stable for {READY_STABLE_SECONDS:g} s.")
    if not args.execute:
        print("PREVIEW: no DDS or motor commands. Experiment bounds are NOT certified hardware clearance.")
        return 0
    if not args.auto and not sys.stdin.isatty():
        raise SafetyStop("interactive terminal required for real hardware")
    output = args.output or ROOT / "gear_sonic_deploy/logs" / (
        "ankle_"+args.experiment+"_"+args.side+"_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    rec = Recorder(output)
    schema = ("h2_ankle_dynamic_v2" if args.sweep_plan else "h2_ankle_dynamic_v1") if dynamic else "h2_ankle_range_v1"
    meta = dict(schema=schema,
        args={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        joints_dds=NAMES, base_q=ready, ready_q=ready, stages=[asdict(s) for s in stages],
        nominal_kp=KP, nominal_kd=KD, publish_hz=PUBLISH_HZ, target_hz=TARGET_HZ,
        imu_orientation_stop_enabled=not dynamic,
        ready_check=dict(position_error_limits_rad=READY_ERROR_LIMITS, speed_limit_rad_s=READY_SPEED_LIMIT,
                         stable_seconds=READY_STABLE_SECONDS, min_settle_seconds=READY_SETTLE_SECONDS,
                         timeout_seconds=READY_TIMEOUT_SECONDS),
        asset=str(ASSET), asset_sha256=hashlib.sha256(ASSET.read_bytes()).hexdigest(),
        profile_sha256=hashlib.sha256(PROFILE.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        started_wall_time=datetime.now().isoformat(), status="initializing", completed_stages=[])
    if dynamic:
        for item, stage in zip(meta["stages"], stages):
            item["excitation_phase"] = command_phase(stage, "excitation", args)
        meta.update(active_index=stages[0].index, waveform="raised_cosine_ramped_fixed_amplitude_sine",
                    identification="local closed-loop q_des -> measured q; not open-loop motor identification",
                    target_speed_limit_rad_s=args.target_speed_limit_rad_s,
                    target_accel_limit_rad_s2=args.target_accel_limit_rad_s2)
    save_json(output / "metadata.json", meta)
    tr, active, mode, sequence = None, False, None, 0
    switcher = None
    stop = threading.Event()
    handlers = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    status, code = "completed", 0
    try:
        if args.target == "real":
            switcher = create_motion_switcher(args)
        tr = Transport(args, rec, domain_initialized=switcher is not None)
        print(f"Logs: {output}")
        deadline = time.monotonic()+10
        while tr.get() is None:
            if tr.error or stop.is_set() or time.monotonic()>deadline:
                raise SafetyStop(tr.error or "LowState timeout / interrupted")
            time.sleep(.01)
        def preflight(before_release=False):
            if stop.is_set():
                raise SafetyStop("operator / signal stop before takeover")
            tr.healthy(before_release=before_release)
            if before_release and tr.command_writers>1:
                raise SafetyStop(f"{tr.command_writers} LowCmd publishers before release; stop other controllers first")
            for index in (5, 11):
                check_state(tr.get(), time.monotonic_ns(), limits, index=index)
            if before_release and max(abs(v) for v in tr.get().dq)>.15:
                raise SafetyStop("control release requires stationary, supported robot")

        deadline = time.monotonic()+1
        while time.monotonic()<deadline:
            preflight(before_release=switcher is not None)
            time.sleep(.01)
        wait_for_start(stop, lambda: preflight(before_release=switcher is not None), automatic=args.auto)
        rec.event("start_authorized", source="simulation_auto" if args.auto else "operator_y")
        if switcher is not None:
            meta["status"] = "awaiting_control_release"
            save_json(output / "metadata.json", meta)
            release_high_level(switcher, tr, rec, stop, lambda: preflight(before_release=True), authorized=True)
            meta.update(high_level_release_verified=True, status="preparing_takeover")
            save_json(output / "metadata.json", meta)
        if switcher is not None and read_motion_mode(switcher, rec):
            raise SafetyStop("high-level mode reactivated before arming; no control takeover")
        preflight()
        tr.healthy()
        s = tr.get()
        check_state(s, time.monotonic_ns(), limits)
        if max(abs(v) for v in s.dq)>.15 or stop.is_set():
            raise SafetyStop("arming requires stationary state")
        captured, quat, mode = s.q, s.quaternion, s.mode_machine
        prep_duration = max(3., move_duration(captured, ready, .35, .5))
        meta.update(captured_q=captured, initial_quaternion=quat, mode_machine=mode,
                    preparation_seconds=prep_duration, status="armed")
        save_json(output / "metadata.json", meta)
        tr.create_writer()
        active = True
        rec.event("armed", captured_q=captured, ready_q=ready)
        phase, phase_start = "armed_hold", time.monotonic()
        started = deadline = last_send = phase_start
        target = captured
        stage_number = segment_number = 0
        segments = []
        stable_since = None
        next_ready_report = 0.
        trial_started = False
        print("Armed. Preparing after gain ramp. n: "
              + ("start entire frequency sweep" if dynamic else "confirm each range")
              + "; q/Esc/Ctrl+C: damping stop.", flush=True)
        with keyboard(not args.auto) as key:
            while True:
                now = time.monotonic()
                pressed = key()
                if stop.is_set() or pressed in ("q", "\x1b", "\x03", ""):
                    raise SafetyStop("operator / signal stop")
                if now-last_send>LOOP_TIMEOUT or now-started>2400:
                    raise SafetyStop("publisher stalled >20 ms / total duration exceeded")
                tr.healthy()
                s = tr.get()
                stage = stages[stage_number]
                # The selected ankle may traverse its range; do not apply the old 0.10-rad capture bound.
                for i in (5, 11):
                    check_state(s, time.monotonic_ns(), limits, index=i,
                                initial_quat=None if dynamic else quat, mode=mode)
                elapsed = now-phase_start
                if phase == "armed_hold":
                    if elapsed>=3:
                        # Start from the latest actual pose, not a stale startup measurement.
                        captured = s.q
                        prep_duration = max(3., move_duration(captured, ready, .35, .5))
                        phase, phase_start = "prepare", now
                        rec.event("prepare_start", from_q=captured, target_q=ready, duration=prep_duration)
                        print(f"Moving to deploy ready pose in {prep_duration:.1f} s", flush=True)
                elif phase == "prepare" and elapsed>=prep_duration:
                    phase, phase_start, stable_since = "ready_settle", now, None
                    next_ready_report = now+2
                elif phase == "ready_settle":
                    stable_since = (now if stable_since is None else stable_since) if reached_ready(s, ready) else None
                    if (stable_since is not None and now-stable_since>=READY_STABLE_SECONDS
                            and elapsed>=READY_SETTLE_SECONDS):
                        phase, phase_start = "waiting", now
                        rec.event("ready_reached", q=s.q, dq=s.dq,
                                  position_error=[q-target for q, target in zip(s.q, ready)])
                        print(f"Init Done. {stage.name}: inspect clearance, then n", flush=True)
                    elif elapsed>READY_TIMEOUT_SECONDS:
                        blockers = ready_blockers(s, ready)
                        rec.event("ready_timeout", blockers=blockers, elapsed_seconds=elapsed)
                        raise SafetyStop(f"ready pose not reached / not stationary within {READY_TIMEOUT_SECONDS:g} s: "
                                         + ready_failure_summary(blockers))
                    elif now>=next_ready_report:
                        blockers = ready_blockers(s, ready)
                        if blockers:
                            rec.event("ready_wait", blockers=blockers, elapsed_seconds=elapsed)
                            print(f"Waiting for ready ({elapsed:.1f}/{READY_TIMEOUT_SECONDS:g} s): "
                                  + ready_failure_summary(blockers), flush=True)
                        next_ready_report = now+2
                elif phase == "waiting":
                    if elapsed>WAIT_SECONDS:
                        raise SafetyStop("experiment start timed out")
                    if args.auto or pressed == "n" or (dynamic and trial_started):
                        if not reached_ready(s, ready):
                            raise SafetyStop("robot must be settled at ready before excitation: "
                                             + ready_failure_summary(ready_blockers(s, ready)))
                        segments = trial_segments(stage, ready, args)
                        phase, phase_start, segment_number = "scanning", now, 0
                        trial_started = True
                        rec.event("frequency_start" if dynamic else "range_start", **asdict(stage))
                        print(f"{stage.name}: [{math.degrees(stage.lower):.2f}, {math.degrees(stage.upper):.2f}] deg", flush=True)
                elif phase == "scanning" and elapsed>=segments[segment_number][3]:
                    segment_number += 1
                    if segment_number == len(segments):
                        if not reached_ready(s, ready):
                            raise SafetyStop("did not return to ready after scan: "
                                             + ready_failure_summary(ready_blockers(s, ready)))
                        rec.event("frequency_complete" if dynamic else "range_complete", **asdict(stage))
                        meta["completed_stages"].append(asdict(stage))
                        stage_number += 1
                        if stage_number == len(stages):
                            break
                        stage = stages[stage_number]
                        phase, phase_start = "waiting", now
                        print(f"Returned to ready. Next {stage.name}: "
                              + ("automatic next frequency" if dynamic else "press n (no automatic expansion/side switch)"), flush=True)
                    else:
                        phase_start = now
                elapsed = now-phase_start
                sample_t = int(elapsed*TARGET_HZ)/TARGET_HZ
                measure = False
                if phase == "prepare":
                    target = smooth_move(captured, ready, sample_t, prep_duration)
                elif phase == "armed_hold":
                    target = captured
                elif phase == "scanning":
                    target, measure = trial_target(stage, segments[segment_number], sample_t, args)
                else:
                    target = ready
                kp, kd = gains(stage.index, args.kp_scale, args.kd_scale)
                # Both ankle gains are fixed throughout prep, side switching and scanning.
                for i in (5, 11):
                    kp[i], kd[i] = 40*args.kp_scale, 2*args.kd_scale
                if phase == "armed_hold":
                    kp = [v*min(1, elapsed/3) for v in kp]
                check_tracking(s, target, ready, stage.index,
                               preparation=phase in ("prepare", "armed_hold", "ready_settle"))
                for i in (5, 11):
                    pd_tau = kp[i]*(target[i]-s.q[i])-kd[i]*s.dq[i]
                    if abs(pd_tau)>10:
                        raise SafetyStop(f"DDS {i} predicted PD torque >10 Nm")
                label = phase
                frequency = 0.
                if phase == "scanning":
                    label = (command_phase(stage, segments[segment_number][0], args) if dynamic
                             else stage.name+"_"+segments[segment_number][0])
                    frequency = stage.frequency_hz if dynamic else 0.
                write_time = tr.send(target, kp, kd, mode, label, frequency, elapsed, measure, sequence)
                sequence += 1
                last_send = time.monotonic()
                if write_time>LOOP_TIMEOUT:
                    raise SafetyStop("DDS write blocked >20 ms")
                deadline = max(deadline+1/PUBLISH_HZ, last_send)
                stop.wait(max(0, deadline-time.monotonic()))
    except (Exception, KeyboardInterrupt) as exc:
        status, code = str(exc) or "interrupted", 1
        print(f"STOP: {status}", file=sys.stderr)
    finally:
        if active:
            rec.event("damping_start", reason=status)
            end = time.monotonic()+1
            while time.monotonic()<end:
                s = tr.get()
                if s is None or s.mode_machine!=mode or tr.command_writers>1:
                    break
                try:
                    tr.send([0.]*31, [0.]*31, [8.]*31, mode, "damping", 0, 0, False, sequence)
                    sequence += 1
                except Exception:
                    break
                time.sleep(.002)
        rec.event("finished", reason=status, exit_code=code)
        if tr is not None:
            with tr.callback_lock:
                tr.closed = True
            tr.state_reader = tr.command_reader = tr.writer = None
        rec.close()
        if rec.error:
            status, code = rec.error, 1
        meta.update(status=status, exit_code=code, log_counts=rec.counts, ended_wall_time=datetime.now().isoformat())
        save_json(output / "metadata.json", meta)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        print(f"Finished: {status}. Logs: {output}")
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except (SafetyStop, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
