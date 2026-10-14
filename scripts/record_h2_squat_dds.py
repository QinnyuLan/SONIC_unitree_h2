#!/usr/bin/env python3
"""Read-only H2 LowCmd/LowState/IMU/optional odometry recorder. Never publishes."""

import argparse
import json
from pathlib import Path
import signal
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external_dependencies/unitree_sdk2_python"))

from gear_sonic.utils.diagnostic_log import DiagnosticLog


PREFIX = ["time_monotonic_ms", "time_realtime_ms", "dds_source_timestamp_ns", "publisher_handle"]
IMU = [f"imu_{group}_{axis}" for group, axes in
       (("quat_wxyz", "wxyz"), ("gyro", "xyz"), ("accel", "xyz")) for axis in axes]
HEADERS = {
    "lowstate": PREFIX + ["tick", "mode_pr", "mode_machine", "crc_ok"] +
        [f"{group}_{i}" for group in ("q", "dq", "tau_est", "motor_error", "temperature_0", "temperature_1") for i in range(31)] + IMU,
    "lowcmd": PREFIX + ["mode_pr", "mode_machine", "crc_ok"] +
        [f"{group}_{i}" for group in ("mode", "q_target", "dq_target", "kp", "kd", "tau_ff") for i in range(31)],
    "secondary_imu": PREFIX + IMU,
    "odometry_unverified": PREFIX + ["tick"] +
        [f"{group}_{axis}" for group, axes in (("position", "xyz"), ("orientation_raw", "0123"),
                                              ("linear_velocity", "xyz"), ("angular_velocity", "xyz")) for axis in axes],
}


def imu_values(msg):
    return [*msg.quaternion, *msg.gyroscope, *msg.accelerometer]


def message_row(name, msg, crc, clock=None):
    info = getattr(msg, "sample_info", None)
    prefix = [time.monotonic_ns()/1e6 if clock is None else clock, time.time_ns()/1e6,
              int(getattr(info, "source_timestamp", 0)), int(getattr(info, "publication_handle", 0))]
    if name == "lowstate":
        motors = msg.motor_state[:31]
        return prefix + [msg.tick, msg.mode_pr, msg.mode_machine, int(crc.Crc(msg) == msg.crc)] + [
            value for group in ((m.q for m in motors), (m.dq for m in motors),
                                (m.tau_est for m in motors), (m.motorstate for m in motors),
                                (m.temperature[0] for m in motors), (m.temperature[1] for m in motors))
            for value in group] + imu_values(msg.imu_state)
    if name == "lowcmd":
        motors = msg.motor_cmd[:31]
        return prefix + [msg.mode_pr, msg.mode_machine, int(crc.Crc(msg) == msg.crc)] + [
            getattr(m, group) for group in ("mode", "q", "dq", "kp", "kd", "tau") for m in motors]
    if name == "secondary_imu":
        return prefix + imu_values(msg)
    return prefix + [msg.tick, *msg.position, *msg.orientation, *msg.linear_velocity, *msg.angular_velocity]


def create_reader(participant, topic, qos, listener):
    from cyclonedds.core import Listener
    from cyclonedds.sub import DataReader

    # Constructor callbacks can precede Python entity registration and receive None.
    # Status callbacks don't dereference the reader; keep early discovery counts.
    startup_listener = Listener(on_subscription_matched=listener.on_subscription_matched,
                                on_sample_lost=listener.on_sample_lost,
                                on_sample_rejected=listener.on_sample_rejected)
    reader = DataReader(participant, topic, qos=qos, listener=startup_listener)
    reader.set_listener(listener)
    # Data may already be queued before the listener is attached.
    listener.on_data_available(reader)
    return reader


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=("real", "sim"))
    parser.add_argument("--interface", required=True)
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=0, help="0: until Ctrl+C")
    args = parser.parse_args(argv)
    if not 0 <= args.domain <= 232 or not 0 <= args.seconds <= 86400:
        parser.error("invalid domain or duration")
    if args.target == "sim" and args.interface != "lo":
        parser.error("simulation recorder must use lo")
    if args.target == "real" and args.interface == "lo":
        parser.error("real recorder must use the robot NIC, not lo")

    from cyclonedds.core import Listener
    from cyclonedds.domain import Domain, DomainParticipant
    from cyclonedds.qos import Qos, Policy
    from cyclonedds.topic import Topic
    from unitree_sdk2py.core.channel_config import ChannelConfigHasInterface
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_, IMUState_, OdoState_
    from unitree_sdk2py.utils.crc import CRC
    from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import H2_JOINT_NAMES_DDS

    config = ChannelConfigHasInterface.replace("$__IF_NAME__$", args.interface)
    domain = Domain(args.domain, config)
    participant = DomainParticipant(args.domain)
    log = DiagnosticLog(args.output, HEADERS, dict(
        schema="h2_squat_readonly_dds_v1", target=args.target, interface=args.interface,
        domain=args.domain, joint_order=H2_JOINT_NAMES_DDS,
        source="DDS received samples; no writer, no RPC, no control authority",
        semantics={"clock": "CLOCK_MONOTONIC ms on this host; arrival time, NOT acquisition time",
                   "tau_est": "drive-reported estimate, NOT calibrated output torque",
                   "position_contact_force_cop": "unavailable in H2 LowState; not inferred or filled with zero",
                   "odometry_unverified": "optional rt/odostate; provenance/frame unverified, not ground truth",
                   "crc": "recorded, not filtered; simulation may intentionally lack valid CRC"}))
    lock = threading.Lock()
    stopping = threading.Event()
    accepting = True
    quality = dict(received=dict.fromkeys(HEADERS, 0), matched_publishers=dict.fromkeys(HEADERS, 0),
                   max_matched_publishers=dict.fromkeys(HEADERS, 0),
                   lost_or_rejected=0, invalid_crc=dict(lowstate=0, lowcmd=0), callback_errors=[])
    crc = CRC()

    def receive(name, reader):
        nonlocal accepting
        with lock:
            if not accepting:
                return
            try:
                for msg in reader.take(1024):
                    if not getattr(getattr(msg, "sample_info", None), "valid_data", True):
                        continue
                    row = message_row(name, msg, crc)
                    quality['received'][name] += 1
                    if name in ("lowcmd", "lowstate") and not row[7 if name == "lowstate" else 6]:
                        quality['invalid_crc'][name] += 1
                    log.put(name, row)
            except Exception as exc:
                quality['callback_errors'].append(f"{name}: {exc}")
                stopping.set()

    def matched(name, status):
        with lock:
            quality['matched_publishers'][name] = status.current_count
            quality['max_matched_publishers'][name] = max(quality['max_matched_publishers'][name], status.current_count)
        print(f"[Recorder] {name}: {status.current_count} publisher(s)", flush=True)

    def lost(reader, status):
        with lock:
            quality['lost_or_rejected'] += int(status.total_count_change)

    topics, readers = [], []
    qos = Qos(Policy.History.KeepLast(4096), Policy.Reliability.BestEffort)
    try:
        for name, topic_name, cls in (("lowstate", "rt/lowstate", LowState_),
                                      ("lowcmd", "rt/lowcmd", LowCmd_),
                                      ("secondary_imu", "rt/secondary_imu", IMUState_),
                                      ("odometry_unverified", "rt/odostate", OdoState_)):
            topic = Topic(participant, topic_name, cls)
            topics.append(topic)
            reader = create_reader(participant, topic, qos=qos, listener=Listener(
                on_data_available=lambda r, n=name: receive(n, r),
                on_subscription_matched=lambda r, s, n=name: matched(n, s),
                on_sample_lost=lost, on_sample_rejected=lost))
            readers.append(reader)
        signal.signal(signal.SIGTERM, lambda *_: stopping.set())
        signal.signal(signal.SIGINT, lambda *_: stopping.set())
        print(f"Read-only recorder ready: {args.output}. Ctrl+C stops recording only.", flush=True)
        start = time.monotonic()
        last_status = start
        while not stopping.wait(.1):
            now = time.monotonic()
            if args.seconds and now - start >= args.seconds:
                break
            if log.error:
                print(f"Recorder disk error: {log.error}", file=sys.stderr)
                break
            if now - last_status >= 10:
                print(f"[Recorder] {quality['received']}, dropped={log.dropped}", flush=True)
                last_status = now
    finally:
        with lock:
            accepting = False
        for reader in readers:
            reader.set_listener(None)
        readers.clear()
        quality['missing_required_topics'] = [n for n in ('lowstate', 'lowcmd') if not quality['received'][n]]
        log.metadata['quality'] = quality
        log.close(incomplete=bool(quality['callback_errors']))
        print(f"Recorder finished: {args.output}; {json.dumps(quality)}")
    return 1 if log.error or log.dropped or quality['callback_errors'] else 0


if __name__ == "__main__":
    raise SystemExit(main())
