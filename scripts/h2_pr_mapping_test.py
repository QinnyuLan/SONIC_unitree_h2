#!/usr/bin/env python3
"""Diagnose Unitree H2 PR-mode ankle/waist command mapping.

This script is intentionally independent from the SONIC policy deploy path. It
publishes a small PR-mode LowCmd step to one motor-command slot at a time while
holding the other motors near their initial LowState positions. Run it only with
the robot safely suspended or otherwise mechanically supported.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import signal
import sys
import threading
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_UNITREE_PY = REPO_ROOT / "external_dependencies" / "unitree_sdk2_python"
if LOCAL_UNITREE_PY.exists():
    sys.path.insert(0, str(LOCAL_UNITREE_PY))

try:
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC
except Exception as exc:  # pragma: no cover - depends on robot SDK runtime.
    raise SystemExit(
        "Failed to import unitree_sdk2py. Install/configure the Unitree Python SDK "
        "or run from the robot environment with CycloneDDS available.\n"
        f"Import error: {exc}"
    ) from exc


MODE_PR = 0
MODE_AB = 1
HG_LOW_CMD_MOTORS = 35
DEFAULT_H2_MOTORS = 31

WATCH_NAMES = {
    4: "slot4 / code-assumed L ankle roll / Unitree-PR-example L ankle pitch",
    5: "slot5 / code-assumed L ankle pitch / Unitree-PR-example L ankle roll",
    10: "slot10 / code-assumed R ankle roll / Unitree-PR-example R ankle pitch",
    11: "slot11 / code-assumed R ankle pitch / Unitree-PR-example R ankle roll",
    12: "slot12 / waist yaw",
    13: "slot13 / waist roll or waist A",
    14: "slot14 / waist pitch or waist B",
}


@dataclass
class Sample:
    t: float
    phase: str
    commanded_index: int
    target: list[float]
    q: list[float]
    dq: list[float]
    tau: list[float]
    mode_pr: int
    mode_machine: int


class LowStateBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = None

    def callback(self, msg: LowState_) -> None:
        with self._lock:
            self._state = msg

    def get(self):
        with self._lock:
            return self._state

    def wait(self, timeout_s: float):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            state = self.get()
            if state is not None:
                return state
            time.sleep(0.02)
        raise TimeoutError("Timed out waiting for rt/lowstate.")


def parse_indices(text: str) -> list[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise argparse.ArgumentTypeError("at least one index is required")
    return values


def read_q_dq_tau(state, count: int) -> tuple[list[float], list[float], list[float]]:
    q = [float(state.motor_state[i].q) for i in range(count)]
    dq = [float(state.motor_state[i].dq) for i in range(count)]
    tau = [float(state.motor_state[i].tau_est) for i in range(count)]
    return q, dq, tau


def build_cmd(
    *,
    crc: CRC,
    base_q: list[float],
    target_q: list[float],
    mode_machine: int,
    num_motors: int,
    active_indices: set[int],
    kp: float,
    kd: float,
    hold_kp: float,
    hold_kd: float,
) -> LowCmd_:
    cmd = unitree_hg_msg_dds__LowCmd_()
    cmd.mode_pr = MODE_PR
    cmd.mode_machine = int(mode_machine)
    for i in range(HG_LOW_CMD_MOTORS):
        motor = cmd.motor_cmd[i]
        if i < num_motors:
            motor.mode = 1
            motor.q = float(target_q[i])
            motor.dq = 0.0
            motor.tau = 0.0
            if i in active_indices:
                motor.kp = float(kp)
                motor.kd = float(kd)
            else:
                motor.kp = float(hold_kp)
                motor.kd = float(hold_kd)
        else:
            motor.mode = 0
            motor.q = 0.0
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = 0.0
            motor.kd = 0.0
    cmd.crc = crc.Crc(cmd)
    return cmd


def publish_phase(
    *,
    publisher,
    state_buffer: LowStateBuffer,
    crc: CRC,
    base_q: list[float],
    target_q: list[float],
    mode_machine: int,
    num_motors: int,
    active_indices: set[int],
    kp: float,
    kd: float,
    hold_kp: float,
    hold_kd: float,
    duration_s: float,
    rate_hz: float,
    phase: str,
    commanded_index: int,
    samples: list[Sample],
    stop_event: threading.Event,
) -> None:
    period_s = 1.0 / rate_hz
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline and not stop_event.is_set():
        cmd = build_cmd(
            crc=crc,
            base_q=base_q,
            target_q=target_q,
            mode_machine=mode_machine,
            num_motors=num_motors,
            active_indices=active_indices,
            kp=kp,
            kd=kd,
            hold_kp=hold_kp,
            hold_kd=hold_kd,
        )
        publisher.Write(cmd)
        state = state_buffer.get()
        if state is not None:
            q, dq, tau = read_q_dq_tau(state, num_motors)
            samples.append(
                Sample(
                    t=time.time(),
                    phase=phase,
                    commanded_index=commanded_index,
                    target=target_q[:],
                    q=q,
                    dq=dq,
                    tau=tau,
                    mode_pr=int(getattr(state, "mode_pr", MODE_PR)),
                    mode_machine=int(getattr(state, "mode_machine", mode_machine)),
                )
            )
        time.sleep(period_s)


def mean_tail(samples: list[Sample], phase: str, index: int, tail_s: float) -> list[float] | None:
    phase_samples = [s for s in samples if s.phase == phase and s.commanded_index == index]
    if not phase_samples:
        return None
    t_last = phase_samples[-1].t
    tail = [s for s in phase_samples if s.t >= t_last - tail_s]
    if not tail:
        tail = phase_samples
    width = len(tail[0].q)
    return [sum(s.q[i] for s in tail) / len(tail) for i in range(width)]


def write_log(path: Path, samples: list[Sample], watch_indices: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["time_wall_s", "phase", "commanded_index", "mode_pr", "mode_machine"]
        for idx in watch_indices:
            header.extend([f"target_{idx}", f"q_{idx}", f"dq_{idx}", f"tau_{idx}"])
        writer.writerow(header)
        for sample in samples:
            row = [f"{sample.t:.6f}", sample.phase, sample.commanded_index, sample.mode_pr, sample.mode_machine]
            for idx in watch_indices:
                row.extend(
                    [
                        f"{sample.target[idx]:.9f}",
                        f"{sample.q[idx]:.9f}",
                        f"{sample.dq[idx]:.9f}",
                        f"{sample.tau[idx]:.9f}",
                    ]
                )
            writer.writerow(row)


def maybe_release_motion_mode(skip_release: bool) -> None:
    if skip_release:
        print("[INFO] Skipping MotionSwitcher ReleaseMode.")
        return
    print("[INFO] Releasing motion-control service via MotionSwitcherClient...")
    try:
        client = MotionSwitcherClient()
        client.SetTimeout(5.0)
        client.Init()
        for _ in range(5):
            code, result = client.CheckMode()
            if code != 0:
                print(f"[WARN] MotionSwitcher CheckMode failed with code {code}.")
                print("[WARN] This usually means the Python DDS/RPC path is not reaching the robot on the selected interface.")
                return
            name = result.get("name", "") if code == 0 and result else ""
            form = result.get("form", "") if code == 0 and result else ""
            if not name:
                print("[INFO] Motion-control service already released.")
                return
            print(f"[INFO] Active service form={form!r}, name={name!r}; calling ReleaseMode().")
            client.ReleaseMode()
            time.sleep(1.0)
        print("[WARN] MotionSwitcher still reported an active mode after retries. Continue only if you know it is safe.")
    except Exception as exc:
        print(f"[WARN] MotionSwitcher release failed: {exc}")
        print("[WARN] Continue only if high-level motion service is already disabled.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("network_interface", nargs="?", help="Robot network interface, e.g. enp2s0. Omit to use SDK default.")
    parser.add_argument("--i-understand-suspended", action="store_true", help="Required. Confirms the robot is suspended/supported.")
    parser.add_argument("--indices", type=parse_indices, default=parse_indices("4,5,10,11"), help="Comma-separated command slots to test.")
    parser.add_argument("--include-waist", action="store_true", help="Also test slots 13 and 14 with the same small PR-mode step.")
    parser.add_argument("--num-motors", type=int, default=DEFAULT_H2_MOTORS, help="Number of active H2 motor slots to command.")
    parser.add_argument("--delta", type=float, default=0.10, help="Step size in radians.")
    parser.add_argument("--negative", action="store_true", help="Also test a negative step after each positive step.")
    parser.add_argument("--warmup-duration", type=float, default=1.0)
    parser.add_argument("--settle-duration", type=float, default=0.7)
    parser.add_argument("--step-duration", type=float, default=1.2)
    parser.add_argument("--rate-hz", type=float, default=250.0, help="LowCmd publish rate for this Python diagnostic.")
    parser.add_argument("--kp", type=float, default=40.0, help="Kp for the actively stepped slot.")
    parser.add_argument("--kd", type=float, default=1.0, help="Kd for the actively stepped slot.")
    parser.add_argument("--hold-kp", type=float, default=20.0, help="Kp for all non-tested slots held at initial q.")
    parser.add_argument("--hold-kd", type=float, default=1.0, help="Kd for all non-tested slots held at initial q.")
    parser.add_argument("--tail-s", type=float, default=0.35, help="Tail window used for response summary.")
    parser.add_argument("--log", type=Path, help="CSV output path. Default: /tmp/h2_pr_mapping_test/<timestamp>.csv")
    parser.add_argument("--skip-release-mode", action="store_true", help="Do not call MotionSwitcher ReleaseMode.")
    parser.add_argument("--no-prompt", action="store_true", help="Do not pause before each individual step.")
    parser.add_argument("--read-only", action="store_true", help="Only subscribe to rt/lowstate and print received joint states; do not publish LowCmd.")
    parser.add_argument("--lowstate-timeout", type=float, default=5.0, help="Seconds to wait for the first rt/lowstate sample.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain id. Must match deploy/robot domain; default is 0.")
    args = parser.parse_args()

    if not args.i_understand_suspended and not args.read_only:
        print("Refusing to run: this diagnostic must be run with the H2 safely suspended or mechanically supported.")
        print("Re-run with --i-understand-suspended after confirming the robot cannot fall or walk.")
        return 2
    if args.num_motors <= 0 or args.num_motors > HG_LOW_CMD_MOTORS:
        raise SystemExit(f"--num-motors must be in [1, {HG_LOW_CMD_MOTORS}], got {args.num_motors}")

    indices = list(args.indices)
    if args.include_waist:
        indices.extend([13, 14])
    indices = sorted(dict.fromkeys(indices))
    for idx in indices:
        if idx < 0 or idx >= args.num_motors:
            raise SystemExit(f"Test index {idx} is outside active motor range 0..{args.num_motors - 1}")

    watch_indices = sorted(set([4, 5, 10, 11, 13, 14]).intersection(range(args.num_motors)) | set(indices))
    log_path = args.log
    if log_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path("/tmp/h2_pr_mapping_test") / f"h2_pr_mapping_{stamp}.csv"

    stop_event = threading.Event()

    def handle_signal(_signum, _frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print("\n=== H2 PR Mapping Diagnostic ===")
    if args.read_only:
        print("Read-only mode: subscribing to rt/lowstate only. No LowCmd packets will be sent.")
    else:
        print("Safety: robot must be suspended/supported. This script sends real LowCmd packets.")
    print(f"Mode: PR ({MODE_PR}); active motors: {args.num_motors}; step: {args.delta:+.4f} rad")
    print("Watch slots:")
    for idx in watch_indices:
        print(f"  {idx:2d}: {WATCH_NAMES.get(idx, 'unnamed slot')}")
    print("")

    if args.network_interface:
        ChannelFactoryInitialize(args.domain_id, args.network_interface)
    else:
        ChannelFactoryInitialize(args.domain_id)

    state_buffer = LowStateBuffer()
    publisher = None
    if not args.read_only:
        maybe_release_motion_mode(args.skip_release_mode)
        publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        publisher.Init()
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(state_buffer.callback, 10)
    crc = CRC()

    print("[INFO] Waiting for rt/lowstate...")
    try:
        first_state = state_buffer.wait(timeout_s=args.lowstate_timeout)
    except TimeoutError:
        print(f"[ERROR] Timed out after {args.lowstate_timeout:.1f}s waiting for rt/lowstate.")
        print("")
        print("Checklist:")
        print("  1. Confirm the network interface name is the one connected to the robot, e.g. `ip -4 addr`.")
        print("  2. Try SDK autodetect by omitting the interface argument.")
        print("  3. If deploy used GEAR_SONIC_DOMAIN_ID, pass the same value with `--domain-id <id>`.")
        print("  4. Make sure no other DDS config env var is forcing a different interface:")
        print("       echo $CYCLONEDDS_URI")
        print("  5. Use this script's read-only mode first; the Unitree Python read_lowstate.py in this repo is GO-typed, not H2/HG-typed.")
        return 3
    mode_machine = int(first_state.mode_machine)
    base_q, _, _ = read_q_dq_tau(first_state, args.num_motors)
    print(f"[INFO] Received lowstate. mode_pr={int(first_state.mode_pr)}, mode_machine={mode_machine}")
    print("[INFO] Captured baseline q for tested/watch slots:")
    for idx in watch_indices:
        print(f"  q[{idx:2d}] = {base_q[idx]:+.6f}  ({WATCH_NAMES.get(idx, 'unnamed slot')})")

    if args.read_only:
        print("")
        print("[INFO] Streaming lowstate watch slots for 3 seconds:")
        start = time.monotonic()
        while time.monotonic() - start < 3.0:
            state = state_buffer.get()
            if state is not None:
                q, dq, tau = read_q_dq_tau(state, args.num_motors)
                values = " ".join(f"{idx}:{q[idx]:+.4f}" for idx in watch_indices)
                print(f"  mode_pr={int(state.mode_pr)} mode_machine={int(state.mode_machine)} {values}")
            time.sleep(0.5)
        return 0

    samples: list[Sample] = []
    target_q = base_q[:]
    print(f"[INFO] Holding baseline for {args.warmup_duration:.2f}s...")
    publish_phase(
        publisher=publisher,
        state_buffer=state_buffer,
        crc=crc,
        base_q=base_q,
        target_q=target_q,
        mode_machine=mode_machine,
        num_motors=args.num_motors,
        active_indices=set(indices),
        kp=args.kp,
        kd=args.kd,
        hold_kp=args.hold_kp,
        hold_kd=args.hold_kd,
        duration_s=args.warmup_duration,
        rate_hz=args.rate_hz,
        phase="baseline",
        commanded_index=-1,
        samples=samples,
        stop_event=stop_event,
    )

    results: list[tuple[int, float, list[float]]] = []
    step_values = [args.delta]
    if args.negative:
        step_values.append(-args.delta)

    for idx in indices:
        for delta in step_values:
            if stop_event.is_set():
                break
            print("")
            print(f"[STEP] About to command slot {idx} by {delta:+.4f} rad.")
            print(f"       {WATCH_NAMES.get(idx, 'unnamed slot')}")
            print("       Visually note whether the physical motion is pitch (toe/front-back) or roll (side tilt).")
            if not args.no_prompt:
                input("       Press Enter to run this step, or Ctrl+C to stop...")

            phase_base = f"pre_{idx}_{delta:+.3f}"
            publish_phase(
                publisher=publisher,
                state_buffer=state_buffer,
                crc=crc,
                base_q=base_q,
                target_q=base_q[:],
                mode_machine=mode_machine,
                num_motors=args.num_motors,
                active_indices={idx},
                kp=args.kp,
                kd=args.kd,
                hold_kp=args.hold_kp,
                hold_kd=args.hold_kd,
                duration_s=args.settle_duration,
                rate_hz=args.rate_hz,
                phase=phase_base,
                commanded_index=idx,
                samples=samples,
                stop_event=stop_event,
            )
            before = mean_tail(samples, phase_base, idx, args.tail_s)

            target_q = base_q[:]
            target_q[idx] += delta
            phase_step = f"step_{idx}_{delta:+.3f}"
            publish_phase(
                publisher=publisher,
                state_buffer=state_buffer,
                crc=crc,
                base_q=base_q,
                target_q=target_q,
                mode_machine=mode_machine,
                num_motors=args.num_motors,
                active_indices={idx},
                kp=args.kp,
                kd=args.kd,
                hold_kp=args.hold_kp,
                hold_kd=args.hold_kd,
                duration_s=args.step_duration,
                rate_hz=args.rate_hz,
                phase=phase_step,
                commanded_index=idx,
                samples=samples,
                stop_event=stop_event,
            )
            after = mean_tail(samples, phase_step, idx, args.tail_s)
            if before is not None and after is not None:
                response = [after[i] - before[i] for i in range(args.num_motors)]
                results.append((idx, delta, response))
                print("[RESULT] LowState q response over tail window:")
                ranked = sorted(((abs(response[j]), response[j], j) for j in watch_indices), reverse=True)
                for _, signed, j in ranked:
                    print(f"  dq_meas[{j:2d}] = {signed:+.6f} rad  ({WATCH_NAMES.get(j, 'unnamed slot')})")
            else:
                print("[WARN] Not enough samples to summarize this step.")

            print(f"[INFO] Returning slot {idx} to baseline for {args.settle_duration:.2f}s...")
            publish_phase(
                publisher=publisher,
                state_buffer=state_buffer,
                crc=crc,
                base_q=base_q,
                target_q=base_q[:],
                mode_machine=mode_machine,
                num_motors=args.num_motors,
                active_indices={idx},
                kp=args.kp,
                kd=args.kd,
                hold_kp=args.hold_kp,
                hold_kd=args.hold_kd,
                duration_s=args.settle_duration,
                rate_hz=args.rate_hz,
                phase=f"post_{idx}_{delta:+.3f}",
                commanded_index=idx,
                samples=samples,
                stop_event=stop_event,
            )

    print("")
    print("[INFO] Holding captured baseline briefly before exit...")
    publish_phase(
        publisher=publisher,
        state_buffer=state_buffer,
        crc=crc,
        base_q=base_q,
        target_q=base_q[:],
        mode_machine=mode_machine,
        num_motors=args.num_motors,
        active_indices=set(),
        kp=args.kp,
        kd=args.kd,
        hold_kp=args.hold_kp,
        hold_kd=args.hold_kd,
        duration_s=0.5,
        rate_hz=args.rate_hz,
        phase="final_hold",
        commanded_index=-1,
        samples=samples,
        stop_event=stop_event,
    )

    write_log(log_path, samples, watch_indices)
    print(f"[INFO] Wrote CSV log: {log_path}")

    if results:
        print("")
        print("=== Summary ===")
        for idx, delta, response in results:
            ranked = sorted(((abs(response[j]), response[j], j) for j in watch_indices), reverse=True)
            best_abs, best_signed, best_idx = ranked[0]
            print(
                f"Command slot {idx:2d} {delta:+.4f} rad -> largest lowstate response "
                f"slot {best_idx:2d}: {best_signed:+.6f} rad"
            )
        print("")
        print("Interpretation aid:")
        print("  If physical slot 4 is ankle pitch and slot 5 is ankle roll, swap H2 ankle DDS mapping in deploy.")
        print("  If physical slot 4 is ankle roll and slot 5 is ankle pitch, current H2 ankle order matches the code.")
        print("  Repeat the same logic for slots 10/11 on the right ankle.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
