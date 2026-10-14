#!/usr/bin/env python3
"""Verify a transient user service's watchdog/restart on loopback with no serial device."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]


def run(*args, check=True):
    result = subprocess.run(args, capture_output=True, text=True, timeout=15)
    if check and result.returncode:
        raise RuntimeError(f"{args[0]} failed: {result.stderr.strip()}")
    return result


def state(unit):
    result = run("systemctl", "--user", "show", unit,
                 "--property=ActiveState,MainPID,NRestarts", check=False)
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def await_running(unit, previous_pid=None):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        current = state(unit)
        if (current.get("ActiveState") == "active" and current.get("MainPID", "0") != "0"
                and current.get("MainPID") != previous_pid):
            return current
        time.sleep(0.1)
    raise RuntimeError(f"Service did not recover: {state(unit)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path,
                        default=ROOT / "build/inspire_hands/bin/inspire_hand_service")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/validation/inspire_watchdog.json")
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    if run("systemctl", "--user", "show", "--property=Version", check=False).returncode:
        parser.error("A running systemd user manager is required; no robot connection is used")
    unit = f"sonic-inspire-test-{uuid.uuid4().hex[:12]}.service"
    report = {"binary": str(binary), "unit": unit, "network": "lo", "domain": 218}
    with tempfile.TemporaryDirectory(prefix="sonic-inspire-test-") as tmp:
        try:
            run("systemd-run", "--user", "--collect", f"--unit={unit}",
                "--property=Type=notify", "--property=NotifyAccess=main",
                "--property=Restart=always", "--property=RestartSec=300ms",
                "--property=WatchdogSec=2s", "--property=WatchdogSignal=SIGKILL",
                "--property=TimeoutStartSec=10s", "--property=TimeoutStopSec=3s",
                "--property=StartLimitIntervalSec=0", f"--working-directory={tmp}",
                str(binary), "--network", "lo", "--domain", "218",
                "--namespace", unit.removesuffix(".service").replace("-", "_"), "--hands", "both",
                "--right-port", f"{tmp}/missing-right", "--left-port", f"{tmp}/missing-left")
            initial = await_running(unit)
            time.sleep(2.5)
            healthy = state(unit)
            if healthy != initial:
                raise AssertionError(f"USB absence caused a restart: {initial} -> {healthy}")
            report["missing_usb"] = healthy
            run("systemctl", "--user", "kill", "--signal=SIGSTOP", "--kill-who=main", unit)
            recovered = await_running(unit, initial["MainPID"])
            if int(recovered["NRestarts"]) <= int(initial["NRestarts"]):
                raise AssertionError("Watchdog did not restart the stopped process")
            report["watchdog_recovery"] = recovered
            run("systemctl", "--user", "kill", "--signal=SIGKILL", "--kill-who=main", unit)
            crashed = await_running(unit, recovered["MainPID"])
            if int(crashed["NRestarts"]) <= int(recovered["NRestarts"]):
                raise AssertionError("Crashed process did not restart")
            report["crash_recovery"] = crashed
            run("systemctl", "--user", "stop", unit)
            time.sleep(0.7)
            if state(unit).get("ActiveState") == "active":
                raise AssertionError("Explicit stop unexpectedly restarted the service")
            report["explicit_stop"] = "passed"
            report["result"] = "passed"
        finally:
            run("systemctl", "--user", "stop", unit, check=False)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: missing USB, watchdog recovery, crash recovery and explicit stop; {args.output}")


if __name__ == "__main__":
    main()
