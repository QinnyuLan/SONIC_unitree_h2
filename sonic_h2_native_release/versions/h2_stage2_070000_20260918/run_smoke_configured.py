"""Bounded loopback-only smoke test of this experimental model."""
import json
import os
from pathlib import Path
import signal
import shlex
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = HERE / "smoke_configured_g1"
RUN.mkdir(exist_ok=False)
env = {k: v for k, v in os.environ.items() if not k.startswith("GEAR_SONIC_")}
env.update(GEAR_SONIC_DOMAIN_ID="73", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
           MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
sim_env = dict(env, GEAR_SONIC_SIM_DIAGNOSTICS_DIR=str(RUN / "sim"),
               GEAR_SONIC_MUJOCO_ENABLE_ELASTIC="1")
deploy_env = dict(env, GEAR_SONIC_AUTO_START_AFTER_SEC="3",
    GEAR_SONIC_AUTO_PLAY_AFTER_SEC="7", GEAR_SONIC_AUTO_STOP_AFTER_SEC="18",
    GEAR_SONIC_SKIP_MOTION_SWITCHER="1")
sim_args = [str(ROOT / ".venv_sim/bin/python"), "gear_sonic/scripts/run_sim_loop.py",
    "--wbc-version", "h2_native_unitree_ros_loop", "--no-enable-onscreen", "--visual-style", "paper"]
printed = subprocess.check_output(
    ["bash", "scripts/run_h2_sim2sim.sh", "--print-command"],
    cwd=ROOT / "gear_sonic_deploy", text=True)
command = shlex.split(printed.strip().splitlines()[-1])
command[command.index("--logs-dir") + 1] = str(RUN / "deploy")
deploy_args = ["bash", "-c", 'source scripts/setup_env.sh >/dev/null\nexec "$@"',
    "smoke", *command]

report = dict(dds_domain=73, interface="lo", elastic_band_enabled=True, headless=True,
    full_free_motion_stability_validated=False, sim_command=sim_args, deploy_command=deploy_args)
sim = deploy = None

def stop(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()

try:
    with (RUN / "sim.log").open("x") as sim_log, (RUN / "deploy.log").open("x") as deploy_log:
        sim = subprocess.Popen(sim_args, cwd=ROOT, env=sim_env, stdout=sim_log,
            stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(3)
        assert sim.poll() is None, "simulator exited during startup"
        deploy = subprocess.Popen(deploy_args, cwd=ROOT / "gear_sonic_deploy", env=deploy_env,
            stdin=subprocess.PIPE, stdout=deploy_log, stderr=subprocess.STDOUT,
            start_new_session=True)
        deploy.stdin.write(b"y\n")
        deploy.stdin.flush()
        deadline = time.monotonic() + 120
        while deploy.poll() is None and time.monotonic() < deadline:
            assert sim.poll() is None, "simulator exited during deployment"
            time.sleep(.2)
        assert deploy.poll() is not None, "deployment timeout"
        report["deployment_exit_code"] = deploy.returncode
        assert deploy.returncode == 0, "deployment failed"
finally:
    stop(deploy)
    stop(sim)
    report["simulator_exit_code"] = sim.returncode if sim else None
    (RUN / "process_result.json").write_text(json.dumps(report, indent=2) + "\n")

log = (RUN / "deploy.log").read_text()
for required in ("Found 7 motion folders", "ONNX shapes and metadata validated",
                 "Starting policy control", "Program exiting normally"):
    assert required in log, required
assert "[ERROR]" not in log
print(json.dumps(report, indent=2))
