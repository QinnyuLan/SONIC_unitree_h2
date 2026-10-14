#!/usr/bin/env python3
"""Automatic timing smoke test, exclusively on loopback DDS domain 188.

Runs the current H2 release policy, simulator, viewer (optional), and passive
recorder. This never accepts a robot interface or connects to a real robot.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import pty
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from h2_squat_compare import launch_args, model_reference_hashes, MOTION

DOMAIN = 188


def load_motor_response_config(path):
    """Read only selected actuator parameters, never arbitrary simulator overrides."""
    from gear_sonic.utils.mujoco_sim.motor_response import H2_MOTOR_GROUP_TRANSMISSIONS, MotorResponse
    import numpy as np

    raw = Path(path).read_bytes()
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise ValueError("motor-response JSON must be an object")
    config = document.get('config', document)
    groups = {name: [i] for i, name in enumerate(H2_MOTOR_GROUP_TRANSMISSIONS)}
    MotorResponse(np.ones(len(groups)), groups, config)
    return config, dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                       config=config)


def sim_worker(args):
    from gear_sonic.utils.mujoco_sim.base_sim import BaseSimulator
    from gear_sonic.utils.mujoco_sim.configs import BaseConfig

    version = 'h2_native_unitree_ros_loop' + ('_ankle_sysid' if args.variant == 'identified' else '')
    config = BaseConfig(wbc_version=version, interface='lo', enable_onscreen=args.onscreen).load_wbc_yaml()
    config.update(DOMAIN_ID=DOMAIN, USE_JOYSTICK=False, PRINT_SCENE_INFORMATION=False,
                  DIAGNOSTICS_DIR=str(args.output/'sim'), DIAGNOSTICS_HZ=200)
    if args.motor_response_config:
        config['H2_MOTOR_RESPONSE'], _ = load_motor_response_config(args.motor_response_config)
    sim = BaseSimulator(config, onscreen=args.onscreen)
    stopped = threading.Event()

    def release_band():
        while not stopped.wait(.05):
            if (args.output/'release_band').exists():
                sim.sim_env.elastic_band.enable = False
                print('[Smoke] ElasticBand enable: False', flush=True)
                return

    thread = threading.Thread(target=release_band, daemon=True)
    thread.start()
    try:
        sim.start()
    finally:
        stopped.set()
        thread.join(timeout=1)


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', required=True, choices=('baseline', 'identified'))
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--onscreen', action='store_true')
    p.add_argument('--repeats', type=int, choices=(1, 3), default=1)
    p.add_argument('--motor-response-config', type=Path,
                   help='Opt-in candidate motor-response JSON; no general simulator overrides')
    p.add_argument('--sim-worker', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    args.output = args.output.resolve()
    motor_response_fingerprint = None
    if args.motor_response_config:
        args.motor_response_config = args.motor_response_config.resolve()
        _, motor_response_fingerprint = load_motor_response_config(args.motor_response_config)
    if args.sim_worker:
        return sim_worker(args)
    args.output.mkdir(parents=True, exist_ok=False)
    fingerprints = model_reference_hashes()
    launch = SimpleNamespace(command='deploy', target='sim', interface='lo',
                             variant=args.variant, run='timing_smoke')
    _, env, deploy_cmd = launch_args(launch)
    deploy_cmd[deploy_cmd.index('--logs-dir') + 1] = str(args.output/'deploy')
    env.update(GEAR_SONIC_DOMAIN_ID=str(DOMAIN), GEAR_SONIC_SKIP_MOTION_SWITCHER='1',
               GEAR_SONIC_AUTO_START_AFTER_SEC='8', GEAR_SONIC_AUTO_PLAY_AFTER_SEC='14',
               GEAR_SONIC_AUTO_STOP_AFTER_SEC=str(27.5 + 14 * (args.repeats - 1)), PYTHONUNBUFFERED='1')
    # Explicit CLI interface/domain overrides are used in every child.
    env.pop('CYCLONEDDS_URI', None)
    sim_cmd = [sys.executable, str(Path(__file__).resolve()), '--sim-worker',
               '--variant', args.variant, '--output', str(args.output)]
    if args.onscreen:
        sim_cmd.append('--onscreen')
    if args.motor_response_config:
        sim_cmd += ['--motor-response-config', str(args.motor_response_config)]
    recorder_cmd = [sys.executable, str(ROOT/'scripts/record_h2_squat_dds.py'),
                    '--target', 'sim', '--interface', 'lo', '--domain', str(DOMAIN),
                    '--output', str(args.output/'dds')]
    processes, streams = [], []
    master, slave = pty.openpty()

    for name, cmd in [('deploy', deploy_cmd), ('sim', sim_cmd), ('record', recorder_cmd)]:
        metadata = dict(target='sim', command=cmd, variant=args.variant,
                        reference_type='h2', motion=MOTION, files=fingerprints,
                        gains='all five scaling arguments explicitly 1.0',
                        automatic_repeats=args.repeats,
                        motor_response_candidate=motor_response_fingerprint,
                        overrides={k: v for k, v in env.items() if k.startswith('GEAR_SONIC_')})
        (args.output/(name+'_launch.json')).write_text(json.dumps(metadata, indent=2)+'\n')

    def start(name, cmd, stdin=None):
        stream = (args.output/(name+'.log')).open('x')
        streams.append(stream)
        process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=stdin,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(process)
        return process

    try:
        # Finish heavyweight model loading before physics starts. A loaded policy
        # safely waits for LowState; this avoids GPU initialization distorting time.
        deploy = start('deploy', deploy_cmd, slave)
        os.write(master, b'y\n')
        deadline = time.monotonic() + 150
        while 'Deploy controller ready' not in (args.output/'deploy.log').read_text():
            if deploy.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('policy did not initialize; see deploy.log')
            time.sleep(.1)
        ready_at = time.monotonic()
        recorder = start('recorder', recorder_cmd)
        while 'Read-only recorder ready' not in (args.output/'recorder.log').read_text():
            if recorder.poll() is not None or time.monotonic() - ready_at > 5:
                raise RuntimeError('recorder did not start promptly; see recorder.log')
            time.sleep(.1)
        sim = start('sim', sim_cmd)
        while '[Realtime] Absolute deadlines' not in (args.output/'sim.log').read_text():
            if sim.poll() is not None or time.monotonic() - ready_at > 5:
                raise RuntimeError('simulator did not start; see sim.log')
            time.sleep(.1)
        next_repeat = 1
        released = False
        while deploy.poll() is None:
            if sim.poll() is not None or recorder.poll() is not None:
                raise RuntimeError('simulator or recorder stopped early; see component logs')
            if time.monotonic() > deadline:
                raise RuntimeError('smoke test timeout')
            if 'Warning: Robot has fallen' in (args.output/'sim.log').read_text():
                raise RuntimeError('robot fell; stop instead of collecting reset-contaminated clips')
            elapsed = time.monotonic() - ready_at
            if not released and elapsed > 9:
                if 'Init Done' not in (args.output/'deploy.log').read_text():
                    raise RuntimeError('policy ready pose did not complete before band release')
                (args.output/'release_band').touch(exist_ok=True)
                released = True
            if next_repeat < args.repeats and elapsed >= 14 + 14 * next_repeat:
                os.write(master, b't')
                next_repeat += 1
                print(f'[{args.variant}] Start playback {next_repeat}/{args.repeats}', flush=True)
            time.sleep(.1)
        if deploy.returncode:
            raise RuntimeError(f'deploy exited {deploy.returncode}; see deploy.log')
    finally:
        # Stop the command producer first; do not leave it running after physics.
        for process in processes:
            stop(process)
        for stream in streams:
            stream.close()
        os.close(master)
        os.close(slave)

    metadata = json.loads((args.output/'sim/metadata.json').read_text())
    dds_metadata = json.loads((args.output/'dds/metadata.json').read_text())
    with (args.output/'deploy/motion_playing.csv').open() as f:
        playing = list(csv.DictReader(f))
    print(json.dumps(metadata['realtime'], indent=2))
    if metadata['status'] != 'completed' or not metadata['realtime']['valid']:
        raise RuntimeError('invalid simulation timing or incomplete log')
    if dds_metadata['status'] != 'completed':
        raise RuntimeError('incomplete DDS log')
    # Verify playback with the existing CSV layout, not just process exit status.
    active = [row for row in playing if float(row['playing_0']) > .5]
    print(f'Playback active rows: {len(active)}; logs: {args.output}')
    starts = sum(float(row['playing_0']) > .5 and (i == 0 or float(playing[i-1]['playing_0']) <= .5)
                 for i, row in enumerate(playing))
    if len(active) < 400 * args.repeats or starts != args.repeats:
        raise RuntimeError(f'expected {args.repeats} complete reference clips; got {starts} starts')
    if model_reference_hashes() != fingerprints:
        raise RuntimeError('model or reference files changed during collection')
    if args.motor_response_config:
        _, final_response_fingerprint = load_motor_response_config(args.motor_response_config)
        if final_response_fingerprint != motor_response_fingerprint:
            raise RuntimeError('motor-response candidate file changed during collection')


if __name__ == '__main__':
    main()
