#!/usr/bin/env python3
"""Exercise the actual GPU runtime through private IPC sockets, without DDS/devices."""
import argparse
import os
import pty
from contextlib import ExitStack
from pathlib import Path
import subprocess
import tempfile
import sys
import time
import termios
import numpy as np
import zmq
from interface.commands import RuntimeClient, h2_reference, smpl_reference, vr_reference
from interface.commands.protocol import token_command
from interface.teleop.bridge import standing_g1

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--planner', action='store_true', help='also run the downloaded planner model')
    parser.add_argument('--start-key', choices=[']', 's'], default='s', help='exercise either runtime start key')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='sonic-runtime-') as temp, ExitStack() as stack:
        temp = Path(temp)
        endpoints = {key: f'ipc://{temp}/{key}' for key in ('body', 'head', 'hands', 'state', 'planner', 'planned')}
        log = stack.enter_context((temp / 'runtime.log').open('w+'))
        master, slave = pty.openpty()
        stack.callback(os.close, master)
        stack.callback(os.close, slave)
        terminal_before = termios.tcgetattr(slave)
        process = subprocess.Popen([str(ROOT / 'build/runtime/bin/sonic_h2_runtime'),
            '--dry-run', '--model-dir', str(ROOT / 'models/sonic_h2_released'),
            '--command-bind', endpoints['body'], '--head-bind', endpoints['head'],
            '--hands-bind', endpoints['hands'], '--state-bind', endpoints['state']],
            cwd=temp, stdin=slave, stdout=log, stderr=log, start_new_session=True)
        def stop():
            if process.poll() is not None: return
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=3)
        stack.callback(stop)
        sub = zmq.Context.instance().socket(zmq.SUB)
        sub.setsockopt(zmq.LINGER, 0); sub.setsockopt_string(zmq.SUBSCRIBE, '')
        sub.setsockopt(zmq.CONFLATE, 1); sub.connect(endpoints['state']); stack.callback(sub.close)
        def expect(predicate, send=None, timeout=4):
            deadline = time.monotonic() + timeout
            latest = None
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    log.seek(0); raise AssertionError(log.read())
                if send: send()
                if sub.poll(30):
                    latest = sub.recv_json()
                    if predicate(latest): return latest
            log.seek(0)
            raise AssertionError(f'Timed out; latest={latest}\n{log.read()[-2000:]}')
        expect(lambda s: s['phase'] == 'initializing' and not s['policy_enabled'], timeout=180)
        os.write(master, args.start_key.encode())  # An early start must be discarded.
        expect(lambda s: s['phase'] == 'ready' and not s['policy_enabled'])
        body = stack.enter_context(RuntimeClient('test_body', endpoints['body']))
        head = stack.enter_context(RuntimeClient('test_head', endpoints['head']))
        hands = stack.enter_context(RuntimeClient('test_hands', endpoints['hands']))
        reference = standing_g1()
        body.send('g1', reference)
        time.sleep(.1)
        expect(lambda s: s['phase'] == 'ready' and s['body_kind'] == 'stand')
        os.write(master, args.start_key.encode())  # No newline: test the actual terminal input.
        expect(lambda s: s['phase'] == 'running' and s['policy_enabled'] and s['body_kind'] == 'stand')
        def all_three():
            body.send('g1', reference)
            head.send('head', {'pitch': .3, 'yaw': -.2})
            hands.send('hands', {'left': [0] * 7})
        expect(lambda s: s['body_kind'] == 'g1' and s['head_active'] and s['hands_active'] and abs(s['head_target'][0] - .3) < .01, all_three)
        # A body release cannot clear independent head/hand commands.
        body.stand()
        expect(lambda s: s['body_kind'] == 'stand' and s['head_active'],
               lambda: head.send('head', {'pitch': .3, 'yaw': -.2}))
        expect(lambda s: not s['head_active'] and not s['hands_active'] and abs(s['head_target'][0]) < .01)
        for kind, data in [
            ('h2', h2_reference(**reference)),
            ('vr', vr_reference(np.zeros((3, 3)), np.tile([1, 0, 0, 0], (3, 1)), reference)),
            ('smpl', smpl_reference(np.zeros((10, 24, 3)), np.tile([1, 0, 0, 0], (10, 1)), np.zeros((10, 6)))),
            ('token', token_command(np.zeros(64))),
        ]:
            expect(lambda s: s['body_kind'] == kind, lambda: body.send(kind, data))
            expect(lambda s: s['body_kind'] == 'stand')
        # Two producers cannot silently interleave; an explicit takeover changes ownership.
        expect(lambda s: s['source'] == 'test_body', lambda: body.send('g1', reference))
        rival = stack.enter_context(RuntimeClient('rival', endpoints['body']))
        def compete(): body.send('g1', reference); rival.send('g1', reference)
        expect(lambda s: s['source'] == 'test_body' and s['rejected'] > 0, compete)
        rival.takeover = True
        expect(lambda s: s['source'] == 'rival', lambda: rival.send('g1', reference))
        body.stand()
        expect(lambda s: s['source'] == 'rival', lambda: rival.send('g1', reference))
        rival.stand()
        expect(lambda s: s['body_kind'] == 'stand')
        # Run the actual teleoperation adapter against synthetic PICO wire packets.
        from interface.teleop.transport.packets import pack_pose_message, build_head_pose_message
        tracking_endpoint = f'ipc://{temp}/tracking'
        tracking = zmq.Context.instance().socket(zmq.PUB)
        tracking.setsockopt(zmq.LINGER, 0); tracking.bind(tracking_endpoint)
        stack.callback(tracking.close)
        bridge = subprocess.Popen([sys.executable, '-m', 'interface.teleop.bridge',
            '--tracking', tracking_endpoint, '--runtime', endpoints['body'],
            '--head', endpoints['head'], '--hands', endpoints['hands'],
            '--planner-input', endpoints['planner'], '--planner-output', endpoints['planned']],
            cwd=ROOT, stdout=log, stderr=log)
        def stop_bridge():
            if bridge.poll() is None:
                bridge.terminate()
                try: bridge.wait(timeout=5)
                except subprocess.TimeoutExpired: bridge.kill(); bridge.wait()
        stack.callback(stop_bridge)
        tracking_data = {'smpl_joints': np.zeros((10, 24, 3)),
            'smpl_root_quat': np.tile([1., 0, 0, 0], (10, 1)),
            'joint_pos': np.zeros((10, 29)), 'left_hand_joints': np.zeros(7)}
        def send_tracking():
            assert bridge.poll() is None, 'teleoperation bridge exited'
            tracking.send(pack_pose_message(tracking_data))
            tracking.send(build_head_pose_message(.2, .1))
        expect(lambda s: s['source'] == 'teleop' and s['body_kind'] == 'smpl' and s['head_active'] and s['hands_active'], send_tracking)
        stop_bridge()
        expect(lambda s: s['body_kind'] == 'stand' and not s['head_active'] and not s['hands_active'])
        if args.planner:
            planner_log = stack.enter_context((temp / 'planner.log').open('w+'))
            planner = subprocess.Popen([str(ROOT / '.venv_planner/bin/python'), '-m', 'interface.planner.service',
                '--model', str(ROOT / 'models/motion_planner_g1/planner_sonic.onnx'),
                '--input-bind', endpoints['planner'], '--output-bind', endpoints['planned'],
                '--runtime', endpoints['body'], '--state', endpoints['state']], cwd=temp, stdout=planner_log, stderr=planner_log)
            def stop_planner():
                planner.terminate()
                try: planner.wait(timeout=10)
                except subprocess.TimeoutExpired: planner.kill(); planner.wait()
            stack.callback(stop_planner)
            planner_control = stack.enter_context(RuntimeClient('planner_test', endpoints['planner'], ttl_ms=500))
            def request_planning():
                if planner.poll() is not None:
                    planner_log.seek(0); raise AssertionError(planner_log.read())
                planner_control.send('planner', {'mode': 0, 'movement': [0, 0, 0], 'facing': [1, 0, 0], 'speed': 0, 'height': -1})
            expect(lambda s: s['source'] == 'planner' and s['body_kind'] == 'g1', request_planning, timeout=300)
            expect(lambda s: s['body_kind'] == 'stand', timeout=4)
        os.write(master, b'o')
        assert process.wait(timeout=5) == 0, 'runtime did not stop cleanly on o'
        assert termios.tcgetattr(slave) == terminal_before, 'runtime did not restore terminal settings'
        log.seek(0)
        output = log.read()
        assert '[STOPPED]' in output and '[READY]' in output and 'Start ignored' in output
        print('PASS: manual policy start (' + args.start_key + '), early-key discard, pre-start command discard, '
              'stop on o, terminal restoration, G1/H2/VR/SMPL/token, expiry, ownership, independent head/hands, '
              'teleoperation adapter' + (', planner inference' if args.planner else ''))


if __name__ == '__main__':
    main()
