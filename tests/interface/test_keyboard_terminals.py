"""Real terminal/IPC tests. Runtime state and planner inference are test doubles."""
import fcntl
import json
import os
from pathlib import Path
import pty
import select
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import time

import pytest
import zmq

from interface.commands.receiver import CommandInbox

ROOT = Path(__file__).resolve().parents[2]


class Harness:
    def __init__(self, directory):
        self.endpoints = {key: f'ipc://{directory}/{key}' for key in ('body', 'state', 'planner', 'planned')}
        self.context = zmq.Context()
        self.pub, self.body = self.context.socket(zmq.PUB), self.context.socket(zmq.PULL)
        self.pub.bind(self.endpoints['state'])
        self.body.bind(self.endpoints['body'])
        self.master, self.slave = pty.openpty()
        fcntl.ioctl(self.slave, termios.TIOCSWINSZ, struct.pack('HHHH', 30, 120, 0, 0))
        self.attributes = termios.tcgetattr(self.slave)
        self.inbox = CommandInbox({'g1', 'h2', 'smpl', 'stand'})
        self.messages, self.output = [], b''
        self.publish_state = True
        self.process = None

    def start(self, argv, extra_env=None):
        env = dict(os.environ, TERM='xterm', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        env.update(extra_env or {})
        self.process = subprocess.Popen([sys.executable, *argv], cwd=ROOT, env=env,
            stdin=self.slave, stdout=self.slave, stderr=self.slave, start_new_session=True)

    def key(self, value):
        os.write(self.master, value.encode())

    def pump(self, seconds=.1, until=None):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            active = self.inbox.active()
            if self.publish_state:
                self.pub.send_json({'schema': 'sonic.h2.state.v1', 'sent_at_ms': time.time_ns() // 1_000_000,
                    'phase': 'running', 'joint_pos': [0] * 31, 'root_quat': [1, 0, 0, 0],
                    'body_kind': active['kind'] if active else 'stand', 'source': active['source'] if active else ''})
            while self.body.poll(0):
                message = self.body.recv_json()
                assert self.inbox.accept(message), self.inbox.last_error
                self.messages.append(message)
            if select.select([self.master], [], [], 0)[0]:
                self.output += os.read(self.master, 65536)
            if until is not None and until():
                return
            time.sleep(.01)
        if until is not None:
            pytest.fail(f'Terminal condition timed out; exit={self.process.poll()}\n{self.output[-4000:]!r}')

    def expect_kind(self, kind, start):
        self.pump(5, lambda: any(m['kind'] == kind for m in self.messages[start:]))

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=6)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        os.close(self.master)
        os.close(self.slave)
        self.pub.close(linger=0)
        self.body.close(linger=0)
        self.context.term()


@pytest.fixture
def terminal():
    # Unix socket paths have a small length limit; pytest's temp path can exceed it.
    with tempfile.TemporaryDirectory(prefix='h2-keys-') as directory:
        harness = Harness(directory)
        try:
            yield harness
        finally:
            harness.close()


def test_replay_terminal_selection_pause_switch_disconnect_and_exit(terminal):
    t = terminal
    t.start(['-m', 'interface.motion_replay', '--keyboard', '--endpoint', t.endpoints['body'],
             '--state', t.endpoints['state']])
    t.pump(5, lambda: b'Choose a collection' in t.output)
    t.key('1'); t.pump(.15)
    assert not t.messages
    t.key('\n'); t.expect_kind('g1', 0)
    start = len(t.messages)
    t.key(' '); t.expect_kind('stand', start)
    start = len(t.messages)
    t.pump(.2)
    assert not t.messages[start:]
    t.key(' '); t.expect_kind('g1', start)
    for collection, kind in [('2', 'g1'), ('3', 'h2'), ('4', 'smpl')]:
        start = len(t.messages)
        t.key(collection); t.expect_kind('stand', start)
        t.pump(.1)
        start = len(t.messages)
        t.key('\n'); t.expect_kind(kind, start)
    t.publish_state = False
    start = len(t.messages)
    t.expect_kind('stand', start)
    t.publish_state = True
    start = len(t.messages)
    t.pump(.2)
    assert not t.messages[start:]
    t.key('q')
    t.pump(3, lambda: t.process.poll() is not None)
    assert t.process.returncode == 0
    assert termios.tcgetattr(t.slave)[3] == t.attributes[3]


def test_planner_terminal_owns_worker_and_releases_runtime(terminal, tmp_path):
    # Patch only the inference object in both spawned interpreters. All service,
    # keyboard, transport, trajectory and shutdown code is the released code.
    (tmp_path / 'sitecustomize.py').write_text('''
import json, os
from pathlib import Path
import numpy as np
import interface.planner.motion as motion
import interface.planner.keyboard as keyboard
keyboard.ROOT = Path(os.environ['KEYBOARD_TEST_DIR'])
class Planner:
    def __init__(self, *args, **kwargs):
        (keyboard.ROOT / 'worker.pid').write_text(str(os.getpid()))
    def initialize(self, joints):
        return motion.Trajectory(np.zeros((100,29)), np.zeros((100,29)), np.tile([1,0,0,0],(100,1)), np.zeros((100,3)))
    def update(self, previous, cursor, command):
        with (keyboard.ROOT / 'requests.jsonl').open('a') as log:
            log.write(json.dumps(command)+'\\n')
        return self.initialize(None), cursor+2
    def __enter__(self): return self
    def __exit__(self, *args): pass
motion.G1Planner = Planner
''')
    t = terminal
    t.start(['-m', 'interface.planner.service', '--keyboard', '--input-bind', t.endpoints['planner'],
             '--output-bind', t.endpoints['planned'], '--runtime', t.endpoints['body'], '--state', t.endpoints['state']],
            {'PYTHONPATH': str(tmp_path) + os.pathsep + str(ROOT / 'src'), 'KEYBOARD_TEST_DIR': str(tmp_path)})
    t.pump(5, lambda: b'ready' in t.output)
    assert not t.messages
    t.key('w'); t.expect_kind('g1', 0)
    t.key('2+w'); t.pump(.3)
    requests = [json.loads(line) for line in (tmp_path / 'requests.jsonl').read_text().splitlines()]
    assert any(r['mode'] == 2 and r['speed'] == .4 and r['movement'] == [1, 0, 0] for r in requests)
    start = len(t.messages)
    t.key(' '); t.expect_kind('stand', start)
    start = len(t.messages)
    t.pump(.25)
    assert not t.messages[start:]
    t.key('w'); t.expect_kind('g1', start)
    t.publish_state = False
    start = len(t.messages)
    t.expect_kind('stand', start)
    t.publish_state = True
    t.pump(.2)
    start = len(t.messages)
    t.pump(.2)
    assert not t.messages[start:]
    t.key('w'); t.expect_kind('g1', start)
    t.key('\x1b')
    t.pump(5, lambda: t.process.poll() is not None)
    assert t.process.returncode == 0
    assert t.messages[-1]['kind'] == 'stand'
    assert not Path('/proc/' + (tmp_path / 'worker.pid').read_text()).exists()
    assert termios.tcgetattr(t.slave)[3] == t.attributes[3]


def test_planner_cancel_during_engine_startup(terminal, tmp_path):
    (tmp_path / 'sitecustomize.py').write_text('''
import os, time
from pathlib import Path
import interface.planner.motion as motion
import interface.planner.keyboard as keyboard
keyboard.ROOT = Path(os.environ['KEYBOARD_TEST_DIR'])
class Planner:
    def __init__(self, *args, **kwargs):
        (keyboard.ROOT / 'worker.pid').write_text(str(os.getpid()))
        time.sleep(60)
motion.G1Planner = Planner
''')
    t = terminal
    t.start(['-m', 'interface.planner.service', '--keyboard', '--input-bind', t.endpoints['planner'],
             '--output-bind', t.endpoints['planned'], '--runtime', t.endpoints['body'], '--state', t.endpoints['state']],
            {'PYTHONPATH': str(tmp_path) + os.pathsep + str(ROOT / 'src'), 'KEYBOARD_TEST_DIR': str(tmp_path)})
    t.pump(5, lambda: (tmp_path / 'worker.pid').is_file() and b'preparing TensorRT' in t.output)
    assert not t.messages
    t.key('\x1b')
    t.pump(6, lambda: t.process.poll() is not None)
    assert t.process.returncode == 0
    assert not Path('/proc/' + (tmp_path / 'worker.pid').read_text()).exists()
    assert termios.tcgetattr(t.slave)[3] == t.attributes[3]
