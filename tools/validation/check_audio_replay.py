#!/usr/bin/env python3
"""Exercise motion/audio replay against loopback mocks; no robot or sound output."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import wave

import zmq
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_api.msg.dds_ import Request_, Response_, ResponseHeader_, ResponseStatus_

from interface.commands.protocol import g1_reference
from interface.paths import ROOT
from interface.motion_replay.playback import ReplayBundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', type=int, default=219, help='unused loopback DDS domain, 0–230')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/validation/audio_replay.json')
    args = parser.parse_args()
    if not 0 <= args.domain <= 230:
        parser.error('DDS domain must be between 0 and 230')
    binary = ROOT / 'build/runtime/bin/unitree_audio_player'
    if not binary.is_file():
        parser.error('Build the native audio player first: bash scripts/build_controller.sh')
    bundle = ROOT / 'motions/example_text2motion_replay/echo_g_en'
    assets = ReplayBundle.load(bundle)
    with wave.open(str(assets.audio), 'rb') as audio:
        expected = audio.readframes(audio.getnframes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ChannelFactoryInitialize(args.domain, 'lo')
    writer = ChannelPublisher('rt/api/voice/response', Response_)
    writer.Init()
    requests = []

    def reply(request):
        api = request.header.identity.api_id
        requests.append((api, request.parameter, bytes(request.binary), time.time_ns() / 1e6))
        writer.Write(Response_(ResponseHeader_(request.header.identity, ResponseStatus_(0)),
                               '{"volume":50}' if api == 1005 else '{}', []))

    reader = ChannelSubscriber('rt/api/voice/request', Request_)
    reader.Init(reply, 32)
    command = shlex.join([str(binary), '--interface', 'lo', '--domain', str(args.domain),
                          '--play', '--wait-for-go'])
    cases = []
    try:
        for interrupt in (False, True):
            name = 'interrupted' if interrupt else 'complete'
            request_start = len(requests)
            with tempfile.TemporaryDirectory(prefix='sonic-audio-check-') as temporary:
                socket = zmq.Context.instance().socket(zmq.PULL)
                socket.setsockopt(zmq.LINGER, 0)
                endpoint = f'ipc://{temporary}/body'
                socket.bind(endpoint)
                log_path = args.output.with_name(f'{args.output.stem}_{name}.log')
                with log_path.open('w') as log:
                    process = subprocess.Popen(
                        [sys.executable, '-m', 'interface.motion_replay', str(bundle), '--endpoint', endpoint,
                         '--audio-ready', '--audio-command', command], stdout=log, stderr=log,
                        start_new_session=True)
                    messages, interrupted = [], False
                    try:
                        deadline = time.monotonic() + 20
                        while time.monotonic() < deadline:
                            if socket.poll(20):
                                messages.append(socket.recv_json())
                            if process.poll() is not None:
                                while socket.poll(100):
                                    messages.append(socket.recv_json())
                                break
                            if (interrupt and not interrupted and len(messages) >= 20
                                    and any(r[0] == 1003 for r in requests[request_start:])):
                                process.send_signal(signal.SIGINT)
                                interrupted = True
                        assert process.poll() == 0, f'{name}: exit={process.poll()}; see {log_path}'
                        motion = [m for m in messages if m['kind'] == 'g1']
                        assert len(motion) >= 20
                        for message in motion:
                            g1_reference(**message['data'])
                        chunks = [r for r in requests[request_start:] if r[0] == 1003]
                        payload = b''.join(r[2] for r in chunks)
                        assert payload and payload == expected[:len(payload)]
                        assert len(payload) < len(expected) if interrupt else payload == expected
                        assert motion[0]['sent_at_ms'] <= chunks[0][3]
                        stops = [r for r in requests[request_start:] if r[0] == 1004]
                        assert len(stops) == 1
                        assert json.loads(stops[0][1])['app_name'] == json.loads(chunks[0][1])['app_name']
                        assert messages[-1]['kind'] == 'stand'
                        cases.append({'case': name, 'g1_messages': len(motion), 'audio_chunks': len(chunks),
                                      'pcm_bytes': len(payload), 'pcm_identical': True,
                                      'motion_before_audio': True, 'stop_matches_app': True,
                                      'final_command': 'stand'})
                    finally:
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGINT)
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait(timeout=5)
                        socket.close()
    finally:
        reader.Close()
        writer.Close()
    report = {'scope': 'Loopback mock DDS voice service and private IPC body receiver; '
                       'no hardware, sound output or policy inference', 'cases': cases}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
