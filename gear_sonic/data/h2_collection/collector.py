"""Read-only DDS/ZMQ/RTP collection. No publishers, RPC calls or motor commands."""
import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import shutil
import signal
import socket
import threading
import time

import cv2
import msgpack
import numpy as np
import zmq

from .core import Sample, RecordingEvents, decode_dds, unpack_wire, scalar, problems, make_row, TOPICS
from .writer import EpisodeWriter

ROOT = Path(__file__).resolve().parents[3]


class Hub:
    def __init__(self):
        self.lock = threading.RLock()
        self.samples = {}
        self.raw_latest = {}
        self.counts = Counter()
        self.publishers = {}
        self.errors = []
        self.episode_errors = Counter()
        self.events = deque()
        self.record_events = RecordingEvents()
        self.writer = None
        self.config = None
        self.stopping = threading.Event()

    def put(self, name, data, source_ns=0, publisher=0, raw=None, times=None):
        mono, wall = times or (time.monotonic_ns(), time.time_ns())
        with self.lock:
            self.counts[name] += 1
            sample = Sample(data, mono, wall, int(source_ns), self.counts[name], int(publisher))
            self.samples[name] = sample
            if raw is not None:
                event = dict(topic=name, arrival_monotonic_ns=mono, arrival_wall_ns=wall,
                             source_timestamp_ns=int(source_ns), sequence=sample.sequence, wire=raw)
                self.raw_latest[name] = event
                if self.writer:
                    if self.episode_errors:
                        self.counts['writer.raw_not_queued'] += 1
                    else:
                        try:
                            self.writer.put('raw', event)
                        except Exception as exc:
                            self.episode_errors[str(exc)] += 1
                            self.counts['writer.raw_not_queued'] += 1
            if name == 'manager_state':
                for event in self.record_events.update(data):
                    if len(self.events) < 128:
                        self.events.append((event, mono))
                    else:
                        self.errors.append('record event queue overflow')

    def snapshot(self):
        with self.lock:
            return dict(self.samples)

    def detach_writer(self):
        with self.lock:
            self.writer = None
            errors = dict(self.episode_errors)
            self.episode_errors.clear()
            return errors


class DDSInput:
    def __init__(self, hub, interface, domain_id):
        from cyclonedds.core import Listener
        from cyclonedds.domain import Domain, DomainParticipant
        from cyclonedds.qos import Qos, Policy
        from cyclonedds.topic import Topic
        from unitree_sdk2py.core.channel_config import ChannelConfigHasInterface
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_, IMUState_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
        from unitree_sdk2py.utils.crc import CRC
        from scripts.record_h2_squat_dds import create_reader
        self.domain = Domain(domain_id, ChannelConfigHasInterface.replace('$__IF_NAME__$', interface))
        self.participant = DomainParticipant(domain_id)
        self.topics, self.readers = [], []
        crc = CRC()

        def receive(name, reader):
            if reader is None or hub.stopping.is_set():
                return
            try:
                for msg in reader.take(1024):
                    info = getattr(msg, 'sample_info', None)
                    if not getattr(info, 'valid_data', True):
                        continue
                    times = (time.monotonic_ns(), time.time_ns())
                    data = decode_dds(name, msg)
                    if name in ('lowstate', 'lowcmd'):
                        data['crc_ok'] = crc.Crc(msg) == msg.crc
                    hub.put(name, data, getattr(info, 'source_timestamp', 0),
                            getattr(info, 'publication_handle', 0), times=times)
            except Exception as exc:
                with hub.lock:
                    hub.errors.append(f'DDS {name}: {exc}')

        def matched(name, status):
            with hub.lock:
                hub.publishers[name] = int(status.current_count)

        def lost(name, status):
            with hub.lock:
                hub.counts[f'{name}.lost_or_rejected'] += int(status.total_count_change)
                if hub.writer:
                    hub.errors.append(f'DDS {name} sample lost/rejected')

        for name, topic, cls in (
            ('lowstate', 'rt/lowstate', LowState_), ('lowcmd', 'rt/lowcmd', LowCmd_),
            ('hand_state', 'rt/inspire/state', MotorStates_), ('hand_cmd', 'rt/inspire/cmd', MotorCmds_),
            ('secondary_imu', 'rt/secondary_imu', IMUState_),
        ):
            t = Topic(self.participant, topic, cls)
            self.topics.append(t)
            listener = Listener(on_data_available=lambda r, n=name: receive(n, r),
                                on_subscription_matched=lambda r, s, n=name: matched(n, s),
                                on_sample_lost=lambda r, s, n=name: lost(n, s),
                                on_sample_rejected=lambda r, s, n=name: lost(n, s))
            self.readers.append(create_reader(self.participant, t,
                Qos(Policy.History.KeepLast(256), Policy.Reliability.BestEffort), listener))

    def close(self):
        for reader in self.readers:
            reader.set_listener(None)
        self.readers.clear()


def zmq_input(hub, teleop_address, state_address):
    context = zmq.Context()
    sockets = []
    try:
        poller = zmq.Poller()
        for address, topics in ((teleop_address, TOPICS), (state_address, ('g1_debug', 'robot_config'))):
            sub = context.socket(zmq.SUB)
            sub.setsockopt(zmq.LINGER, 0)
            sub.setsockopt(zmq.RCVHWM, 10000)  # Do NOT conflate one-frame record button edges.
            for topic in topics:
                sub.setsockopt_string(zmq.SUBSCRIBE, topic)
            sub.connect(address)
            poller.register(sub, zmq.POLLIN)
            sockets.append(sub)
        while not hub.stopping.is_set():
            for sub, _ in poller.poll(100):
                # Bounded draining keeps both sockets responsive under load.
                for _ in range(256):
                    try:
                        raw = sub.recv(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    times = (time.monotonic_ns(), time.time_ns())
                    if raw.startswith(b'robot_config'):
                        data = msgpack.unpackb(raw[len(b'robot_config'):], raw=False)
                        with hub.lock:
                            if hub.config is not None and hub.config != data and hub.writer:
                                hub.errors.append('robot_config changed during episode')
                            hub.config = data
                        hub.put('robot_config', data, raw=raw, times=times)
                    elif raw.startswith(b'g1_debug'):
                        data = msgpack.unpackb(raw[len(b'g1_debug'):], raw=False)
                        hub.put('g1_debug', data, raw=raw, times=times)
                    else:
                        name, data = unpack_wire(raw)
                        source = int(float(scalar(data, 'timestamp_monotonic')) * 1e9)
                        hub.put(name, data, source_ns=source, raw=raw, times=times)
    except Exception as exc:
        with hub.lock:
            hub.errors.append(f'ZMQ input stopped: {exc}')
        hub.stopping.set()
    finally:
        for sub in sockets:
            sub.close()
        context.term()


def camera_input(hub, port, width, height):
    # Decoding is the only reader of this UDP port; the collector also provides preview.
    pipeline = (f'udpsrc port={port} buffer-size=2097152 '
        'caps="application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000" ! '
        'rtpjitterbuffer latency=50 drop-on-latency=true ! rtph264depay ! h264parse ! avdec_h264 ! '
        'videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false')
    capture = None
    try:
        capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not capture.isOpened():
            raise RuntimeError('cannot open GStreamer RGB stream')
        while not hub.stopping.is_set():
            ok, frame = capture.read()
            times = (time.monotonic_ns(), time.time_ns())
            if not ok:
                raise RuntimeError('RGB decode stopped')
            native_size = [int(frame.shape[1]), int(frame.shape[0])]
            if native_size != [width, height]:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            hub.put('camera', dict(image=frame, native_size=native_size), times=times)
    except Exception as exc:
        with hub.lock:
            hub.errors.append(f'camera: {exc}')
        hub.stopping.set()
    finally:
        if capture is not None:
            capture.release()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(args):
    model = args.model.resolve()
    result = dict(task=args.task, robot='h2_native', hand_backend='inspire',
                  interface=args.interface, dds_domain=args.domain, model_directory=str(model),
                  model_files={name: dict(path=str(model/name), sha256=file_hash(model/name))
                               for name in ('model_encoder_full.onnx', 'model_decoder.onnx')},
                  config_files={str(p.resolve()): p.read_text() for p in args.config},
                  camera=dict(port=args.rgb_port, source='H2 PC1 stereo_patch_pc1 original RGB',
                              calibration=json.loads(args.calibration.read_text()) if args.calibration else None,
                              calibration_status='provided_unverified' if args.calibration else 'not_provided'),
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'config'})
    manifest = model/'deployment_manifest.json'
    if manifest.exists():
        result['deployment_manifest'] = json.loads(manifest.read_text())
    return result


def config_problem(config, model):
    if not config:
        return 'waiting_robot_config'
    if config.get('robot_profile') != 'h2_native' or config.get('command_motor_count') != 31:
        return 'wrong_robot_profile'
    for key, name in (('model_path', 'model_decoder.onnx'), ('encoder_file', 'model_encoder_full.onnx')):
        path = Path(str(config.get(key, '')))
        if not path.is_absolute():
            path = ROOT/'gear_sonic_deploy'/path
        if path.resolve() != (model/name).resolve():
            return f'running_{key}_does_not_match_configured_model'
    if not config.get('h2_vr_head_control'):
        return 'head_control_disabled'
    return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--config', type=Path, action='append', default=[])
    p.add_argument('--interface', default='eno1')
    p.add_argument('--domain', type=int, default=0)
    p.add_argument('--fps', type=int, default=50)
    p.add_argument('--width', type=int, default=1920)
    p.add_argument('--height', type=int, default=1080)
    p.add_argument('--rgb-port', type=int, default=5002)
    p.add_argument('--teleop-address', default='tcp://127.0.0.1:5556')
    p.add_argument('--state-address', default='tcp://127.0.0.1:5557')
    p.add_argument('--min-free-gb', type=float, default=2)
    p.add_argument('--calibration', type=Path)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--check', action='store_true', help='offline dependency/config check only; no DDS/ZMQ/camera connection')
    args = p.parse_args(argv)
    if not args.task.strip() or args.task == 'CHANGE_ME':
        p.error('请在 h2_data_collection.env 填写本次任务 TASK_PROMPT')
    if args.fps < 1 or args.fps > 100 or min(args.width, args.height) < 16 or args.width % 2 or args.height % 2:
        p.error('invalid fps/video size (H264 requires even dimensions)')
    if not 0 <= args.domain <= 232 or args.min_free_gb < .5:
        p.error('invalid DDS domain or min-free-gb (<0.5)')
    metadata = provenance(args)
    if 'GStreamer:                   YES' not in cv2.getBuildInformation():
        # Different OpenCV builds use different padding.
        if not any('GStreamer:' in line and 'YES' in line for line in cv2.getBuildInformation().splitlines()):
            raise RuntimeError('OpenCV needs GStreamer; use .venv_data_collection configured from .venv_teleop')
    if args.check:
        import av
        import pyarrow
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        av.codec.Codec('libx264', 'w')
        print(json.dumps(dict(check='offline_ok', model=metadata['model_directory'], task=args.task,
                              pyarrow=pyarrow.__version__, camera='not_opened', dds='not_connected'), ensure_ascii=False))
        return 0
    # Fail before subscribing if another local viewer already owns the video port.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(('', args.rgb_port))
    args.root.mkdir(parents=True, exist_ok=True)
    # Exclusive session lock prevents two collectors from interleaving one dataset directory.
    import fcntl
    lock_file = (args.root/'.collector.lock').open('a')
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for partial in args.root.glob('*.partial'):
        print(f'[NOTICE] Unfinished episode excluded from training: {partial}', flush=True)
    hub = Hub()
    signal.signal(signal.SIGINT, lambda *_: hub.stopping.set())
    signal.signal(signal.SIGTERM, lambda *_: hub.stopping.set())
    dds = None
    writer = None
    outcome = 'unlabeled'
    frame_index = 0
    next_tick = time.monotonic()
    last_status = 0
    last_preview = 0
    last_frame_ns = None
    first_frame_ns = None
    free_gb = shutil.disk_usage(args.root).free/1e9
    initial_counts = {}

    def finish(status, reason='', event_ns=None):
        nonlocal writer, frame_index, last_frame_ns
        if writer is None:
            return
        print(f'[SAVING] status={status} frames={frame_index} reason={reason}', flush=True)
        writer.metadata['end_event_monotonic_ns'] = event_ns
        writer.metadata['end_receive_monotonic_ns'] = time.monotonic_ns()
        with hub.lock:
            episode_errors = hub.detach_writer()  # Detach producers before sentinel.
            quality = dict(received_delta={k: v-initial_counts.get(k, 0) for k, v in hub.counts.items()},
                           publishers=dict(hub.publishers), errors=list(hub.errors),
                           episode_writer_errors=episode_errors,
                           free_gb=free_gb, last_valid_frame_ns=last_frame_ns)
        path = writer.finish(status, outcome, reason, quality)
        print(f'[{writer.metadata["status"].upper()}] {path} outcome={outcome}', flush=True)
        writer = None
        frame_index, last_frame_ns = 0, None

    try:
        dds = DDSInput(hub, args.interface, args.domain)
        threading.Thread(target=zmq_input, args=(hub, args.teleop_address, args.state_address), daemon=True).start()
        threading.Thread(target=camera_input, args=(hub, args.rgb_port, args.width, args.height), daemon=True).start()
        print('[IDLE] Read-only H2 recorder. Left Grip+A=start/save; Left Grip+B=discard WHILE recording.\n'
              'Preview keys: S=success, F=failure, U=unlabeled (label only); Q=quit recorder.\n'
              'Closing this program DOES NOT stop the robot.', flush=True)
        while not hub.stopping.is_set():
            samples = hub.snapshot()
            now_ns, wall_ns = time.monotonic_ns(), time.time_ns()
            errors = problems(samples, now_ns)
            with hub.lock:
                events = list(hub.events)
                hub.events.clear()
                config = hub.config
                errors.extend(hub.errors)
                errors.extend(hub.episode_errors)
                for name, count in hub.publishers.items():
                    if count > 1:
                        errors.append(f'{name}_multiple_publishers')
            error = config_problem(config, args.model)
            if error:
                errors.append(error)
            if now_ns/1e9-last_status >= 1:
                free_gb = shutil.disk_usage(args.root).free/1e9
            if free_gb < args.min_free_gb:
                errors.append('disk_space_low')
            if writer and writer.error:
                errors.append(writer.error)
            if writer and errors:
                finish('incomplete', '; '.join(dict.fromkeys(errors)))
                # A save button in this same batch must not restart an invalid episode.
                events = []
            for event, event_ns in events:
                if now_ns-event_ns > 500_000_000:
                    finish('incomplete', 'record event delayed >500ms')
                    continue
                if event == 'discard':
                    finish('discarded', 'PICO Left Grip+B', event_ns)
                elif event == 'toggle':
                    if writer:
                        finish('saved', event_ns=event_ns)
                    elif errors:
                        print('[START REFUSED] ' + '; '.join(errors), flush=True)
                    else:
                        outcome = 'unlabeled'
                        with hub.lock:
                            writer = EpisodeWriter(args.root, {**metadata, 'robot_config': config,
                                'camera_native_size': samples['camera'].data['native_size'],
                                'start_event_monotonic_ns': event_ns}, args.fps, args.width, args.height)
                            initial_counts = dict(hub.counts)
                            # Seed sidecar with last pre-start messages as well as subsequent arrivals.
                            for value in hub.raw_latest.values():
                                writer.put('raw', {**value, 'pre_start_snapshot': True})
                            hub.writer = writer
                        frame_index, last_frame_ns = 0, None
                        first_frame_ns = now_ns
                        print('[REC] Recording; saved video has no overlay.', flush=True)
                else:
                    finish('incomplete', event)
                    if event == 'legacy_manager':
                        errors.append('restart_updated_pico_manager')
            if writer:
                if ((last_frame_ns and now_ns-last_frame_ns > max(2/args.fps, .05)*1e9)
                        or abs((now_ns-first_frame_ns)/1e9-frame_index/args.fps) > max(2/args.fps, .05)):
                    finish('incomplete', 'sampling deadline missed')
                else:
                    row = make_row(samples, now_ns, wall_ns, frame_index, args.fps, first_frame_ns)
                    try:
                        writer.put('frame', (row, samples['camera'].data['image']))
                    except RuntimeError as exc:
                        finish('incomplete', str(exc))
                    else:
                        frame_index += 1
                        last_frame_ns = now_ns
            if now_ns/1e9-last_status >= 1:
                state = 'REC' if writer else ('READY' if not errors else 'WAIT')
                print(f'[{state}] frames={frame_index} outcome={outcome} free={free_gb:.1f}GB '
                      + ('; '.join(dict.fromkeys(errors)) if errors else 'all required streams fresh'), flush=True)
                last_status = now_ns/1e9
            if not args.headless and now_ns/1e9-last_preview >= .1:
                camera = samples.get('camera')
                view = cv2.resize(camera.data['image'], (960, 540)) if camera else np.zeros((540, 960, 3), np.uint8)
                state = 'REC' if writer else ('READY' if not errors else 'WAIT')
                color = (30, 30, 255) if writer else ((40, 230, 40) if not errors else (0, 180, 255))
                cv2.putText(view, f'{state} | {frame_index/args.fps:.1f}s | {outcome} | {free_gb:.1f}GB',
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2)
                cv2.putText(view, 'Grip+A start/save; Grip+B discard; S/F/U label; Q quit recorder',
                            (15, 520), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
                cv2.imshow('H2 VLA collection - RGB', view)
                key = cv2.waitKey(1) & 0xff
                if ord('A') <= key <= ord('Z'):
                    key += 32
                if key == ord('q'):
                    hub.stopping.set()
                elif writer and key in (ord('s'), ord('f'), ord('u')):
                    outcome = {ord('s'): 'success', ord('f'): 'failure', ord('u'): 'unlabeled'}[key]
                    print(f'[LABEL] {outcome}', flush=True)
                last_preview = now_ns/1e9
            next_tick += 1/args.fps
            delay = next_tick-time.monotonic()
            if delay < -1/args.fps:
                next_tick = time.monotonic()
            hub.stopping.wait(max(0, delay))
    except Exception as exc:
        with hub.lock:
            hub.errors.append(str(exc))
        raise
    finally:
        hub.stopping.set()
        finish('incomplete', 'recorder interrupted/stopped before save')
        if dds:
            dds.close()
        cv2.destroyAllWindows()
        lock_file.close()
    return 0
