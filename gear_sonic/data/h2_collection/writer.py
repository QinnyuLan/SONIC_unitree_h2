"""Bounded, single-owner Parquet/MP4 writer. Only finalized episodes are training candidates."""
import json
import os
from pathlib import Path
import queue
import threading
import time
from fractions import Fraction

import av
import msgpack
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .core import BODY_NAMES, HAND_NAMES, STATE_NAMES, SOURCES

SCHEMA_INFO = {
    'format': 'h2_vla_raw_v1',
    'state_action_order': STATE_NAMES,
    'body_joint_order': BODY_NAMES, 'hand_actuator_order': HAND_NAMES,
    'body_unit': 'rad', 'hand_unit': 'normalized: 0 closed, 1 open; six actuators per hand',
    'action_semantics': 'absolute q targets observed on rt/lowcmd and rt/inspire/cmd; not measured actions or policy residuals',
    'hand_command_semantics': 'DDS setpoints sent to bridge; not a serial execution acknowledgement',
    'head_indices': {'pitch': 29, 'yaw': 30},
    'imu_quaternion': 'wxyz',
    'temperature_order': [f'{n}/sensor{s}' for n in BODY_NAMES for s in (0, 1)],
    'tau_est': 'drive estimate, not a calibrated output torque sensor',
    'video': 'head_rgb.mp4, BGR input encoded as RGB-visible H264/yuv420p; no overlay in saved images',
    'alignment': 'latest received sample at each tick; nominal frame_index/fps plus actual monotonic and wall times',
    'camera_time': 'host decoded-frame arrival, NOT sensor exposure time; repeated image IDs identify reused frames',
    'source_time': 'DDS source_timestamp in ns if available; ZMQ timestamp_monotonic converted to ns if available; zero means unavailable',
    'source_clock_warning': 'DDS publisher clocks may not be synchronized; do not subtract across clock domains',
    'teleop': 'input.*.json contains full latest windows; raw_zmq.msgpack is a stream of original received wire messages',
    'freshness': 'timing.*.fresh and age_ms; inactive teleop inputs can be stale and must be masked by teleop_mode',
    'unavailable': ['camera hardware exposure timestamp', 'camera pre-decode packet/frame loss count', 'hardware synchronization', 'depth', 'tactile/contact forces',
                    'hand motor current/temperature/torque (not reported by current bridge)', 'world-frame ground-truth pose'],
    'training_filter': 'metadata.status == saved; outcome is independently success/failure/unlabeled',
    'compatibility': 'native H2 raw format, not a stock G1 or LeRobot dataset; adapt feature mapping before training',
}


def write_json(path, value):
    temporary = Path(str(path) + '.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def row_schema(row):
    fields = []
    int_lists = {'observation.temperature', 'observation.motor_error', 'observation.hand_lost', 'action.motor_mode'}
    for k, v in row.items():
        if k.startswith('input.'):
            dtype = pa.string()
        elif isinstance(v, (list, tuple, np.ndarray)):
            dtype = pa.list_(pa.int64() if k in int_lists else pa.float32(), len(v))
        elif isinstance(v, bool):
            dtype = pa.bool_()
        elif isinstance(v, (int, np.integer)):
            # CycloneDDS instance handles are unsigned 64-bit.
            dtype = pa.uint64() if k.endswith('.publisher') else pa.int64()
        else:
            dtype = pa.float64()
        fields.append(pa.field(k, dtype))
    return pa.schema(fields)


class EpisodeWriter:
    def __init__(self, root, metadata, fps, width, height, queue_size=256):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.name = f'episode_{time.strftime("%Y%m%d_%H%M%S")}_{time.time_ns() % 1000000000:09d}'
        self.stage = self.root / (self.name + '.partial')
        self.stage.mkdir()
        self.metadata = {**metadata, 'format': 'h2_vla_raw_v1', 'status': 'recording',
                         'outcome': 'unlabeled', 'fps': fps, 'video_size': [width, height],
                         'created_wall_ns': time.time_ns()}
        write_json(self.stage / 'metadata.json', self.metadata)
        write_json(self.stage / 'schema.json', SCHEMA_INFO)
        self.fps, self.width, self.height = fps, width, height
        self.queue = queue.Queue(maxsize=queue_size)
        self.queue_high_watermark = 0
        self.error = None
        self.frames = 0
        self.raw_count = 0
        self.reused_camera_frames = 0
        self.previous_camera = None
        self.thread = threading.Thread(target=self._run, name='h2-episode-writer', daemon=True)
        self.thread.start()

    def put(self, kind, value):
        if self.error:
            raise RuntimeError(f'writer failed: {self.error}')
        try:
            self.queue.put_nowait((kind, value))
            self.queue_high_watermark = max(self.queue_high_watermark, self.queue.qsize())
        except queue.Full as exc:
            raise RuntimeError(f'writer queue full ({self.queue.maxsize} items, incoming={kind}); '
                               'episode cannot remain valid') from exc

    def _run(self):
        container = parquet = raw = None
        rows = []
        schema = None
        try:
            raw = (self.stage / 'raw_zmq.msgpack').open('wb')
            container = av.open(str(self.stage / 'head_rgb.mp4'), mode='w')
            video = container.add_stream('libx264', rate=self.fps)
            video.width, video.height, video.pix_fmt = self.width, self.height, 'yuv420p'
            video.options = {'crf': '18', 'preset': 'veryfast', 'tune': 'zerolatency'}
            while True:
                item = self.queue.get()
                if item is None:
                    break
                kind, value = item
                if kind == 'raw':
                    raw.write(msgpack.packb(value, use_bin_type=True))
                    self.raw_count += 1
                    continue
                row, bgr = value
                if bgr.shape != (self.height, self.width, 3):
                    raise ValueError(f'camera size changed: {bgr.shape}')
                frame = av.VideoFrame.from_ndarray(bgr, format='bgr24')
                frame.pts = self.frames
                frame.time_base = Fraction(1, self.fps)
                for packet in video.encode(frame):
                    container.mux(packet)
                if schema is None:
                    schema = row_schema(row)
                    parquet = pq.ParquetWriter(self.stage / 'frames.parquet', schema, compression='zstd')
                rows.append(row)
                self.frames += 1
                if self.previous_camera == row['camera_frame_id']:
                    self.reused_camera_frames += 1
                self.previous_camera = row['camera_frame_id']
                if len(rows) >= 128:
                    parquet.write_table(pa.Table.from_pylist(rows, schema=schema))
                    rows.clear()
            for packet in video.encode(None):
                container.mux(packet)
            if rows:
                parquet.write_table(pa.Table.from_pylist(rows, schema=schema))
        except Exception as exc:
            self.error = repr(exc)
        finally:
            for resource in (parquet, container, raw):
                if resource:
                    try:
                        resource.close()
                    except Exception as exc:
                        self.error = self.error or repr(exc)

    def finish(self, status, outcome='unlabeled', reason='', quality=None):
        # Stop producers before calling. Sentinel + join prevent flush/encode races.
        while self.thread.is_alive():
            try:
                self.queue.put(None, timeout=.1)
                break
            except queue.Full:
                continue
        self.thread.join()
        if self.error:
            status, reason = 'incomplete', f'{reason}; writer: {self.error}'
        if not self.frames and status == 'saved':
            status, reason = 'incomplete', 'empty episode'
        self.metadata.update(status=status, outcome=outcome, reason=reason, frames=self.frames,
                             raw_zmq_messages=self.raw_count, repeated_camera_frames=self.reused_camera_frames,
                             closed_wall_ns=time.time_ns(), quality=quality or {})
        self.metadata['writer_queue'] = dict(capacity=self.queue.maxsize, high_watermark=self.queue_high_watermark)
        write_json(self.stage / 'metadata.json', self.metadata)
        # A failed encode stays visibly .partial; never advertises a valid MP4.
        if self.error:
            return self.stage
        for path in self.stage.iterdir():
            if path.is_file():
                with path.open('rb') as f:
                    os.fsync(f.fileno())
        destination = self.root / self.name
        self.stage.rename(destination)
        fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return destination
