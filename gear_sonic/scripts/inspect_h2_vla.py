#!/usr/bin/env python3
"""Offline validation of H2 VLA raw episodes; never connects to a robot."""
import argparse
import json
from pathlib import Path
import sys

import av
import msgpack
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gear_sonic.data.h2_collection.core import unpack_wire


def inspect_episode(path):
    path = Path(path)
    metadata = json.loads((path/'metadata.json').read_text())
    errors = []
    status = metadata.get('status')
    result = dict(path=str(path.resolve()), status=status, outcome=metadata.get('outcome'),
                  task=metadata.get('task'), frames=metadata.get('frames', 0), errors=errors)
    if path.name.endswith('.partial') or status == 'recording':
        errors.append('unfinished_episode')
    try:
        table = pq.read_table(path/'frames.parquet', columns=[
            'frame_index', 'timestamp', 'host_monotonic_ns', 'camera_frame_id',
            'sampling_offset_ms',
            'observation.state', 'action', 'action.motion_token',
            *[f'timing.{s}.fresh' for s in ('lowstate', 'lowcmd', 'hand_state', 'hand_cmd',
                                          'g1_debug', 'manager_state', 'camera', 'head_pose')]])
        count = len(table)
        if count != result['frames'] or count == 0:
            errors.append('frame_count_mismatch_or_empty')
        rows = table.to_pydict()
        if rows['frame_index'] != list(range(count)):
            errors.append('frame_index_not_contiguous')
        if not np.allclose(rows['timestamp'], np.arange(count)/metadata['fps'], atol=1e-8):
            errors.append('nominal_timestamps_invalid')
        for name in ('observation.state', 'action', 'action.motion_token'):
            data = np.asarray(rows[name])
            expected = 64 if name.endswith('motion_token') else 43
            if data.shape != (count, expected) or not np.all(np.isfinite(data)):
                errors.append(f'{name}_invalid')
        for name in rows:
            if name.endswith('.fresh') and not all(rows[name]):
                errors.append(f'{name}_false')
        dt = np.diff(np.asarray(rows['host_monotonic_ns'], np.int64))/1e6
        result['sample_interval_ms'] = dict(mean=float(dt.mean()), max=float(dt.max()), min=float(dt.min())) if len(dt) else None
        if len(dt) and (np.any(dt <= 0) or dt.max() > max(2/metadata['fps'], .05)*1000):
            errors.append('sampling_timing_invalid')
        if np.max(np.abs(rows['sampling_offset_ms']), initial=0) > max(2/metadata['fps'], .05)*1000:
            errors.append('sampling_rate_drift')
        result['unique_rgb_frames'] = len(set(rows['camera_frame_id']))
        with av.open(str(path/'head_rgb.mp4')) as video:
            video_count = 0
            for frame in video.decode(video=0):
                if abs(float(frame.time)-video_count/metadata['fps']) > 1e-5:
                    errors.append('video_timestamp_mismatch')
                    break
                video_count += 1
        result['video_frames'] = video_count
        if video_count != count:
            errors.append('video_table_count_mismatch')
        raw_count = 0
        topics = set()
        with (path/'raw_zmq.msgpack').open('rb') as f:
            for event in msgpack.Unpacker(f, raw=False):
                raw_count += 1
                topics.add(event['topic'])
                if event['topic'] not in ('g1_debug', 'robot_config'):
                    unpack_wire(event['wire'])
        result['raw_topics'] = sorted(topics)
        if raw_count != metadata.get('raw_zmq_messages'):
            errors.append('raw_message_count_mismatch')
    except Exception as exc:
        errors.append(f'read_failed: {exc}')
    result['training_eligible'] = status == 'saved' and not errors
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path, help='one episode or dataset root')
    args = parser.parse_args()
    paths = [args.path] if (args.path/'metadata.json').exists() else sorted(args.path.glob('episode_*'))
    if not paths:
        parser.error('no episodes found')
    reports = [inspect_episode(p) for p in paths]
    print(json.dumps(reports, indent=2, ensure_ascii=False))
    return int(any(r['errors'] for r in reports))


if __name__ == '__main__':
    raise SystemExit(main())
