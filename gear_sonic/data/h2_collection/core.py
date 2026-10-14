"""Pure data mapping, freshness checks and PICO recording events (no hardware I/O)."""
from dataclasses import dataclass
import json
import numpy as np

BODY_NAMES = [
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint',
    'left_ankle_roll_joint', 'left_ankle_pitch_joint', 'right_hip_pitch_joint',
    'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint',
    'right_ankle_roll_joint', 'right_ankle_pitch_joint', 'waist_yaw_joint',
    'waist_roll_joint', 'waist_pitch_joint', 'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint',
    'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
    'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint',
    'right_wrist_yaw_joint', 'head_pitch_joint', 'head_yaw_joint',
]
HAND_NAMES = [f'{side}_{joint}' for side in ('left', 'right')
              for joint in ('pinky', 'ring', 'middle', 'index', 'thumb_bend', 'thumb_rotate')]
STATE_NAMES = BODY_NAMES + HAND_NAMES
TOPICS = ('manager_state', 'head_pose', 'command', 'planner', 'pose')
SOURCES = ('lowstate', 'lowcmd', 'hand_state', 'hand_cmd', 'g1_debug', 'manager_state',
           'camera', 'head_pose', 'pose', 'planner', 'secondary_imu')
MAX_AGE = dict(lowstate=.1, lowcmd=.1, hand_state=.3, hand_cmd=.3, g1_debug=.15,
               manager_state=.25, camera=.35, head_pose=.25, pose=.2, planner=.25)


def unpack_wire(raw):
    topic = next((t for t in TOPICS if raw.startswith(t.encode())), None)
    if topic is None or len(raw) < len(topic) + 1280:
        raise ValueError('unknown topic or truncated header')
    offset = len(topic) + 1280
    header = json.loads(raw[len(topic):offset].split(b'\0', 1)[0])
    endian = {'le': '<', 'be': '>'}[header.get('endian', 'le')]
    types = dict(f32='f4', f64='f8', i32='i4', i64='i8', u8='u1', bool='?')
    result = {}
    for field in header['fields']:
        dtype = np.dtype(endian + types[field['dtype']])
        shape = field['shape']
        if not all(isinstance(n, int) and 0 <= n <= 100000 for n in shape):
            raise ValueError('invalid field shape')
        size = int(np.prod(shape)) * dtype.itemsize
        if offset + size > len(raw) or field['name'] in result:
            raise ValueError('truncated or duplicate field')
        result[field['name']] = np.frombuffer(raw, dtype, int(np.prod(shape)), offset).reshape(shape).copy()
        offset += size
    if offset != len(raw):
        raise ValueError('unexpected trailing payload')
    return topic, result


def scalar(data, key, default=0):
    return np.asarray(data.get(key, [default])).reshape(-1)[0].item()


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


@dataclass(frozen=True)
class Sample:
    data: dict
    mono_ns: int
    wall_ns: int
    source_ns: int = 0
    sequence: int = 0
    publisher: int = 0


def decode_dds(name, msg):
    """Use DDS order, never the legacy G1-transformed debug state."""
    if name == 'lowstate':
        m = msg.motor_state[:31]
        if len(m) != 31:
            raise ValueError('H2 needs 31 motors')
        return dict(q=[x.q for x in m], dq=[x.dq for x in m], tau=[x.tau_est for x in m],
                    temperature=[t for x in m for t in x.temperature],
                    error=[x.motorstate for x in m], tick=int(msg.tick),
                    mode_pr=int(msg.mode_pr), mode_machine=int(msg.mode_machine),
                    imu=decode_dds('secondary_imu', msg.imu_state))
    if name == 'lowcmd':
        m = msg.motor_cmd[:31]
        return {**{k: [getattr(x, k) for x in m] for k in ('q', 'dq', 'tau', 'kp', 'kd', 'mode')},
                'mode_pr': int(msg.mode_pr), 'mode_machine': int(msg.mode_machine)}
    if name == 'secondary_imu':
        return dict(quaternion=list(msg.quaternion), gyroscope=list(msg.gyroscope),
                    accelerometer=list(msg.accelerometer))
    m = list(msg.states if name == 'hand_state' else msg.cmds)
    if len(m) != 12:
        raise ValueError('Inspire requires 12 actuators')
    m = m[6:] + m[:6]  # DDS right/left -> dataset left/right
    result = dict(q=[x.q for x in m])
    if name == 'hand_state':
        result['lost'] = [x.lost for x in m]
    return result


class RecordingEvents:
    """Cumulative counters survive isolated PUB/SUB packet loss; no phantom late-join start."""
    def __init__(self):
        self.previous = None

    def update(self, data):
        keys = ('record_session_id', 'record_toggle_count', 'record_abort_count')
        if not all(k in data for k in keys):
            return ['legacy_manager']
        current = tuple(int(scalar(data, k)) for k in keys)
        previous, self.previous = self.previous, current
        if previous is None:
            return []
        if current[0] != previous[0]:
            return ['manager_restarted']
        delta = tuple(a-b for a, b in zip(current[1:], previous[1:]))
        if delta == (0, 0):
            return []
        if delta == (0, 1):
            return ['discard']
        if delta == (1, 0):
            return ['toggle']
        return ['ambiguous_record_events']


def problems(samples, now_ns):
    required = list(SOURCES[:8])  # includes fixed head command
    manager = samples.get('manager_state')
    mode = int(scalar(manager.data, 'stream_mode')) if manager else 0
    if mode == 1:
        required.append('pose')
    elif mode in (2, 3, 5):
        required.append('planner')
    errors = []
    if mode not in (1, 2, 3, 5):
        errors.append('teleop_not_active')
    if manager and not all(k in manager.data for k in
                           ('record_session_id', 'record_toggle_count', 'record_abort_count')):
        errors.append('restart_updated_pico_manager')
    for name in required:
        sample = samples.get(name)
        if sample is None or not 0 <= (now_ns-sample.mono_ns)/1e9 <= MAX_AGE[name]:
            errors.append(f'{name}_missing_or_stale')
    for name in ('hand_state', 'hand_cmd'):
        if name in samples:
            data = samples[name].data
            q = np.asarray(data['q'])
            if q.shape != (12,) or not np.all(np.isfinite(q)) or np.any((q < 0) | (q > 1)):
                errors.append(f'{name}_invalid_position')
            if any(data.get('lost', [])):
                errors.append('hand_state_offline')
    if 'g1_debug' in samples:
        token = np.asarray(samples['g1_debug'].data.get('token_state', []))
        if token.shape != (64,) or not np.all(np.isfinite(token)):
            errors.append('motion_token_missing_or_invalid')
    for name in ('lowstate', 'lowcmd'):
        if name in samples:
            q = np.asarray(samples[name].data['q'])
            if q.shape != (31,) or not np.all(np.isfinite(q)):
                errors.append(f'{name}_invalid_position')
            if samples[name].data.get('crc_ok') is False:
                errors.append(f'{name}_bad_crc')
            for key in ('dq', 'tau', 'kp', 'kd'):
                if key in samples[name].data and not np.all(np.isfinite(samples[name].data[key])):
                    errors.append(f'{name}_nonfinite_{key}')
    if 'lowstate' in samples:
        if not all(np.all(np.isfinite(v)) for v in samples['lowstate'].data['imu'].values()):
            errors.append('imu_nonfinite')
    return errors


def make_row(samples, now_ns, wall_ns, index, fps, start_ns=None):
    state, cmd = samples['lowstate'].data, samples['lowcmd'].data
    hand, hand_cmd = samples['hand_state'].data, samples['hand_cmd'].data
    row = {
        'frame_index': index, 'timestamp': index/fps, 'host_monotonic_ns': now_ns,
        'elapsed_seconds': (now_ns-start_ns)/1e9 if start_ns is not None else index/fps,
        'sampling_offset_ms': ((now_ns-start_ns)/1e9-index/fps)*1000 if start_ns is not None else 0.0,
        'host_wall_ns': wall_ns, 'observation.state': state['q'] + hand['q'],
        'action': cmd['q'] + hand_cmd['q'], 'observation.velocity': state['dq'],
        'observation.tau_est': state['tau'], 'observation.temperature': state['temperature'],
        'observation.motor_error': state['error'], 'observation.hand_lost': hand['lost'],
        'observation.imu.quaternion_wxyz': state['imu']['quaternion'],
        'observation.imu.gyroscope': state['imu']['gyroscope'],
        'observation.imu.accelerometer': state['imu']['accelerometer'],
        'action.velocity': cmd['dq'], 'action.kp': cmd['kp'], 'action.kd': cmd['kd'],
        'action.tau_ff': cmd['tau'], 'action.motor_mode': cmd['mode'],
        'action.motion_token': samples['g1_debug'].data['token_state'],
        'body_tick': state['tick'], 'state_mode_pr': state['mode_pr'],
        'command_mode_pr': cmd['mode_pr'], 'policy_index': samples['g1_debug'].data.get('index', -1),
        'camera_frame_id': samples['camera'].sequence,
        'teleop_mode': int(scalar(samples['manager_state'].data, 'stream_mode')),
    }
    for name in SOURCES:
        s = samples.get(name)
        row[f'timing.{name}.arrival_ns'] = s.mono_ns if s else -1
        row[f'timing.{name}.wall_ns'] = s.wall_ns if s else -1
        row[f'timing.{name}.source_ns'] = s.source_ns if s else -1
        row[f'timing.{name}.sequence'] = s.sequence if s else -1
        row[f'timing.{name}.publisher'] = s.publisher if s else 0
        row[f'timing.{name}.age_ms'] = (now_ns-s.mono_ns)/1e6 if s else -1.0
        row[f'timing.{name}.fresh'] = bool(s and 0 <= (now_ns-s.mono_ns)/1e9 <= MAX_AGE.get(name, .1))
    # The complete original shapes/windows remain in JSON and raw ZMQ msgpack sidecar.
    for name in ('pose', 'planner', 'manager_state', 'head_pose', 'secondary_imu'):
        row[f'input.{name}.json'] = json.dumps(plain(samples[name].data), allow_nan=False) if name in samples else None
    return row
