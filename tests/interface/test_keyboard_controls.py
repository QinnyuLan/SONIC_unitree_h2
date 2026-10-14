"""Keyboard playback timing and planner intent, without GPU or robot connections."""
import numpy as np
import pytest

from interface.motion_replay import Motion
from interface.motion_replay.keyboard import ReplayControl
from interface.planner.keyboard import MODE_NAMES, PlannerControl


class Client:
    def __init__(self):
        self.messages = []
        self.connected = True

    def send(self, kind, data):
        self.messages.append((kind, data))
        return self.connected

    def stand(self):
        self.messages.append(("stand", {}))


def control():
    client = Client()
    replay = ReplayControl(client)
    replay.motion = Motion(np.arange(101)[:, None] * np.ones((1, 29)) * .02,
                           np.tile([1., 0., 0., 0.], (101, 1)), 50)
    return client, replay


def test_pause_resumes_same_motion_time_and_releases_body():
    client, replay = control()
    replay.key("\n", 10, True)
    replay.tick(10.4, True)
    assert client.messages[-1][1]["joint_pos"][0][0] == pytest.approx(.4)
    replay.key(" ", 10.5, True)
    assert client.messages[-1][0] == "stand"
    count = len(client.messages)
    replay.tick(50, True)
    assert len(client.messages) == count
    replay.key(" ", 50, True)
    replay.tick(50.1, True)
    assert client.messages[-1][1]["joint_pos"][0][0] == pytest.approx(.6)
    replay.key("r", 51, True)
    replay.tick(51, True)
    assert client.messages[-1][1]["joint_pos"][0][0] == 0


@pytest.mark.parametrize("key,kind,count", [("1", "g1", 13), ("2", "g1", 7), ("3", "h2", 8), ("4", "smpl", 8)])
def test_collection_selection_never_autoplays(key, kind, count):
    client = Client()
    replay = ReplayControl(client)
    replay.key(key, 0, True)
    replay.tick(0, True)
    assert not client.messages
    assert len(replay.clips) == count
    replay.key("\n", 1, True)
    replay.tick(1, True)
    assert client.messages[-1][0] == kind
    replay.key("down", 2, True)
    assert not replay.playing and replay.index == 1
    assert client.messages[-1][0] == "stand"
    replay.key("n", 3, True)
    replay.tick(3, True)
    assert replay.index == 2 and client.messages[-1][0] == kind


def test_replay_end_and_disconnect_do_not_restart_automatically():
    client, replay = control()
    replay.key("\n", 0, False)
    assert not replay.playing
    replay.play(0)
    replay.tick(2, True)
    assert not replay.playing and client.messages[-1][0] == "stand"
    replay.play(3)
    replay.tick(3.5, False)
    replay.tick(6, True)
    assert not replay.playing and replay.offset == .5
    replay.play(7)
    client.connected = False
    replay.tick(7.1, True)
    assert not replay.playing


def test_loop_wraps_time_without_losing_velocity_scaling():
    client, replay = control()
    replay.loop, replay.speed = True, 2.
    replay.play(0)
    replay.tick(1.1, True)
    assert replay.playing
    data = client.messages[-1][1]
    assert data['joint_pos'][0][0] == pytest.approx(.2)
    assert data['joint_vel'][0][0] == pytest.approx(2.)


def test_planner_movement_expires_and_heading_rotates_direction():
    planner = PlannerControl()
    assert planner.request(0) is None
    for _ in range(9):
        planner.key('q', 0)
    planner.key('w', 0)
    request = planner.request(.1)
    np.testing.assert_allclose(request['movement'], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(request['facing'], [0, 1, 0], atol=1e-12)
    assert request['mode'] == 1 and request['speed'] == .3
    assert planner.request(.36)['movement'] == [0, 0, 0]
    assert planner.request(.36)['mode'] == 0
    planner.key('d', 1)
    np.testing.assert_allclose(planner.request(1)['movement'], [1, 0, 0], atol=1e-12)
    planner.key(' ', 1.1)
    assert planner.request(1.1) is None


def test_planner_modes_speed_and_stationary_activation():
    planner = PlannerControl()
    planner.key('3', 0)
    planner.key('+', 0)
    planner.key('w', 0)
    assert planner.request(.1)['mode'] == 3
    assert planner.request(.1)['speed'] == .4
    planner.key('x', 1)
    planner.key('4', 1)
    assert planner.request(1) is None
    planner.key('\n', 1)
    assert planner.request(1)['mode'] == 4
    for _ in range(100):
        planner.key('-', 2)
    assert planner.speed == 0
    for _ in range(100):
        planner.key('+', 2)
    assert planner.speed == 7.5
    for _ in range(27):
        planner.key(']', 2)
    assert planner.mode == 4 and len(MODE_NAMES) == 27
    planner.key('\n', 2)
    assert planner.request(2) is None
