"""MotionSwitcher handoff tests with fake RPC; no robot communication."""

from contextlib import nullcontext, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import h2_ankle_range_test as r


class ClockStop:
    now = 0.
    stopped = False

    def is_set(self):
        return self.stopped

    def wait(self, duration):
        self.now += duration


class ReleaseControl(unittest.TestCase):
    def setUp(self):
        self.clock = ClockStop()
        self.name = "sport"
        self.tr = SimpleNamespace(own_writer_created=False, command_writers=1,
                                  healthy=Mock(), create_writer=Mock(), send=Mock())
        self.rec = Mock()
        self.client = Mock()
        self.client.CheckMode.side_effect = lambda: (0, {"name": self.name})
        self.client.ReleaseMode.side_effect = self.release

    def release(self):
        self.name = ""
        self.tr.command_writers = 0
        return 0, None

    def check(self):
        if self.clock.is_set():
            raise r.SafetyStop("operator stop")

    def handoff(self, authorized=True):
        with patch.object(r.time, "monotonic", side_effect=lambda: self.clock.now), \
             redirect_stdout(io.StringIO()):
            r.release_high_level(self.client, self.tr, self.rec, self.clock, self.check, authorized=authorized)
        self.tr.send.assert_not_called()
        self.tr.create_writer.assert_not_called()

    def test_authorized_release_and_writer_drain(self):
        self.handoff()
        self.client.ReleaseMode.assert_called_once()
        self.client.SelectMode.assert_not_called()
        self.assertGreaterEqual(self.clock.now, 3)
        self.rec.event.assert_any_call("motion_switcher_released", command_writers=0)
        self.tr.healthy.assert_called_once_with()

    def test_already_released_does_not_call_release(self):
        self.name, self.tr.command_writers = "", 0
        self.handoff()
        self.client.ReleaseMode.assert_not_called()

    def test_missing_start_authorization_never_releases(self):
        with self.assertRaisesRegex(r.SafetyStop, "operator start key"):
            self.handoff(False)
        self.client.CheckMode.assert_not_called()
        self.client.ReleaseMode.assert_not_called()
        self.tr.create_writer.assert_not_called()

    def test_one_start_key_replaces_typed_confirmations(self):
        for key, succeeds in (("y", True), ("q", False), ("\x1b", False), ("", False)):
            with self.subTest(key=key), patch.object(r, "keyboard", return_value=nullcontext(lambda: key)), \
                 redirect_stdout(io.StringIO()):
                if succeeds:
                    r.wait_for_start(self.clock, self.check)
                else:
                    with self.assertRaisesRegex(r.SafetyStop, "cancelled"):
                        r.wait_for_start(self.clock, self.check)
                self.client.ReleaseMode.assert_not_called()

    def test_start_timeout_and_signal_do_not_authorize(self):
        with patch.object(r.time, "monotonic", side_effect=lambda: self.clock.now), \
             patch.object(r, "keyboard", return_value=nullcontext(lambda: None)), \
             redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(r.SafetyStop, "timed out"):
                r.wait_for_start(self.clock, self.check)
        self.clock.stopped = True
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(r.SafetyStop, "operator stop"):
            r.wait_for_start(self.clock, self.check, automatic=True)
        self.client.ReleaseMode.assert_not_called()

    def test_check_failure_and_invalid_mode_never_release(self):
        for reply in ((3103, None), (0, {}), (0, {"name": None})):
            with self.subTest(reply=reply):
                self.client.CheckMode.side_effect = None
                self.client.CheckMode.return_value = reply
                with self.assertRaises(r.SafetyStop):
                    self.handoff()
                self.client.ReleaseMode.assert_not_called()

    def test_release_failure_stops_without_takeover(self):
        self.client.ReleaseMode.side_effect = lambda: (3103, None)
        with self.assertRaisesRegex(r.SafetyStop, "ReleaseMode failed: code=3103"):
            self.handoff()
        self.tr.create_writer.assert_not_called()
        self.tr.send.assert_not_called()

    def test_retries_are_bounded_if_mode_remains_active(self):
        self.client.ReleaseMode.side_effect = lambda: (0, None)
        with self.assertRaisesRegex(r.SafetyStop, "still active"):
            self.handoff()
        self.assertEqual(self.client.ReleaseMode.call_count, 3)

    def test_lingering_lowcmd_writer_is_not_ignored(self):
        def release():
            self.name = ""
            return 0, None
        self.client.ReleaseMode.side_effect = release
        with self.assertRaisesRegex(r.SafetyStop, "LowCmd publishers=1, expected 0"):
            self.handoff()
        self.tr.create_writer.assert_not_called()
        self.tr.send.assert_not_called()

    def test_stop_during_check_rpc_prevents_release(self):
        checks = 0
        def check_mode():
            nonlocal checks
            checks += 1
            if checks == 2:
                self.clock.stopped = True
            return 0, {"name": "sport"}
        self.client.CheckMode.side_effect = check_mode
        with self.assertRaisesRegex(r.SafetyStop, "operator stop"):
            self.handoff()
        self.client.ReleaseMode.assert_not_called()

    def test_mode_reactivation_prevents_takeover(self):
        checks = 0
        def check_mode():
            nonlocal checks
            checks += 1
            return 0, {"name": "sport" if checks >= 5 else self.name}
        self.client.CheckMode.side_effect = check_mode
        with self.assertRaisesRegex(r.SafetyStop, "reactivated"):
            self.handoff()
        self.tr.create_writer.assert_not_called()

    def test_no_release_once_lowcmd_writer_exists(self):
        self.tr.own_writer_created = True
        with self.assertRaisesRegex(r.SafetyStop, "before creating"):
            self.handoff()
        self.client.CheckMode.assert_not_called()
        self.client.ReleaseMode.assert_not_called()

    def test_publisher_exception_only_applies_before_takeover(self):
        tr = object.__new__(r.Transport)
        tr.error, tr.recorder = None, SimpleNamespace(error=None)
        tr.args = SimpleNamespace(target="sim")
        tr.state_writers, tr.command_writers, tr.own_writer_created = 1, 1, False
        tr.healthy(before_release=True)
        with self.assertRaisesRegex(r.SafetyStop, "1, expected 0"):
            tr.healthy()
        tr.own_writer_created = True
        tr.healthy()
        with self.assertRaisesRegex(r.SafetyStop, "cannot relax"):
            tr.healthy(before_release=True)
        tr.command_writers = 2
        with self.assertRaisesRegex(r.SafetyStop, "2, expected 1"):
            tr.healthy()


def rpc_smoke():
    # All endpoints, including the fake high-level writer, are on private lo/98.
    args = SimpleNamespace(domain=98, interface="lo", target="sim")
    client = r.create_motion_switcher(args)
    rec = SimpleNamespace(error=None, put=lambda *a: None, event=lambda *a, **k: None)
    tr = r.Transport(args, rec, domain_initialized=True)
    from cyclonedds.pub import DataWriter
    from cyclonedds.sub import DataReader
    from cyclonedds.topic import Topic
    from unitree_sdk2py.idl.unitree_api.msg.dds_ import Request_, Response_, ResponseHeader_, ResponseStatus_
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_

    request_topic = Topic(tr.participant, "rt/api/motion_switcher/request", Request_)
    response_topic = Topic(tr.participant, "rt/api/motion_switcher/response", Response_)
    requests, responses = DataReader(tr.participant, request_topic), DataWriter(tr.participant, response_topic)
    state_writer = DataWriter(tr.participant, tr.state_topic)
    external_writer = DataWriter(tr.participant, tr.command_topic)
    state = unitree_hg_msg_dds__LowState_()
    state.mode_machine = 1
    state.imu_state.quaternion = [1., 0., 0., 0.]
    for i, q in enumerate(r.ready_pose()):
        state.motor_state[i].q = q
    done = threading.Event()
    calls = []
    failures = []

    def server():
        nonlocal external_writer
        name = "sport"
        try:
            while not done.is_set():
                state.tick += 5
                state.crc = tr.crc.Crc(state)
                state_writer.write(state)
                for req in requests.take(10):
                    api = req.header.identity.api_id
                    calls.append(api)
                    if api == 1003:
                        name = ""
                        external_writer = None
                    if api not in (1001, 1003):
                        raise AssertionError(f"unexpected RPC API {api}")
                    result = {"name": name, "form": "0"} if api == 1001 else {}
                    responses.write(Response_(ResponseHeader_(req.header.identity, ResponseStatus_(0)),
                                               json.dumps(result), []))
                done.wait(.005)
        except Exception as exc:
            failures.append(exc)

    worker = threading.Thread(target=server, daemon=True)
    worker.start()
    try:
        deadline = time.monotonic()+5
        while tr.get() is None or tr.state_writers != 1 or tr.command_writers != 1:
            if time.monotonic()>deadline or failures:
                raise AssertionError(f"DDS fixture discovery failed: {failures}")
            time.sleep(.01)
        r.release_high_level(client, tr, rec, done, lambda: tr.healthy(before_release=True), authorized=True)
        assert calls.count(1003) == 1
        assert not tr.own_writer_created and tr.writer is None
        assert tr.command_writers == 0 and not failures
        print("RPC_SMOKE_PASSED: SDK release + writer drain, no test LowCmd writer")
    finally:
        done.set()
        worker.join(timeout=5)
        with tr.callback_lock:
            tr.closed = True
        tr.state_reader = tr.command_reader = None


@unittest.skipUnless(os.environ.get("H2_SYSID_DDS_TESTS") == "1", "private loopback RPC integration")
class ReleaseRpcDds(unittest.TestCase):
    def test_sdk_rpc_and_transport_share_configured_domain(self):
        result = subprocess.run([sys.executable, __file__, "--rpc-smoke"],
                                capture_output=True, text=True, timeout=25)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn("RPC_SMOKE_PASSED", result.stdout)


if __name__ == "__main__":
    if "--rpc-smoke" in sys.argv:
        rpc_smoke()
    else:
        unittest.main()
