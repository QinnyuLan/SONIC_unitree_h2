import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from interface.teleop.xr.offline_xrt import (
    OfflineXRoboToolkit,
    load_tracking_data_recording,
    load_xrobotoolkit_recording,
)


def _recorded_fixture() -> dict[str, np.ndarray]:
    identity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    headset = np.stack((identity, identity.copy()))
    headset[1, 2] = 0.1
    left = headset.copy()
    left[:, 0] = -0.2
    right = headset.copy()
    right[:, 0] = 0.2
    return {
        "fps": np.asarray(50.0),
        "headset_pose": headset,
        "left_controller_pose": left,
        "right_controller_pose": right,
        "a_button": np.array([False, True]),
    }


class OfflineXRoboToolkitTest(unittest.TestCase):
    def test_three_recorded_poses_are_required(self):
        data = _recorded_fixture()
        del data["headset_pose"]
        with self.assertRaisesRegex(ValueError, "headset_pose"):
            OfflineXRoboToolkit(data)

    def test_playback_uses_recorded_frames(self):
        now = [100.0]
        backend = OfflineXRoboToolkit(_recorded_fixture(), clock=lambda: now[0])
        backend.init()

        np.testing.assert_allclose(backend.get_headset_pose()[:3], [0.0, 0.0, 0.0])
        self.assertFalse(backend.get_A_button())
        now[0] += 0.021
        np.testing.assert_allclose(backend.get_headset_pose()[:3], [0.0, 0.0, 0.1])
        self.assertTrue(backend.get_A_button())
        self.assertFalse(backend.is_body_data_available())
        self.assertTrue(backend.is_vr_3pt_data_available())

    def test_npz_replay(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "recorded_vr.npz"
            np.savez_compressed(path, **_recorded_fixture())
            backend = OfflineXRoboToolkit.from_path(path)
            self.assertEqual(backend.frame_count, 2)
            self.assertEqual(backend.fps, 50.0)
            self.assertEqual(np.asarray(backend.get_left_controller_pose()).shape, (7,))

    def test_official_robot_data_recorder_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            heads = [
                {"pose": "0,1,2,0,0,0,1"},
                {"pose": "0,1,3,0,0,0,1"},
            ]
            controllers = []
            for pressed in (False, True):
                controllers.append(
                    {
                        "left": {
                            "pose": "-1,1,2,0,0,0,1",
                            "primaryButton": pressed,
                            "axisX": 0.25,
                            "axisY": 0.5,
                        },
                        "right": {
                            "pose": "1,1,2,0,0,0,1",
                            "primaryButton": pressed,
                            "trigger": 0.75,
                        },
                    }
                )
            path.joinpath("head.txt").write_text(
                "".join(json.dumps(item, indent=2) + "\n" for item in heads), encoding="utf-8"
            )
            path.joinpath("controller.txt").write_text(
                "".join(json.dumps(item, indent=2) + "\n" for item in controllers),
                encoding="utf-8",
            )

            data = load_xrobotoolkit_recording(path, fps=50.0)
            self.assertEqual(data["headset_pose"].shape, (2, 7))
            self.assertEqual(data["left_controller_pose"].shape, (2, 7))
            self.assertTrue(data["x_button"][1])
            self.assertTrue(data["a_button"][1])
            self.assertAlmostEqual(float(data["right_trigger"][0]), 0.75)
            np.testing.assert_allclose(data["left_axis"][0], [0.25, 0.5])

    def test_pico_tracking_data_full_body_recording(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trackingData_test.txt"
            identity = "0,0,0,0,0,0,1"
            lines = [json.dumps({"notice": "metadata", "timeStampNs": 999})]
            for index, timestamp in enumerate((1_000_000_000, 1_020_000_000)):
                joints = [{"p": identity, "t": 0} for _ in range(24)]
                joints[0]["p"] = f"{index},0,0,0,0,0,1"
                lines.append(
                    json.dumps(
                        {
                            "Head": {"pose": identity},
                            "Controller": {
                                "left": {
                                    "pose": identity,
                                    "primaryButton": bool(index),
                                    "axisX": 0.25,
                                },
                                "right": {"pose": identity, "trigger": 0.75},
                            },
                            "Body": {"joints": joints},
                            "timeStampNs": timestamp,
                        }
                    )
                )
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            data = load_tracking_data_recording(path)
            self.assertEqual(data["body_joints_pose"].shape, (2, 24, 7))
            np.testing.assert_array_equal(data["timestamps_ns"], [0, 20_000_000])
            self.assertAlmostEqual(float(data["fps"]), 50.0)

            backend = OfflineXRoboToolkit.from_path(path)
            self.assertTrue(backend.is_body_data_available())
            self.assertFalse(backend.is_offline_3pt)
            self.assertEqual(np.asarray(backend.get_body_joints_pose()).shape, (24, 7))
            self.assertAlmostEqual(backend.get_right_trigger(), 0.75)


if __name__ == "__main__":
    unittest.main()
