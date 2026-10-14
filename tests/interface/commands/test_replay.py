"""Replay data provenance and the native H2 / SMPL input contracts."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from interface.commands.protocol import envelope
from interface.replay import Motion, SmplMotion

ROOT = Path(__file__).resolve().parents[3]
COLLECTIONS = {"example_g1_input": ("g1", 13), "example_g1_explore_input": ("g1", 7),
               "example_h2_input": ("h2", 8), "example_smpl_input": ("smpl", 8)}


def csv(path):
    try:
        return np.loadtxt(path, delimiter=",", ndmin=2)
    except ValueError:
        return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


@pytest.mark.parametrize("collection", COLLECTIONS)
def test_released_collections_have_complete_unchanged_data(collection):
    folder = ROOT / "motions" / collection
    manifest = json.loads((folder / "manifest.json").read_text())
    kind, count = COLLECTIONS[collection]
    assert len(manifest["motions"]) == count
    assert {p.name for p in folder.iterdir() if p.is_dir()} == {m["motion_name"] for m in manifest["motions"]}
    for entry in manifest["motions"]:
        clip = folder / entry["motion_name"]
        for name, expected in entry["sha256"].items():
            assert hashlib.sha256((clip / name).read_bytes()).hexdigest() == expected
        motion = Motion.load(clip)
        assert motion.reference_type == kind
        assert len(motion.times) == entry["frames"]
        for elapsed in (0, motion.duration / 2, motion.duration):
            data = motion.sample(elapsed)
            envelope(kind, data, source="test", session="test", seq=1)
        if kind in {"g1", "h2"}:
            np.testing.assert_array_equal(data["joint_vel"], np.zeros((10, 29)))


def test_h2_native_joint_mapping_excludes_head_and_scales_velocity(tmp_path):
    # Distinct values in every source column detect ankle/arm ordering mistakes.
    q = np.tile(np.arange(31, dtype=float), (101, 1)) + np.arange(101)[:, None] * .02
    dq = np.full_like(q, 1.)
    root = np.tile([1., 0, 0, 0], (101, 1))
    path = tmp_path / "h2.npz"
    np.savez(path, h2_joint_pos=q, h2_joint_vel=dq, root_quat=root, fps=50)
    m = Motion.load(path)
    s = m.sample(.5, 2)
    assert m.reference_type == "h2"
    # Model order: L hip pitch, R hip pitch, waist yaw, ... shoulders, ankles, wrists.
    expected = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 17, 24, 5, 11,
                18, 25, 4, 10, 19, 26, 20, 27, 21, 28, 22, 29, 23, 30]
    np.testing.assert_allclose(s["joint_pos"][0], np.array(expected) + .5)
    np.testing.assert_allclose(np.array(s["joint_vel"])[:8], 2)
    np.testing.assert_allclose(np.array(s["joint_vel"])[8:], 0)
    q[:, 15:17] = 999
    np.savez(path, h2_joint_pos=q, h2_joint_vel=dq, root_quat=root, fps=50)
    np.testing.assert_array_equal(Motion.load(path).sample(.5, 2)["joint_pos"], s["joint_pos"])


def test_smpl_csv_uses_smpl_root_and_semantic_wrists():
    clip = ROOT / "motions/example_smpl_input/forward_lunge_R_001__A359_M"
    m = Motion.load(clip)
    sample = m.sample(.4, 2)
    frames = 20 + np.arange(10) * 2  # 20 ms playback horizon, doubled source speed.
    np.testing.assert_allclose(sample["joints"], csv(clip / "smpl_joint.csv").reshape(-1, 24, 3)[frames])
    np.testing.assert_allclose(sample["wrist_joints"], csv(clip / "joint_pos.csv")[frames, 23:29])
    root = csv(clip / "smpl_root_quat.csv")[frames]
    root /= np.linalg.norm(root, axis=1, keepdims=True)
    np.testing.assert_allclose(sample["root_quat"], root, atol=1e-8)
    assert not np.allclose(root, csv(clip / "body_quat.csv")[frames, :4])


def test_smpl_npz_roundtrip_and_invalid_frame_counts(tmp_path):
    path = tmp_path / "smpl.npz"
    root = np.tile([1., 0, 0, 0], (51, 1))
    joints = np.arange(51)[:, None, None] * np.ones((51, 24, 3)) * .02
    wrists = np.ones((51, 6))
    np.savez(path, joints=joints, wrist_joints=wrists, root_quat=root, fps=50)
    m = Motion.load(path)
    assert m.reference_type == "smpl"
    np.testing.assert_allclose(np.array(m.sample(.1)["joints"])[:, 0, 0], .1 + np.arange(10) * .02)
    with pytest.raises(ValueError, match="frame counts"):
        SmplMotion(joints, root, wrists[:-1], 50)


def test_conflicting_explicit_type_cannot_reinterpret_release_data():
    clip = ROOT / "motions/example_h2_input/squat_001__A359"
    with pytest.raises(ValueError, match="conflicting"):
        Motion.load(clip, reference_type="g1")


def test_unlabelled_mixed_csv_requires_type_and_missing_native_h2_fails(tmp_path):
    q = np.zeros((3, 29))
    root = np.tile([1, 0, 0, 0], (3, 1))
    for name, data in (("joint_pos", q), ("body_quat", root),
                       ("h2_joint_pos", np.zeros((3, 31))), ("smpl_joint", np.zeros((3, 72)))):
        np.savetxt(tmp_path / f"{name}.csv", data, delimiter=",")
    with pytest.raises(ValueError, match="Ambiguous"):
        Motion.load(tmp_path)
    assert Motion.load(tmp_path, reference_type="h2").reference_type == "h2"
    (tmp_path / "h2_joint_pos.csv").unlink()
    (tmp_path / "metadata.txt").write_text("Reference type: h2\n")
    with pytest.raises(FileNotFoundError):
        Motion.load(tmp_path)


def test_csv_fps_metadata_override_and_zero_rejected(tmp_path):
    np.savetxt(tmp_path / "joint_pos.csv", np.zeros((101, 29)), delimiter=",")
    np.savetxt(tmp_path / "body_quat.csv", np.tile([1, 0, 0, 0], (101, 1)), delimiter=",")
    (tmp_path / "metadata.txt").write_text("Reference type: g1\nFPS: 100\n")
    assert Motion.load(tmp_path).duration == 1
    assert Motion.load(tmp_path, fps=50).duration == 2
    with pytest.raises(ValueError, match="fps must be positive"):
        Motion.load(tmp_path, fps=0)


def test_collection_root_and_corrupt_first_csv_row_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="one clip directory"):
        Motion.load(ROOT / "motions/example_h2_input")
    (tmp_path / "joint_pos.csv").write_text(
        "bad," + ",".join(["0"] * 28) + "\n" + (",".join(["0"] * 29) + "\n") * 3)
    np.savetxt(tmp_path / "body_quat.csv", np.tile([1, 0, 0, 0], (3, 1)), delimiter=",")
    with pytest.raises(ValueError, match="Invalid CSV header"):
        Motion.load(tmp_path)


@pytest.mark.parametrize("kind,collection", [("g1", "example_g1_input"), ("h2", "example_h2_input"), ("smpl", "example_smpl_input")])
def test_cli_sends_detected_type_and_inspect_never_connects(monkeypatch, kind, collection):
    import interface.replay as replay
    clip = ROOT / "motions" / collection / "squat_001__A359"
    messages = []

    class Client:
        def __init__(self, *args, **kwargs):
            messages.append("connected")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def send(self, command_kind, data):
            messages.append(envelope(command_kind, data, source="test", session="test", seq=1))

    monkeypatch.setattr(replay, "RuntimeClient", Client)
    monkeypatch.setattr("sys.argv", ["replay", str(clip), "--inspect"])
    replay.main()
    assert not messages
    times = iter([0, .01, 10000])
    monkeypatch.setattr(replay.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(replay.time, "sleep", lambda _: None)
    monkeypatch.setattr("sys.argv", ["replay", str(clip)])
    replay.main()
    assert messages[0] == "connected"
    assert messages[1]["kind"] == kind
