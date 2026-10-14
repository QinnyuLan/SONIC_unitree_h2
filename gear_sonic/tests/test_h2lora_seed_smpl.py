import joblib
import numpy as np
import torch

from gear_sonic.data_process import build_h2lora_seed_motion_lib as builder
from gear_sonic.isaac_utils import rotations
from gear_sonic.trl.utils import torch_transform


class DummySmplModel:
    def __init__(self):
        self.global_orient = None
        self.transl = None

    def __call__(self, body_pose, betas, global_orient, transl):
        assert body_pose.shape == (1, 4, 63)
        assert betas.shape == (1, 4, 10)
        assert transl.shape == (1, 4, 3)
        self.global_orient = global_orient.detach().cpu()
        self.transl = transl.detach().cpu()
        joints = transl.unsqueeze(-2).expand(-1, -1, 24, -1).contiguous()
        return joints, joints


def dummy_human_joints(body_pose, global_orient):
    joints = global_orient.unsqueeze(1).expand(-1, 24, -1).contiguous()
    joints[:, 1:, :] += 1.0
    return joints


def test_smpl_root_axis_angle_yup_to_zup_matches_runtime_base_rotation():
    converted = builder._smpl_root_axis_angle_yup_to_zup(np.zeros((1, 3), dtype=np.float32))
    expected = np.asarray([[np.pi / 2.0, 0.0, 0.0]], dtype=np.float32)
    assert np.allclose(converted, expected, atol=1e-6)


def test_human_fk_joints_are_zup_root_centered_and_ignore_translation(monkeypatch):
    monkeypatch.setattr(builder, "_SMPL_MODE", "human-fk")
    monkeypatch.setattr(builder, "_HUMAN_JOINTS_FN", dummy_human_joints)
    monkeypatch.setattr(builder, "_SMPL_DEVICE", "cpu")
    monkeypatch.setattr(builder, "_SMPL_BATCH_FRAMES", 8)

    poses = np.zeros((4, 77, 3), dtype=np.float32)
    transl = np.asarray(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        dtype=np.float32,
    )
    betas = np.zeros((1, 10), dtype=np.float32)

    joints = builder._smpl_joints_24(poses, transl, betas)

    assert joints.shape == (4, 24, 3)
    assert np.allclose(joints[:, 0, :], 0.0, atol=1e-6)
    assert np.allclose(joints[:, 1:, :], 1.0, atol=1e-6)
    assert not np.allclose(joints[:, 0, :], builder._smpl_yup_points_to_zup(transl), atol=1e-6)


def test_smplx_fk_joints_are_zup_root_centered_and_ignore_translation(monkeypatch):
    dummy_model = DummySmplModel()
    monkeypatch.setattr(builder, "_SMPL_MODE", "smplx-fk")
    monkeypatch.setattr(builder, "_SMPL_MODEL", dummy_model)
    monkeypatch.setattr(builder, "_SMPL_DEVICE", "cpu")
    monkeypatch.setattr(builder, "_SMPL_BATCH_FRAMES", 8)

    poses = np.zeros((4, 77, 3), dtype=np.float32)
    transl = np.asarray(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        dtype=np.float32,
    )
    betas = np.zeros((1, 10), dtype=np.float32)

    joints = builder._smpl_joints_24(poses, transl, betas)

    assert joints.shape == (4, 24, 3)
    assert np.allclose(joints[:, 0, :], 0.0, atol=1e-6)
    assert torch.allclose(
        dummy_model.global_orient,
        torch.tensor([[[np.pi / 2.0, 0.0, 0.0]]]).expand(1, 4, 3),
        atol=1e-6,
    )
    assert torch.allclose(dummy_model.transl, torch.zeros(1, 4, 3))


def test_load_smplx_npz_records_schema6_pelvis_centered_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "_SMPL_MODE", "human-fk")
    monkeypatch.setattr(builder, "_HUMAN_JOINTS_FN", dummy_human_joints)
    monkeypatch.setattr(builder, "_SMPL_DEVICE", "cpu")
    monkeypatch.setattr(builder, "_SMPL_BATCH_FRAMES", 8)

    npz_path = tmp_path / "motion.npz"
    np.savez(
        npz_path,
        poses=np.zeros((4, 77, 3), dtype=np.float32),
        transl=np.ones((4, 3), dtype=np.float32),
        identity_coeffs=np.zeros((1, 10), dtype=np.float32),
    )

    entry = builder.load_smplx_npz(npz_path, fps_source=120, fps_target=60)

    assert entry["schema_version"] == 6
    assert entry["smpl_joints_frame"] == "z_up_pelvis_centered_world_oriented_no_translation"
    assert entry["smpl_joints_root_center"] == "joint0/pelvis subtracted after FK"
    assert np.allclose(entry["smpl_joints"][:, 0, :], 0.0, atol=1e-6)


def test_soma_bvh_mode_uses_name_mapped_smpl24_and_root_only_pose(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "_SMPL_MODE", "soma-bvh")
    monkeypatch.setattr(builder, "_SMPL_ROOT_MODE", "raw-bvh")
    monkeypatch.setattr(builder, "_SMPL_BVH_ROOT", tmp_path)

    bvh_path = tmp_path / "session" / "sample.bvh"
    bvh_path.parent.mkdir()
    bvh_path.write_text("dummy")

    positions = np.zeros((4, len(builder.SMPL24_FROM_SOMA_NAMES), 3), dtype=np.float32)
    for idx in range(len(builder.SMPL24_FROM_SOMA_NAMES)):
        positions[:, idx, :] = np.asarray([idx, 2 * idx, 3 * idx], dtype=np.float32)
    positions[:, 0, :] = np.asarray([10.0, 20.0, 30.0], dtype=np.float32)
    root_quat = np.tile(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (4, 1))

    monkeypatch.setattr(
        builder,
        "_resolve_soma_bvh_path",
        lambda npz_path, source_bvh_relpath: bvh_path,
    )

    import gear_sonic.data_process.extract_soma_joints_from_bvh as soma_extract

    monkeypatch.setattr(
        soma_extract,
        "parse_bvh",
        lambda path: ([], [], np.zeros((4, 1), dtype=np.float32), 4, 1.0 / 120.0),
    )
    monkeypatch.setattr(
        soma_extract,
        "compute_fk_selected",
        lambda joints, channel_order, motion_data, selected_names: (
            positions,
            root_quat,
        ),
    )

    npz_path = tmp_path / "motion.npz"
    np.savez(
        npz_path,
        poses=np.full((4, 77, 3), 999.0, dtype=np.float32),
        transl=np.full((4, 3), 999.0, dtype=np.float32),
        source_bvh_relpath=np.array("session/sample.bvh"),
    )

    entry = builder.load_smplx_npz(npz_path, fps_source=120, fps_target=120)

    assert entry["schema_version"] == 6
    assert entry["smpl_joints_source"] == "soma_bvh_fk_name_mapped_to_smpl24"
    assert entry["smpl_pose_body_source"] == "zeros_nonroot_because_seed_smplx_poses_are_somax_77_not_smpl24"
    assert np.allclose(entry["pose_aa"][:, 3:], 0.0, atol=1e-6)
    assert np.allclose(entry["smpl_joints"][:, 0, :], 0.0, atol=1e-6)
    # The second mapped joint starts at [1,2,3] cm, subtracts Hips
    # [10,20,30] cm, then converts Y-up -> Z-up.
    expected_joint1 = np.asarray([-0.09, 0.27, -0.18], dtype=np.float32)
    assert np.allclose(entry["smpl_joints"][0, 1, :], expected_joint1, atol=1e-6)


def test_soma_bvh_compatible_root_matches_runtime_bvh_root(monkeypatch):
    monkeypatch.setattr(builder, "_SMPL_ROOT_MODE", "soma-bvh-compatible")
    bvh_root_xyzw = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.142127, -0.071063, 0.213190, 0.963968],
        ],
        dtype=np.float32,
    )
    bvh_root_xyzw /= np.linalg.norm(bvh_root_xyzw, axis=-1, keepdims=True)

    stored_rotvec = builder._soma_bvh_root_quat_to_smpl_pose_rotvec(bvh_root_xyzw)
    stored_root_wxyz = torch_transform.angle_axis_to_quaternion(torch.from_numpy(stored_rotvec))
    bvh_root_wxyz = torch.from_numpy(bvh_root_xyzw[:, [3, 0, 1, 2]])

    smpl_runtime = rotations.remove_smpl_base_rot(
        rotations.smpl_root_ytoz_up(stored_root_wxyz),
        w_last=False,
    )
    bvh_runtime = rotations.remove_bvh_base_rot(
        rotations.smpl_root_ytoz_up(bvh_root_wxyz),
        w_last=False,
    )

    # q and -q represent the same rotation; compare sign-invariant alignment.
    alignment = torch.sum(smpl_runtime * bvh_runtime, dim=-1).abs()
    assert torch.allclose(alignment, torch.ones_like(alignment), atol=1e-5)


def test_skip_existing_requires_current_smpl_schema(tmp_path):
    key = "sample_motion"
    old_path = tmp_path / "old.pkl"
    current_path = tmp_path / "current.pkl"
    wrong_fps_path = tmp_path / "wrong_fps.pkl"
    wrong_root_path = tmp_path / "wrong_root.pkl"

    joblib.dump({key: {"fps": 50, "smpl_joints": np.zeros((1, 24, 3), dtype=np.float32)}}, old_path)
    joblib.dump(
        {
            key: {
                "schema_version": builder.H2LORA_SEED_SMPL_SCHEMA_VERSION,
                "fps": 50,
                "smpl_joints_source": "soma_bvh_fk_name_mapped_to_smpl24",
                "smpl_root_mode": "soma-bvh-compatible",
            }
        },
        current_path,
    )
    joblib.dump(
        {
            key: {
                "schema_version": builder.H2LORA_SEED_SMPL_SCHEMA_VERSION,
                "fps": 30,
                "smpl_joints_source": "soma_bvh_fk_name_mapped_to_smpl24",
                "smpl_root_mode": "soma-bvh-compatible",
            }
        },
        wrong_fps_path,
    )
    joblib.dump(
        {
            key: {
                "schema_version": builder.H2LORA_SEED_SMPL_SCHEMA_VERSION,
                "fps": 50,
                "smpl_joints_source": "soma_bvh_fk_name_mapped_to_smpl24",
                "smpl_root_mode": "raw-bvh",
            }
        },
        wrong_root_path,
    )

    assert not builder._existing_smpl_schema_is_current(old_path, key)
    assert builder._existing_smpl_schema_is_current(
        current_path,
        key,
        fps_target=50,
        smpl_mode="soma-bvh",
        smpl_root_mode="soma-bvh-compatible",
    )
    assert not builder._existing_smpl_schema_is_current(
        wrong_fps_path,
        key,
        fps_target=50,
        smpl_mode="soma-bvh",
        smpl_root_mode="soma-bvh-compatible",
    )
    assert not builder._existing_smpl_schema_is_current(
        wrong_root_path,
        key,
        fps_target=50,
        smpl_mode="soma-bvh",
        smpl_root_mode="soma-bvh-compatible",
    )


def test_write_manifest_records_selected_frame_counts(tmp_path):
    class Args:
        seed_root = "/seed"
        output_root = tmp_path
        fps_source = 120
        fps = 50
        selection_mode = "sorted"
        random_seed = 0
        target_frames = None
        smpl_mode = "soma-bvh"
        smpl_bvh_root = "/seed/soma_uniform/bvh"
        smpl_root_mode = "soma-bvh-compatible"

    builder._write_manifest(
        tmp_path,
        ["walk_forward_stop_001__A039", "idle_pose_001__A001"],
        Args,
        selected_frame_counts={
            "walk_forward_stop_001__A039": 371,
            "idle_pose_001__A001": 100,
        },
        total_paired_motions=2,
    )

    import json

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["smpl_schema_version"] == builder.H2LORA_SEED_SMPL_SCHEMA_VERSION
    assert manifest["selected_total_frames"] == 471
    assert manifest["category_motion_counts"] == {"idle": 1, "walk": 1}
