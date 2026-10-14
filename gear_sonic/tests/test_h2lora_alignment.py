import torch

from gear_sonic.envs.env_utils import h2lora_alignment, joint_utils


class DummyAsset:
    def __init__(self):
        self.joint_names = [
            "head_pitch_joint",
            *joint_utils.H2LORA_BODY_JOINTS_IN_G1_ORDER,
            "head_yaw_joint",
        ]
        self.device = "cpu"
        self.name = "dummy_h2"


def test_axis_audit():
    h2lora_alignment.assert_axis_audit()


def test_alignment_composition_matches_paper_projection():
    torch.manual_seed(0)
    values = torch.randn(64, 29)
    h2 = h2lora_alignment.g1_to_h2_physical(values)
    g1 = h2lora_alignment.h2_to_g1_semantic(h2)
    hip_indices = {
        idx
        for pair in joint_utils.HIP_JOINT_PAIRS.values()
        for idx in (joint_utils.H2LORA_BODY_JOINTS_IN_G1_ORDER.index(pair[0]), joint_utils.H2LORA_BODY_JOINTS_IN_G1_ORDER.index(pair[1]))
    }
    non_hip_indices = [idx for idx in range(values.shape[-1]) if idx not in hip_indices]
    assert torch.allclose(values[:, non_hip_indices], g1[:, non_hip_indices], atol=1e-6)

    matrices = h2lora_alignment.hip_alignment_matrices()
    for side, pair in joint_utils.HIP_JOINT_PAIRS.items():
        idx = [
            joint_utils.H2LORA_BODY_JOINTS_IN_G1_ORDER.index(pair[0]),
            joint_utils.H2LORA_BODY_JOINTS_IN_G1_ORDER.index(pair[1]),
        ]
        projection = matrices[side]["target_to_source"] @ matrices[side]["source_to_target"]
        expected = values[:, idx] @ projection.T
        assert torch.allclose(g1[:, idx], expected, atol=1e-6)


def test_h2lora_joint_indices_are_body_plus_head():
    asset = DummyAsset()
    body_indices = joint_utils.get_h2lora_body_joint_indices(asset)
    head_indices = joint_utils.get_h2_head_joint_indices(asset)
    assert body_indices.numel() == 29
    assert head_indices.numel() == 2
    assert asset.joint_names[int(body_indices[0])] == "left_hip_pitch_joint"
    assert [asset.joint_names[int(i)] for i in head_indices] == [
        "head_pitch_joint",
        "head_yaw_joint",
    ]


def test_full_h2_isaaclab_to_h2lora_headless_indices():
    full = torch.arange(31)
    headless = full[joint_utils.H2LORA_FULL_H2_ISAACLAB_DOF_INDICES]
    assert headless.numel() == 29
    assert 11 not in headless.tolist()
    assert 16 not in headless.tolist()
    assert headless[13].item() == 19
    assert headless[17].item() == 14
