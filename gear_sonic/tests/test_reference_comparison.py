import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.render_g1_h2_reference_comparison import (
    G1_ASSET, G1_JOINT_NAMES, H2_ASSET, H2_JOINT_NAMES, Reference,
    joint_qpos_addresses, render_model_xml,
)


class ReferenceComparisonTest(unittest.TestCase):
    def test_alignment_preserves_height_and_joint_values(self):
        q = np.arange(58).reshape(2, 29).astype(float)
        quat = Rotation.from_euler("z", [[90], [110]], degrees=True).as_quat()[:, [3, 0, 1, 2]]
        original = Reference(q, np.array([[3., 4., .9], [3., 5., .5]]), quat, 50)
        aligned = original.aligned()
        np.testing.assert_array_equal(aligned.joints, q)
        np.testing.assert_allclose(aligned.root_pos, [[0, 0, .9], [1, 0, .5]], atol=1e-12)
        np.testing.assert_allclose(aligned.root_quat[0], [1, 0, 0, 0], atol=1e-12)
        np.testing.assert_array_equal(original.root_pos[0], [3, 4, .9])

    def test_sample_uses_time_not_normalized_motion_progress(self):
        q = np.array([[0.], [1.], [2.]])
        reference = Reference(q, np.zeros((3, 3)), np.tile([1., 0, 0, 0], (3, 1)), 50)
        values, _, quat = reference.sample(np.array([0., .01, .02, .04, .06]))
        np.testing.assert_allclose(values[:, 0], [0, .5, 1, 2, 2])
        np.testing.assert_allclose(np.linalg.norm(quat, axis=1), 1)

    def test_quaternion_sampling_handles_sign_equivalence(self):
        reference = Reference(np.zeros((2, 1)), np.zeros((2, 3)),
                              np.array([[1., 0, 0, 0], [-1., 0, 0, 0]]), 50)
        _, _, quat = reference.sample(np.array([.01]))
        np.testing.assert_allclose(np.abs(quat), [[1, 0, 0, 0]], atol=1e-12)

    def test_invalid_references_fail_before_rendering(self):
        with self.assertRaises(ValueError):
            Reference(np.zeros((2, 29)), np.zeros((1, 3)), np.zeros((2, 4)), 50)
        with self.assertRaises(ValueError):
            Reference(np.zeros((2, 29)), np.zeros((2, 3)), np.zeros((2, 4)), 50)

    def test_native_csv_order_maps_to_named_asset_joints(self):
        import mujoco

        for asset, names, count in ((G1_ASSET, G1_JOINT_NAMES, 29), (H2_ASSET, H2_JOINT_NAMES, 31)):
            model = mujoco.MjModel.from_xml_string(render_model_xml(asset))
            addresses = joint_qpos_addresses(model, names)
            self.assertEqual(len(set(addresses)), count)
            data = mujoco.MjData(model)
            values = np.arange(count) * .01
            data.qpos[addresses] = values
            for name, expected in zip(names, values):
                self.assertAlmostEqual(data.joint(name).qpos[0], expected)
        self.assertLess(H2_JOINT_NAMES.index("left_ankle_roll_joint"), H2_JOINT_NAMES.index("left_ankle_pitch_joint"))


if __name__ == "__main__":
    unittest.main()
