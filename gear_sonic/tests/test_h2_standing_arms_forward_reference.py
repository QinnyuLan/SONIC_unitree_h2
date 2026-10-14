import unittest

import numpy as np

from scripts.make_h2_standing_arms_forward_reference import build_reference_pose


class H2StandingArmsForwardReferenceTest(unittest.TestCase):
    def test_reference_is_symmetric_and_geometrically_straight(self):
        joints, geometry = build_reference_pose()
        self.assertEqual(joints.shape, (31,))
        for side in ("left", "right"):
            np.testing.assert_allclose(
                geometry[side]["shoulder_to_wrist_direction"],
                [1.0, 0.0, 0.0],
                atol=1e-6,
            )
            self.assertAlmostEqual(geometry[side]["elbow_internal_angle_deg"], 180.0, places=5)
        self.assertAlmostEqual(joints[17], joints[24])
        self.assertAlmostEqual(joints[18], -joints[25])
        self.assertAlmostEqual(joints[19], -joints[26])
        self.assertAlmostEqual(joints[20], joints[27])


if __name__ == "__main__":
    unittest.main()
