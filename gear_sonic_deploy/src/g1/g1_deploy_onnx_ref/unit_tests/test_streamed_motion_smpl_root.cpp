#include "gtest/gtest.h"

#include "motion_data_reader.hpp"
#include "input_interface/streamed_motion_merger.hpp"

TEST(StreamedMotionMerger, PreservesSmplRootQuaternion) {
  StreamedMotionMerger merger;
  StreamedMotionMerger::IncomingData data;
  data.protocol_version = 3;
  data.num_frames = 2;
  data.num_joints = 29;
  data.num_quat_bodies = 1;
  data.num_smpl_joints = 24;
  data.num_smpl_poses = 21;
  data.num_smpl_root_quaternions = 1;
  data.frame_indices = {10, 11};
  data.joint_pos.assign(2, std::vector<double>(29, 0.0));
  data.joint_vel.assign(2, std::vector<double>(29, 0.0));
  data.body_quat = {{{1.0, 0.0, 0.0, 0.0}}, {{1.0, 0.0, 0.0, 0.0}}};
  data.smpl_joints.assign(2, std::vector<std::array<double, 3>>(24));
  data.smpl_pose.assign(2, std::vector<std::array<double, 3>>(21));
  data.smpl_root_quat = {
      {{0.9, 0.1, 0.2, 0.3}},
      {{0.8, -0.1, 0.4, 0.2}},
  };

  const auto result = merger.MergeIncomingData(data, 0);

  ASSERT_NE(result.motion, nullptr);
  EXPECT_EQ(result.motion->GetNumSmplRootQuaternions(), 1);
  EXPECT_EQ(result.motion->timesteps, 2);
  EXPECT_EQ(result.motion->SmplRootQuaternions(0)[0], data.smpl_root_quat[0][0]);
  EXPECT_EQ(result.motion->SmplRootQuaternions(1)[0], data.smpl_root_quat[1][0]);
}
