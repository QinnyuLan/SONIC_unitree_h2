#include <gtest/gtest.h>

#include "../include/h2_native_contract.hpp"

namespace {
constexpr auto kFull = "../models/released/model_encoder_full.onnx";
constexpr auto kCompact = "../models/released/model_encoder.onnx";
constexpr auto kDecoder = "../models/released/model_decoder.onnx";
constexpr auto kOrientation = "sonic_full_root_relative_v0";
}

TEST(H2ReleasedContract, AcceptsDirectFullEncoderAndDecoder) {
  EXPECT_NO_THROW(h2_native_contract::ValidateDeploymentModels(kFull, kDecoder, kOrientation));
}

TEST(H2ReleasedContract, AcceptsDirectCompactEncoderAndDecoder) {
  EXPECT_NO_THROW(h2_native_contract::ValidateDeploymentModels(kCompact, kDecoder, kOrientation));
}

TEST(H2ReleasedContract, RejectsSwappedEncoderAndDecoder) {
  EXPECT_THROW(h2_native_contract::ValidateDeploymentModels(kDecoder, kFull, kOrientation),
               std::runtime_error);
}

TEST(H2ReleasedContract, RejectsIncompatibleOrientation) {
  EXPECT_THROW(h2_native_contract::ValidateDeploymentModels(kFull, kDecoder, "sonic_robot_heading_v1"),
               std::runtime_error);
}

TEST(H2ReleasedContract, RejectsIncorrectRoleEvenWhenDimensionsMatch) {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "released_contract_test");
  Ort::AllocatorWithDefaultOptions allocator;
  EXPECT_THROW(h2_native_contract::ValidateModel(
      env, allocator, kDecoder, {{1, 994}, {1, 29}}, {{"model_role", "encoder"}}),
      std::runtime_error);
}
