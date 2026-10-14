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
  EXPECT_THROW(h2_native_contract::ValidateModel(
      h2_native_contract::ReadModel(kDecoder), {{1, 994}, {1, 29}}, {{"model_role", "encoder"}}),
      std::runtime_error);
}

TEST(H2ReleasedContract, RejectsModelBytesThatDoNotMatchManifest) {
  const auto dir = std::filesystem::temp_directory_path() / "sonic-contract-digest-test";
  std::filesystem::create_directories(dir);
  const auto model = dir / "model.onnx";
  std::ofstream(model) << "changed model bytes";
  nlohmann::json manifest = {{"model_sha256", {{"model", std::string(64, '0')}}}};
  std::ofstream(dir / "deployment_manifest.json") << manifest.dump();
  EXPECT_THROW(h2_native_contract::ReadModel(model.string()), std::runtime_error);
  std::filesystem::remove_all(dir);
}
