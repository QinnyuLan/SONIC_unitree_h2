#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace h2_native_contract {

struct ModelShape {
  std::array<int64_t, 2> input;
  std::array<int64_t, 2> output;
};

inline std::string ShapeString(const std::vector<int64_t>& shape) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) oss << ",";
    oss << shape[i];
  }
  oss << "]";
  return oss.str();
}

inline void ValidateShape(
    const std::vector<int64_t>& actual,
    const std::array<int64_t, 2>& expected,
    const std::string& model_path,
    const char* tensor_kind) {
  if (actual.size() != expected.size() ||
      !std::equal(actual.begin(), actual.end(), expected.begin())) {
    std::ostringstream oss;
    oss << "H2-native " << tensor_kind << " shape mismatch for " << model_path
        << ": got " << ShapeString(actual) << ", expected ["
        << expected[0] << "," << expected[1] << "]";
    throw std::runtime_error(oss.str());
  }
}

inline std::string ReadMetadata(
    const Ort::Session& session,
    Ort::AllocatorWithDefaultOptions& allocator,
    const char* key) {
  const auto metadata = session.GetModelMetadata();
  const auto value = metadata.LookupCustomMetadataMapAllocated(key, allocator);
  return value ? std::string(value.get()) : std::string();
}

inline void ValidateMetadata(
    const Ort::Session& session,
    Ort::AllocatorWithDefaultOptions& allocator,
    const std::string& model_path,
    const std::vector<std::pair<const char*, const char*>>& required) {
  for (const auto& [key, expected] : required) {
    const std::string actual = ReadMetadata(session, allocator, key);
    if (actual != expected) {
      std::ostringstream oss;
      oss << "H2-native ONNX metadata mismatch for " << model_path << ": "
          << key << "='" << (actual.empty() ? "<missing>" : actual)
          << "', expected '" << expected << "'";
      throw std::runtime_error(oss.str());
    }
  }
}

inline void ValidateModel(
    Ort::Env& env,
    Ort::AllocatorWithDefaultOptions& allocator,
    const std::string& model_path,
    const ModelShape& expected,
    const std::vector<std::pair<const char*, const char*>>& required_metadata) {
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(1);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
  const Ort::Session session(env, model_path.c_str(), options);
  if (session.GetInputCount() != 1 || session.GetOutputCount() != 1) {
    std::ostringstream oss;
    oss << "H2-native ONNX must have exactly one input and one output: "
        << model_path << " has " << session.GetInputCount() << " input(s), "
        << session.GetOutputCount() << " output(s)";
    throw std::runtime_error(oss.str());
  }
  ValidateShape(
      session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape(),
      expected.input, model_path, "input");
  ValidateShape(
      session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape(),
      expected.output, model_path, "output");
  ValidateMetadata(session, allocator, model_path, required_metadata);
}

inline void ValidateDeploymentModels(
    const std::string& encoder_path,
    const std::string& decoder_path,
    const std::string& reference_orientation_contract = "") {
  if (encoder_path.empty()) {
    throw std::runtime_error("H2-native profile requires an encoder ONNX file");
  }
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "h2_native_contract");
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(1);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
  const Ort::Session encoder_session(env, encoder_path.c_str(), options);
  const auto encoder_input_shape = encoder_session.GetInputTypeInfo(0)
                                       .GetTensorTypeAndShapeInfo()
                                       .GetShape();
  const bool h2_only_encoder =
      encoder_input_shape == std::vector<int64_t>({1, 640});
  const bool universal_encoder =
      encoder_input_shape == std::vector<int64_t>({1, 1751});
  if (!h2_only_encoder && !universal_encoder) {
    throw std::runtime_error(
        "H2-native encoder input must be [1,640] (H2-only) or [1,1751] "
        "(universal), got " + ShapeString(encoder_input_shape));
  }

  std::vector<std::pair<const char*, const char*>> encoder_metadata = {
      {"policy_interface", h2_only_encoder ? "h2_native" : "universal"},
      {"model_role", "encoder"},
      {"observation_contract", "sonic_h2_native_v1"},
  };
  // Deployment depends on the exported native H2 interface. Training method
  // and adapter-folding tags do not define this interface: a directly
  // fine-tuned model needs no LoRA merge or input/output adapter folding.
  if (universal_encoder) {
    const std::string selector_contract =
        ReadMetadata(encoder_session, allocator, "selector_contract");
    const std::string selector_names =
        ReadMetadata(encoder_session, allocator, "selector_encoder_names");
    const bool full_selector =
        selector_contract == "sonic_h2_native_full_v1" &&
        selector_names == "[\"h2\", \"teleop\", \"smpl\", \"g1\"]";
    if (reference_orientation_contract == "sonic_full_root_relative_v0" && !full_selector) {
      throw std::runtime_error(
          "Non-v1.1 H2-native universal deployment requires the full four-way "
          "selector [h2,teleop,smpl,g1] with native G1 mode 3");
    }
    if (full_selector) {
      encoder_metadata.emplace_back(
          "selector_contract", "sonic_h2_native_full_v1");
      encoder_metadata.emplace_back(
          "selector_encoder_names", "[\"h2\", \"teleop\", \"smpl\", \"g1\"]");
    } else {
      encoder_metadata.emplace_back(
          "selector_contract", "sonic_h2_native_universal_v1");
      encoder_metadata.emplace_back(
          "selector_encoder_names", "[\"h2\", \"teleop\", \"smpl\"]");
    }
  }
  std::vector<std::pair<const char*, const char*>> decoder_metadata = {
      {"policy_interface", "h2_native"},
      {"model_role", "decoder"},
      {"observation_contract", "sonic_h2_native_v1"},
  };
  if (!reference_orientation_contract.empty()) {
    encoder_metadata.emplace_back(
        "reference_orientation_contract",
        reference_orientation_contract.c_str());
    decoder_metadata.emplace_back(
        "reference_orientation_contract",
        reference_orientation_contract.c_str());
  }
  ValidateModel(
      env,
      allocator,
      encoder_path,
      {h2_only_encoder ? std::array<int64_t, 2>{1, 640}
                       : std::array<int64_t, 2>{1, 1751},
       {1, 64}},
      encoder_metadata);
  ValidateModel(
      env,
      allocator,
      decoder_path,
      {{1, 994}, {1, 29}},
      decoder_metadata);
}

}  // namespace h2_native_contract
