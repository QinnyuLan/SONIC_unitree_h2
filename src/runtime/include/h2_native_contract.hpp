#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <inference/picosha2.h>

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

// The release manifest records ONNX interfaces and selected metadata. Verify the
// model digest before trusting those records; TensorRT validates the executable
// tensor names, types and dimensions when loading the engine.
struct ModelInfo {
  std::string path;
  nlohmann::json record;
};

inline ModelInfo ReadModel(const std::string& model_path) {
  const std::filesystem::path path(model_path);
  std::ifstream manifest(path.parent_path() / "deployment_manifest.json");
  if (!manifest) throw std::runtime_error("Missing deployment_manifest.json beside " + model_path);
  const auto data = nlohmann::json::parse(manifest);
  std::ifstream model(path, std::ios::binary);
  if (!model) throw std::runtime_error("Cannot read model " + model_path);
  const auto expected = data.at("model_sha256").at(path.stem().string()).get<std::string>();
  const auto actual = picosha2::hash256_hex_string(
      std::istreambuf_iterator<char>(model), std::istreambuf_iterator<char>());
  if (actual != expected) throw std::runtime_error("Model SHA256 mismatch: " + model_path);
  return {model_path, data.at("onnx_check").at(path.filename().string())};
}

inline std::string ReadMetadata(const ModelInfo& model, const char* key) {
  return model.record.at("metadata").value(key, std::string());
}

inline std::vector<int64_t> TensorShape(const ModelInfo& model, const char* kind) {
  const auto& tensors = model.record.at(kind);
  if (tensors.size() != 1 || !tensors.at(0).is_array() || tensors.at(0).size() != 3)
    throw std::runtime_error("Expected one rank-two tensor: " + model.path);
  return {tensors.at(0).at(1).get<int64_t>(), tensors.at(0).at(2).get<int64_t>()};
}

inline void ValidateModel(
    const ModelInfo& model,
    const ModelShape& expected,
    const std::vector<std::pair<const char*, const char*>>& required_metadata) {
  ValidateShape(TensorShape(model, "inputs"), expected.input, model.path, "input");
  ValidateShape(TensorShape(model, "outputs"), expected.output, model.path, "output");
  if (model.record.at("input_dtype") != "float32" || model.record.at("output_dtype") != "float32")
    throw std::runtime_error("Expected float32 model tensors: " + model.path);
  for (const auto& [key, expected_value] : required_metadata) {
    if (ReadMetadata(model, key) != expected_value)
      throw std::runtime_error("Model contract mismatch for " + model.path + ": " + key);
  }
}

inline void ValidateDeploymentModels(
    const std::string& encoder_path,
    const std::string& decoder_path,
    const std::string& reference_orientation_contract = "") {
  if (encoder_path.empty()) {
    throw std::runtime_error("H2-native profile requires an encoder ONNX file");
  }
  const auto encoder = ReadModel(encoder_path);
  const auto decoder = ReadModel(decoder_path);
  const auto encoder_input_shape = TensorShape(encoder, "inputs");
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
        ReadMetadata(encoder, "selector_contract");
    const std::string selector_names =
        ReadMetadata(encoder, "selector_encoder_names");
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
      encoder,
      {h2_only_encoder ? std::array<int64_t, 2>{1, 640}
                       : std::array<int64_t, 2>{1, 1751},
       {1, 64}},
      encoder_metadata);
  ValidateModel(
      decoder,
      {{1, 994}, {1, 29}},
      decoder_metadata);
}

}  // namespace h2_native_contract
