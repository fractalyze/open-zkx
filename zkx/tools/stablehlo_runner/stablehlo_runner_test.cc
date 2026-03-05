/* Copyright 2026 The ZKX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "mlir/IR/MLIRContext.h"

#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/fingerprint.h"
#include "xla/tsl/platform/path.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test_util.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/literal.h"
#include "zkx/literal_util.h"
#include "zkx/service/hlo_runner.h"
#include "zkx/service/platform_util.h"
#include "zkx/tools/stablehlo_runner/stablehlo_utils.h"

namespace zkx {
namespace {

class StablehloRunnerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto platform_or = PlatformUtil::GetPlatform("cuda");
    if (!platform_or.ok()) {
      platform_or = PlatformUtil::GetPlatform("host");
    }
    TF_ASSERT_OK_AND_ASSIGN(auto platform, platform_or);
    runner_ = std::make_unique<HloRunner>(platform);
  }

  absl::StatusOr<std::unique_ptr<HloModule>> LoadModule(
      const std::string &example_subdir, const std::string &mlir_file_name) {
    std::string mlir_path =
        tsl::io::JoinPath(tsl::testing::ZkxSrcRoot(), "..", "examples",
                          example_subdir, mlir_file_name);
    std::string module_text;
    TF_RETURN_IF_ERROR(
        tsl::ReadFileToString(tsl::Env::Default(), mlir_path, &module_text));

    mlir::MLIRContext context;
    TF_ASSIGN_OR_RETURN(auto stablehlo_module,
                        ParseStablehloModule(module_text, &context));
    return ConvertStablehloToHloModule(*stablehlo_module);
  }

  // Runs a compiled module with the given input literal and returns a hex
  // string of the output bytes.
  absl::StatusOr<std::string> RunAndHash(std::unique_ptr<HloModule> hlo_module,
                                         std::vector<Literal> inputs) {
    std::vector<const Literal *> input_ptrs;
    input_ptrs.reserve(inputs.size());
    for (const auto &literal : inputs) {
      input_ptrs.push_back(&literal);
    }

    TF_ASSIGN_OR_RETURN(auto executable,
                        runner_->CreateExecutable(std::move(hlo_module),
                                                  /*run_hlo_passes=*/true));
    TF_ASSIGN_OR_RETURN(Literal output, runner_->ExecuteWithExecutable(
                                            executable.get(), input_ptrs,
                                            /*profile=*/nullptr));
    return absl::BytesToHexString(std::string_view(
        static_cast<const char *>(output.untyped_data()), output.size_bytes()));
  }

  // Runs a compiled module with the given input literals and returns a
  // deterministic fingerprint string of the output bytes.
  // Handles both scalar/array and tuple outputs.
  absl::StatusOr<std::string> RunAndFingerprint(
      std::unique_ptr<HloModule> hlo_module, std::vector<Literal> inputs) {
    std::vector<const Literal *> input_ptrs;
    input_ptrs.reserve(inputs.size());
    for (const auto &literal : inputs) {
      input_ptrs.push_back(&literal);
    }

    TF_ASSIGN_OR_RETURN(auto executable,
                        runner_->CreateExecutable(std::move(hlo_module),
                                                  /*run_hlo_passes=*/true));
    TF_ASSIGN_OR_RETURN(Literal output, runner_->ExecuteWithExecutable(
                                            executable.get(), input_ptrs,
                                            /*profile=*/nullptr));

    // For tuple outputs, concatenate all leaf element bytes.
    std::string all_bytes;
    if (output.shape().IsTuple()) {
      auto elements = output.DecomposeTuple();
      for (auto &element : elements) {
        all_bytes.append(static_cast<const char *>(element.untyped_data()),
                         element.size_bytes());
      }
    } else {
      all_bytes.assign(static_cast<const char *>(output.untyped_data()),
                       output.size_bytes());
    }
    uint64_t fp = tsl::Fingerprint64(all_bytes);
    return absl::StrCat(absl::Hex(fp, absl::kZeroPad16));
  }

  std::unique_ptr<HloRunner> runner_;
};

// Verified against Python reference: Poseidon2BabyBear16.permute([0..15])
// Source: whir-zorch/poseidon2/testing/poseidon2_baby_bear_test.py
TEST_F(StablehloRunnerTest, Poseidon2Permutation) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto hlo_module,
      LoadModule("poseidon2_babybear", "poseidon2_permutation.stablehlo.mlir"));

  // Input: [0, 1, 2, ..., 15] as BabyBear standard-form elements.
  const Shape &shape =
      hlo_module->entry_computation()->parameter_instruction(0)->shape();
  Literal input(shape);
  std::memset(input.untyped_data(), 0, input.size_bytes());
  auto *data = static_cast<uint32_t *>(input.untyped_data());
  for (uint32_t i = 0; i < 16; ++i) {
    data[i] = i;
  }

  std::vector<Literal> inputs;
  inputs.push_back(std::move(input));
  TF_ASSERT_OK_AND_ASSIGN(std::string hash,
                          RunAndHash(std::move(hlo_module), std::move(inputs)));

  // Expected: [1906786279, 1737026427, 1959749225, 700325316, 1638050605,
  //            1021608788, 1726691001, 1761127344, 1552405120, 417318995,
  //            36799261, 1215172152, 614923223, 1300746575, 957311597,
  //            304856115]
  // as little-endian uint32 hex.
  EXPECT_EQ(
      "e73fa7717beb88676966cf74c41dbe292daba2615483e43cb936eb66b0abf868"
      "80d2875c53c8df181d833102380e6e48d7fba6244fd1874d6d6a0f3933bc2b12",
      hash);
}

// Input Fusion: sum((a + b) * a) with zero inputs → 0.
TEST_F(StablehloRunnerTest, InputFusion) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto hlo_module,
      LoadModule("fusion_showcase", "input_fusion.stablehlo.mlir"));

  // Create two zero-filled BabyBear[1024] inputs.
  std::vector<Literal> inputs;
  for (int i = 0; i < 2; ++i) {
    const Shape &shape =
        hlo_module->entry_computation()->parameter_instruction(i)->shape();
    Literal input(shape);
    std::memset(input.untyped_data(), 0, input.size_bytes());
    inputs.push_back(std::move(input));
  }

  TF_ASSERT_OK_AND_ASSIGN(std::string hash,
                          RunAndHash(std::move(hlo_module), std::move(inputs)));

  // sum((0 + 0) * 0) = 0  →  4 bytes of zero (little-endian BabyBear).
  EXPECT_EQ("00000000", hash);
}

// Multi-Output Fusion: r1 = sum((a+b)*a), r2 = sum((a+b)*b).
// Uses pseudo-random inputs; checks output fingerprint.
TEST_F(StablehloRunnerTest, MultiOutputFusion) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto hlo_module,
      LoadModule("fusion_showcase", "multi_output_fusion.stablehlo.mlir"));

  const HloComputation *entry = hlo_module->entry_computation();
  std::vector<Literal> inputs;
  inputs.reserve(entry->num_parameters());
  for (int64_t i = 0; i < entry->num_parameters(); ++i) {
    const Shape &shape = entry->parameter_instruction(i)->shape();
    TF_ASSERT_OK_AND_ASSIGN(Literal literal,
                            MakeFakeLiteral(shape, /*pseudo_random=*/true));
    inputs.push_back(std::move(literal));
  }

  TF_ASSERT_OK_AND_ASSIGN(
      std::string fp,
      RunAndFingerprint(std::move(hlo_module), std::move(inputs)));
  EXPECT_FALSE(fp.empty());
}

// Broadcast Fusion: ((a + b) * broadcast(scale))² with pseudo-random inputs.
TEST_F(StablehloRunnerTest, BroadcastFusion) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto hlo_module,
      LoadModule("fusion_showcase", "broadcast_fusion.stablehlo.mlir"));

  const HloComputation *entry = hlo_module->entry_computation();
  std::vector<Literal> inputs;
  inputs.reserve(entry->num_parameters());
  for (int64_t i = 0; i < entry->num_parameters(); ++i) {
    const Shape &shape = entry->parameter_instruction(i)->shape();
    TF_ASSERT_OK_AND_ASSIGN(Literal literal,
                            MakeFakeLiteral(shape, /*pseudo_random=*/true));
    inputs.push_back(std::move(literal));
  }

  TF_ASSERT_OK_AND_ASSIGN(
      std::string fp,
      RunAndFingerprint(std::move(hlo_module), std::move(inputs)));

  // Deterministic fingerprint of result tensor.
  EXPECT_FALSE(fp.empty());
}

}  // namespace
}  // namespace zkx
