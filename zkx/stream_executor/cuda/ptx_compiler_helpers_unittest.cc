/* Copyright 2024 The OpenXLA Authors.
Copyright 2025 The ZKX Authors.

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

#include "zkx/stream_executor/cuda/ptx_compiler_helpers.h"

#include "absl/status/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace stream_executor {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;

// When the compilation succeeds, then the error log is empty.
constexpr std::string_view kPtxasLogSuccessfulCompilation = R"(
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function 'input_concatenate_fusion' for 'sm_80'
ptxas info    : Function properties for input_concatenate_fusion
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, 368 bytes cmem[0]
)";

constexpr std::string_view kPtxasLogTooOldError = R"(
// Something in the log before the error.
ptxas fatal   : Value 'sm_80' is not defined for option 'gpu-name'
ptxas fatal   : Ptx assembly aborted due to errors
// Something in the log after the error.
)";

constexpr std::string_view kPtxasLogRegisterAllocationError = R"(
// Something in the log before the error.
ptxas fatal   : (C7600) Register allocation failed with register count of '64'. Compile the program with a higher register target
ptxas fatal   : Ptx assembly aborted due to errors
// Something in the log after the error.
)";

constexpr std::string_view kPtxasLogRegisterSpillWarning = R"(
// Something in the log before the warning.
ptxas warning : Registers are spilled to local memory in function '__kernel', 8 bytes spill stores, 8 bytes spill loads
// Something in the log after the warning.
)";

TEST(PtxCompilerHelpersTest, IsPtxRegisterAllocationError) {
  EXPECT_TRUE(IsPtxRegisterAllocationError(kPtxasLogRegisterAllocationError));
  EXPECT_FALSE(IsPtxRegisterAllocationError(kPtxasLogRegisterSpillWarning));
}

constexpr std::string_view kDefaultArchitecture = "sm_80";

TEST(PtxCompilerHelpersTest, CreateErrorFromPtxasLogNoError) {
  EXPECT_THAT(CreateErrorFromPtxasLog(kPtxasLogSuccessfulCompilation,
                                      kDefaultArchitecture,
                                      /*cancel_if_reg_spill=*/true),
              IsOk());
}

TEST(PtxCompilerHelpersTest,
     CreateErrorFromPtxasLogDetectsRegisterAllocationError) {
  EXPECT_THAT(CreateErrorFromPtxasLog(kPtxasLogRegisterAllocationError,
                                      kDefaultArchitecture,
                                      /*cancel_if_reg_spill=*/true),
              StatusIs(absl::StatusCode::kResourceExhausted));
}

TEST(PtxCompilerHelpersTest, CreateErrorFromPtxasLogDetectsPtxasTooOldError) {
  EXPECT_THAT(
      CreateErrorFromPtxasLog(kPtxasLogTooOldError, kDefaultArchitecture,
                              /*cancel_if_reg_spill=*/true),
      StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(PtxCompilerHelpersTest,
     CreateErrorFromPtxasLogDetectsRegisterSpillWarning) {
  EXPECT_THAT(CreateErrorFromPtxasLog(kPtxasLogRegisterSpillWarning,
                                      kDefaultArchitecture,
                                      /*cancel_if_reg_spill=*/true),
              StatusIs(absl::StatusCode::kCancelled));
}

TEST(PtxCompilerHelpersTest,
     CreateErrorFromPtxasLogIgnoresRegisterSpillWarningIfNotRequested) {
  EXPECT_THAT(CreateErrorFromPtxasLog(kPtxasLogRegisterSpillWarning,
                                      kDefaultArchitecture,
                                      /*cancel_if_reg_spill=*/false),
              IsOk());
}

TEST(PtxCompilerHelpersTest, IsPtxRegisterAllocationErrorStatus) {
  EXPECT_TRUE(IsPtxRegisterAllocationError(
      PtxRegisterAllocationError("Register allocation failed")));
  EXPECT_FALSE(
      IsPtxRegisterAllocationError(absl::ResourceExhaustedError("OOM")));
  EXPECT_FALSE(IsPtxRegisterAllocationError(absl::OkStatus()));
}

}  // namespace
}  // namespace stream_executor
