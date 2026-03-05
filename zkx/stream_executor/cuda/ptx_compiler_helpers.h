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
#ifndef ZKX_STREAM_EXECUTOR_CUDA_PTX_COMPILER_HELPERS_H_
#define ZKX_STREAM_EXECUTOR_CUDA_PTX_COMPILER_HELPERS_H_

#include "absl/status/status.h"

#include "zkx/stream_executor/cuda/cuda_compute_capability.h"
#include "zkx/stream_executor/semantic_version.h"

namespace stream_executor {

// Creates a status with a payload indicating a register allocation error.
absl::Status PtxRegisterAllocationError(std::string_view message);

// Checks whether ptxas log contains errors related to register allocation.
bool IsPtxRegisterAllocationError(std::string_view);

// Checks whether the status is a register allocation error.
bool IsPtxRegisterAllocationError(absl::Status status);

// Identifies errors in the ptxas log and creates an error status.
// `architecture` is the name of the GPU architecture, e.g. "sm_80" and is only
// used for error message generation. If `cancel_if_reg_spill` is true, then a
// register spill warning will be treated as an error, otherwise it will be
// ignored.
absl::Status CreateErrorFromPtxasLog(std::string_view log,
                                     std::string_view architecture,
                                     bool cancel_if_reg_spill);

// Warns if the ptxas version should be upgraded.
void WarnIfBadPtxasVersion(std::string_view method,
                           const CudaComputeCapability& cc,
                           SemanticVersion compiler_version);

// Determine whether the PTX extension for a compute capability should be used.
//
// Returns true if the argument compute capability has PTX extensions that are
// only valid for that compute capability. For example, "sm_90" only includes
// features that are forward compatible, whereas "sm_90a" (the extension) also
// includes Hopper-specific features, such as WGMMA. We want to use the latter.
bool ShouldUsePtxExtension(const CudaComputeCapability& cc);

}  // namespace stream_executor

#endif  // ZKX_STREAM_EXECUTOR_CUDA_PTX_COMPILER_HELPERS_H_
