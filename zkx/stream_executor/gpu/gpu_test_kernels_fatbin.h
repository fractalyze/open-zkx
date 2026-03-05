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

#ifndef ZKX_STREAM_EXECUTOR_GPU_GPU_TEST_KERNELS_FATBIN_H_
#define ZKX_STREAM_EXECUTOR_GPU_GPU_TEST_KERNELS_FATBIN_H_

#include <stdint.h>

#include <vector>

#include "absl/status/statusor.h"

#include "zkx/stream_executor/namespace_alias.h"

namespace stream_executor::gpu {

// Returns the NVIDIA or HIP fatbin for the :gpu_test_kernels target.
// The fatbin is being extracted at compile time from the compilation artifact.
// Note that this function will read the extracted fatbin from the file system
// at runtime and will only be able to succeed when the test is being invoked by
// `bazel test`.
absl::StatusOr<std::vector<uint8_t>> GetGpuTestKernelsFatbin();

}  // namespace stream_executor::gpu

#endif  // ZKX_STREAM_EXECUTOR_GPU_GPU_TEST_KERNELS_FATBIN_H_
