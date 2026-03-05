/* Copyright 2018 The OpenXLA Authors.
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

#ifndef ZKX_SERVICE_GPU_STREAM_EXECUTOR_UTIL_H_
#define ZKX_SERVICE_GPU_STREAM_EXECUTOR_UTIL_H_

#include <stdint.h>

#include <memory>

#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"

#include "zkx/service/gpu/launch_dimensions.h"
#include "zkx/service/hlo_module_config.h"
#include "zkx/stream_executor/device_memory.h"
#include "zkx/stream_executor/kernel.h"
#include "zkx/stream_executor/launch_dim.h"
#include "zkx/stream_executor/stream_executor.h"

// Helper functions for interacting with StreamExecutor.

namespace zkx::gpu {

// Generates and returns a unique lock per the provided executor.
// Guarantees that blocks of code running for the same provided executor will
// not be running concurrently if they lock the returned mutex.
//
// This is used to prevent other ZKX instances from trying to autotune on a
// device while another thread is using it.
absl::Mutex& GetGpuMutex(const se::StreamExecutor* stream_exec);

// Creates a kernel with a provided name, based from provided PTX in ptx.
// The kernel should be executed using the provided executor.
// The argument cubin_data represents compiled PTX and may be left empty.
//
// The canonical storage for both ptx and cubin_data should outlive
// the lifetime of the kernel.
absl::StatusOr<std::unique_ptr<se::Kernel>> CreateKernel(
    std::string_view kernel_name, uint64_t num_args, std::string_view ptx,
    absl::Span<const uint8_t> cubin_data, se::StreamExecutor* stream_exec,
    uint32_t shared_mem_bytes = 0);

// Runs loaded kernel on the stream with the provided arguments.
absl::Status ExecuteKernelOnStream(se::Kernel& kernel,
                                   absl::Span<const se::DeviceMemoryBase> args,
                                   const LaunchDimensions& dims,
                                   se::Stream* stream);

// Runs loaded kernel on the stream with the provided arguments.
absl::Status ExecuteKernelOnStream(se::Kernel& kernel,
                                   absl::Span<const se::DeviceMemoryBase> args,
                                   const LaunchDimensions& dims,
                                   const se::ClusterDim& cluster_dim,
                                   se::Stream* stream);

}  // namespace zkx::gpu

#endif  // ZKX_SERVICE_GPU_STREAM_EXECUTOR_UTIL_H_
