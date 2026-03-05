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

#ifndef ZKX_STREAM_EXECUTOR_CUDA_ASSEMBLE_COMPILATION_PROVIDER_H_
#define ZKX_STREAM_EXECUTOR_CUDA_ASSEMBLE_COMPILATION_PROVIDER_H_

#include <memory>

#include "absl/status/statusor.h"

#include "zkx/stream_executor/cuda/compilation_provider.h"
#include "zkx/stream_executor/cuda/compilation_provider_options.h"

namespace stream_executor::cuda {

// Returns the best available CompilationProvider while considering the
// following flags from DebugOptions:
// - zkx_gpu_enable_libnvptxcompiler
// - zkx_gpu_libnvjitlink_mode
// - zkx_gpu_cuda_data_dir
// - zkx_gpu_enable_llvm_module_compilation_parallelism
// - zkx_gpu_unsafe_fallback_to_driver_on_ptxas_not_found
//
// Considered compilation methods are:
// - nvptxcompiler
// - nvjitlink
// - subprocess(ptxas, nvlink)
// - driver
//
// Returns an error if either no compilation method is available or if
// requested features like compilation parallelism are not supported.
// Also returns an error if contradicting flags are set.
absl::StatusOr<std::unique_ptr<CompilationProvider>>
AssembleCompilationProvider(const CompilationProviderOptions& options);

}  // namespace stream_executor::cuda

#endif  // ZKX_STREAM_EXECUTOR_CUDA_ASSEMBLE_COMPILATION_PROVIDER_H_
