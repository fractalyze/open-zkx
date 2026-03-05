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

#ifndef ZKX_STREAM_EXECUTOR_CUDA_NVJITLINK_COMPILATION_PROVIDER_H_
#define ZKX_STREAM_EXECUTOR_CUDA_NVJITLINK_COMPILATION_PROVIDER_H_

#include "zkx/stream_executor/cuda/compilation_provider.h"

namespace stream_executor::cuda {

class NvJitLinkCompilationProvider : public CompilationProvider {
 public:
  NvJitLinkCompilationProvider() = default;

  bool SupportsCompileToRelocatableModule() const override { return false; }
  bool SupportsCompileAndLink() const override { return true; }

  absl::StatusOr<Assembly> Compile(
      const CudaComputeCapability& cc, std::string_view ptx,
      const CompilationOptions& options) const override;

  absl::StatusOr<RelocatableModule> CompileToRelocatableModule(
      const CudaComputeCapability& cc, std::string_view ptx,
      const CompilationOptions& options) const override;

  absl::StatusOr<Assembly> CompileAndLink(
      const CudaComputeCapability& cc,
      absl::Span<const RelocatableModuleOrPtx> inputs,
      const CompilationOptions& options) const override;

  std::string name() const override { return "NvJitLinkCompilationProvider"; }
};

}  // namespace stream_executor::cuda

#endif  // ZKX_STREAM_EXECUTOR_CUDA_NVJITLINK_COMPILATION_PROVIDER_H_
