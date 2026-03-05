/* Copyright 2017 The OpenXLA Authors.
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

#ifndef ZKX_SERVICE_GPU_NVPTX_COMPILER_H_
#define ZKX_SERVICE_GPU_NVPTX_COMPILER_H_

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"

#include "zkx/service/gpu/gpu_compiler.h"
#include "zkx/stream_executor/cuda/compilation_provider.h"
#include "zkx/stream_executor/cuda/compilation_provider_options.h"

namespace zkx::gpu {

void WarnIfBadDriverJITVersion();

// NVPTXCompiler generates efficient GPU executables for NVPTX target.
class NVPTXCompiler : public GpuCompiler {
 public:
  explicit NVPTXCompiler();

  HloDataflowAnalysis::CanShareBuffer GetCanShareBuffer(
      const se::DeviceDescription& device_description) const override;

  absl::StatusOr<BackendCompileResult> CompileTargetBinary(
      const HloModuleConfig& module_config, llvm::Module* llvm_module,
      const se::DeviceDescription& device_description, bool relocatable,
      const HloModule* debug_module, const CompileOptions& options,
      std::optional<int> shard_number) override;

  absl::StatusOr<bool> CanUseLinkModules(
      const HloModuleConfig& module_config,
      const se::DeviceDescription& device_description) override;

 private:
  absl::StatusOr<std::vector<uint8_t>> LinkModules(
      const se::DeviceDescription& device_description,
      se::StreamExecutor* stream_exec,
      std::vector<std::vector<uint8_t>> modules,
      const DebugOptions& debug_options) override;

  absl::Mutex compilation_providers_mutex_;
  absl::flat_hash_map<se::cuda::CompilationProviderOptions,
                      std::unique_ptr<se::cuda::CompilationProvider>>
      compilation_providers_ ABSL_GUARDED_BY(compilation_providers_mutex_);

  absl::StatusOr<const se::cuda::CompilationProvider*> GetCompilationProvider(
      const DebugOptions& debug_options);

  NVPTXCompiler(const NVPTXCompiler&) = delete;
  NVPTXCompiler& operator=(const NVPTXCompiler&) = delete;

  std::vector<std::string> GetLLVMCommandLineOptions(
      const DebugOptions& debug_options) const override;
};

}  // namespace zkx::gpu

#endif  // ZKX_SERVICE_GPU_NVPTX_COMPILER_H_
