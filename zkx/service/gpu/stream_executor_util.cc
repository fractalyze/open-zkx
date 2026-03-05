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

#include "zkx/service/gpu/stream_executor_util.h"

#include <map>
#include <utility>

#include "absl/base/const_init.h"
#include "absl/debugging/leak_check.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/stream_executor/kernel_spec.h"

namespace zkx::gpu {

// Returns a mutex that can be used to lock the given stream executor.
absl::Mutex& GetGpuMutex(const se::StreamExecutor* stream_exec) {
  static absl::Mutex mu(absl::kConstInit);
  // se::Platform*s are global singletons guaranteed to live forever.
  static auto* mutexes = absl::IgnoreLeak(
      new std::map<std::pair<const se::Platform*, /*device_ordinal*/ int64_t>,
                   absl::Mutex>());

  absl::MutexLock global_lock(&mu);
  auto it = mutexes
                ->emplace(std::piecewise_construct,
                          std::make_tuple(stream_exec->GetPlatform(),
                                          stream_exec->device_ordinal()),
                          std::make_tuple())
                .first;

  return it->second;
}

absl::StatusOr<std::unique_ptr<se::Kernel>> CreateKernel(
    std::string_view kernel_name, uint64_t num_args, std::string_view ptx,
    absl::Span<const uint8_t> cubin_data, se::StreamExecutor* stream_exec,
    uint32_t shared_mem_bytes) {
  se::MultiKernelLoaderSpec loader_spec(num_args);
  loader_spec.AddCudaPtxInMemory(ptx, kernel_name);

  if (!cubin_data.empty()) {
    loader_spec.AddCudaCubinInMemory(cubin_data, kernel_name);
  }

  TF_ASSIGN_OR_RETURN(std::unique_ptr<se::Kernel> kernel,
                      stream_exec->LoadKernel(loader_spec));

  se::KernelMetadata m;
  m.set_shared_memory_bytes(shared_mem_bytes);
  kernel->set_metadata(m);
  return kernel;
}

absl::Status ExecuteKernelOnStream(se::Kernel& kernel,
                                   absl::Span<const se::DeviceMemoryBase> args,
                                   const LaunchDimensions& dims,
                                   se::Stream* stream) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<se::KernelArgsPackedArrayBase> kernel_args,
      se::PackKernelArgs(args, kernel.metadata()));

  return kernel.Launch(dims.thread_counts_per_block(), dims.block_counts(),
                       stream, *kernel_args);
}

absl::Status ExecuteKernelOnStream(se::Kernel& kernel,
                                   absl::Span<const se::DeviceMemoryBase> args,
                                   const LaunchDimensions& dims,
                                   const se::ClusterDim& cluster_dim,
                                   se::Stream* stream) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<se::KernelArgsPackedArrayBase> kernel_args,
      se::PackKernelArgs(args, kernel.metadata()));

  return kernel.Launch(dims.thread_counts_per_block(), dims.block_counts(),
                       cluster_dim, stream, *kernel_args);
}

}  // namespace zkx::gpu
