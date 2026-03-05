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

#ifndef ZKX_BACKENDS_GPU_RUNTIME_KERNEL_THUNK_H_
#define ZKX_BACKENDS_GPU_RUNTIME_KERNEL_THUNK_H_

#include <stdint.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"

#include "zkx/backends/gpu/runtime/thunk.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/service/buffer_assignment.h"
#include "zkx/service/gpu/kernel_arguments.h"
#include "zkx/service/gpu/launch_dimensions.h"
#include "zkx/stream_executor/kernel.h"
#include "zkx/stream_executor/launch_dim.h"
#include "zkx/stream_executor/stream_executor.h"
#include "zkx/types.h"  // IWYU pragma: keep

namespace zkx::gpu {

class GpuExecutable;

// TODO(ezhulenev): Unify KernelThunk and CustomKernelThunk as they are very
// similar. ZKX:GPU should use more of kernel loading APIs provided by
// StreamExecutor out of the box and less custom kernel loading solutions.
//
// Today KernelThunk is required for lowering to ZKX runtime, and
// CustomKernelThunk is only supported for thunk execution.

//===----------------------------------------------------------------------===//
// KernelThunk
//===----------------------------------------------------------------------===//

// This class stores everything that StreamExecutor needs for launching a
// kernel. It implements the ExecuteOnStream interface for GpuExecutable to
// invoke the corresponding kernel.
//
// This is thread-compatible.
class KernelThunk : public Thunk {
 public:
  // Constructs a thunk for the given kernel.
  //
  // KernelThunk takes args as `BufferAllocation::Slice`s (wrapped in
  // `KernelArgument`s). Each slice directly corresponds to an argument or
  // output of the computation. Also, the values must correspond to each arg
  // directly, not to their base allocation (e.g. they can be the result of an
  // `mlir::memref::ViewOp`).
  KernelThunk(const HloInstruction *instr, std::string kernel_name,
              absl::Span<const KernelArgument> kernel_arguments,
              LaunchDimensions launch_dimensions,
              std::optional<se::ClusterDim> cluster_dim, int64_t shmem_bytes);
  KernelThunk(const KernelThunk &) = delete;
  KernelThunk &operator=(const KernelThunk &) = delete;
  ~KernelThunk() override = default;

  std::string ToString(int indent) const override;

  absl::Status Initialize(const InitializeParams &params) override;
  absl::Status ExecuteOnStream(const ExecuteParams &params) override;

  const std::vector<BufferAllocation::Slice> &arguments() const {
    return args_;
  }
  const std::vector<bool> &written() const { return written_; }

  const std::string &kernel_name() const { return kernel_name_; }
  const LaunchDimensions &launch_dimensions() const {
    return launch_dimensions_;
  }
  // The shared memory required by the kernel.
  int64_t shmem_bytes() const { return shmem_bytes_; }

 private:
  // Buffer slices passed to the kernel as arguments.
  std::vector<BufferAllocation::Slice> args_;

  // args_[i] is written iff (written_[i] == true).
  std::vector<bool> written_;

  // Entry kernel name for the computation.
  const std::string kernel_name_;

  // The thread and block dimension used to launch the kernel.
  const LaunchDimensions launch_dimensions_;

  // The cluster dimensions used to launch the kernel.
  const std::optional<se::ClusterDim> cluster_dim_;

  int64_t shmem_bytes_;

  // Loaded kernels for each `StreamExecutor`.
  mutable absl::Mutex mutex_;
  absl::flat_hash_map<se::StreamExecutor *, std::unique_ptr<se::Kernel>>
      kernel_cache_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace zkx::gpu

#endif  // ZKX_BACKENDS_GPU_RUNTIME_KERNEL_THUNK_H_
