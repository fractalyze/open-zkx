/* Copyright 2020 The OpenXLA Authors.
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

#ifndef ZKX_SERVICE_GPU_GPU_EXECUTABLE_RUN_OPTIONS_H_
#define ZKX_SERVICE_GPU_GPU_EXECUTABLE_RUN_OPTIONS_H_

#include <functional>

#include "zkx/executable_run_options.h"

namespace zkx::gpu {

// Forward declaration for removed collectives type.
class GpuCollectives;

// Stub callback type replacing GpuCollectives::CliqueIdCallback.
using CliqueIdCallback = std::function<void()>;

// GPU-specific executable options.
// We keep these separate from ExecutableRunOptions to avoid adding
// dependencies to ExecutableRunOptions.
class GpuExecutableRunOptions {
 public:
  // Callback that returns a unique clique id for a given clique key.
  GpuExecutableRunOptions &set_clique_id_callback(
      CliqueIdCallback clique_id_callback);
  const CliqueIdCallback &clique_id_callback() const;

  // Collectives API for running collective operations on the GPU devices.
  GpuExecutableRunOptions &set_collectives(GpuCollectives *collectives);
  GpuCollectives *collectives() const;

  // Whether the run requires an exclusive lock on the GPU.
  bool requires_exclusive_lock_on_gpu() const {
    return requires_exclusive_lock_on_gpu_;
  }

  // Require writers lock on the GPU.
  GpuExecutableRunOptions &set_requires_exclusive_lock_on_gpu() {
    requires_exclusive_lock_on_gpu_ = true;
    return *this;
  }

  bool enable_mock_collectives() const { return enable_mock_collectives_; }

  // Enables mocking nccl collective operations on the GPU.
  GpuExecutableRunOptions &set_enable_mock_collectives() {
    enable_mock_collectives_ = true;
    return *this;
  }

 private:
  bool requires_exclusive_lock_on_gpu_ = false;
  bool enable_mock_collectives_ = false;
  CliqueIdCallback clique_id_callback_;
  GpuCollectives *collectives_ = nullptr;
};

}  // namespace zkx::gpu

#endif  // ZKX_SERVICE_GPU_GPU_EXECUTABLE_RUN_OPTIONS_H_
