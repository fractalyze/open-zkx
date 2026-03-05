/* Copyright 2017 The OpenXLA Authors.
Copyright 2026 The ZKX Authors.

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

#ifndef ZKX_SERVICE_GPU_TRANSFORMS_LAYOUT_ASSIGNMENT_H_
#define ZKX_SERVICE_GPU_TRANSFORMS_LAYOUT_ASSIGNMENT_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"

#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/hlo/pass/hlo_pass_interface.h"
#include "zkx/service/computation_layout.h"
#include "zkx/stream_executor/device_description.h"

namespace zkx::gpu {

// GPU-specific layout assignment pass.
// Stubbed out: the base LayoutAssignment class has been removed.
class GpuLayoutAssignment : public HloModulePass {
 public:
  explicit GpuLayoutAssignment(
      ComputationLayout* /*entry_computation_layout*/,
      const se::DeviceDescription& /*device_description*/,
      void* /*channel_constraints*/ = nullptr) {}
  ~GpuLayoutAssignment() override = default;

  std::string_view name() const override { return "gpu-layout-assignment"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* /*module*/,
      const absl::flat_hash_set<std::string_view>& /*execution_threads*/)
      override {
    // Stubbed: base LayoutAssignment has been removed.
    return false;
  }
};

}  // namespace zkx::gpu

#endif  // ZKX_SERVICE_GPU_TRANSFORMS_LAYOUT_ASSIGNMENT_H_
