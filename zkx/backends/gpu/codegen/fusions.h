/* Copyright 2023 The OpenXLA Authors.
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
#ifndef ZKX_BACKENDS_GPU_CODEGEN_FUSIONS_H_
#define ZKX_BACKENDS_GPU_CODEGEN_FUSIONS_H_

#include <memory>

#include "absl/status/statusor.h"

#include "zkx/backends/gpu/codegen/fusion_emitter.h"
#include "zkx/hlo/ir/hlo_instructions.h"
#include "zkx/service/buffer_assignment.h"
#include "zkx/service/gpu/hlo_fusion_analysis.h"
#include "zkx/service/gpu/ir_emission_utils.h"

namespace zkx::gpu {

class FusionInfo {
 public:
  explicit FusionInfo(const HloFusionAnalysis &analysis)
      : analysis_(analysis) {}
  virtual ~FusionInfo() = default;

  const HloFusionAnalysis &analysis() const { return analysis_; }

  // If the fusion is a DUS fusion, returns whether it can be emitted in place.
  // Undefined if the fusion is not a DUS fusion.
  virtual bool CanEmitDynamicUpdateSliceInPlace() const = 0;

 private:
  const HloFusionAnalysis &analysis_;
};

class HloFusionInfo : public FusionInfo {
 public:
  HloFusionInfo(const HloFusionAnalysis &analysis,
                const HloFusionInstruction *instr,
                const BufferAssignment *buffer_assignment)
      : FusionInfo(analysis),
        instr_(instr),
        buffer_assignment_(buffer_assignment) {}

  bool CanEmitDynamicUpdateSliceInPlace() const override;

 private:
  const HloFusionInstruction *instr_;          // not owned
  const BufferAssignment *buffer_assignment_;  // not owned
};

class PreBufferAssignmentFusionInfo : public FusionInfo {
 public:
  explicit PreBufferAssignmentFusionInfo(const HloFusionAnalysis &analysis)
      : FusionInfo(analysis) {}

  bool CanEmitDynamicUpdateSliceInPlace() const override {
    auto ret = CanEmitFusedDynamicUpdateSliceInPlaceForGpu(
        analysis().fusion(), /*get_allocation_slice=*/{});
    return ret.value_or(false);
  }
};

// Returns the emitter for the given fusion.
std::unique_ptr<FusionInterface> GetFusionEmitter(
    const FusionInfo &fusion_info);

}  // namespace zkx::gpu

#endif  // ZKX_BACKENDS_GPU_CODEGEN_FUSIONS_H_
