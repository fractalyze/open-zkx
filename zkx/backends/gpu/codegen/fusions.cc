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
#include "zkx/backends/gpu/codegen/fusions.h"

#include <memory>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/strings/match.h"

#include "zkx/backends/gpu/codegen/emitters/concatenate.h"
#include "zkx/backends/gpu/codegen/emitters/in_place_dynamic_update_slice.h"
#include "zkx/backends/gpu/codegen/emitters/loop.h"
#include "zkx/backends/gpu/codegen/emitters/reduction.h"
#include "zkx/backends/gpu/codegen/emitters/scatter.h"
#include "zkx/backends/gpu/codegen/emitters/transpose.h"

namespace zkx::gpu {
namespace {

bool IsDynamicUpdateSliceFusion(const HloFusionAnalysis& analysis) {
  return absl::c_all_of(
      analysis.fusion_roots(), [](const HloInstructionAdaptor& root) {
        return root.opcode() == HloOpcode::kDynamicUpdateSlice ||
               (root.opcode() == HloOpcode::kBitcast &&
                root.GetOperand(0).opcode() == HloOpcode::kDynamicUpdateSlice);
      });
}

}  // namespace

bool HloFusionInfo::CanEmitDynamicUpdateSliceInPlace() const {
  auto ret = CanEmitFusedDynamicUpdateSliceInPlaceForGpu(
      analysis().fusion(),
      [this](const HloInstruction* instruction, const ShapeIndex& index) {
        return GetAllocationSlice(*buffer_assignment_, instruction, index);
      },
      instr_);
  return ret.ok() && *ret;
}

std::unique_ptr<FusionInterface> GetFusionEmitter(
    const FusionInfo& fusion_info) {
  const auto& analysis = fusion_info.analysis();

  switch (analysis.GetEmitterFusionKind()) {
    case HloFusionAnalysis::EmitterFusionKind::kLoop: {
      if (IsDynamicUpdateSliceFusion(analysis) &&
          fusion_info.CanEmitDynamicUpdateSliceInPlace()) {
        return std::make_unique<InPlaceDynamicUpdateSliceFusion>(analysis);
      }
      return std::make_unique<LoopFusion>(analysis);
    }
    case HloFusionAnalysis::EmitterFusionKind::kReduction:
      return CreateReductionFusion(analysis);
    case HloFusionAnalysis::EmitterFusionKind::kScatter:
      return CreateScatterFusion(analysis);
    case HloFusionAnalysis::EmitterFusionKind::kTranspose:
      return std::make_unique<TransposeFusion>(analysis);
    case HloFusionAnalysis::EmitterFusionKind::kConcatenate:
      return std::make_unique<ConcatenateFusion>(analysis);
  }
}

}  // namespace zkx::gpu
