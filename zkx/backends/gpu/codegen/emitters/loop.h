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
#ifndef ZKX_BACKENDS_GPU_CODEGEN_EMITTERS_LOOP_H_
#define ZKX_BACKENDS_GPU_CODEGEN_EMITTERS_LOOP_H_

#include <cstdint>
#include <optional>

#include "absl/status/status.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"

#include "zkx/backends/gpu/codegen/emitters/emitter_base.h"
#include "zkx/codegen/emitters/computation_partitioner.h"
#include "zkx/hlo/analysis/indexing_map.h"
#include "zkx/hlo/ir/hlo_instructions.h"
#include "zkx/service/gpu/gpu_fusible.h"
#include "zkx/service/gpu/hlo_fusion_analysis.h"
#include "zkx/service/gpu/launch_dimensions.h"

namespace zkx::gpu {

// Generic loop fusion. Lowers to LLVM via MLIR.
class LoopFusion : public EmitterBase {
 public:
  explicit LoopFusion(const HloFusionAnalysis& analysis)
      : analysis_(analysis), config_(ComputeLoopFusionConfig(analysis)) {}
  LaunchDimensions launch_dimensions() const override;

  std::optional<IndexingMap> ComputeThreadIdToOutputIndexing(
      int64_t root_index, mlir::MLIRContext* ctx) const override;

  std::optional<IndexingMap> ComputeThreadIdToInputIndexing(
      int64_t root_index, int64_t hero_operand_index,
      mlir::MLIRContext* ctx) const override;

 protected:
  absl::Status EmitEntryFunction(
      const emitters::PartitionedComputations& computations,
      const emitters::CallTargetProvider& call_targets,
      mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion) const override;

 private:
  const HloFusionAnalysis& analysis_;
  LaunchDimensionsConfig config_;
};

}  // namespace zkx::gpu

#endif  // ZKX_BACKENDS_GPU_CODEGEN_EMITTERS_LOOP_H_
