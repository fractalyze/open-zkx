/* Copyright 2019 The OpenXLA Authors.
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

#ifndef ZKX_MLIR_HLO_MHLO_TRANSFORMS_PASSES_H_
#define ZKX_MLIR_HLO_MHLO_TRANSFORMS_PASSES_H_

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::mhlo {

#define GEN_PASS_DECL
#include "zkx/mlir_hlo/mhlo/transforms/mhlo_passes.h.inc"

// Sinks constants implicitly captured in control flow regions. This is
// necessary to export to ZKX.
std::unique_ptr<OperationPass<func::FuncOp>>
createSinkConstantsToControlFlowPass();

// Creates a pass to analyze shapes and to use that information for
// shape-related optimizations.
std::unique_ptr<OperationPass<func::FuncOp>>
createSymbolicShapeOptimizationPass();

// Pass to replace unsigned types with signless integers.
std::unique_ptr<OperationPass<ModuleOp>> createConvertToSignlessPass();

// Legalizes from the StableHLO dialect to the MHLO dialect.
std::unique_ptr<OperationPass<ModuleOp>> createStablehloLegalizeToHloPass();

// Legalizes from the Shape dialect to the MHLO dialect.
std::unique_ptr<OperationPass<func::FuncOp>>
createShapeLegalizeToHloPass(bool legalizeConstraints = false);

#define GEN_PASS_REGISTRATION
#include "zkx/mlir_hlo/mhlo/transforms/mhlo_passes.h.inc" // NOLINT(build/include)

} // namespace mlir::mhlo

#endif // ZKX_MLIR_HLO_MHLO_TRANSFORMS_PASSES_H_
