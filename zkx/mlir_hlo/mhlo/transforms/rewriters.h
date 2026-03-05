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

#ifndef ZKX_MLIR_HLO_MHLO_TRANSFORMS_REWRITERS_H_
#define ZKX_MLIR_HLO_MHLO_TRANSFORMS_REWRITERS_H_

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::stablehlo {

// Populates StableHLO ops to MHLO ops rewriting patterns.
// Also see `stablehlo::registerFuncOpsForTypeConversion` for helper patterns
// which make sure `func.func`, `func.call` and `func.return` which involve
// illegal types also get converted.
void populateStablehloToHloPatterns(RewritePatternSet *patterns,
                                    TypeConverter *converter,
                                    MLIRContext *context);

} // namespace mlir::stablehlo

#endif // ZKX_MLIR_HLO_MHLO_TRANSFORMS_REWRITERS_H_
