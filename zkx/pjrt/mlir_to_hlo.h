/* Copyright 2021 The OpenXLA Authors.
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

#ifndef ZKX_PJRT_MLIR_TO_HLO_H_
#define ZKX_PJRT_MLIR_TO_HLO_H_

#include "absl/status/status.h"
#include "mlir/IR/BuiltinOps.h"

#include "zkx/hlo/builder/zkx_computation.h"

namespace zkx {

// Converts an MHLO module to ZKX HLO.
// TODO(b/345414638): Delete `use_shardy` when we move Shardy as the first pass
// in the ZKX pipeline.
absl::Status MlirToZkxComputation(mlir::ModuleOp module,
                                  ZkxComputation& xla_computation,
                                  bool use_tuple_args, bool return_tuple,
                                  bool use_shardy);

}  // namespace zkx

#endif  // ZKX_PJRT_MLIR_TO_HLO_H_
