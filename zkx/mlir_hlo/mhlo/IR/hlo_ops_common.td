/* Copyright 2022 The OpenXLA Authors.
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

#ifndef ZKX_MLIR_HLO_MHLO_IR_HLO_OPS_COMMON_TD
#define ZKX_MLIR_HLO_MHLO_IR_HLO_OPS_COMMON_TD

def MHLO_Dialect : Dialect {
  let name = "mhlo";
  let cppNamespace = "::mlir::mhlo";

  let useDefaultAttributePrinterParser = 0;
  let useDefaultTypePrinterParser = 0;
}

include "zkx/mlir_hlo/mhlo/IR/hlo_base.td"
include "zkx/mlir_hlo/mhlo/IR/hlo_ops_enums.td"
include "zkx/mlir_hlo/mhlo/IR/hlo_ops_attrs.td"

#endif // ZKX_MLIR_HLO_MHLO_IR_HLO_OPS_COMMON_TD
