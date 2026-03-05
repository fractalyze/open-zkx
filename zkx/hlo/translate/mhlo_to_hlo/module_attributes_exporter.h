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

#ifndef ZKX_HLO_TRANSLATE_MHLO_TO_HLO_MODULE_ATTRIBUTES_EXPORTER_H_
#define ZKX_HLO_TRANSLATE_MHLO_TO_HLO_MODULE_ATTRIBUTES_EXPORTER_H_

#include "absl/status/status.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

#include "zkx/service/hlo.pb.h"
#include "zkx/service/hlo_module_config.h"

namespace mlir::mhlo {

// Exports HLO Module Config info stored in the MHLO module as module
// attributes prefixed with `mhlo.`.
void ExportHloModuleConfig(zkx::HloModuleConfig& config, mlir::ModuleOp module);

absl::Status ExportModuleEntryComputationParameterLayouts(
    const mlir::ArrayAttr& zkx_entry_computation_parameter_layout,
    zkx::HloModuleProto& hlo_module);

absl::Status ExportModuleEntryComputationParameterTiles(
    const mlir::ArrayAttr& zkx_entry_computation_parameter_tiles,
    zkx::HloModuleProto& hlo_module);

absl::Status ExportModuleEntryComputationResultLayout(
    const mlir::ArrayAttr& zkx_entry_computation_result_layout,
    zkx::HloModuleProto& hlo_module);

absl::Status ExportModuleEntryComputationResultTiles(
    const mlir::ArrayAttr& zkx_entry_computation_result_tiles,
    zkx::HloModuleProto& hlo_module);

}  // namespace mlir::mhlo

#endif  // ZKX_HLO_TRANSLATE_MHLO_TO_HLO_MODULE_ATTRIBUTES_EXPORTER_H_
