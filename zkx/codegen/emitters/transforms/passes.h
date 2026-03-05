/* Copyright 2025 The OpenXLA Authors.
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
#ifndef ZKX_CODEGEN_EMITTERS_TRANSFORMS_PASSES_H_
#define ZKX_CODEGEN_EMITTERS_TRANSFORMS_PASSES_H_

#include <cstdint>
#include <memory>
#include <string_view>

#include "mlir/Pass/Pass.h"

#include "zkx/stream_executor/device_description.h"

namespace zkx::emitters {

#define GEN_PASS_DECL
#include "zkx/codegen/emitters/transforms/passes.h.inc"

std::unique_ptr<mlir::Pass> CreateConvertPureCallOpsPass();
std::unique_ptr<mlir::Pass> CreateEraseDeadFunctionsPass();
std::unique_ptr<mlir::Pass> CreateFlattenTensorsPass();
std::unique_ptr<mlir::Pass> CreateLowerTensorsPass(
    std::string_view target_type = "gpu",
    std::string_view gpu_device_info = "");
std::unique_ptr<mlir::Pass> CreateLowerTensorsPass(
    const se::DeviceDescription& device_description);
std::unique_ptr<mlir::Pass> CreateLowerToLLVMPass(
    std::string_view target_type = "gpu",
    std::string_view gpu_device_info = "");
std::unique_ptr<mlir::Pass> CreateLowerToLLVMPass(
    const se::DeviceDescription& device_description);
std::unique_ptr<mlir::Pass> CreateLowerZkxToScfPass(int64_t warp_size = 32);
std::unique_ptr<mlir::Pass> CreateLowerZkxLoopsToScfPass();
std::unique_ptr<mlir::Pass> CreateMergePointersToSameSlicePass();
std::unique_ptr<mlir::Pass> CreatePropagateSliceIndicesPass();
std::unique_ptr<mlir::Pass> CreateSimplifyAffinePass();
std::unique_ptr<mlir::Pass> CreateSimplifyArithPass();
std::unique_ptr<mlir::Pass> CreateUnswitchLoopsPass();

#define GEN_PASS_REGISTRATION
#include "zkx/codegen/emitters/transforms/passes.h.inc"  // NOLINT(build/include)

}  // namespace zkx::emitters

#endif  // ZKX_CODEGEN_EMITTERS_TRANSFORMS_PASSES_H_
