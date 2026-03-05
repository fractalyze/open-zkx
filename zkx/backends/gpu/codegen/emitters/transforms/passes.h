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
#ifndef ZKX_BACKENDS_GPU_CODEGEN_EMITTERS_TRANSFORMS_PASSES_H_
#define ZKX_BACKENDS_GPU_CODEGEN_EMITTERS_TRANSFORMS_PASSES_H_

#include <memory>

namespace zkx::gpu {

#define GEN_PASS_DECL
#include "zkx/backends/gpu/codegen/emitters/transforms/passes.h.inc"

std::unique_ptr<mlir::Pass> CreateConvertIndexTypePass();
std::unique_ptr<mlir::Pass> CreateOptimizeLoopsPass();
std::unique_ptr<mlir::Pass> CreatePeelLoopsPass();
std::unique_ptr<mlir::Pass> CreateVectorizeLoadsAndStoresPass();

#define GEN_PASS_REGISTRATION
#include "zkx/backends/gpu/codegen/emitters/transforms/passes.h.inc"  // NOLINT(build/include)

}  // namespace zkx::gpu

#endif  // ZKX_BACKENDS_GPU_CODEGEN_EMITTERS_TRANSFORMS_PASSES_H_
