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

#ifndef ZKX_SERVICE_GPU_TARGET_UTIL_H_
#define ZKX_SERVICE_GPU_TARGET_UTIL_H_

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"

namespace zkx::gpu {

// Annotate the kernel as GPU kernel according to the GPU target.
void AnnotateFunctionAsGpuKernel(llvm::Module* module, llvm::Function* func,
                                 llvm::IRBuilderBase* b);

}  // namespace zkx::gpu

#endif  // ZKX_SERVICE_GPU_TARGET_UTIL_H_
