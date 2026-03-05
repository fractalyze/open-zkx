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
// Provide helper routine for obtaining gpu target information useful
// for llvm IR construction.

#include "zkx/service/gpu/target_util.h"

#include "absl/log/log.h"
#include "llvm/TargetParser/Triple.h"

namespace zkx::gpu {

void AnnotateFunctionAsGpuKernel(llvm::Module* module, llvm::Function* func,
                                 llvm::IRBuilderBase* b) {
  llvm::Triple target_triple = llvm::Triple(module->getTargetTriple());
  if (target_triple.isNVPTX()) {
    // Attach information so NVPTX can recognize function as a CUDA kernel.
    func->setCallingConv(llvm::CallingConv::PTX_Kernel);

  } else if (target_triple.getArch() == llvm::Triple::amdgcn) {
    // Attach information so AMDGPU can recognize function as a AMDGPU kernel.
    func->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
    func->addFnAttr("uniform-work-group-size", "true");
  } else if (target_triple.isSPIR()) {
    // Attach information so that it can be recognized as a SPIR kernel.
    func->setCallingConv(llvm::CallingConv::SPIR_KERNEL);
  } else {
    LOG(FATAL) << "Invalid triple " << target_triple.str();
  }
}

}  // namespace zkx::gpu
