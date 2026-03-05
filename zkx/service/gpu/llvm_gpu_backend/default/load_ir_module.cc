/* Copyright 2017 The OpenXLA Authors.
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

#include "zkx/service/gpu/llvm_gpu_backend/load_ir_module.h"

#include "absl/log/log.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"

namespace {

void DieWithSMDiagnosticError(llvm::SMDiagnostic* diagnostic) {
  LOG(FATAL) << diagnostic->getFilename().str() << ":"
             << diagnostic->getLineNo() << ":" << diagnostic->getColumnNo()
             << ": " << diagnostic->getMessage().str();
}

}  // namespace

namespace zkx::gpu {

std::unique_ptr<llvm::Module> LoadIRModule(std::string_view filename,
                                           llvm::LLVMContext* llvm_context) {
  llvm::SMDiagnostic diagnostic_err;
  std::unique_ptr<llvm::Module> module =
      llvm::getLazyIRFileModule(filename, diagnostic_err, *llvm_context,
                                /*ShouldLazyLoadMetadata=*/true);

  if (module == nullptr) {
    DieWithSMDiagnosticError(&diagnostic_err);
  }

  return module;
}

}  // namespace zkx::gpu
