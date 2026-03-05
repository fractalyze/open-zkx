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

#include <string>

#include "gtest/gtest.h"

#include "xla/tsl/platform/path.h"
#include "xla/tsl/platform/test_util.h"

namespace zkx::gpu {
namespace {

std::string SaxpyIRFile() {
  return tsl::io::JoinPath(tsl::testing::ZkxSrcRoot(), "service", "gpu",
                           "llvm_gpu_backend", "tests_data", "saxpy.ll");
}

TEST(LoadIrModuleTest, TestLoadIRModule) {
  llvm::LLVMContext llvm_context;
  std::unique_ptr<llvm::Module> module =
      LoadIRModule(SaxpyIRFile(), &llvm_context);
  // Sanity check that the module was loaded properly.
  ASSERT_NE(nullptr, module);
  ASSERT_NE(std::string::npos, module->getModuleIdentifier().find("saxpy.ll"));
  ASSERT_NE(nullptr, module->getFunction("cuda_saxpy"));
}

}  // namespace
}  // namespace zkx::gpu
