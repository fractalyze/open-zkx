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

// LLVM-based compiler backend.
#ifndef ZKX_SERVICE_GPU_LLVM_GPU_BACKEND_GPU_BACKEND_LIB_H_
#define ZKX_SERVICE_GPU_LLVM_GPU_BACKEND_GPU_BACKEND_LIB_H_

#include <functional>
#include <memory>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "llvm/IR/Module.h"
#include "llvm/PassRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"

#include "zkx/stream_executor/device_description.h"
#include "zkx/zkx.pb.h"

namespace zkx::gpu {

// Initializes LLVM passes. Uses the PassRegistry mechanism.
void InitializePasses(llvm::PassRegistry* pass_registry);

// Returns the TargetMachine, given a triple.
std::unique_ptr<llvm::TargetMachine> GetTargetMachine(
    llvm::Triple triple, std::string_view cpu_name,
    const DebugOptions& debug_options, std::string_view feature_str);

// Returns whether the module could use any device bitcode library functions.
bool CouldNeedDeviceBitcode(const llvm::Module& module);

// Links the module with a vector of path to bitcode modules.
// The caller must guarantee that the paths exist.
absl::Status LinkWithBitcodeVector(
    llvm::Module* module,
    const std::vector<std::string_view>& bitcode_path_vector);

using TargetModuleLinker =
    std::function<absl::Status(llvm::Module*, se::GpuComputeCapability,
                               const DebugOptions&, std::string_view)>;

// Links and optimizes the module.
absl::Status LinkAndOptimizeModule(
    llvm::Module* module, se::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options, std::string_view device_bitcode_path,
    TargetModuleLinker module_linker, llvm::Triple default_target_triple,
    llvm::TargetMachine* target_machine, int inline_threshold);

}  // namespace zkx::gpu

#endif  // ZKX_SERVICE_GPU_LLVM_GPU_BACKEND_GPU_BACKEND_LIB_H_
