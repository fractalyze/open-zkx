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

#ifndef ZKX_CODEGEN_KERNEL_EMITTER_H_
#define ZKX_CODEGEN_KERNEL_EMITTER_H_

#include <memory>

#include "absl/status/statusor.h"

#include "zkx/codegen/kernel_definition.h"

namespace zkx {

// TODO(ezhulenev): Do we need virtual KernelEmitterContext in API?

// KernelEmitter is an API that emits kernel definition from a given input
// (i.e. it emits kernels compiled from HLO fusions).
class KernelEmitter {
 public:
  virtual ~KernelEmitter() = default;

  virtual absl::StatusOr<KernelDefinition> EmitKernelDefinition() = 0;
};

// A base class for backend-specific kernel emitters.
//
// Example: ZKX:GPU backend kernel emitter.
//
//   class zkx::gpu::GpuPlatform;
//
//   class zkx::gpu::HloFusionEmitter :
//     public KernelEmitter<GpuPlatform, const HloFusionInstruction*>;
//
template <typename Platform, typename Operation>
class KernelEmitterBase {
 public:
  KernelEmitterBase(std::shared_ptr<Platform> platform, Operation operation)
      : platform_(std::move(platform)), operation_(std::move(operation)) {}

  const Operation& operation() const { return operation_; }
  const Platform& platform() const { return *platform_; }

 private:
  std::shared_ptr<Platform> platform_;
  Operation operation_;
};

}  // namespace zkx

#endif  // ZKX_CODEGEN_KERNEL_EMITTER_H_
