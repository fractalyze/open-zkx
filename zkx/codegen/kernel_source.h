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

#ifndef ZKX_CODEGEN_KERNEL_SOURCE_H_
#define ZKX_CODEGEN_KERNEL_SOURCE_H_

#include <string>

namespace zkx {

// KernelSource is a base class for generated kernel source. Concrete types of
// kernel source are backends specific, i.e. on GPU backend it can be PTX (if
// already compiled) or an LLVM IR (if ZKX itself will compile it to PTX).
class KernelSource {
 public:
  virtual ~KernelSource() = default;

  // Get a human readable string representation of the kernel source.
  virtual std::string ToString() const = 0;
};

}  // namespace zkx

#endif  // ZKX_CODEGEN_KERNEL_SOURCE_H_
