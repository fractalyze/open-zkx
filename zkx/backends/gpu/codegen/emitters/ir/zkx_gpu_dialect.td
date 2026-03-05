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

#ifndef ZKX_BACKENDS_GPU_CODEGEN_EMITTERS_IR_ZKX_GPU_DIALECT_TD
#define ZKX_BACKENDS_GPU_CODEGEN_EMITTERS_IR_ZKX_GPU_DIALECT_TD

include "mlir/IR/DialectBase.td"

def ZkxGpuDialect : Dialect {
  let name = "zkx_gpu";

  let description = [{
    This dialect contains ops required for lowering HLO to LLVM.
  }];

  let cppNamespace = "::zkx::gpu";
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

#endif // ZKX_BACKENDS_GPU_CODEGEN_EMITTERS_IR_ZKX_GPU_DIALECT_TD
