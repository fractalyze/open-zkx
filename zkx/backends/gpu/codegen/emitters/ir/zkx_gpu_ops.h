/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
#ifndef ZKX_BACKENDS_GPU_CODEGEN_EMITTERS_IR_ZKX_GPU_OPS_H_
#define ZKX_BACKENDS_GPU_CODEGEN_EMITTERS_IR_ZKX_GPU_OPS_H_

#include "mlir/IR/Dialect.h"  // IWYU pragma: keep

#include "zkx/backends/gpu/codegen/emitters/ir/zkx_gpu_dialect.h.inc"
#include "zkx/backends/gpu/codegen/emitters/ir/zkx_gpu_enums.h.inc"
#include "zkx/codegen/emitters/ir/zkx_ops.h"
#define GET_ATTRDEF_CLASSES
#include "zkx/backends/gpu/codegen/emitters/ir/zkx_gpu_attrs.h.inc"
#define GET_TYPEDEF_CLASSES
#include "zkx/backends/gpu/codegen/emitters/ir/zkx_gpu_types.h.inc"
#define GET_OP_CLASSES
#include "zkx/backends/gpu/codegen/emitters/ir/zkx_gpu_ops.h.inc"

#endif  // ZKX_BACKENDS_GPU_CODEGEN_EMITTERS_IR_ZKX_GPU_OPS_H_
