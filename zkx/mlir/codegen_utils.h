/* Copyright 2025 The ZKX Authors.

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

#ifndef ZKX_MLIR_CODEGEN_UTILS_H_
#define ZKX_MLIR_CODEGEN_UTILS_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

namespace zkx::mlir_utils {

mlir::Value ConvertInteger(
    mlir::ImplicitLocOpBuilder& b, mlir::ArrayRef<mlir::Type> result_types,
    mlir::Type source_type, mlir::Type target_type, mlir::ValueRange args,
    bool is_signed, mlir::ArrayRef<mlir::NamedAttribute> attributes = {});

mlir::Value DivideInteger(mlir::ImplicitLocOpBuilder& b, mlir::Value lhs,
                          mlir::Value rhs, bool is_signed);

mlir::Value PowerInteger(mlir::ImplicitLocOpBuilder& b, mlir::Value lhs,
                         mlir::Value rhs, bool is_signed);

mlir::Value RemainderInteger(mlir::ImplicitLocOpBuilder& b, mlir::Value lhs,
                             mlir::Value rhs, bool is_signed);

mlir::Value SignInteger(mlir::ImplicitLocOpBuilder& b, mlir::Value value);

mlir::Value ShiftLeftInteger(mlir::ImplicitLocOpBuilder& b, mlir::Value lhs,
                             mlir::Value rhs);

mlir::Value ShiftRightArithmeticInteger(mlir::ImplicitLocOpBuilder& b,
                                        mlir::Value lhs, mlir::Value rhs);

mlir::Value ShiftRightLogicalInteger(mlir::ImplicitLocOpBuilder& b,
                                     mlir::Value lhs, mlir::Value rhs);

mlir::Value ConvertField(mlir::ImplicitLocOpBuilder& b,
                         mlir::ArrayRef<mlir::Type> result_types,
                         mlir::Type source_type, mlir::Type target_type,
                         mlir::ValueRange args,
                         mlir::ArrayRef<mlir::NamedAttribute> attributes = {});

mlir::Value ConvertEcPoint(
    mlir::ImplicitLocOpBuilder& b, mlir::ArrayRef<mlir::Type> result_types,
    mlir::Type source_type, mlir::Type target_type, mlir::ValueRange args,
    mlir::ArrayRef<mlir::NamedAttribute> attributes = {});

}  // namespace zkx::mlir_utils

#endif  // ZKX_MLIR_CODEGEN_UTILS_H_
