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

// This file defines the operations used in the MHLO dialect.

#ifndef ZKX_MLIR_HLO_MHLO_IR_HLO_OPS_H_
#define ZKX_MLIR_HLO_MHLO_IR_HLO_OPS_H_

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TensorEncoding.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/Base.h"

// Include order below matters.
#include "zkx/mlir_hlo/mhlo/IR/hlo_ops_enums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "zkx/mlir_hlo/mhlo/IR/hlo_ops_attrs.h.inc"
#define GET_TYPEDEF_CLASSES

namespace mlir::mhlo {

class MhloDialect : public Dialect {
public:
  explicit MhloDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "mhlo"; }

  // Registered hook to materialize a constant operation from a given attribute
  // value with the desired resultant type.
  Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                 Location loc) override;

  // Registered hook to verify region arg attributes on operations.
  LogicalResult verifyRegionArgAttribute(mlir::Operation *op,
                                         unsigned regionIndex,
                                         unsigned argIndex,
                                         mlir::NamedAttribute attr) override;

  // Registered hook to verify an attribute from this dialect on operations.
  LogicalResult verifyOperationAttribute(mlir::Operation *op,
                                         mlir::NamedAttribute attr) override;

  // Parses a type registered to this dialect.
  Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;

  // Prints an attribute registered to this dialect.
  void printAttribute(Attribute attr, DialectAsmPrinter &os) const override;
};

class TokenType : public Type::TypeBase<TokenType, Type, TypeStorage> {
public:
  using Base::Base;
  static constexpr StringLiteral name = "mhlo.token";
};

} // namespace mlir::mhlo

#define GET_OP_CLASSES
#include "zkx/mlir_hlo/mhlo/IR/hlo_ops.h.inc"

#endif // ZKX_MLIR_HLO_MHLO_IR_HLO_OPS_H_
