/* Copyright 2025 The OpenXLA Authors.
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

#include "zkx/codegen/emitters/transforms/atomic_rmw_utils.h"

#include "absl/log/check.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/UseDefLists.h"

namespace zkx::emitters {
namespace {

using mlir::Block;
using mlir::Operation;
using mlir::Type;
using mlir::Value;

namespace arith = ::mlir::arith;
namespace ml = ::mlir::LLVM;

bool IsAtomicIntegral(Type element_type) {
  return element_type.isInteger(32) || element_type.isInteger(64);
}

std::optional<ml::AtomicBinOp> GetAtomicBinOp(Operation* modifier_op,
                                              Type element_type) {
  if (!IsAtomicIntegral(element_type)) {
    return std::nullopt;
  }
  return llvm::TypeSwitch<Operation*, std::optional<ml::AtomicBinOp>>(
             modifier_op)
      .Case<arith::AddIOp>([](auto) { return ml::AtomicBinOp::add; })
      .Case<arith::MaxUIOp>([](auto) { return ml::AtomicBinOp::umax; })
      .Case<arith::MinUIOp>([](auto) { return ml::AtomicBinOp::umin; })
      .Case<arith::MaxSIOp>([](auto) { return ml::AtomicBinOp::max; })
      .Case<arith::MinSIOp>([](auto) { return ml::AtomicBinOp::min; })
      .Default([](Operation*) { return std::nullopt; });
}

}  // namespace

std::optional<std::pair<Value, ml::AtomicBinOp>> GetAtomicModifierParameters(
    AtomicRMWOp op) {
  Type element_type = op.getInput().getType().getElementType();
  Block::OpListType& operations = op.getBody()->getOperations();
  Operation* terminator = op.getBody()->getTerminator();
  if (operations.size() > 2) {
    return std::nullopt;
  }
  // If the body contains only the terminator, then it is an atomic store.
  if (operations.size() == 1) {
    if (IsAtomicIntegral(element_type)) {
      return std::make_pair(terminator->getOperand(0), ml::AtomicBinOp::xchg);
    }
    return std::nullopt;
  }
  CHECK_EQ(operations.size(), 2);
  // Match the kind of the atomic op.
  Operation* modifier_op = &operations.front();
  std::optional<ml::AtomicBinOp> kind =
      GetAtomicBinOp(modifier_op, element_type);
  if (!kind.has_value()) {
    return std::nullopt;
  }
  // Find the modifier arg that does not match the argument of `atomic_rmw`
  // body.
  Value block_arg = op.getBody()->getArgument(0);
  Value modifier_arg = modifier_op->getOperand(0) == block_arg
                           ? modifier_op->getOperand(1)
                           : modifier_op->getOperand(0);
  return std::make_pair(modifier_arg, *kind);
}

}  // namespace zkx::emitters
