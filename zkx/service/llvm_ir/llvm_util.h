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

#ifndef ZKX_SERVICE_LLVM_IR_LLVM_UTIL_H_
#define ZKX_SERVICE_LLVM_IR_LLVM_UTIL_H_

#include <stdint.h>

#include <optional>
#include <string>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

#include "zkx/hlo/ir/hlo_instruction.h"

namespace zkx::llvm_ir {

// We have different DumpToString functions for each type for findability. We
// use pointers / values based on the usual semantics of the parameter type.

std::string DumpToString(const llvm::Module* module);
std::string DumpToString(const llvm::Type* type);
std::string DumpToString(const llvm::Value* value);

// This also works for mlir::Op<...> descendants, such as mlir::ModuleOp.
//
// For findability:
//   std::string DumpToString(mlir::Op<...>& op);
//   std::string DumpToString(mlir::ModuleOp& module_op);
//
// The `operation` parameter is not const, because the used print() method is
// not const.
std::string DumpToString(mlir::Operation* operation);
std::string DumpToString(mlir::Type type);
std::string DumpToString(mlir::Value value);

// Constructs a human-friendly name from the given inputs.  The result is
// suitable for use as an llvm::Value's name.
//
// This is equivalent to
//
//   - changing the HloInstruction* to its name() (if we called that overload),
//   - joining all of the nonempty inputs by '.', and then
//   - removing all '%'s.
//
std::string IrName(std::string_view a);
std::string IrName(std::string_view a, std::string_view b);
std::string IrName(const HloInstruction* a, std::string_view b = "");

// Construct a module from the given location with an optional name.
//
// The underlying "create" method is unsafe, because it leaks the new module by
// default. This function avoids this by always returning an OwningOpRef.
mlir::OwningOpRef<mlir::ModuleOp> CreateMlirModuleOp(
    mlir::Location loc, std::optional<llvm::StringRef> name = std::nullopt);

// Removes special characters from a function name.
//
// Note that this can cause different inputs to map to the same output, so after
// sanitizing a function name, you must run it through a uniquer.
std::string SanitizeFunctionName(std::string_view function_name);

template <typename T>
llvm::APInt ConvertBigIntToAPInt(const T& value) {
  return {T::kBitWidth, static_cast<unsigned>(T::kLimbNums), value.limbs()};
}

// Tells LLVM `inst >= lower && inst < upper`. Returns `inst` for convenience.
llvm::Instruction* AddRangeMetadata(int32_t lower, int32_t upper,
                                    llvm::Instruction* inst,
                                    llvm::Module* module);

}  // namespace zkx::llvm_ir

#endif  // ZKX_SERVICE_LLVM_IR_LLVM_UTIL_H_
