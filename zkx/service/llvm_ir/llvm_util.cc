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

#include "zkx/service/llvm_ir/llvm_util.h"

#include <algorithm>
#include <cctype>
#include <utility>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

namespace zkx::llvm_ir {
namespace {

// This works for most llvm / mlir types. This also accepts a const pointer to
// objects which have a const print() method.
template <typename T>
std::string DumpToStringTempl(T* entity) {
  CHECK_NE(entity, nullptr);

  std::string s;
  llvm::raw_string_ostream ostream(s);
  ostream << *entity;
  return s;
}

}  // namespace

std::string DumpToString(const llvm::Module* module) {
  return DumpToStringTempl(module);
}

std::string DumpToString(const llvm::Type* type) {
  return DumpToStringTempl(type);
}

std::string DumpToString(const llvm::Value* value) {
  return DumpToStringTempl(value);
}

std::string DumpToString(mlir::Operation* operation) {
  return DumpToStringTempl(operation);
}

std::string DumpToString(mlir::Type type) { return DumpToStringTempl(&type); }

std::string DumpToString(mlir::Value value) {
  return DumpToStringTempl(&value);
}

llvm::Instruction* AddRangeMetadata(int32_t lower, int32_t upper,
                                    llvm::Instruction* inst,
                                    llvm::Module* module) {
  if (llvm::Triple(module->getTargetTriple()).isSPIR()) {
    return inst;
  }
  llvm::LLVMContext& context = inst->getParent()->getContext();
  llvm::IntegerType* i32 = llvm::Type::getInt32Ty(context);
  inst->setMetadata(
      llvm::LLVMContext::MD_range,
      llvm::MDNode::get(
          context,
          {llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32, lower)),
           llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32, upper))}));
  return inst;
}

std::string IrName(std::string_view a) {
  std::string s(a);
  s.erase(std::remove(s.begin(), s.end(), '%'), s.end());
  return s;
}

std::string IrName(std::string_view a, std::string_view b) {
  if (!a.empty() && !b.empty()) {
    return IrName(absl::StrCat(a, ".", b));
  }
  return IrName(absl::StrCat(a, b));
}

std::string IrName(const HloInstruction* a, std::string_view b) {
  return IrName(a->name(), b);
}

mlir::OwningOpRef<mlir::ModuleOp> CreateMlirModuleOp(
    mlir::Location loc, std::optional<llvm::StringRef> name) {
  return mlir::OwningOpRef<mlir::ModuleOp>(
      mlir::ModuleOp::create(std::move(loc), std::move(name)));
}

std::string SanitizeFunctionName(std::string_view old_function_name) {
  std::string function_name(old_function_name);
  // The backend with the strictest requirements on function names is NVPTX, so
  // we sanitize to its requirements.
  //
  // A slightly stricter version of the NVPTX requirements is that names match
  // /[a-zA-Z_$][a-zA-Z0-9_$]*/, with the exception that the names "_" and "$"
  // are illegal.

  // Sanitize chars in function_name.
  std::transform(function_name.begin(), function_name.end(),
                 function_name.begin(), [](char c) {
                   if (std::isalnum(static_cast<unsigned char>(c)) ||
                       c == '_' || c == '$') {
                     return c;
                   }
                   return '_';
                 });

  // Ensure the name isn't empty.
  if (function_name.empty()) {
    function_name = "__unnamed";
  }

  // Ensure the name doesn't start with a number.
  if (!function_name.empty() && function_name[0] >= '0' &&
      function_name[0] <= '9') {
    function_name.insert(function_name.begin(), '_');
  }

  // Ensure the name isn't "_" or "$".
  if (function_name == "_" || function_name == "$") {
    function_name += '_';
  }

  return function_name;
}

}  // namespace zkx::llvm_ir
