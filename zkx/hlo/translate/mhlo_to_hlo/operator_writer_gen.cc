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

#include <string>
#include <utility>

#include "absl/debugging/leak_check.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include "mlir/TableGen/Argument.h"
#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/Operator.h"

using llvm::interleaveComma;
using llvm::raw_ostream;
using llvm::RecordKeeper;
using llvm::StringRef;
using mlir::tblgen::Attribute;
using mlir::tblgen::NamedAttribute;
using mlir::tblgen::NamedTypeConstraint;
using mlir::tblgen::Operator;

namespace {

std::string GetDefaultAttrExport(
    const mlir::tblgen::NamedAttribute &named_attr) {
  Attribute attr = named_attr.attr;
  StringRef storage_type = attr.getStorageType();
  // For some attribute types we have a general conversion, so use that.
  if (!attr.isEnumAttr() && (storage_type.ends_with("BoolAttr") ||
                             storage_type.ends_with("FloatAttr") ||
                             storage_type.ends_with("IntegerAttr") ||
                             storage_type.ends_with("StringAttr"))) {
    // The return type may contains qualified namespaces. Split to remove them.
    std::pair<StringRef, StringRef> splits = attr.getReturnType().rsplit("::");
    StringRef symbol = splits.second;
    if (symbol.empty()) symbol = splits.first;
    return "Convert" + symbol.str();
  }
  return "Convert_" + named_attr.name.str();
}

StringRef GetClientBuilder(const Operator &op) {
  static const auto *kOpToZKXBuilderMap = absl::IgnoreLeak(
      new llvm::StringMap<StringRef>{{"ReverseOp", "Rev"},
                                     {"ConcatenateOp", "ConcatInDim"},
                                     {"ConvOp", "ConvGeneralDilated"}});

  StringRef op_name = op.getCppClassName();

  // Default case where the client builder method names closely follow the op
  // names in the dialect. For e.g., AddOp -> zkx::Add method.
  if (!kOpToZKXBuilderMap->count(op_name)) return op_name.drop_back(2);

  // Otherwise, if the op to client builder method mapping is provided.
  return kOpToZKXBuilderMap->lookup(op_name);
}

void BuildOperator(const Operator &op, raw_ostream &os) {
  os << "mlir::LogicalResult ExportZkxOp(mlir::mhlo::" << op.getCppClassName()
     << " op, OpLoweringContext ctx) {\n"
     << "  auto& value_map = *ctx.values;\n"
     << "  auto result = op.getResult();\n";

  // Build a conversion for each of the arguments.
  int operand_number = 0;
  for (int index : llvm::seq<int>(0, op.getNumArgs())) {
    auto arg = op.getArg(index);

    // Emit an argument for an operand.
    if (auto *operand_cst = arg.dyn_cast<NamedTypeConstraint *>()) {
      std::string zkx_arg = "zkx_arg_" + std::to_string(index);
      // Handle a non-variadic operand.
      if (!operand_cst->isVariableLength()) {
        os << "  zkx::ZkxOp " << zkx_arg << ";\n";
        os << "  if (failed(GetZkxOp(*op.getODSOperands(" << operand_number++
           << ").begin(), value_map, &" << zkx_arg << ", op)))\n";
        os << "    return mlir::failure();\n";
        continue;
      }

      // Otherwise, this is a variadic operand list.
      os << "  std::vector<zkx::ZkxOp> " << zkx_arg << ";\n"
         << "  for (auto operand : op.getODSOperands(" << operand_number++
         << ")) {\n";
      os << "    zkx::ZkxOp result;\n";
      os << "    if (failed(GetZkxOp(operand, value_map, &result, op)))\n";
      os << "      return mlir::failure();\n";
      os << "    " << zkx_arg << ".push_back(result);\n";
      os << "  }\n";
      continue;
    }

    // Otherwise, this is an attribute.
    auto named_attr = arg.get<NamedAttribute *>();
    os << "  auto zkx_arg_" << index << " = "
       << GetDefaultAttrExport(*named_attr) << "(op.get"
       << convertToCamelFromSnakeCase(op.getArgName(index),
                                      /*capitalizeFirst=*/true)
       << "());\n";
  }

  // Emit call to client API
  os << "  auto zkx_result = zkx::" << GetClientBuilder(op) << "(";

  // If all operands are variadic, then pass the builder explicitly to zkx
  // client API call
  if (op.getNumOperands() == op.getNumVariableLengthOperands()) {
    os << "ctx.builder";
    if (op.getNumArgs() != 0) os << ", ";
  }

  // Emit each of the arguments.
  interleaveComma(llvm::seq<int>(0, op.getNumArgs()), os,
                  [&](int i) { os << "Unwrap(zkx_arg_" << i << ')'; });
  os << ");\n";

  os << "  value_map[result] = zkx_result;\n";
  os << "  return mlir::success();\n";
  os << "}\n";
}

// The function below has a non-constant reference as that is required by LLVM's
// TableGenMain.
bool OperatorWritersMain(raw_ostream &os, const RecordKeeper &records) {
  emitSourceFileHeader("MLIR ZKX Builders", os);

  // Emit all the helper functions.
  for (const auto *def : records.getAllDerivedDefinitions("MHLO_Op")) {
    Operator op(def);

    // Skip operations that have a custom exporter.
    if (!def->getValueAsBit("hasCustomHLOConverter")) BuildOperator(op, os);
  }

  // Emit a function to generate an ZKX operation for the operations with
  // auto-generated builders.
  os << "mlir::LogicalResult ExportZkxOperator(\n"
        "mlir::Operation* op, OpLoweringContext lowering_context) {\n\n";

  // Create a scoped object to assign frontend attributes to generated ZKX ops.
  // Any HLO can have an attribute of "frontend_attributes", which are used to
  // pass hints / configuration options.
  os << "  zkx::ZkxScopedFrontendAttributesAssignment "
        "frontend_attributes(lowering_context.builder, "
        "CreateZkxFrontendAttributesFromOp(op));\n\n";

  // Create a scoped object to assign op metadata to generated ZKX ops.
  os << "  zkx::ZkxScopedOpMetadataAssignment "
        "op_metadata(lowering_context.builder, "
        "mlir::mhlo::CreateOpMetadataFromLocation("
        "op, lowering_context.frame_index_builder));\n\n";

  // Retrieve all the definitions derived from MHLO_Op and sort by record name.
  for (const auto *def : records.getAllDerivedDefinitions("MHLO_Op")) {
    // Skip operations that have a custom exporter.
    Operator op(def);

    // Cast to the current operation and build the exporter.
    os << "  if (auto zkx_op = llvm::dyn_cast<mlir::mhlo::"
       << op.getCppClassName() << ">(op)) {\n";
    os << "    return ";
    // The autogenerated converters aren't in the same namespace.
    // TODO(jpienaar): Reconsider this.
    if (def->getValueAsBit("hasCustomHLOConverter")) os << "mlir::mhlo::";
    os << "ExportZkxOp(zkx_op, lowering_context);\n";
    os << "  }\n";
  }

  os << "  return mlir::failure();\n"
        "}\n";
  return false;
}

}  // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  return TableGenMain(argv[0], &OperatorWritersMain);
}
