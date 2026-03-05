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

#include "zkx/backends/gpu/codegen/emitters/ir/zkx_gpu_ops.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

#include "zkx/backends/gpu/codegen/emitters/ir/zkx_gpu_dialect.cc.inc"
#include "zkx/hlo/analysis/indexing_map.h"

namespace zkx::gpu {
namespace {

using llvm::ArrayRef;
using mlir::DenseI64ArrayAttr;
using mlir::failure;
using mlir::Location;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::OpAsmParser;
using mlir::OpAsmPrinter;
using mlir::OperationState;
using mlir::ParseResult;
using mlir::RankedTensorType;
using mlir::SmallVector;
using mlir::success;
using mlir::Type;
using mlir::TypeRange;
using mlir::Value;
using mlir::ValueRange;

}  // namespace

//===----------------------------------------------------------------------===//
// AllocateSharedOp
//===----------------------------------------------------------------------===//

void AllocateSharedOp::getAsmResultNames(
    llvm::function_ref<void(Value, mlir::StringRef)> setNameFn) {
  setNameFn(getResult(), "shmem");
}

//===----------------------------------------------------------------------===//
// MaterializeOp
//===----------------------------------------------------------------------===//

LogicalResult MaterializeOp::verify() {
  IndexingMap map_in = getMap().getIndexingMap();
  IndexingMap map_out =
      getResult().getType().getIndexingMapAttr().getIndexingMap();
  if (getIndices().size() != map_in.GetDimVarsCount()) {
    return emitOpError() << "number of indices must match number of dimensions "
                            "of indexing map";
  }

  // The thread dimension must have the same domain (range and constraints)
  if (map_in.GetDimVarsCount() == 0 || map_out.GetDimVarsCount() == 0) {
    return emitOpError()
           << "must have thread_id dimension in both indexing maps";
  }
  if (map_in.GetDimVar(0).bounds != map_out.GetDimVar(0).bounds) {
    return emitOpError() << "thread_id dimension must have the same bounds in "
                            "both indexing maps";
  }

  VariableConstraints variable_constraints_in =
      GetConstraintsForVariables(map_in);
  VariableConstraints variable_constraints_out =
      GetConstraintsForVariables(map_out);
  if (variable_constraints_in.constraints_for_dims[0] !=
      variable_constraints_out.constraints_for_dims[0]) {
    return emitOpError() << "constraints of indexing maps must be equal for "
                         << "the thread_id dimension";
  }

  // The two maps must have the same symbols and they must have the same domain
  if (map_in.GetRangeVarsCount() != map_out.GetRangeVarsCount()) {
    return emitOpError()
           << "number of symbols in both indexing_maps must match";
  }
  for (auto const& [range_in, range_out] :
       llvm::zip(map_in.GetRangeVars(), map_out.GetRangeVars())) {
    if (range_in.bounds != range_out.bounds) {
      return emitOpError() << "domain of symbols of indexing_maps must match";
    }
  }
  if (variable_constraints_in.constraints_for_symbols !=
      variable_constraints_out.constraints_for_symbols) {
    return emitOpError()
           << "constraints of indexing maps must be equal for all symbols";
  }

  // The vector mapping indices must not depend on the block ID
  if (map_out.GetDimVarsCount() > 1) {
    for (auto expr : map_out.GetAffineMap().getResults()) {
      if (expr.isFunctionOfDim(1)) {
        return emitOpError() << "vector mapping indices must not depend on the "
                             << "block ID";
      }
    }
  }
  // If there are constraints on the block ID, they must be the same in both
  // maps
  if (map_in.GetDimVarsCount() > 1 && map_out.GetDimVarsCount() > 1) {
    if (variable_constraints_in.constraints_for_dims[1] !=
        variable_constraints_out.constraints_for_dims[1]) {
      return emitOpError() << "constraints of indexing maps must be equal for "
                           << "the block_id dimension";
    }
  } else if (map_in.GetDimVarsCount() > 1 &&
             !variable_constraints_in.constraints_for_dims[1].empty()) {
    return emitOpError() << "constraints of indexing maps must be equal for "
                         << "the block_id dimension";
  } else if (map_out.GetDimVarsCount() > 1 &&
             !variable_constraints_out.constraints_for_dims[1].empty()) {
    return emitOpError() << "constraints of indexing maps must be equal for "
                         << "the block_id dimension";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// InsertOp
//===----------------------------------------------------------------------===//

LogicalResult InsertOp::verify() {
  if (!getMap().getIndexingMap().GetRangeVars().empty()) {
    return emitOpError() << "insert_op map must not have any symbols";
  }
  int64_t vector_map_num_results =
      getSource().getType().getIndexingMapAttr().getNumResults();
  if (vector_map_num_results != getMap().getIndexingMap().GetDimVars().size()) {
    return emitOpError() << "source map result count must equal insert_op's "
                            "map's dimension count";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//

SmallVector<Type, 2> inferReductionResultTypes(TypeRange input_types,
                                               ArrayRef<int64_t> reduced_dims) {
  auto input_shape =
      mlir::cast<RankedTensorType>(input_types.front()).getShape();
  size_t num_reduced_dims = reduced_dims.size();
  SmallVector<int64_t, 4> output_shape;
  output_shape.reserve(input_shape.size() - num_reduced_dims);
  int reduce_dim = 0;
  for (int64_t i = 0; i < input_shape.size(); ++i) {
    if (reduce_dim < num_reduced_dims && i == reduced_dims[reduce_dim]) {
      ++reduce_dim;
      continue;
    }
    output_shape.push_back(input_shape[i]);
  }
  SmallVector<Type, 2> result_types;
  result_types.reserve(input_types.size());
  for (auto input_type : input_types) {
    result_types.push_back(RankedTensorType::get(
        output_shape,
        mlir::cast<RankedTensorType>(input_type).getElementType()));
  }
  return result_types;
}

SmallVector<Type, 2> inferReductionInitTypes(TypeRange input_types) {
  SmallVector<Type, 2> init_types;
  init_types.reserve(input_types.size());
  for (auto input_type : input_types) {
    init_types.push_back(
        mlir::cast<RankedTensorType>(input_type).getElementType());
  }
  return init_types;
}

LogicalResult ReduceOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location, ValueRange operands,
    mlir::DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions,
    mlir::SmallVectorImpl<Type>& inferredReturnTypes) {
  ReduceOp::Adaptor adaptor(operands, attributes, properties, regions);
  inferredReturnTypes.append(inferReductionResultTypes(
      TypeRange{adaptor.getInputs()}, adaptor.getDimensions()));
  return success();
}

ParseResult ReduceOp::parse(OpAsmParser& parser, OperationState& result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> inputs;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> inits;
  SmallVector<int64_t, 2> dimensions;
  mlir::StringAttr combiner;
  SmallVector<Type, 2> input_types;
  SmallVector<Type, 2> result_types;

  if (failed(parser.parseLParen()) || failed(parseOperands(parser, &inputs)) ||
      failed(parser.parseRParen()) || failed(parser.parseKeyword("inits")) ||
      failed(parser.parseLParen()) || failed(parseOperands(parser, &inits)) ||
      failed(parser.parseRParen()) ||
      failed(parser.parseKeyword("dimensions")) ||
      failed(parser.parseEqual()) ||
      failed(parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Square,
                                            [&]() -> ParseResult {
                                              return parser.parseInteger(
                                                  dimensions.emplace_back());
                                            })) ||
      failed(parser.parseKeyword("combiner")) || failed(parser.parseEqual()) ||
      failed(parser.parseSymbolName(combiner)) ||
      failed(parser.parseOptionalAttrDict(result.attributes)) ||
      failed(parser.parseColonTypeList(input_types)) ||
      failed(parser.parseKeyword("to")) ||
      failed(parser.parseTypeList(result_types))) {
    return failure();
  }
  MLIRContext* ctx = result.getContext();
  mlir::OperationName opname(ReduceOp::getOperationName(), ctx);
  result.addAttribute(ReduceOp::getDimensionsAttrName(opname),
                      DenseI64ArrayAttr::get(ctx, dimensions));
  result.addAttribute(ReduceOp::getCombinerAttrName(opname),
                      mlir::FlatSymbolRefAttr::get(ctx, combiner));
  result.addTypes(result_types);

  SmallVector<Type, 2> init_types = inferReductionInitTypes(input_types);
  mlir::SMLoc loc = parser.getCurrentLocation();
  if (failed(
          parser.resolveOperands(inputs, input_types, loc, result.operands)) ||
      failed(parser.resolveOperands(inits, init_types, loc, result.operands))) {
    return failure();
  }
  return success();
}

void ReduceOp::print(OpAsmPrinter& p) {
  p << '(' << getInputs() << ") inits(" << getInits() << ") dimensions=["
    << getDimensions() << "] combiner=@" << getCombiner();
  p.printOptionalAttrDict((*this)->getAttrs(),
                          {getCombinerAttrName(), getDimensionsAttrName()});
  p << " : " << TypeRange(getInputs()) << " to " << TypeRange(getResults());
}

LogicalResult ReduceOp::verify() {
  // Check init types.
  auto inferred_init_types = inferReductionInitTypes(TypeRange(getInputs()));
  for (auto [inferred_init_type, init_type] :
       llvm::zip(inferred_init_types, TypeRange(getInits()))) {
    if (inferred_init_type != init_type) {
      return emitOpError() << "init type " << init_type
                           << " does not match inferred type "
                           << inferred_init_type;
    }
  }
  // Check combiner.
  auto module = this->getOperation()->getParentOfType<mlir::ModuleOp>();
  auto combiner = module.lookupSymbol<mlir::func::FuncOp>(getCombinerAttr());
  if (!combiner) {
    return emitOpError() << "combiner `@" << getCombiner() << "` not found";
  }
  SmallVector<Type, 2> combiner_operand_types;
  combiner_operand_types.reserve(getNumOperands());
  combiner_operand_types.append(inferred_init_types);
  combiner_operand_types.append(inferred_init_types);
  auto expected_combiner_type = mlir::FunctionType::get(
      getContext(), combiner_operand_types, inferred_init_types);
  if (expected_combiner_type != combiner.getFunctionType()) {
    return emitOpError() << "provided combiner `@" << getCombiner()
                         << " expected to have type " << expected_combiner_type
                         << " but got " << combiner.getFunctionType();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ShuffleReduceOp
//===----------------------------------------------------------------------===//

ParseResult ShuffleReduceOp::parse(OpAsmParser& parser,
                                   OperationState& result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> inputs;
  mlir::StringAttr combiner;
  int64_t max_distance;
  SmallVector<Type, 2> operand_types;
  mlir::SMLoc loc = parser.getCurrentLocation();
  if (failed(parser.parseLParen()) || failed(parseOperands(parser, &inputs)) ||
      failed(parser.parseRParen()) || failed(parser.parseKeyword("to")) ||
      failed(parser.parseInteger(max_distance)) ||
      failed(parser.parseKeyword("combiner")) || failed(parser.parseEqual()) ||
      failed(parser.parseSymbolName(combiner)) ||
      failed(parser.parseOptionalAttrDict(result.attributes)) ||
      failed(parser.parseColonTypeList(operand_types)) ||
      failed(parser.resolveOperands(inputs, operand_types, loc,
                                    result.operands))) {
    return failure();
  }
  MLIRContext* ctx = result.getContext();
  mlir::OperationName opname(ShuffleReduceOp::getOperationName(), ctx);
  result.addAttribute(ShuffleReduceOp::getCombinerAttrName(opname),
                      mlir::FlatSymbolRefAttr::get(ctx, combiner));
  result.addAttribute(
      ShuffleReduceOp::getMaxDistanceAttrName(opname),
      mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), max_distance));
  result.addTypes(operand_types);
  return success();
}

void ShuffleReduceOp::print(OpAsmPrinter& p) {
  p << '(' << getOperands() << ") to " << getMaxDistance() << " combiner=@"
    << getCombiner();
  p.printOptionalAttrDict((*this)->getAttrs(),
                          {getCombinerAttrName(), getMaxDistanceAttrName()});
  p << " : " << TypeRange(getResultTypes());
}

//===----------------------------------------------------------------------===//
// SyncThreadsOp
//===----------------------------------------------------------------------===//

void SyncThreadsOp::getAsmResultNames(
    llvm::function_ref<void(Value, mlir::StringRef)> setNameFn) {
  for (auto result : getResults()) {
    setNameFn(result, "synced_tensor");
  }
}

}  // namespace zkx::gpu

#define GET_OP_CLASSES
#include "zkx/backends/gpu/codegen/emitters/ir/zkx_gpu_ops.cc.inc"
