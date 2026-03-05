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

#include "zkx/mlir_hlo/mhlo/IR/hlo_ops.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <optional>
#include <utility>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"
#include "stablehlo/dialect/AssemblyFormat.h"
#include "stablehlo/dialect/TypeInference.h"

#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "zkx/mlir_hlo/mhlo/IR/hlo_ops.h.inc"
#include "zkx/mlir_hlo/utils/hlo_utils.h" // IWYU pragma: keep

using mlir::hlo::parseDimSizes;
using mlir::hlo::printDimSizes;

#include "zkx/mlir_hlo/mhlo/IR/hlo_ops_enums.cc.inc"
#define GET_ATTRDEF_CLASSES
#include "zkx/mlir_hlo/mhlo/IR/hlo_ops_attrs.cc.inc"
#define GET_TYPEDEF_CLASSES

namespace mlir::mhlo {
namespace {

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

hlo::HloDialectInterface *getMhloDialect(MLIRContext *context) {
  MhloDialect *dialect = context->getLoadedDialect<MhloDialect>();
  return dialect->getRegisteredInterface<hlo::HloDialectInterface>();
}

// Replaces the given op with the contents of the given single-block region,
// using the operands of the block terminator to replace operation results.
void replaceOpWithRegion(PatternRewriter &rewriter, Operation *op,
                         Region &region, ValueRange blockArgs = {}) {
  assert(llvm::hasSingleElement(region) && "expected single-block region");
  Block *block = &region.front();
  Operation *terminator = block->getTerminator();
  ValueRange results = terminator->getOperands();
  rewriter.inlineBlockBefore(block, op, blockArgs);
  rewriter.replaceOp(op, results);
  rewriter.eraseOp(terminator);
}

// Returns a new scalar integer value having type `type`. Here `type` must be
// an integer or index type.
Value maybeCastTo(OpBuilder &b, Location loc, Value value, Type type) {
  if (type == value.getType())
    return value;
  assert(type.isIndex() || value.getType().isIndex());
  return b.create<arith::IndexCastOp>(loc, type, value);
}

Value castToIndexTensor(OpBuilder &builder, Location loc, Value shapeOp) {
  ShapedType resultTy = shape::getExtentTensorType(
      builder.getContext(), cast<ShapedType>(shapeOp.getType()).getDimSize(0));
  if (shapeOp.getType() == resultTy)
    return shapeOp; // Nothing to do.
  return builder.create<arith::IndexCastOp>(loc, resultTy, shapeOp);
}

// Verifies that dimension attribute for the op correctly indexes in operand or
// result shape.
template <typename OpT>
LogicalResult verifyDimAttr(OpT op) {
  int64_t rank = -1;
  if (auto ty = dyn_cast<RankedTensorType>(op.getOperand().getType())) {
    rank = ty.getRank();
  } else if (auto ty = dyn_cast<RankedTensorType>(op.getType())) {
    rank = ty.getRank();
  } else {
    return success();
  }

  int64_t dim = op.getDimension();
  if (dim < 0 || dim >= rank)
    return op.emitOpError() << "requires dimension attribute in range [0, "
                            << rank << "); found (" << dim << ")";
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// Utilities for verifiers
//===----------------------------------------------------------------------===//

namespace {

LogicalResult verify1dTensor(std::optional<Location> loc,
                             DenseIntElementsAttr attr,
                             std::string_view attrName) {
  int64_t rank = attr.getType().getRank();
  if (rank != 1) {
    return emitOptionalError(loc, attrName, " has rank ", rank,
                             " instead of required rank 1.");
  }
  return success();
}

} // namespace

LogicalResult TypeExtensionsAttr::verifyEncoding(
    ArrayRef<int64_t> shape, mlir::Type elementType,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) const {
  return hlo::verifyBounds(
      getBounds(), RankedTensorType::get(shape, elementType), emitError);
}

//===----------------------------------------------------------------------===//
// CompatibleOperandsAndResultType
//===----------------------------------------------------------------------===//

// TODO(b/231358795): Review the use of InferTypeOpInterface for ops that
// support sparsity.
#define INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(Op)                         \
  LogicalResult Op::inferReturnTypeComponents(                                 \
      MLIRContext *context, std::optional<Location> location,                  \
      ValueShapeRange operands, DictionaryAttr attributes,                     \
      OpaqueProperties properties, RegionRange regions,                        \
      SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {           \
    return inferReturnTypeComponentsFromOperands(                              \
        context, location, operands, attributes, properties, regions,          \
        inferredReturnShapes);                                                 \
  }

INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AddOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AndOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(BitReverseOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ClzOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(DivOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(InverseOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(MaxOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(MinOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(MulOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(NegOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(NotOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(OrOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(PopulationCountOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(PowOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(RemOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ReverseOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ShiftLeftOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ShiftRightArithmeticOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ShiftRightLogicalOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SignOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SubtractOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(XorOp)

//===----------------------------------------------------------------------===//
// AbsOp
//===----------------------------------------------------------------------===//

LogicalResult
AbsOp::inferReturnTypes(MLIRContext *, std::optional<Location> location,
                        ValueRange operands, DictionaryAttr attributes,
                        OpaqueProperties properties, RegionRange regions,
                        SmallVectorImpl<Type> &inferredReturnTypes) {
  AbsOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferAbsOp(location, adaptor.getOperand(), inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// BitcastConvertOp
//===----------------------------------------------------------------------===//

LogicalResult BitcastConvertOp::reifyReturnTypeShapes(
    OpBuilder &builder, ValueRange operands,
    SmallVectorImpl<Value> &reifiedReturnShapes) {
  auto operandType = dyn_cast<RankedTensorType>(operands[0].getType());
  auto resultType = dyn_cast<RankedTensorType>(getType());

  // Only ranked tensors are supported.
  if (!operandType || !resultType)
    return failure();

  // Shape-changing bitcast convert is not implemented.
  // TODO(kramerb): This could be done by adjusting the last dimension.
  DataLayout dataLayout = DataLayout::closest(*this);
  unsigned operandElementSize =
      dataLayout.getTypeSizeInBits(operandType.getElementType());
  unsigned resultElementSize =
      dataLayout.getTypeSizeInBits(resultType.getElementType());
  if (operandElementSize != resultElementSize)
    return failure();

  return hlo::deriveShapeFromOperand(&builder, getOperation(), operands.front(),
                                     &reifiedReturnShapes);
}

LogicalResult BitcastConvertOp::verify() {
  return hlo::verifyBitcastConvertOp(getLoc(), getOperand(), getResult());
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

LogicalResult BroadcastOp::inferReturnTypeComponents(
    MLIRContext *, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  BroadcastOp::Adaptor adaptor(operands, attributes, properties, regions);
  if (failed(verify1dTensor(location, adaptor.getBroadcastSizes(),
                            "broadcast_sizes")))
    return failure();
  return hlo::inferBroadcastOp(
      location, adaptor.getOperand(),
      llvm::to_vector(adaptor.getBroadcastSizes().getValues<int64_t>()),
      inferredReturnShapes);
}

LogicalResult BroadcastOp::reifyReturnTypeShapes(
    OpBuilder &builder, ValueRange operands,
    SmallVectorImpl<Value> &reifiedReturnShapes) {
  BroadcastOp::Adaptor adaptor(operands);
  Value operand = adaptor.getOperand();

  auto operandType = dyn_cast<RankedTensorType>(operand.getType());
  // Unranked tensors are not supported.
  if (!operandType)
    return failure();

  Location loc = getLoc();
  SmallVector<Value, 4> shapeValues;

  // Collect the broadcast sizes.
  for (const auto &size : getBroadcastSizes()) {
    shapeValues.push_back(
        builder.create<arith::ConstantIndexOp>(loc, size.getZExtValue()));
  }

  // Collect the operand sizes.
  for (auto index : llvm::seq<int64_t>(0, operandType.getRank())) {
    shapeValues.push_back(
        builder.createOrFold<tensor::DimOp>(loc, operand, index));
  }

  reifiedReturnShapes.push_back(builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            builder.getIndexType()),
      shapeValues));

  return success();
}

//===----------------------------------------------------------------------===//
// BroadcastInDimOp
//===----------------------------------------------------------------------===//

LogicalResult BroadcastInDimOp::verify() {
  return hlo::verifyBroadcastInDimOp(
      getLoc(), getOperand(),
      llvm::to_vector(getBroadcastDimensions().getValues<int64_t>()),
      getResult());
}

namespace {

// Simplify BroadcastInDim has the following behaviors: replace BroadcastInDim
// with Reshape or Transpose if they are equivalent or replace
// BroadcastInDim(BroadcastInDim(X)) with BroadcastInDim(X)
class BroadcastInDimSimplifier : public OpRewritePattern<BroadcastInDimOp> {
public:
  using OpRewritePattern<BroadcastInDimOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(BroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
    auto operandType = dyn_cast<RankedTensorType>(op.getOperand().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!operandType || !resultType) {
      return failure();
    }
    auto bsDimIndices = op.getBroadcastDimensions().getValues<int64_t>();
    if (operandType.hasStaticShape() && resultType.hasStaticShape()) {
      bool sameTotalElements =
          operandType.getNumElements() == resultType.getNumElements();
      // BroadcastInDim equivalent to reshape
      if (llvm::is_sorted(bsDimIndices) && sameTotalElements) {
        rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(),
                                               op.getOperand());
        return success();
      }
      // BroadcastInDim equivalent to transpose
      if (operandType.getRank() == resultType.getRank() && sameTotalElements) {
        rewriter.replaceOpWithNewOp<TransposeOp>(
            op, op.getType(), op.getOperand(), op.getBroadcastDimensions());
        return success();
      }
    }
    // eliminate redundant BroadcastInDim
    if (auto broadcastInDimOp = llvm::dyn_cast_or_null<BroadcastInDimOp>(
            op.getOperand().getDefiningOp())) {
      auto newIndices = cast<DenseIntElementsAttr>(
          broadcastInDimOp.getBroadcastDimensions().mapValues(
              op.getBroadcastDimensions().getElementType(),
              [&bsDimIndices](const APInt &dim) -> APInt {
                return APInt(dim.getBitWidth(),
                             bsDimIndices[dim.getSExtValue()], true);
              }));
      rewriter.replaceOpWithNewOp<BroadcastInDimOp>(
          op, op.getType(), broadcastInDimOp.getOperand(), newIndices);
      return success();
    }
    return failure();
  }
};

} // namespace

void BroadcastInDimOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
  results.add<BroadcastInDimSimplifier>(context);
}

//===----------------------------------------------------------------------===//
// Case Op
//===----------------------------------------------------------------------===//

LogicalResult
CaseOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                         ValueRange operands, DictionaryAttr attributes,
                         OpaqueProperties properties, RegionRange regions,
                         SmallVectorImpl<Type> &inferredReturnTypes) {
  CaseOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferCaseOp(location, adaptor.getIndex(), adaptor.getRegions(),
                          inferredReturnTypes);
}

namespace {

LogicalResult inlineCaseConstantCondition(CaseOp caseOp,
                                          PatternRewriter &rewriter) {
  DenseIntElementsAttr indexAttr;
  if (!matchPattern(caseOp.getIndex(), m_Constant(&indexAttr))) {
    return failure();
  }
  int64_t index =
      indexAttr.getSplatValue<IntegerAttr>().getValue().getSExtValue();
  // For an OOB index, the last branch is executed as the default branch:
  // https://www.tensorflow.org/xla/operation_semantics#conditional
  if (index < 0 || index >= caseOp.getNumRegions())
    index = caseOp.getNumRegions() - 1;

  Region &region = caseOp.getRegion(index);
  if (!llvm::hasSingleElement(region))
    return failure();
  replaceOpWithRegion(rewriter, caseOp, region);
  return success();
}

} // namespace

void CaseOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add(&inlineCaseConstantCondition);
}

//===----------------------------------------------------------------------===//
// ClampOp
//===----------------------------------------------------------------------===//

LogicalResult ClampOp::inferReturnTypeComponents(
    MLIRContext *, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ClampOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferClampOp(location, adaptor.getMin(), adaptor.getOperand(),
                           adaptor.getMax(), inferredReturnShapes);
}

LogicalResult
ClampOp::reifyReturnTypeShapes(OpBuilder &builder, ValueRange operands,
                               SmallVectorImpl<Value> &reifiedReturnShapes) {
  // For `mhlo.clamp`, the first operand may be a scalar.
  return hlo::deriveShapeFromOperand(&builder, getOperation(), operands[1],
                                     &reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// CompareOp
//===----------------------------------------------------------------------===//

LogicalResult CompareOp::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  CompareOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferCompareOp(context, location, adaptor.getLhs(),
                             inferredReturnShapes);
}

LogicalResult
CompareOp::reifyReturnTypeShapes(OpBuilder &builder, ValueRange operands,
                                 SmallVectorImpl<Value> &reifiedReturnShapes) {
  return hlo::deriveShapeFromOperand(&builder, getOperation(), operands.front(),
                                     &reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// ConcatenateOp
//===----------------------------------------------------------------------===//

namespace {

class SingleOperandConcatenateToCast : public OpRewritePattern<ConcatenateOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConcatenateOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getVal().size() != 1)
      return failure();

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, op.getType(),
                                                op.getVal().front());
    return success();
  }
};

class ConcatenateOperandRemoval : public OpRewritePattern<ConcatenateOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConcatenateOp op,
                                PatternRewriter &rewriter) const override {
    auto axis = op.getDimension();
    llvm::SmallVector<Value, 6> newOperands;
    for (auto operand : op.getOperands()) {
      auto ty = cast<ShapedType>(operand.getType());
      if (!ty.hasRank() || ty.getDimSize(axis) != 0) {
        newOperands.push_back(operand);
      }
    }

    if (!newOperands.empty() && newOperands.size() < op.getNumOperands()) {
      rewriter.replaceOpWithNewOp<ConcatenateOp>(
          op, op.getResult().getType(), newOperands, op.getDimension());
      return success();
    }

    return failure();
  }
};

class ConcatenateForwarding : public OpRewritePattern<ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConcatenateOp op,
                                PatternRewriter &rewriter) const override {
    auto getFlattenedOperands = [&](const Value &val) -> ValueRange {
      auto definingOp = dyn_cast_or_null<ConcatenateOp>(val.getDefiningOp());
      // To avoid inflate the memory footprint, only flatten the ConcatenateOp
      // when it has only one use.
      if (definingOp && definingOp->hasOneUse() &&
          definingOp.getDimension() == op.getDimension())
        return definingOp.getVal();
      return val;
    };

    bool needToFlatten = false;
    int operandCount = 0;
    llvm::for_each(op.getVal(), [&](Value val) {
      auto result = getFlattenedOperands(val);
      if (result.size() != 1 || result[0] != val)
        needToFlatten = true;
      operandCount += result.size();
    });

    if (!needToFlatten)
      return failure();

    llvm::SmallVector<Value, 6> newOperands;
    newOperands.reserve(operandCount);

    for (auto operand : op.getVal()) {
      auto flattenedOperands = getFlattenedOperands(operand);
      newOperands.append(flattenedOperands.begin(), flattenedOperands.end());
    }

    rewriter.replaceOpWithNewOp<ConcatenateOp>(op, op.getResult().getType(),
                                               newOperands, op.getDimension());
    return success();
  }
};

} // namespace

LogicalResult ConcatenateOp::inferReturnTypes(
    MLIRContext *, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  ConcatenateOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferConcatenateOp(location, adaptor.getVal().getTypes(),
                                 adaptor.getDimension(), inferredReturnTypes);
}

void ConcatenateOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.add<ConcatenateOperandRemoval, ConcatenateForwarding,
              SingleOperandConcatenateToCast>(context);
}

LogicalResult ConcatenateOp::reifyReturnTypeShapes(
    OpBuilder &builder, ValueRange operands,
    SmallVectorImpl<Value> &reifiedReturnShapes) {
  ConcatenateOp::Adaptor adaptor(operands);
  auto inputs = adaptor.getVal();

  auto operandType = dyn_cast<RankedTensorType>(inputs[0].getType());
  // Not support unranked type a.t.m.
  if (!operandType)
    return failure();

  Location loc = this->getLoc();
  Type shapeScalarType = builder.getIndexType();
  auto toShapeScalarType = [&](Value v) {
    return maybeCastTo(builder, loc, v, shapeScalarType);
  };

  SmallVector<SmallVector<Value, 4>, 4> allShapeValues;
  for (size_t inputId = 0; inputId < inputs.size(); ++inputId) {
    Value operand = inputs[inputId];
    auto operandType = dyn_cast<RankedTensorType>(operand.getType());
    if (!operandType)
      return failure();

    SmallVector<Value, 4> shapeVals;
    for (const auto &element : llvm::enumerate(operandType.getShape())) {
      Value valueDim = toShapeScalarType(
          builder.create<tensor::DimOp>(loc, operand, element.index()));
      shapeVals.push_back(valueDim);
    }
    allShapeValues.emplace_back(std::move(shapeVals));
  }

  int axis = this->getDimension();
  auto &shapeValues = allShapeValues[0];
  for (size_t vecId = 1; vecId < allShapeValues.size(); ++vecId) {
    auto &otherShapeValues = allShapeValues[vecId];
    if (otherShapeValues.size() != shapeValues.size()) {
      this->emitOpError()
          << "Concatenate expects all operands must be of the same rank";
      return failure();
    }
    shapeValues[axis] = builder.create<arith::AddIOp>(loc, shapeValues[axis],
                                                      otherShapeValues[axis]);
  }

  Value outputShape = builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            shapeScalarType),
      shapeValues);
  reifiedReturnShapes.push_back(outputShape);

  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

// static
// Builds a constant op with the specified attribute `value`.
void ConstantOp::build(OpBuilder & /*builder*/, OperationState &result,
                       Attribute value) {
  Properties &properties = result.getOrAddProperties<Properties>();
  Type type;
  if (auto elemAttr = dyn_cast<ElementsAttr>(value)) {
    type = elemAttr.getType();
    properties.value = elemAttr;
  } else if (isa<BoolAttr, IntegerAttr>(value)) {
    // All ZKX types must be tensor types. In the build() method, we want to
    // provide more flexibility by allowing attributes of scalar types. But we
    // need to wrap it up with ElementsAttr to construct valid ZKX constants.
    type =
        RankedTensorType::get(/*shape=*/{}, cast<TypedAttr>(value).getType());
    properties.value = DenseElementsAttr::get(cast<TensorType>(type), value);
  }

  // TODO: support other ZKX specific types.
  assert(type && "unsupported attribute type for building mhlo.constant");
  result.types.push_back(type);
}

LogicalResult
ConstantOp::inferReturnTypes(MLIRContext *, std::optional<Location> location,
                             ValueRange operands, DictionaryAttr attributes,
                             OpaqueProperties properties, RegionRange regions,
                             SmallVectorImpl<Type> &inferredReturnTypes) {
  ConstantOpAdaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferConstantOp(location, adaptor.getValue(),
                              inferredReturnTypes);
}

bool ConstantOp::isCompatibleReturnTypes(TypeRange l, TypeRange r) {
  if (l.size() != r.size() || l.size() != 1)
    return false;
  auto lhsTy = cast<ShapedType>(l.front());
  auto rhsTy = cast<ShapedType>(r.front());
  if (!lhsTy || !rhsTy)
    return false;

  if (lhsTy == rhsTy)
    return true;

  Type lhsElementType = getElementTypeOrSelf(lhsTy);
  Type rhsElementType = getElementTypeOrSelf(rhsTy);
  // NOTE(chokobole): This allows us to create constants of prime field from
  // integer constants.
  if (isa<IntegerType>(lhsElementType) &&
      isa<prime_ir::field::PrimeFieldType>(rhsElementType)) {
    return lhsTy.clone(rhsElementType) == rhsTy;
  }
  return false;
}

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  return hlo::parseConstantOp(parser, result);
}

void ConstantOp::print(OpAsmPrinter &p) {
  hlo::printConstantOp(p, getOperation(), getValue());
}

OpFoldResult ConstantOp::fold(FoldAdaptor) { return getValue(); }

//===----------------------------------------------------------------------===//
// ConvertOp
//===----------------------------------------------------------------------===//

// static
void ConvertOp::build(OpBuilder &builder, OperationState &result, Value operand,
                      Type resultElementTy) {
  auto rankedTy = cast<RankedTensorType>(operand.getType());
  auto resultTy = RankedTensorType::get(rankedTy.getShape(), resultElementTy);
  build(builder, result, resultTy, operand);
}

namespace {

struct EliminateRedundantConvert : public OpRewritePattern<ConvertOp> {
  using OpRewritePattern<ConvertOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ConvertOp op,
                                PatternRewriter &rewriter) const override {
    auto convertOp = op.getOperand().getDefiningOp<ConvertOp>();
    if (!convertOp) {
      return failure();
    }
    auto firstType =
        cast<TensorType>(convertOp.getOperand().getType()).getElementType();
    auto secondType =
        cast<TensorType>(op.getOperand().getType()).getElementType();
    auto thirdType =
        cast<TensorType>(op.getResult().getType()).getElementType();
    Location loc = rewriter.getFusedLoc({convertOp->getLoc(), op->getLoc()});
    if (isa<IntegerType>(firstType) && isa<IntegerType>(secondType) &&
        isa<IntegerType>(thirdType)) {
      // fold when the second integer type's width is longer than first,
      // like i16 -> i32 -> i64, u16 -> i32 -> u32
      if (cast<IntegerType>(secondType).getWidth() >
          cast<IntegerType>(firstType).getWidth()) {
        Value result = rewriter.create<ConvertOp>(loc, op.getResult().getType(),
                                                  convertOp.getOperand());
        rewriter.replaceOp(op, result);
        return success();
      }
    }
    return failure();
  }
};

} // namespace

void ConvertOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<EliminateRedundantConvert>(context);
}

//===----------------------------------------------------------------------===//
// CreateTokenOp
//===----------------------------------------------------------------------===//

LogicalResult
CreateTokenOp::inferReturnTypes(MLIRContext *context,
                                std::optional<Location> location, ValueRange,
                                DictionaryAttr, OpaqueProperties, RegionRange,
                                SmallVectorImpl<Type> &inferredReturnTypes) {
  return hlo::inferCreateTokenOp(getMhloDialect(context), location,
                                 inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// DotGeneralOp
//===----------------------------------------------------------------------===//

LogicalResult DotGeneralOp::verify() {
  return hlo::verifyDotGeneralOp(
      getLoc(), getLhs(), getRhs(),
      getDotDimensionNumbersAttr().getLhsBatchingDimensions(),
      getDotDimensionNumbersAttr().getRhsBatchingDimensions(),
      getDotDimensionNumbersAttr().getLhsContractingDimensions(),
      getDotDimensionNumbersAttr().getRhsContractingDimensions(), getResult());
}

LogicalResult DotGeneralOp::reifyReturnTypeShapes(
    OpBuilder &builder, ValueRange operands,
    SmallVectorImpl<Value> &reifiedReturnShapes) {
  auto lhsRankedType = dyn_cast<RankedTensorType>(getLhs().getType());
  auto rhsRankedType = dyn_cast<RankedTensorType>(getRhs().getType());
  if (!lhsRankedType || !rhsRankedType)
    return failure();

  Adaptor adaptor(operands);
  auto dimNumbers = getDotDimensionNumbers();
  SmallVector<Value> dimensions;

  for (const int64_t lhsDim : dimNumbers.getLhsBatchingDimensions())
    dimensions.push_back(
        builder.create<tensor::DimOp>(getLoc(), adaptor.getLhs(), lhsDim));

  for (int64_t i = 0; i < lhsRankedType.getRank(); i++)
    if (!llvm::is_contained(dimNumbers.getLhsContractingDimensions(), i) &&
        !llvm::is_contained(dimNumbers.getLhsBatchingDimensions(), i))
      dimensions.push_back(
          builder.create<tensor::DimOp>(getLoc(), adaptor.getLhs(), i));

  for (int64_t i = 0; i < rhsRankedType.getRank(); i++)
    if (!llvm::is_contained(dimNumbers.getRhsContractingDimensions(), i) &&
        !llvm::is_contained(dimNumbers.getRhsBatchingDimensions(), i))
      dimensions.push_back(
          builder.create<tensor::DimOp>(getLoc(), adaptor.getRhs(), i));

  reifiedReturnShapes.push_back(
      builder.create<tensor::FromElementsOp>(getLoc(), dimensions));
  return success();
}

//===----------------------------------------------------------------------===//
// DynamicBroadcastInDimOp
//===----------------------------------------------------------------------===//

LogicalResult DynamicBroadcastInDimOp::verify() {
  // Check for unranked dynamism. Unranked dynamism is not supported by
  // StableHLO (hlo::verifyReshapeOp will fail) and we can't verify
  // anything statically in that case anyway.
  auto outputdimensionsType = cast<ShapedType>(getOutputDimensions().getType());
  auto resultType = cast<ShapedType>(getResult().getType());
  if (!outputdimensionsType.hasRank() || !resultType.hasRank()) {
    return success();
  }

  return hlo::verifyDynamicBroadcastInDimOp(
      getLoc(), getOperand(), getOutputDimensions(),
      llvm::to_vector(getBroadcastDimensions().getValues<int64_t>()),
      getKnownExpandingDimensionsAttr()
          ? std::optional<SmallVector<int64_t>>(llvm::to_vector(
                getKnownExpandingDimensions()->getValues<int64_t>()))
          : std::nullopt,
      getKnownNonexpandingDimensions()
          ? std::optional<SmallVector<int64_t>>(llvm::to_vector(
                getKnownNonexpandingDimensions()->getValues<int64_t>()))
          : std::nullopt,
      getResult());
}

namespace {

// Does the same as PatternRewriter::replaceOpWithNewOp, but with a twist.
//
// Sometimes, we want to replace an op with a new op and simultaneously refine
// the result type from a dynamically-shaped type to a statically-shaped type.
// (Search for usages of this function for examples).
//
// Oftentimes, this works just fine because MHLO is designed to accommodate
// this kind of type refinements. But sometimes, this doesn't work - when
// the op is used outside of the MHLO dialect (e.g. in func.return). In these
// cases, we insert a tensor.cast to smooth things out.
template <typename OpTy, typename... Args>
OpTy refineOpWithNewOp(PatternRewriter &rewriter, Operation *op,
                       Args &&...args) {
  auto newOp = rewriter.create<OpTy>(op->getLoc(), std::forward<Args>(args)...);

  llvm::SmallVector<Value> replacementResults;
  assert(op->getNumResults() == newOp->getNumResults() &&
         "replacement op doesn't match results of original op");
  for (auto [opResult, newOpResult] :
       llvm::zip(op->getResults(), newOp->getResults())) {
    Value replacementResult = newOpResult;
    if (llvm::any_of(opResult.getUsers(), [&](Operation *user) {
          return user->getDialect() != op->getDialect();
        })) {
      replacementResult = rewriter.create<tensor::CastOp>(
          op->getLoc(), opResult.getType(), newOpResult);
    }
    replacementResults.push_back(replacementResult);
  }

  rewriter.replaceOp(op, replacementResults);
  return newOp;
}

// If a DynamicBroadCastInDimOp is not actually dynamic, use an ordinary
// BroadcastInDimOp.
class DynamicBroadcastInDimOpNotActuallyDynamic
    : public OpRewritePattern<DynamicBroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicBroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    auto operandType = dyn_cast<RankedTensorType>(op.getOperand().getType());
    auto *outputDimOp = op.getOutputDimensions().getDefiningOp();
    if (!type || !operandType || !operandType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op, "requires operand static shape");
    }
    // output has static shape, replace with broadcast_in_dim
    if (type.hasStaticShape()) {
      rewriter.replaceOpWithNewOp<BroadcastInDimOp>(
          op, type, op.getOperand(), op.getBroadcastDimensions());
      return success();
    }
    // output_dimensions are constant, set output shape with output_dimensions,
    // then replace with broadcast_in_dim
    if (outputDimOp && outputDimOp->hasTrait<mlir::OpTrait::ConstantLike>()) {
      DenseIntElementsAttr shapeAttr;
      if (matchPattern(outputDimOp, m_Constant(&shapeAttr))) {
        SmallVector<int64_t> outputShape;
        for (APInt shape : shapeAttr.getValues<APInt>()) {
          outputShape.push_back(shape.getZExtValue());
        }
        refineOpWithNewOp<BroadcastInDimOp>(
            rewriter, op,
            RankedTensorType::get(outputShape, type.getElementType()),
            op.getOperand(), op.getBroadcastDimensions());
        return success();
      }
    }
    return rewriter.notifyMatchFailure(
        op, "requires output static shape or constant broadcast dimensions");
  }
};

class ChainedDynamicBroadcastInDimCanonicalization
    : public OpRewritePattern<DynamicBroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicBroadcastInDimOp bcast,
                                PatternRewriter &rewriter) const override {
    auto precedingBcast =
        bcast.getOperand().getDefiningOp<DynamicBroadcastInDimOp>();
    if (!precedingBcast)
      return failure();

    // Compose broadcast dimensions.
    DenseIntElementsAttr precedingBcastDims =
        precedingBcast.getBroadcastDimensions();
    DenseIntElementsAttr bcastDims = bcast.getBroadcastDimensions();
    SmallVector<APInt, 4> composition;
    for (APInt precedingDim : precedingBcastDims) {
      composition.push_back(
          bcastDims.getValues<APInt>()[precedingDim.getZExtValue()]);
    }
    auto composedBcastDims =
        DenseIntElementsAttr::get(precedingBcastDims.getType(), composition);

    rewriter.replaceOpWithNewOp<DynamicBroadcastInDimOp>(
        bcast, bcast.getType(), precedingBcast.getOperand(),
        bcast.getOutputDimensions(), composedBcastDims);
    return success();
  }
};

// If all dimensions are known to be nonexpanding from the attribute, replace
// the dynamic broadcast with a cast.
class DynamicBroadcastInDimAllDimsNonExpanding
    : public OpRewritePattern<DynamicBroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicBroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "requires ranked result type");

    if (!op.getKnownNonexpandingDimensions().has_value() ||
        op.getKnownNonexpandingDimensions()->size() != resultType.getRank()) {
      return rewriter.notifyMatchFailure(
          op, "known_nonexpanding_dimensions don't cover all output dims");
    }

    auto cast = rewriter.createOrFold<tensor::CastOp>(op.getLoc(), resultType,
                                                      op.getOperand());
    rewriter.replaceOp(op, cast);
    return success();
  }
};
} // namespace

void DynamicBroadcastInDimOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<ChainedDynamicBroadcastInDimCanonicalization,
              DynamicBroadcastInDimOpNotActuallyDynamic,
              DynamicBroadcastInDimAllDimsNonExpanding>(context);
}

LogicalResult DynamicBroadcastInDimOp::reifyReturnTypeShapes(
    OpBuilder &builder, ValueRange operands,
    SmallVectorImpl<Value> &reifiedReturnShapes) {
  DynamicBroadcastInDimOp::Adaptor adaptor(operands);
  reifiedReturnShapes.push_back(
      castToIndexTensor(builder, getLoc(), adaptor.getOutputDimensions()));
  return success();
}

//===----------------------------------------------------------------------===//
// DynamicSliceOp
//===----------------------------------------------------------------------===//

namespace {

// Given the start indices and slice sizes for a dynamic-slice that can be
// converted to a static slice, returns the limits for the static slice.
DenseIntElementsAttr buildSliceLimits(DenseIntElementsAttr startIndices,
                                      DenseIntElementsAttr sliceSizes,
                                      Builder *builder) {
  SmallVector<int64_t, 4> sliceLimits;
  for (int64_t i = 0; i < sliceSizes.getNumElements(); ++i) {
    int64_t startIndex = startIndices.getValues<IntegerAttr>()[i].getInt();
    int64_t sliceSize = sliceSizes.getValues<IntegerAttr>()[i].getInt();
    sliceLimits.push_back(startIndex + sliceSize);
  }
  return builder->getI64TensorAttr(sliceLimits);
}

// Canonicalizes DynamicSlice ops that can be replaced instead with Slice ops.
// This canonicalization is applied the case when the `begin` input values are
// compile time constants and thus can be made into a tensor.
struct DynamicSliceToSlice : public OpRewritePattern<DynamicSliceOp> {
  using OpRewritePattern<DynamicSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DynamicSliceOp dynamicSlice,
                                PatternRewriter &rewriter) const override {
    Value input = dynamicSlice.getOperand();
    auto inputTensor = dyn_cast<RankedTensorType>(input.getType());
    if (!inputTensor || !inputTensor.hasStaticShape())
      return failure();

    auto sliceSizes = dynamicSlice.getSliceSizes().getValues<int64_t>();
    SmallVector<int64_t, 4> tempStartIndices;
    for (const auto &indexAndSliceStart :
         llvm::enumerate(dynamicSlice.getStartIndices())) {
      APInt val;
      Value start = indexAndSliceStart.value();
      int64_t index = indexAndSliceStart.index();
      if (!matchPattern(start, m_ConstantInt(&val))) {
        return failure();
      }
      // Clamp the indices within bounds to faithfully mirror dynamic slice
      // semantics.
      int64_t clampedStart =
          std::max(static_cast<int64_t>(0),
                   std::min(val.getSExtValue(),
                            inputTensor.getDimSize(index) - sliceSizes[index]));
      tempStartIndices.push_back(clampedStart);
    }

    // At this point we've determined that the start indices are all constants;
    // pack them into a single tensor.
    auto loc = dynamicSlice.getLoc();
    int64_t inputRank = inputTensor.getRank();
    auto sliceStartIndices = rewriter.getI64TensorAttr(tempStartIndices);
    DenseIntElementsAttr sliceLimits = buildSliceLimits(
        sliceStartIndices, dynamicSlice.getSliceSizes(), &rewriter);
    DenseIntElementsAttr sliceStrides =
        rewriter.getI64TensorAttr(SmallVector<int64_t, 4>(inputRank, 1));
    auto result = rewriter.create<SliceOp>(loc, input, sliceStartIndices,
                                           sliceLimits, sliceStrides);
    rewriter.replaceOp(dynamicSlice, result);
    return success();
  }
};

} // namespace

void DynamicSliceOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.add<DynamicSliceToSlice>(context);
}

LogicalResult DynamicSliceOp::inferReturnTypeComponents(
    MLIRContext *, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  DynamicSliceOp::Adaptor adaptor(operands, attributes, properties, regions);
  if (failed(verify1dTensor(location, adaptor.getSliceSizes(), "slice_sizes")))
    return failure();
  return hlo::inferDynamicSliceOp(
      location, adaptor.getOperand().getType(),
      adaptor.getStartIndices().getTypes(),
      llvm::to_vector(adaptor.getSliceSizes().getValues<int64_t>()),
      inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// DynamicUpdateSliceOp
//===----------------------------------------------------------------------===//

LogicalResult DynamicUpdateSliceOp::inferReturnTypeComponents(
    MLIRContext *, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  DynamicUpdateSliceOp::Adaptor adaptor(operands, attributes, properties,
                                        regions);
  return hlo::inferDynamicUpdateSliceOp(
      location, adaptor.getOperand(), adaptor.getUpdate(),
      adaptor.getStartIndices(), inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// GetDimensionSizeOp
//===----------------------------------------------------------------------===//

LogicalResult GetDimensionSizeOp::inferReturnTypeComponents(
    MLIRContext *, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  GetDimensionSizeOp::Adaptor adaptor(operands, attributes, properties,
                                      regions);
  return hlo::inferGetDimensionSizeOp(location, adaptor.getOperand().getType(),
                                      adaptor.getDimension(),
                                      inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// GetTupleElementOp
//===----------------------------------------------------------------------===//

LogicalResult GetTupleElementOp::inferReturnTypes(
    MLIRContext *, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  GetTupleElementOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferGetTupleElementOp(location, adaptor.getOperand(),
                                     adaptor.getIndex(), inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// If Op
//===----------------------------------------------------------------------===//

LogicalResult
IfOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                       ValueRange operands, DictionaryAttr attributes,
                       OpaqueProperties properties, RegionRange regions,
                       SmallVectorImpl<Type> &inferredReturnTypes) {
  IfOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferIfOp(location, adaptor.getPred(), adaptor.getRegions(),
                        inferredReturnTypes);
}

static LogicalResult inlineIfConstantCondition(IfOp ifOp,
                                               PatternRewriter &rewriter) {
  DenseIntElementsAttr predAttr;
  if (!matchPattern(ifOp.getPred(), m_Constant(&predAttr)))
    return failure();

  if (predAttr.getSplatValue<BoolAttr>().getValue()) {
    replaceOpWithRegion(rewriter, ifOp, ifOp.getTrueBranch());
  } else {
    replaceOpWithRegion(rewriter, ifOp, ifOp.getFalseBranch());
  }
  return success();
}

void IfOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                       MLIRContext *context) {
  results.add(&inlineIfConstantCondition);
}

//===----------------------------------------------------------------------===//
// IotaOp
//===----------------------------------------------------------------------===//

LogicalResult IotaOp::verify() {
  return hlo::verifyIotaOp(getLoc(), getIotaDimension(), getResult());
}

namespace {

// Iota operations across multiple dimensions can be reduced to an iota and a
// ranked broadcast.
struct IotaBroadcast : public OpRewritePattern<IotaOp> {
  using OpRewritePattern<IotaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IotaOp iota,
                                PatternRewriter &rewriter) const override {
    auto resultTy = cast<ShapedType>(iota.getType());
    if (!resultTy.hasRank() || resultTy.getRank() < 2) {
      return failure();
    }

    auto iotaDimension = iota.getIotaDimension();

    auto iotaType = RankedTensorType::get({resultTy.getDimSize(iotaDimension)},
                                          resultTy.getElementType());

    auto newIota = rewriter.create<IotaOp>(iota.getLoc(), iotaType,
                                           rewriter.getI64IntegerAttr(0));

    auto broadcastAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({1}, rewriter.getIntegerType(64)),
        {iotaDimension});
    rewriter.replaceOpWithNewOp<BroadcastInDimOp>(iota, resultTy, newIota,
                                                  broadcastAttr);
    return success();
  }
};

} // namespace

void IotaOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<IotaBroadcast>(context);
}

//===----------------------------------------------------------------------===//
// MapOp
//===----------------------------------------------------------------------===//

LogicalResult MapOp::inferReturnTypeComponents(
    MLIRContext *, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  MapOp::Adaptor adaptor(operands, attributes, properties, regions);
  if (failed(verify1dTensor(location, adaptor.getDimensions(), "dimensions")))
    return failure();
  return hlo::inferMapOp(
      location, adaptor.getInputs(),
      llvm::to_vector(adaptor.getDimensions().getValues<int64_t>()),
      adaptor.getComputation(), inferredReturnShapes);
}

LogicalResult
MapOp::reifyReturnTypeShapes(OpBuilder &builder, ValueRange operands,
                             SmallVectorImpl<Value> &reifiedReturnShapes) {
  return hlo::deriveShapeFromOperand(&builder, getOperation(), operands.front(),
                                     &reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

LogicalResult
PadOp::inferReturnTypes(MLIRContext *, std::optional<Location> location,
                        ValueRange operands, DictionaryAttr attributes,
                        OpaqueProperties properties, RegionRange regions,
                        SmallVectorImpl<Type> &inferredReturnTypes) {
  PadOp::Adaptor adaptor(operands, attributes, properties, regions);
  if (failed(verify1dTensor(location, adaptor.getEdgePaddingLow(),
                            "edge_padding_low")) ||
      failed(verify1dTensor(location, adaptor.getEdgePaddingHigh(),
                            "edge_padding_high")))
    return failure();
  return hlo::inferPadOp(
      location, adaptor.getOperand().getType(),
      adaptor.getPaddingValue().getType(),
      llvm::to_vector(adaptor.getEdgePaddingLow().getValues<int64_t>()),
      llvm::to_vector(adaptor.getEdgePaddingHigh().getValues<int64_t>()),
      inferredReturnTypes);
}

LogicalResult
PadOp::reifyReturnTypeShapes(OpBuilder &builder, ValueRange operands,
                             SmallVectorImpl<Value> &reifiedReturnShapes) {
  PadOp::Adaptor adaptor(operands, this->getOperation()->getAttrDictionary(),
                         this->getOperation()->getPropertiesStorage());
  auto loc = this->getLoc();
  Value operand = adaptor.getOperand();
  auto operandTy = cast<RankedTensorType>(operand.getType());

  llvm::SmallVector<int32_t> padHigh;
  llvm::SmallVector<int32_t> padLow;
  llvm::SmallVector<int32_t> padInterior;

  auto padHighAttr = adaptor.getEdgePaddingHigh();
  auto padLowAttr = adaptor.getEdgePaddingLow();

  padHigh.reserve(padHighAttr.getNumElements());
  padLow.reserve(padLowAttr.getNumElements());

  for (const APInt &val : padHighAttr.getValues<APInt>())
    padHigh.push_back(val.getSExtValue());

  for (const APInt &val : padLowAttr.getValues<APInt>())
    padLow.push_back(val.getSExtValue());

  llvm::SmallVector<Value> dimensions;
  dimensions.reserve(operandTy.getRank());
  for (int i = 0, s = operandTy.getRank(); i < s; ++i) {
    Value padEdge =
        builder.create<arith::ConstantIndexOp>(loc, padHigh[i] + padLow[i]);

    // First we grab the initial interior size.
    Value dim = builder.create<tensor::DimOp>(loc, operand, i).getResult();

    // Then we add the padding on the edge of the tensor.
    dim = builder.create<arith::AddIOp>(loc, dim, padEdge).getResult();
    dimensions.push_back(dim);
  }

  Value dimensionTensor =
      builder.create<tensor::FromElementsOp>(loc, dimensions).getResult();
  reifiedReturnShapes.push_back(dimensionTensor);
  return success();
}

namespace {

// If the input tensor has a dimension of length-0, the input tensor is
// irrelevant. Instead we can broadcast the pad value to the output size rather
// than pad the input tensor.
struct PadEmptyTensor : public OpRewritePattern<PadOp> {
  using OpRewritePattern<PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PadOp op,
                                PatternRewriter &rewriter) const override {
    auto operand = op.getOperand();
    auto padVal = op.getPaddingValue();

    auto operandTy = cast<RankedTensorType>(operand.getType());
    auto resultTy = cast<RankedTensorType>(op.getType());

    if (llvm::all_of(operandTy.getShape(), [](int64_t d) { return d != 0; })) {
      return failure();
    }

    if (resultTy.hasStaticShape()) {
      auto dimsType = RankedTensorType::get({0}, rewriter.getIntegerType(64));
      auto dims =
          DenseIntElementsAttr::get(dimsType, SmallVector<int64_t, 1>{});
      rewriter.replaceOpWithNewOp<BroadcastInDimOp>(op, resultTy, padVal, dims);
      return success();
    }

    llvm::SmallVector<Value> reifiedShapes;
    if (failed(op.reifyReturnTypeShapes(rewriter, op.getOperands(),
                                        reifiedShapes))) {
      return failure();
    }

    auto dimsType = RankedTensorType::get({0}, rewriter.getIntegerType(64));
    auto broadcastDims =
        DenseIntElementsAttr::get(dimsType, SmallVector<int64_t, 1>{});
    rewriter.replaceOpWithNewOp<DynamicBroadcastInDimOp>(
        op, op.getType(), padVal, reifiedShapes.front(), broadcastDims);
    return success();
  }
};

} // namespace

void PadOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<PadEmptyTensor>(context);
}

//===----------------------------------------------------------------------===//
// RealDynamicSliceOp
//===----------------------------------------------------------------------===//
// Verifies that operand rank matches start_indices/limit_indices/strides size
LogicalResult RealDynamicSliceOp::verify() {
  return hlo::verifyRealDynamicSliceOp(getLoc(), getOperand(),
                                       getStartIndices(), getLimitIndices(),
                                       getStrides());
}

namespace {

struct RealDSliceToDSlice : public OpRewritePattern<RealDynamicSliceOp> {
  using OpRewritePattern<RealDynamicSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(RealDynamicSliceOp op,
                                PatternRewriter &rewriter) const override {
    // This rewrite only works for unit strides because DynamicSliceOp
    // doesn't support strides (i.e. it implicitly has unit strides).
    DenseIntElementsAttr stridesAttr;
    if (!matchPattern(op.getStrides(), m_Constant(&stridesAttr)))
      return rewriter.notifyMatchFailure(op, "requires constant strides");
    if (!llvm::all_of(stridesAttr.getValues<APInt>(),
                      [&](APInt stride) { return stride == 1; }))
      return rewriter.notifyMatchFailure(op, "requires unit strides");

    // Check that slice sizes are fully static (DynamicSliceOp style).
    // To detect that, we check whether `limit_indices` is defined as
    // `start_indices + constant` or `constant + start_indices`.
    DenseIntElementsAttr sliceSizesAttr;
    auto m_startIndices = matchers::m_Val(op.getStartIndices());
    if (!matchPattern(
            op.getLimitIndices(),
            m_Op<AddOp>(m_startIndices, m_Constant(&sliceSizesAttr))) &&
        !matchPattern(op.getLimitIndices(),
                      m_Op<AddOp>(m_Constant(&sliceSizesAttr), m_startIndices)))
      return rewriter.notifyMatchFailure(
          op, "requires limit indices equal to start indices plus constant");

    // RealDynamicSliceOp can take tensors of integer or index element types.
    // DynamicSliceOp::slice_sizes only supports i64 element type.
    // Adapt accordingly in order to be compatible with DynamicSliceOp.
    SmallVector<int64_t> sliceSizes;
    for (auto element : sliceSizesAttr.getValues<APInt>()) {
      sliceSizes.push_back(element.getSExtValue());
    }

    // RealDynamicSliceOp::start_indices is a 1-dimensional tensor.
    // DynamicSliceOp::start_indices is a vararg of 0-dimensional tensors.
    // Adapt accordingly in order to be compatible with DynamicSliceOp.
    SmallVector<Value> startIndices;
    for (auto i = 0; i < static_cast<int64_t>(sliceSizes.size()); ++i) {
      auto startIndex1D = rewriter.create<SliceOp>(
          op.getLoc(), op.getStartIndices(), rewriter.getI64TensorAttr(i),
          rewriter.getI64TensorAttr(i + 1), rewriter.getI64TensorAttr(1));
      auto startIndex0DType = RankedTensorType::get(
          {},
          cast<ShapedType>(op.getStartIndices().getType()).getElementType());
      auto startIndex0D = rewriter.create<ReshapeOp>(
          op.getLoc(), startIndex0DType, startIndex1D);
      startIndices.push_back(startIndex0D);
    }

    rewriter.replaceOpWithNewOp<DynamicSliceOp>(
        op, op.getOperand(), startIndices,
        rewriter.getI64TensorAttr(sliceSizes));
    return success();
  }
};

} // namespace

void RealDynamicSliceOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                     MLIRContext *context) {
  results.add<RealDSliceToDSlice>(context);
}

LogicalResult RealDynamicSliceOp::reifyReturnTypeShapes(
    OpBuilder &builder, ValueRange operands,
    SmallVectorImpl<Value> &reifiedReturnShapes) {
  RealDynamicSliceOp::Adaptor adaptor(operands);
  Value operand = adaptor.getOperand();
  Value startIndices = adaptor.getStartIndices();
  Value limitIndices = adaptor.getLimitIndices();
  Value strides = adaptor.getStrides();

  auto operandType = dyn_cast<RankedTensorType>(operand.getType());
  // Not support unranked type a.t.m.
  if (!operandType)
    return failure();

  Location loc = this->getLoc();
  SmallVector<Value, 4> shapeValues;
  shapeValues.reserve(operandType.getRank());
  Type shapeScalarType =
      cast<ShapedType>(startIndices.getType()).getElementType();
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  one = maybeCastTo(builder, loc, one, shapeScalarType);
  for (const auto &element : llvm::enumerate(operandType.getShape())) {
    Value offset = builder.create<arith::ConstantIndexOp>(loc, element.index());
    Value valueStart =
        builder.create<tensor::ExtractOp>(loc, startIndices, offset);
    Value valueLimit =
        builder.create<tensor::ExtractOp>(loc, limitIndices, offset);
    Value valueStride = builder.create<tensor::ExtractOp>(loc, strides, offset);
    // size = (limit - start + stride - 1) / stride
    shapeValues.push_back(builder.create<arith::DivSIOp>(
        loc,
        builder.create<arith::SubIOp>(
            loc,
            builder.create<arith::AddIOp>(
                loc, valueStride,
                builder.create<arith::SubIOp>(loc, valueLimit, valueStart)),
            one),
        valueStride));
  }

  reifiedReturnShapes.push_back(builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            shapeScalarType),
      shapeValues));
  return success();
}

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//

void ReduceOp::print(OpAsmPrinter &p) {
  auto dimensions = llvm::to_vector(getDimensions().getValues<int64_t>());
  hlo::printReduceOp(p, getOperation(), getInputs(), dimensions, getBody());
}

ParseResult ReduceOp::parse(OpAsmParser &parser, OperationState &result) {
  auto parseDenseElements = [](OpBuilder &b,
                               ArrayRef<int64_t> dims) -> Attribute {
    return b.getI64TensorAttr(dims);
  };
  return hlo::parseReduceOp(parser, result, parseDenseElements);
}

LogicalResult ReduceOp::inferReturnTypeComponents(
    MLIRContext *, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ReduceOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferReduceOp(
      location, adaptor.getInputs().getTypes(),
      llvm::to_vector(adaptor.getDimensions().getValues<int64_t>()),
      adaptor.getBody(), inferredReturnShapes);
}

void ReduceOp::build(OpBuilder &, OperationState &odsState, ValueRange inputs,
                     ValueRange initValues, DenseIntElementsAttr dimensions,
                     TypeRange elementTypes) {
  odsState.addOperands(inputs);
  odsState.addOperands(initValues);
  Properties &properties = odsState.getOrAddProperties<Properties>();
  properties.dimensions = dimensions;
  (void)odsState.addRegion();

  SmallVector<int64_t> newDimensions;
  Attribute encoding;
  ReduceOp::Adaptor adaptor(
      odsState.operands,
      odsState.attributes.getDictionary(odsState.getContext()), {},
      odsState.regions);

  SmallVector<ShapedType> inputArgTensorTypes{
      llvm::map_range(adaptor.getInputs().getTypes(),
                      [](Type t) { return cast<ShapedType>(t); })};
  SmallVector<ShapedType> initValueTensorTypes{
      llvm::map_range(adaptor.getInitValues().getTypes(),
                      [](Type t) { return cast<ShapedType>(t); })};

  if (succeeded(hlo::verifyReduceOpInputsAndInferShape(
          odsState.location, inputArgTensorTypes,
          llvm::to_vector(dimensions.getValues<int64_t>()), newDimensions,
          encoding))) {
    SmallVector<Type> inferredReturnTypes;
    for (uint64_t inputIdx = 0; inputIdx < inputArgTensorTypes.size();
         ++inputIdx) {
      Type elementTy = elementTypes[inputIdx];
      ShapedType inputType = inputArgTensorTypes[inputIdx];
      if (inputType.hasRank()) {
        inferredReturnTypes.push_back(
            RankedTensorType::get(newDimensions, elementTy, encoding));
      } else {
        assert(encoding == nullptr && "attribute not supported");
        inferredReturnTypes.push_back(UnrankedTensorType::get(elementTy));
      }
    }
    odsState.addTypes(inferredReturnTypes);
  } else {
    llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

LogicalResult ReduceOp::verify() {
  if (failed(verify1dTensor(getLoc(), getDimensions(), "dimensions")))
    return failure();
  return hlo::verifyReduceOp(
      getLoc(), getInputs(), getInitValues(),
      llvm::to_vector(getDimensions().getValues<int64_t>()), getBody());
}

//===----------------------------------------------------------------------===//
// ReduceWindowOp
//===----------------------------------------------------------------------===//

LogicalResult ReduceWindowOp::inferReturnTypeComponents(
    MLIRContext *, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ReduceWindowOp::Adaptor adaptor(operands, attributes, properties, regions);
  auto optToVec = [](std::optional<DenseIntElementsAttr> attr)
      -> std::optional<SmallVector<int64_t>> {
    if (!attr)
      return std::nullopt;
    return llvm::to_vector(attr->getValues<int64_t>());
  };
  return hlo::inferReduceWindowOp(
      location, adaptor.getInputs(), adaptor.getInitValues(),
      llvm::to_vector(adaptor.getWindowDimensions().getValues<int64_t>()),
      optToVec(adaptor.getWindowStrides()),
      optToVec(adaptor.getBaseDilations()),
      optToVec(adaptor.getWindowDilations()), adaptor.getPadding(),
      adaptor.getBody(), inferredReturnShapes);
}

LogicalResult ReduceWindowOp::verify() {
  auto optToVec = [](std::optional<DenseIntElementsAttr> attr)
      -> std::optional<SmallVector<int64_t>> {
    if (!attr)
      return std::nullopt;
    return llvm::to_vector(attr->getValues<int64_t>());
  };
  return hlo::verifyReduceWindowOp(
      getLoc(), getInputs(), getInitValues(),
      llvm::to_vector(getWindowDimensions().getValues<int64_t>()),
      optToVec(getWindowStrides()), optToVec(getBaseDilations()),
      optToVec(getWindowDilations()), getPadding(), getBody());
}

namespace {

// Enable constant folding to occur within the region of the ReduceOp
// by replacing block argument uses with constants if:
//  1. All the ReduceOp operands are splat constants.
//  2. The ReduceOp region consists of a single logical AND or logical OR.
// The pattern leverages the idempotent property of the AND and OR operators
// to determine the value of a reduction on splat constants. Other boolean
// operators do not have this property, and need separate patterns to resolve
// reductions of their splat constants.
struct LowerBoolSplatConstantsIntoRegion : public OpRewritePattern<ReduceOp> {
  using OpRewritePattern<ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter &rewriter) const override {
    mlir::Block &bb = op.getBody().front();

    // Ensure only a compute op and return op exist and the
    // compute op is an AND or OR op.
    if (bb.getOperations().size() != 2)
      return failure();
    if (!mlir::isa<AndOp, OrOp>(bb.front()))
      return failure();

    // Ensure all operands are splat constants.
    SmallVector<DenseElementsAttr, 4> bargCstAttrs;
    for (auto inpAndBarg : llvm::zip(op.getOperands(), bb.getArguments())) {
      Value inp = std::get<0>(inpAndBarg);
      BlockArgument barg = std::get<1>(inpAndBarg);
      ConstantOp cst = inp.getDefiningOp<ConstantOp>();
      if (!cst)
        return failure();

      auto cstAttr = dyn_cast_or_null<DenseElementsAttr>(cst.getValue());
      if (!cstAttr.isSplat()) {
        return rewriter.notifyMatchFailure(op, "Must be splat constant.");
      }

      auto bargShapedType = dyn_cast<ShapedType>(barg.getType());
      if (!bargShapedType)
        return failure();

      auto bargCstAttr = DenseElementsAttr::get(
          bargShapedType, cstAttr.getSplatValue<mlir::Attribute>());
      bargCstAttrs.push_back(bargCstAttr);
    }

    // Create new splat constants to replace block arguments.
    for (BlockArgument barg : bb.getArguments()) {
      int argIdx = barg.getArgNumber();
      ConstantOp newCst = rewriter.create<mhlo::ConstantOp>(
          bb.front().getLoc(), barg.getType(), bargCstAttrs[argIdx]);
      barg.replaceAllUsesWith(newCst);
    }
    return success();
  }
};

LogicalResult convertEmptyReduces(ReduceOp op, PatternRewriter &rewriter) {
  // We require all reduce shapes to be the same, up to the element types, so we
  // can just the first operand and the first result as a representative.
  RankedTensorType t =
      dyn_cast<RankedTensorType>(op.getInputs().getType().front());
  if (!t)
    return rewriter.notifyMatchFailure(op.getLoc(),
                                       "unranked input unsupported");
  bool zeroExtent = any_of(t.getShape(), [](int64_t d) { return d == 0; });
  if (zeroExtent) {
    auto empty = rewriter.getI64TensorAttr({});
    if (t.hasStaticShape()) {
      for (auto [init, out] : llvm::zip(op.getInitValues(), op.getResults())) {
        out.replaceAllUsesWith(rewriter.create<BroadcastInDimOp>(
            op.getLoc(), out.getType(), init, empty));
      }
      return success();
    }

    SmallVector<Value, 4> shapes;
    if (failed(op.reifyReturnTypeShapes(rewriter, op.getOperands(), shapes)))
      return failure();
    for (auto [init, shape, out] :
         llvm::zip(op.getInitValues(), shapes, op.getResults())) {
      out.replaceAllUsesWith(rewriter.create<DynamicBroadcastInDimOp>(
          op.getLoc(), out.getType(), init, shape, empty));
    }
    return success();
  }
  return rewriter.notifyMatchFailure(op.getLoc(), "non-empty input");
}

} // namespace

void ReduceOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<LowerBoolSplatConstantsIntoRegion>(context);
  results.add(convertEmptyReduces);
}

LogicalResult
ReduceOp::reifyReturnTypeShapes(OpBuilder &builder, ValueRange operands,
                                SmallVectorImpl<Value> &reifiedReturnShapes) {
  ReduceOp::Adaptor adaptor(operands);
  auto inputs = adaptor.getInputs();

  auto operandType = dyn_cast<RankedTensorType>(inputs[0].getType());
  // Not support unranked type a.t.m.
  if (!operandType)
    return failure();

  Location loc = this->getLoc();
  SmallVector<Value, 4> shapeValues;
  SmallVector<int64_t, 4> dimensions(
      this->getDimensions().getValues<int64_t>());
  shapeValues.reserve(operandType.getRank());
  Type shapeScalarType = builder.getIndexType();
  auto toShapeScalarType = [&](Value v) {
    return maybeCastTo(builder, loc, v, shapeScalarType);
  };

  for (const auto &element : llvm::enumerate(operandType.getShape())) {
    int64_t idx = element.index();
    auto *it = std::find(dimensions.begin(), dimensions.end(), idx);
    if (it != dimensions.end()) {
      continue;
    }
    Value valueDim = toShapeScalarType(
        builder.create<tensor::DimOp>(loc, inputs[0], element.index()));
    shapeValues.push_back(valueDim);
  }

  Value outputShape = builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            shapeScalarType),
      shapeValues);
  for (size_t i = 0; i < inputs.size(); ++i) {
    reifiedReturnShapes.push_back(outputShape);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

LogicalResult ReshapeOp::verify() {
  // Check for unranked dynamism. Unranked dynamism is not supported by
  // StableHLO (hlo::verifyReshapeOp will fail) and we can't verify
  // anything statically in that case anyway.
  auto operandType = cast<ShapedType>(getOperand().getType());
  auto resultType = cast<ShapedType>(getResult().getType());
  if (!operandType.hasRank() || !resultType.hasRank()) {
    return success();
  }
  return hlo::verifyReshapeOp(getLoc(), getOperand(), getResult());
}

//===----------------------------------------------------------------------===//
// BitReverseOp
//===----------------------------------------------------------------------===//

LogicalResult BitReverseOp::verify() {
  if (failed(verify1dTensor(getLoc(), getDimensions(), "dimensions")))
    return failure();
  return hlo::verifyBitReverseOp(
      getLoc(), getOperand(),
      llvm::to_vector(getDimensions().getValues<int64_t>()));
}

//===----------------------------------------------------------------------===//
// ReverseOp
//===----------------------------------------------------------------------===//

LogicalResult ReverseOp::verify() {
  if (failed(verify1dTensor(getLoc(), getDimensions(), "dimensions")))
    return failure();
  return hlo::verifyReverseOp(
      getLoc(), getOperand(),
      llvm::to_vector(getDimensions().getValues<int64_t>()));
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

// Makes it such that a SelectOp that is a non-root operation in a DRR infers
// the return type based on operand type.
LogicalResult SelectOp::inferReturnTypeComponents(
    MLIRContext *, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  SelectOp::Adaptor op(operands, attributes, properties, regions);
  return hlo::inferSelectOp(location, op.getPred(), op.getOnTrue(),
                            op.getOnFalse(), inferredReturnShapes);
}

LogicalResult
SelectOp::reifyReturnTypeShapes(OpBuilder &builder, ValueRange operands,
                                SmallVectorImpl<Value> &reifiedReturnShapes) {
  // For `hlo.select`, the first operand may be a scalar.
  return hlo::deriveShapeFromOperand(&builder, getOperation(), operands[1],
                                     &reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// SetDimensionSizeOp
//===----------------------------------------------------------------------===//

LogicalResult SetDimensionSizeOp::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  SetDimensionSizeOp::Adaptor adaptor(operands, attributes, properties,
                                      regions);
  return hlo::inferSetDimensionSizeOp(
      getMhloDialect(context), location, adaptor.getOperand().getType(),
      adaptor.getSize(), adaptor.getDimension(), inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

LogicalResult SliceOp::inferReturnTypes(
    MLIRContext * /*context*/, std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  SliceOpAdaptor adaptor(operands, attributes, properties, regions);
  if (failed(verify1dTensor(location, adaptor.getStartIndices(),
                            "start_indices")) ||
      failed(verify1dTensor(location, adaptor.getLimitIndices(),
                            "limit_indices")) ||
      failed(verify1dTensor(location, adaptor.getStrides(), "strides")))
    return failure();
  return hlo::inferSliceOp(
      location, adaptor.getOperand().getType(),
      llvm::to_vector(adaptor.getStartIndices().getValues<int64_t>()),
      llvm::to_vector(adaptor.getLimitIndices().getValues<int64_t>()),
      llvm::to_vector(adaptor.getStrides().getValues<int64_t>()),
      inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

namespace {

// transpose(transpose(X)) => transpose(X)
class EliminateRedundantTranspose : public OpRewritePattern<TransposeOp> {
public:
  using OpRewritePattern<TransposeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto tranposeOperand = op.getOperand().getDefiningOp<TransposeOp>();
    if (!tranposeOperand) {
      return failure();
    }
    auto operandPermutation =
        tranposeOperand.getPermutation().getValues<APInt>();
    auto newPermutation =
        cast<DenseIntElementsAttr>(op.getPermutation().mapValues(
            op.getPermutation().getElementType(),
            [&operandPermutation](const APInt &index) -> APInt {
              return operandPermutation[index.getSExtValue()];
            }));
    rewriter.replaceOpWithNewOp<TransposeOp>(op, op.getResult().getType(),
                                             tranposeOperand.getOperand(),
                                             newPermutation);
    return success();
  }
};

// BroadcastInDim(BroadcastInDim(X)) => BroadcastInDim(X)
class EliminateBroadcastInDimTranspose : public OpRewritePattern<TransposeOp> {
public:
  using OpRewritePattern<TransposeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto broadcastInDimOp = op.getOperand().getDefiningOp<BroadcastInDimOp>();
    if (!broadcastInDimOp) {
      return failure();
    }
    DenseIntElementsAttr broadcastDimensions =
        broadcastInDimOp.getBroadcastDimensions();
    DenseIntElementsAttr permutation = op.getPermutation();
    SmallVector<int64_t> newBroadcastDimensions;
    for (auto dimension : broadcastDimensions.getValues<int64_t>()) {
      int64_t index = 0;
      for (auto p : permutation.getValues<int64_t>()) {
        if (p == dimension) {
          newBroadcastDimensions.push_back(index);
          break;
        }
        index++;
      }
    }
    rewriter.replaceOpWithNewOp<BroadcastInDimOp>(
        op, op->getResultTypes(), broadcastInDimOp.getOperand(),
        rewriter.getI64TensorAttr(newBroadcastDimensions));
    return success();
  }
};

// simplify Transpose: replace Transpose with Reshape if they are equivalent
class SimplifyTranspose : public OpRewritePattern<TransposeOp> {
public:
  using OpRewritePattern<TransposeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto operandType = dyn_cast<RankedTensorType>(op.getOperand().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!operandType || !resultType) {
      return failure();
    }
    // Not support dynamic shape a.t.m. BTW, when it's dynamic shape,
    // maybe Transpose should be replaced by DynamicReshape.
    if (!operandType.hasStaticShape() || !resultType.hasStaticShape()) {
      return failure();
    }
    auto permutation = op.getPermutation().getValues<int64_t>();
    llvm::SmallVector<int64_t> sortedPermutation;
    for (int64_t i = 0, e = resultType.getRank(); i < e; i++) {
      if (resultType.getDimSize(i) != 1) {
        sortedPermutation.push_back(permutation[i]);
      }
    }
    if (llvm::is_sorted(sortedPermutation)) {
      rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(), op.getOperand());
      return success();
    }
    return failure();
  }
};

} // namespace

void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<EliminateRedundantTranspose>(context);
  results.add<EliminateBroadcastInDimTranspose>(context);
  results.add<SimplifyTranspose>(context);
}

LogicalResult TransposeOp::reifyReturnTypeShapes(
    OpBuilder &builder, ValueRange operands,
    SmallVectorImpl<Value> &reifiedReturnShapes) {
  TransposeOp::Adaptor adaptor(operands);
  Value operand = adaptor.getOperand();

  auto operandType = dyn_cast<RankedTensorType>(operand.getType());
  // Not support unranked type a.t.m.
  if (!operandType)
    return failure();

  Location loc = this->getLoc();
  SmallVector<int64_t, 4> permutation(
      this->getPermutation().getValues<int64_t>());
  SmallVector<Value, 4> shapeValues(permutation.size());

  Type shapeScalarType = builder.getIndexType();
  auto toShapeScalarType = [&](Value v) {
    return maybeCastTo(builder, loc, v, shapeScalarType);
  };

  for (const auto &element : llvm::enumerate(operandType.getShape())) {
    int64_t idx = element.index();
    auto *it = std::find(permutation.begin(), permutation.end(), idx);
    Value valueDim = toShapeScalarType(
        builder.createOrFold<tensor::DimOp>(loc, operand, element.index()));
    shapeValues[std::distance(permutation.begin(), it)] = valueDim;
  }

  Value outputShape = builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            shapeScalarType),
      shapeValues);
  reifiedReturnShapes.push_back(outputShape);

  return success();
}

LogicalResult
TransposeOp::inferReturnTypes(MLIRContext *, std::optional<Location> loc,
                              ValueRange operands, DictionaryAttr attributes,
                              OpaqueProperties properties, RegionRange regions,
                              SmallVectorImpl<Type> &inferredReturnTypes) {
  TransposeOp::Adaptor adaptor(operands, attributes, properties, regions);
  if (failed(verify1dTensor(loc, adaptor.getPermutation(), "permutation")))
    return failure();
  return hlo::inferTransposeOp(
      loc, adaptor.getOperand(),
      llvm::to_vector(adaptor.getPermutation().getValues<int64_t>()),
      inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// TupleOp
//===----------------------------------------------------------------------===//

namespace {

// Pattern for unpacking and repacking the same tuple.
struct UnpackRepackSameTuple : public OpRewritePattern<TupleOp> {
  using OpRewritePattern<TupleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TupleOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getVal().empty())
      return failure();

    Value firstElement = op.getVal().front();
    auto firstElementOp = firstElement.getDefiningOp<GetTupleElementOp>();
    if (!firstElementOp || firstElementOp.getIndexAttr().getInt() != 0)
      return failure();

    Value tuplePredecessor = firstElementOp.getOperand();
    if (tuplePredecessor.getType() != op.getType())
      return failure();

    for (const auto &elementAndIdx :
         llvm::enumerate(op.getVal().drop_front(1))) {
      auto elementOp = elementAndIdx.value().getDefiningOp<GetTupleElementOp>();
      if (!elementOp ||
          elementOp.getIndexAttr().getInt() !=
              static_cast<int64_t>(elementAndIdx.index() + 1) ||
          elementOp.getOperand() != tuplePredecessor)
        return failure();
    }

    rewriter.replaceOp(op, tuplePredecessor);
    return success();
  }
};

} // namespace

void TupleOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add<UnpackRepackSameTuple>(context);
}

LogicalResult TupleOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  TupleOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferTupleOp(context, location, adaptor.getVal(),
                           inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

// Converts gather ops to slice ops in case we have a single set of constant
// indices.
struct GatherSlice : public OpRewritePattern<GatherOp> {
  using OpRewritePattern<GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GatherOp gather,
                                PatternRewriter &rewriter) const override {
    DenseIntElementsAttr index;
    if (!matchPattern(gather.getStartIndices(), m_Constant(&index)))
      return failure();

    const auto &dnums = gather.getDimensionNumbers();
    if (dnums.getIndexVectorDim() != 0 || index.getType().getRank() > 1)
      return failure();

    // TODO(tberghammer): Remove when the verifier catches this case what is
    // invalid if all previous condition holds.
    if (index.getNumElements() !=
        static_cast<int64_t>(dnums.getStartIndexMap().size()))
      return failure();

    RankedTensorType operandType =
        dyn_cast<RankedTensorType>(gather->getOperand(0).getType());
    if (!operandType || !operandType.hasStaticShape())
      return failure();

    auto sliceEnd =
        llvm::to_vector<8>(gather.getSliceSizes().getValues<int64_t>());
    llvm::SmallVector<int64_t, 8> sliceStart(sliceEnd.size(), 0);
    for (auto it :
         llvm::zip(dnums.getStartIndexMap(), index.getValues<APInt>())) {
      int64_t mapIndex = std::get<0>(it);
      // Clamp the indices within bounds to faithfully mirror gather semantics.
      int64_t offset = std::max(
          static_cast<int64_t>(0),
          std::min(std::get<1>(it).getSExtValue(),
                   operandType.getDimSize(mapIndex) - sliceEnd[mapIndex]));
      sliceStart[mapIndex] += offset;
      sliceEnd[mapIndex] += offset;
    }

    llvm::SmallVector<int64_t, 8> sliceStride(sliceEnd.size(), 1);
    llvm::SmallVector<int64_t, 8> sliceShape(sliceEnd.size());
    for (size_t i = 0; i < sliceEnd.size(); ++i) {
      sliceShape[i] = sliceEnd[i] - sliceStart[i];
    }
    Type elementType = cast<TensorType>(gather.getType()).getElementType();
    auto sliceType = RankedTensorType::get(sliceShape, elementType);
    Value result = rewriter.create<SliceOp>(
        gather.getLoc(), sliceType, gather.getOperand(),
        rewriter.getI64TensorAttr(sliceStart),
        rewriter.getI64TensorAttr(sliceEnd),
        rewriter.getI64TensorAttr(sliceStride));

    auto collapsedSliceDims = dnums.getCollapsedSliceDims();
    if (!collapsedSliceDims.empty()) {
      llvm::SmallVector<int64_t, 8> reshapeShape;
      for (size_t i = 0; i < sliceShape.size(); ++i) {
        if (llvm::count(collapsedSliceDims, i) == 0) {
          reshapeShape.push_back(sliceShape[i]);
        }
      }
      auto reshapeType = RankedTensorType::get(reshapeShape, elementType);
      result = rewriter.create<ReshapeOp>(gather.getLoc(), reshapeType, result);
    }

    result.setType(gather.getType());
    rewriter.replaceOp(gather, result);
    return success();
  }
};

void GatherOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<GatherSlice>(context);
}

namespace {

// following https://www.tensorflow.org/xla/operation_semantics#gather
// The bounds for the output array along dimension i is computed as follows:
// (1) If i is present in batch_dims (i.e. is equal to batch_dims[k] for some k)
// then we pick
// the corresponding dimension bounds out of start_indices.shape, skipping
// index_vector_dim
// (i.e. pick start_indices.shape.dims[k] if k < index_vector_dim and
// start_indices.shape.dims[k+1] otherwise).
// (2) If i is present in offset_dims (i.e. equal to offset_dims[k] for some k)
// then we pick
// the corresponding bound out of slice_sizes after accounting for
// collapsed_slice_dims
// (i.e. we pick adjusted_slice_sizes[k] where adjusted_slice_sizes is
// slice_sizes with the bounds at indices collapsed_slice_dims removed).

void getSliceSizeValues(GatherOp *gather, OpBuilder &builder, Location loc,
                        ValueRange operands,
                        SmallVectorImpl<Value> &sliceSizes) {
  for (int64_t val : gather->getSliceSizes().getValues<int64_t>()) {
    sliceSizes.push_back(builder.create<arith::ConstantIndexOp>(loc, val));
  }
}

template <typename Op>
LogicalResult reifyGatherShape(Op *op, OpBuilder &builder, ValueRange operands,
                               SmallVectorImpl<Value> &reifiedReturnShapes) {
  // No support for unranked gather output shape a.t.m.
  auto resultTy = mlir::dyn_cast<RankedTensorType>(op->getResult().getType());
  if (!resultTy)
    return failure();

  typename Op::Adaptor adaptor(operands);
  Value startIndices = adaptor.getStartIndices();

  Location loc = op->getLoc();
  int resultRank = resultTy.getRank();
  Type shapeElTy = builder.getIndexType();
  auto toShapeElType = [&](Value v) {
    return maybeCastTo(builder, loc, v, shapeElTy);
  };

  SmallVector<Value, 4> sliceSizes;
  getSliceSizeValues(op, builder, loc, operands, sliceSizes);
  llvm::transform(sliceSizes, sliceSizes.begin(),
                  [&](Value v) { return toShapeElType(v); });

  auto getStartIndicesDim = [&](int64_t index) {
    return toShapeElType(
        builder.create<tensor::DimOp>(loc, startIndices, index));
  };
  SmallVector<Value, 4> shapeValues;
  auto getSliceDim = [&sliceSizes](int64_t index) -> Value {
    auto ret = sliceSizes[index];
    return ret;
  };
  hlo::reifyGatherDimSizes(resultRank, getStartIndicesDim, getSliceDim,
                           op->getDimensionNumbers().getOffsetDims(),
                           op->getDimensionNumbers().getCollapsedSliceDims(),
                           op->getDimensionNumbers().getOperandBatchingDims(),
                           op->getDimensionNumbers().getIndexVectorDim(),
                           shapeValues);

  Value outputShape = builder.create<tensor::FromElementsOp>(
      loc, RankedTensorType::get({resultRank}, shapeElTy), shapeValues);
  reifiedReturnShapes.push_back(outputShape);

  return success();
}

} // namespace

LogicalResult
GatherOp::reifyReturnTypeShapes(OpBuilder &builder, ValueRange operands,
                                SmallVectorImpl<Value> &reifiedReturnShapes) {
  return reifyGatherShape(this, builder, operands, reifiedReturnShapes);
}

LogicalResult GatherOp::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  GatherOp::Adaptor adaptor(operands, attributes, properties, regions);
  if (failed(verify1dTensor(location, adaptor.getSliceSizes(), "slice_sizes")))
    return failure();
  return hlo::inferGatherOp(
      location, adaptor.getOperand(), adaptor.getStartIndices(),
      adaptor.getDimensionNumbers().getOffsetDims(),
      adaptor.getDimensionNumbers().getCollapsedSliceDims(),
      adaptor.getDimensionNumbers().getOperandBatchingDims(),
      adaptor.getDimensionNumbers().getStartIndicesBatchingDims(),
      adaptor.getDimensionNumbers().getStartIndexMap(),
      adaptor.getDimensionNumbers().getIndexVectorDim(),
      llvm::to_vector(adaptor.getSliceSizes().getValues<int64_t>()),
      inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

LogicalResult
ScatterOp::inferReturnTypes(MLIRContext *, std::optional<Location> location,
                            ValueRange operands, DictionaryAttr attributes,
                            OpaqueProperties properties, RegionRange regions,
                            SmallVectorImpl<Type> &inferredReturnTypes) {
  ScatterOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferScatterOp(location, adaptor.getInputs(),
                             adaptor.getUpdateComputation(),
                             inferredReturnTypes);
}

LogicalResult ScatterOp::verify() {
  return hlo::verifyScatterOp(
      getLoc(), getInputs(), getScatterIndices(), getUpdates(),
      getScatterDimensionNumbers().getUpdateWindowDims(),
      getScatterDimensionNumbers().getInsertedWindowDims(),
      getScatterDimensionNumbers().getInputBatchingDims(),
      getScatterDimensionNumbers().getScatterIndicesBatchingDims(),
      getScatterDimensionNumbers().getScatterDimsToOperandDims(),
      getScatterDimensionNumbers().getIndexVectorDim(), getUpdateComputation());
}

namespace {

// Replace mhlo.scatter overwriting the entire input with mhlo.map.
struct ScatterFullReplace : public OpRewritePattern<ScatterOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ScatterOp scatter,
                                PatternRewriter &rewriter) const override {
    // Variadic Scatter not yet implemented
    if (scatter.getInputs().size() != 1 || scatter.getUpdates().size() != 1)
      return failure();

    auto baseType =
        dyn_cast<RankedTensorType>(scatter.getInputs().getTypes()[0]);
    auto updateType =
        dyn_cast<RankedTensorType>(scatter.getUpdates().getTypes()[0]);
    auto indexType =
        dyn_cast<RankedTensorType>(scatter.getScatterIndices().getType());
    if (!baseType || !indexType || !updateType)
      return failure();

    // If scatter_indices has zero elements, the scatter is a no-op.
    // Per StableHLO spec, return the input tensor unchanged.
    if (!indexType.hasStaticShape() || indexType.getNumElements() > 0)
      return failure();

    rewriter.replaceOp(scatter, scatter.getInputs());
    return success();
  }
};

} // namespace

void ScatterOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<ScatterFullReplace>(context);
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

LogicalResult WhileOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  WhileOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferWhileOp(location, adaptor.getOperand(), inferredReturnTypes);
}

LogicalResult WhileOp::verify() {
  return hlo::verifyWhileOp(getLoc(), getOperand(), getCond(), getBody());
}

void WhileOp::print(OpAsmPrinter &p) {
  hlo::printWhileOp(p, getOperation(), getCond(), getBody());
}

ParseResult WhileOp::parse(OpAsmParser &parser, OperationState &result) {
  return hlo::parseWhileOp(parser, result);
}

static LogicalResult whileCanonicalization(WhileOp whileOp,
                                           PatternRewriter &rewriter) {
  // Turn loop invariant values into implicit capture.
  // Check if there is at least one value is forwarded from one iteration to the
  // next, or one of the yielded value is an implicit capture already. Otherwise
  // there is nothing to do here.
  Block *cond = whileOp.SingleBlock::getBody(0);
  Block *body = whileOp.SingleBlock::getBody(1);
  auto bodyReturnOp = cast<ReturnOp>(body->getTerminator());
  if (!llvm::any_of(llvm::zip(whileOp->getOperands(), body->getArguments(),
                              bodyReturnOp->getOperands()),
                    [&](auto zip) {
                      return (std::get<0>(zip) == std::get<2>(zip) ||
                              std::get<1>(zip) == std::get<2>(zip));
                    }))
    return rewriter.notifyMatchFailure(whileOp, "no loop invariant found");

  SmallVector<Value> newOperands, resultsToReplace;
  SmallVector<unsigned> invariantArgIdxs;
  BitVector invariantArgIdxBitVector(cond->getNumArguments());
  for (const auto &enumeratedOperands : llvm::enumerate(llvm::zip(
           whileOp.getOperands(), cond->getArguments(), body->getArguments(),
           bodyReturnOp->getOperands(), whileOp->getResults()))) {
    const auto &operands = enumeratedOperands.value();
    Value whileOperand = std::get<0>(operands);
    BlockArgument condBlockArg = std::get<1>(operands);
    BlockArgument bodyBlockArg = std::get<2>(operands);
    Value bodyReturnOperand = std::get<3>(operands);
    Value whileResult = std::get<4>(operands);

    bool forwarded = (whileOperand == bodyReturnOperand ||
                      bodyBlockArg == bodyReturnOperand);
    if (forwarded) {
      invariantArgIdxs.push_back(enumeratedOperands.index());
      invariantArgIdxBitVector.set(enumeratedOperands.index());
      condBlockArg.replaceAllUsesWith(whileOperand);
      bodyBlockArg.replaceAllUsesWith(whileOperand);
      whileResult.replaceAllUsesWith(whileOperand);
      continue;
    }
    newOperands.push_back(whileOperand);
    resultsToReplace.push_back(whileResult);
  }
  cond->eraseArguments(invariantArgIdxBitVector);
  body->eraseArguments(invariantArgIdxBitVector);
  for (int idx : llvm::reverse(invariantArgIdxs))
    bodyReturnOp->eraseOperand(idx);

  WhileOp newWhileOp = rewriter.create<WhileOp>(
      whileOp.getLoc(), bodyReturnOp->getOperandTypes(), newOperands);
  newWhileOp.getBodyRegion(0).takeBody(whileOp.getBodyRegion(0));
  newWhileOp.getBodyRegion(1).takeBody(whileOp.getBodyRegion(1));
  for (auto results : llvm::zip(resultsToReplace, newWhileOp->getResults()))
    std::get<0>(results).replaceAllUsesWith(std::get<1>(results));
  rewriter.eraseOp(whileOp);
  return success();
}

void WhileOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add(&whileCanonicalization);
}

} // namespace mlir::mhlo

using mlir::hlo::parsePairwiseOpType;
using mlir::hlo::parseSameOperandsAndResultType;
using mlir::hlo::parseSelectOpType;
using mlir::hlo::parseTupleOpType;
using mlir::hlo::parseVariadicSameOperandsAndResultType;
using mlir::hlo::printPairwiseOpType;
using mlir::hlo::printSameOperandsAndResultType;
using mlir::hlo::printSelectOpType;
using mlir::hlo::printTupleOpType;
using mlir::hlo::printVariadicSameOperandsAndResultType;

#define GET_OP_CLASSES
#include "zkx/mlir_hlo/mhlo/IR/hlo_ops.cc.inc"

namespace mlir::mhlo {

//===----------------------------------------------------------------------===//
// mhlo Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {

struct MhloDialectInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // Allow all call operations to be inlined.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }
  // We don't have any special restrictions on what can be inlined into
  // destination regions (e.g. while/conditional bodies). Always allow it.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }
  // Operations in mhlo dialect are always legal to inline since they are
  // pure.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
};

struct MhloHloDialectInterface : public hlo::HloDialectInterface {
  using HloDialectInterface::HloDialectInterface;

  Type createTokenType() const override {
    return TokenType::get(getDialect()->getContext());
  }

  bool isTokenType(Type type) const override { return isa<TokenType>(type); }

  Attribute createTypeExtensions(ArrayRef<int64_t> bounds) const override {
    return TypeExtensionsAttr::get(getDialect()->getContext(), bounds);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// mhlo Dialect Constructor
//===----------------------------------------------------------------------===//

MhloDialect::MhloDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<MhloDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "zkx/mlir_hlo/mhlo/IR/hlo_ops.cc.inc" // NOLINT(build/include)
      >();
  addInterfaces<MhloHloDialectInterface>();
  addInterfaces<MhloDialectInlinerInterface>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "zkx/mlir_hlo/mhlo/IR/hlo_ops_attrs.cc.inc" // NOLINT(build/include)
      >();
}

// Entry point for Attribute parsing, TableGen generated code will handle the
// dispatch to the individual classes.
Attribute MhloDialect::parseAttribute(DialectAsmParser &parser,
                                      Type type) const {
  StringRef attrTag;
  Attribute attr;
  OptionalParseResult parseResult =
      generatedAttributeParser(parser, &attrTag, type, attr);
  if (parseResult.has_value())
    return attr;
  parser.emitError(parser.getNameLoc(), "unknown mhlo attribute");
  return Attribute();
}

// Entry point for Attribute printing, TableGen generated code will handle the
// dispatch to the individual classes.
void MhloDialect::printAttribute(Attribute attr, DialectAsmPrinter &os) const {
  LogicalResult result = generatedAttributePrinter(attr, os);
  std::ignore = result;
  assert(succeeded(result));
}

namespace {

// Helpers for attributes parsing.
ParseResult parseDims(AsmParser &parser, SmallVector<int64_t> &dimSizes) {
  dimSizes.clear();
  auto failOrDims = parseDimSizes(parser);
  if (failed(failOrDims)) {
    return failure();
  }
  dimSizes = std::move(*failOrDims);
  return success();
}

// Parse a custom attribute that resembles a struct of the form
// <
//   foo = something_parsed_by_custom_parser,
//   bar = something_parsed_by_different_custom_parser,
//   baz something_parsed_by_another_custom_parser
// >
// The optional argument `parse_equal` array can be used to denote if
// '=' follows the keyword (see baz in the example above) for a field. If
// not provided, all fields must be followed by a '='.
ParseResult parseStruct(AsmParser &parser, ArrayRef<StringRef> keywords,
                        ArrayRef<llvm::function_ref<ParseResult()>> parseFuncs,
                        ArrayRef<bool> parseEqual = {}) {
  assert(keywords.size() == parseFuncs.size());
  assert(parseEqual.empty() || parseEqual.size() == keywords.size());
  SmallVector<bool> seen(keywords.size(), false);
  while (failed(parser.parseOptionalGreater())) {
    bool foundOne = false;
    for (const auto &it : llvm::enumerate(keywords)) {
      size_t index = it.index();
      StringRef keyword = it.value();
      if (succeeded(parser.parseOptionalKeyword(keyword))) {
        if (seen[index]) {
          return parser.emitError(parser.getCurrentLocation())
                 << "duplicated `" << keyword << "` entry";
        }
        if (parseEqual.empty() || parseEqual[index]) {
          if (failed(parser.parseEqual()))
            return failure();
        }
        if (failed(parseFuncs[index]()))
          return failure();
        if (failed(parser.parseOptionalComma()))
          return parser.parseGreater();
        seen[index] = true;
        foundOne = true;
        break;
      }
    }
    if (!foundOne) {
      auto parseError = parser.emitError(parser.getCurrentLocation())
                        << "expected one of: ";
      llvm::interleaveComma(keywords, parseError, [&](StringRef kw) {
        parseError << '`' << kw << '`';
      });
      return parseError;
    }
  }
  return success();
}

// Helpers to print an optional array or integer field, to simplify writing
// attribute printers.
template <typename T>
void printField(AsmPrinter &printer, StringRef name, T field,
                StringRef &separator) {
  if (field != 0) {
    printer << separator << name << " = " << field;
    separator = ", ";
  }
}
template <typename T>
void printField(AsmPrinter &printer, StringRef name, ArrayRef<T> field,
                StringRef &separator) {
  if (!field.empty()) {
    printer << separator << name << " = [";
    llvm::interleaveComma(field, printer);
    printer << "]";
    separator = ", ";
  }
}

template <typename... Ts>
void printStruct(AsmPrinter &printer, StringRef name, Ts... printFields) {
  printer << "<";
  StringRef separator = "";
  // Fold expression to print each entry in the parameter pack.
  // TODO(mhlo-team): this can be simplified when TF moves to C++17.
  using unused = int[];
  (void)unused{0, (printField(printer, std::get<0>(printFields),
                              std::get<1>(printFields), separator),
                   0)...};
  printer << ">";
}

} // namespace

// Custom printer and parser for ScatterDimensionNumbersAttr.
void ScatterDimensionNumbersAttr::print(AsmPrinter &printer) const {
  printStruct(printer, "scatter",
              std::make_pair("update_window_dims", getUpdateWindowDims()),
              std::make_pair("inserted_window_dims", getInsertedWindowDims()),
              std::make_pair("input_batching_dims", getInputBatchingDims()),
              std::make_pair("scatter_indices_batching_dims",
                             getScatterIndicesBatchingDims()),
              std::make_pair("scatter_dims_to_operand_dims",
                             getScatterDimsToOperandDims()),
              std::make_pair("index_vector_dim", getIndexVectorDim()));
}

Attribute ScatterDimensionNumbersAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess()))
    return {};
  SmallVector<int64_t> updateWindowDims;
  SmallVector<int64_t> insertedWindowDims;
  SmallVector<int64_t> inputBatchingDims;
  SmallVector<int64_t> scatterIndicesBatchingDims;
  SmallVector<int64_t> scatterDimsToOperandDims;
  int64_t indexVectorDim = 0;

  if (failed(parseStruct(
          parser,
          {"update_window_dims", "inserted_window_dims", "input_batching_dims",
           "scatter_indices_batching_dims", "scatter_dims_to_operand_dims",
           "index_vector_dim"},
          {[&]() { return parseDims(parser, updateWindowDims); },
           [&]() { return parseDims(parser, insertedWindowDims); },
           [&]() { return parseDims(parser, inputBatchingDims); },
           [&]() { return parseDims(parser, scatterIndicesBatchingDims); },
           [&]() { return parseDims(parser, scatterDimsToOperandDims); },
           [&]() { return parser.parseInteger(indexVectorDim); }}))) {
    parser.emitError(parser.getCurrentLocation())
        << "failed parsing scatter dimension numbers attribute";
    return {};
  }

  return ScatterDimensionNumbersAttr::get(
      parser.getContext(), updateWindowDims, insertedWindowDims,
      inputBatchingDims, scatterIndicesBatchingDims, scatterDimsToOperandDims,
      indexVectorDim);
}

// Custom printer and parser for GatherDimensionNumbersAttr.
void GatherDimensionNumbersAttr::print(AsmPrinter &printer) const {
  printStruct(printer, "gather", std::make_pair("offset_dims", getOffsetDims()),
              std::make_pair("collapsed_slice_dims", getCollapsedSliceDims()),
              std::make_pair("operand_batching_dims", getOperandBatchingDims()),
              std::make_pair("start_indices_batching_dims",
                             getStartIndicesBatchingDims()),
              std::make_pair("start_index_map", getStartIndexMap()),
              std::make_pair("index_vector_dim", getIndexVectorDim()));
}

Attribute GatherDimensionNumbersAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess()))
    return {};
  SmallVector<int64_t> offsetDims;
  SmallVector<int64_t> collapsedSliceDims;
  SmallVector<int64_t> operandBatchingDims;
  SmallVector<int64_t> startIndicesBatchingDims;
  SmallVector<int64_t> startIndexMap;
  int64_t indexVectorDim = 0;

  if (failed(parseStruct(
          parser,
          {"offset_dims", "collapsed_slice_dims", "operand_batching_dims",
           "start_indices_batching_dims", "start_index_map",
           "index_vector_dim"},
          {[&]() { return parseDims(parser, offsetDims); },
           [&]() { return parseDims(parser, collapsedSliceDims); },
           [&]() { return parseDims(parser, operandBatchingDims); },
           [&]() { return parseDims(parser, startIndicesBatchingDims); },
           [&]() { return parseDims(parser, startIndexMap); },
           [&]() { return parser.parseInteger(indexVectorDim); }}))) {
    parser.emitError(parser.getCurrentLocation())
        << "failed parsing gather dimension numbers attribute";
    return {};
  }

  return GatherDimensionNumbersAttr::get(
      parser.getContext(), offsetDims, collapsedSliceDims, operandBatchingDims,
      startIndicesBatchingDims, startIndexMap, indexVectorDim);
}

// Custom printer and parser for DotDimensionNumbersAttr.
void DotDimensionNumbersAttr::print(AsmPrinter &printer) const {
  printStruct(
      printer, "dot",
      std::make_pair("lhs_batching_dimensions", getLhsBatchingDimensions()),
      std::make_pair("rhs_batching_dimensions", getRhsBatchingDimensions()),
      std::make_pair("lhs_contracting_dimensions",
                     getLhsContractingDimensions()),
      std::make_pair("rhs_contracting_dimensions",
                     getRhsContractingDimensions()));
}

Attribute DotDimensionNumbersAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess()))
    return {};
  SmallVector<int64_t> lhsBatchingDimensions;
  SmallVector<int64_t> rhsBatchingDimensions;
  SmallVector<int64_t> lhsContractingDimensions;
  SmallVector<int64_t> rhsContractingDimensions;

  if (failed(parseStruct(
          parser,
          {"lhs_batching_dimensions", "rhs_batching_dimensions",
           "lhs_contracting_dimensions", "rhs_contracting_dimensions"},
          {[&]() { return parseDims(parser, lhsBatchingDimensions); },
           [&]() { return parseDims(parser, rhsBatchingDimensions); },
           [&]() { return parseDims(parser, lhsContractingDimensions); },
           [&]() { return parseDims(parser, rhsContractingDimensions); }}))) {
    parser.emitError(parser.getCurrentLocation())
        << "failed parsing dot dimension numbers attribute";
    return {};
  }

  return DotDimensionNumbersAttr::get(
      parser.getContext(), lhsBatchingDimensions, rhsBatchingDimensions,
      lhsContractingDimensions, rhsContractingDimensions);
}

//===----------------------------------------------------------------------===//
// MHLO Dialect Hooks
//===----------------------------------------------------------------------===//

Operation *MhloDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  auto elementsAttr = dyn_cast<ElementsAttr>(value);
  // HLO dialect constants only support ElementsAttr unlike standard dialect
  // constant which supports all attributes.
  if (!elementsAttr)
    return nullptr;
  auto resultShapedType = dyn_cast<ShapedType>(type);
  auto attrShapedType = dyn_cast<ShapedType>(elementsAttr.getType());
  if (resultShapedType && attrShapedType) {
    return builder.create<mhlo::ConstantOp>(loc, type, elementsAttr);
  }
  // HLO dialect constants require the type of value and result to match
  if (type != elementsAttr.getType())
    return nullptr;

  return builder.create<mhlo::ConstantOp>(loc, type, elementsAttr);
}

LogicalResult MhloDialect::verifyRegionArgAttribute(Operation *op,
                                                    unsigned /*regionIndex*/,
                                                    unsigned argIndex,
                                                    NamedAttribute attr) {
  return success();
}

LogicalResult MhloDialect::verifyOperationAttribute(Operation *op,
                                                    NamedAttribute attr) {
  if (attr.getName() == "mhlo.spmd_parameters_sharding") {
    auto arrayAttr = dyn_cast<ArrayAttr>(attr.getValue());
    if (!arrayAttr)
      return op->emitOpError() << "spmd_parameters_sharding: must be an array";
    auto module = dyn_cast<ModuleOp>(op);
    if (!module)
      return op->emitOpError()
             << "has spmd_parameters_sharding but is not a module";
    // Check that the "main" function exists:
    auto main = module.lookupSymbol<func::FuncOp>("main");
    if (!main)
      return module.emitOpError() << "spmd_parameters_sharding: main not found";
    if (main.getNumArguments() != arrayAttr.size())
      return module.emitOpError()
             << "spmd_parameters_sharding: main has " << main.getNumArguments()
             << " arguments, but spmd_parameters_sharding expects "
             << arrayAttr.size();
  }
  return success();
}

} // namespace mlir::mhlo
