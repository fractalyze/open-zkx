/* Copyright 2021 The OpenXLA Authors.
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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "zkx/mlir_hlo/analysis/shape_component_analysis.h"
#include "zkx/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir::mhlo {

#define GEN_PASS_DEF_SYMBOLICSHAPEOPTIMIZATION
#include "zkx/mlir_hlo/mhlo/transforms/mhlo_passes.h.inc"

using ShapeOrValueInfo = ShapeComponentAnalysis::ShapeOrValueInfo;
using Symbol = ShapeComponentAnalysis::Symbol;
using SymbolicExpr = ShapeComponentAnalysis::SymbolicExpr;

namespace {

// Temporary data structure to hold a single dimension of the symbolic result of
// `shape.broadcast`.
struct SymbolicBroadcastDimension {
  size_t operandIndex;
  size_t operandDim;
  SymbolicExpr expr;
};

// Replace shape.broadcast with a shape if it's statically known.
struct SimplifyBroadcasts : public mlir::OpRewritePattern<shape::BroadcastOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult
  matchAndRewrite(shape::BroadcastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Require successful shape analysis.
    ShapeComponentAnalysis shapeAnalysis;
    llvm::SmallVector<ArrayRef<SymbolicExpr>> shapesInfo;
    auto shapes = op.getShapes();
    shapesInfo.reserve(shapes.size());
    for (Value s : shapes) {
      auto sInfo = shapeAnalysis.GetValueInfo(s);
      if (!sInfo)
        return failure();
      shapesInfo.push_back(*sInfo);
    }

    // Find the result rank.
    size_t rank = 0;
    for (const auto &sInfo : shapesInfo)
      rank = std::max(rank, sInfo.size());

    // Compute broadcast symbolically.
    SmallVector<std::optional<SymbolicBroadcastDimension>> symResult(
        rank, std::nullopt);
    for (const auto &sInfo : llvm::enumerate(shapesInfo)) {
      size_t dimOffset = rank - sInfo.value().size();
      for (const auto &symExpr : llvm::enumerate(sInfo.value())) {
        // Unit dimensions are neutral to the final result.
        if (symExpr.value().isConstant(1))
          continue;

        // Use unique expression.
        size_t i = dimOffset + symExpr.index();
        if (!symResult[i]) {
          symResult[i] = {sInfo.index(), symExpr.index(), symExpr.value()};
          continue;
        }

        // Bail if the dimensions are neither equal nor 1.
        if (symResult[i]->expr != symExpr.value())
          return failure();
      }
    }

    // Materialize broadcast result.
    auto loc = op.getLoc();
    DenseMap<int64_t, Value> constants;
    auto findOrCreateConstant = [&](int64_t c) {
      auto it = constants.find(c);
      if (it != constants.end())
        return it->second;
      Value newlyCreated = rewriter.create<arith::ConstantIndexOp>(loc, c);
      constants[c] = newlyCreated;
      return newlyCreated;
    };
    auto elements = llvm::to_vector<8>(
        llvm::map_range(symResult, [&](const auto &symResultDim) {
          // If we know the dimension statically, use a constant.
          if (!symResultDim)
            return findOrCreateConstant(1);
          if (auto cexpr =
                  dyn_cast<AffineConstantExpr>(symResultDim->expr.expr)) {
            return findOrCreateConstant(cexpr.getValue());
          }

          // Otherwise, extract the dimension from the unique operand.
          Value operand = shapes[symResultDim->operandIndex];
          Value operandDim = findOrCreateConstant(symResultDim->operandDim);
          return rewriter.create<tensor::ExtractOp>(loc, operand, operandDim)
              .getResult();
        }));
    Type indexTy = rewriter.getIndexType();
    Type concreteResultTy =
        RankedTensorType::get({static_cast<int64_t>(elements.size())}, indexTy);
    Value result = rewriter.create<tensor::FromElementsOp>(
        loc, concreteResultTy, elements);

    // Insert cast, if needed.
    Type expectedTy = op.getResult().getType();
    if (result.getType() != expectedTy) {
      result = rewriter.create<tensor::CastOp>(loc, expectedTy, result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

LogicalResult analyzeDynamicBroadcastInDimExpandingBehavior(
    ShapeComponentAnalysis &analysis, Value value, Value shape,
    llvm::SmallSetVector<int64_t, 4> *knownExpandingDims,
    llvm::SmallSetVector<int64_t, 4> *knownNonexpandingDims) {
  // Require successful analysis of shapes.
  auto shapeIn = analysis.GetShapeInfo(value);
  auto shapeOut = analysis.GetValueInfo(shape);
  if (!shapeIn || !shapeOut)
    return failure();

  // Analyze per argument dimension.
  size_t rankIn = shapeIn->size();
  size_t rankOut = shapeOut->size();
  assert(rankIn <= rankOut);
  size_t dimOutOffset = rankOut - rankIn;
  for (size_t i = 0; i < rankIn; ++i) {
    SymbolicExpr dimIn = (*shapeIn)[i];
    SymbolicExpr dimOut = (*shapeOut)[dimOutOffset + i];
    if (dimIn.isConstant(1) && dimOut.isKnownNotOne())
      knownExpandingDims->insert(i);
    if (dimIn == dimOut || dimOut.isConstant(1))
      knownNonexpandingDims->insert(i);
  }
  return success();
}

// Analyze `mhlo.dynamic_broadcast_in_dim` op and populate attributes for
// statically known expanding and non-expanding dimensions.
struct AnnotateExpandingDimensionsInDynamicBroadcastInDim
    : public mlir::OpRewritePattern<mhlo::DynamicBroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult
  matchAndRewrite(mhlo::DynamicBroadcastInDimOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Analyze shapes and identify expanding and non-expanding dims.
    ShapeComponentAnalysis analysis;
    llvm::SmallSetVector<int64_t, 4> knownExpandingDims, knownNonexpandingDims;
    if (failed(analyzeDynamicBroadcastInDimExpandingBehavior(
            analysis, op.getOperand(), op.getOutputDimensions(),
            &knownExpandingDims, &knownNonexpandingDims))) {
      return failure();
    }

    // Collect possibly already annotated info.
    auto insertAll = [](llvm::SmallSetVector<int64_t, 4> &dst,
                        std::optional<DenseIntElementsAttr> src) {
      if (!src)
        return;
      for (auto it : *src)
        dst.insert(it.getLimitedValue());
    };
    insertAll(knownExpandingDims, op.getKnownExpandingDimensions());
    insertAll(knownNonexpandingDims, op.getKnownNonexpandingDimensions());

    // Fail pattern application if there is nothing new to annotate.
    auto isEqual = [](llvm::SmallSetVector<int64_t, 4> &set,
                      DenseIntElementsAttr attr) {
      return static_cast<int64_t>(set.size()) == attr.size() &&
             llvm::all_of(attr, [&](auto it) {
               return set.count(it.getLimitedValue());
             });
    };
    if (op.getKnownExpandingDimensions() &&
        op.getKnownNonexpandingDimensions() &&
        isEqual(knownExpandingDims, *op.getKnownExpandingDimensions()) &&
        isEqual(knownNonexpandingDims, *op.getKnownNonexpandingDimensions())) {
      return failure();
    }

    // Annotate op in place.
    rewriter.startOpModification(op);
    op.setKnownExpandingDimensionsAttr(
        rewriter.getI64TensorAttr(knownExpandingDims.takeVector()));
    op.setKnownNonexpandingDimensionsAttr(
        rewriter.getI64TensorAttr(knownNonexpandingDims.takeVector()));
    rewriter.finalizeOpModification(op);
    return success();
  }
};

// Returns true if all of bcasted_shapes can be broadcasted with output_shape.
bool isKnownBroadcastable(ShapeComponentAnalysis &analysis,
                          ValueRange bcastedShapes, Value outputShape) {
  auto outputShapeDims = analysis.GetValueInfo(outputShape);
  if (!outputShapeDims)
    return false;
  for (Value shape : bcastedShapes) {
    auto shapeDims = analysis.GetValueInfo(shape);
    if (!shapeDims)
      return false;
    // Iterate backwards over the smallest input shape.
    for (auto zip : llvm::zip(llvm::reverse(*outputShapeDims),
                              llvm::reverse(*shapeDims))) {
      const auto &first = std::get<0>(zip);
      const auto &second = std::get<1>(zip);
      // TODO(ezhulenev): What to do with dimensions statically known to be
      // zero?
      // Numpy can only broadcast [0] with [1], however Tensorflow can broadcast
      // [0] with any dimension size, and produces dimension of size [0].
      // Currently we'll conservatively return failure and will not proceed with
      // a rewrite.
      if (first.isConstant(0) || second.isConstant(0))
        return false;
      // If either shape has a static one dimension the broadcast will always
      // succeed.
      if (first.isConstant(1) || second.isConstant(1))
        continue;
      // Otherwise dims have to be equal.
      if (first != second)
        return false;
    }
  }
  return true;
}

// Rewrite `shape.cstr_broadcastable` with constant witness if can prove that
// shapes are broadcastable from a symbolic analysis.
struct CstrBroadcastableOpLowering
    : public OpRewritePattern<shape::CstrBroadcastableOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(shape::CstrBroadcastableOp op,
                                PatternRewriter &rewriter) const override {
    ShapeComponentAnalysis shapeComponentAnalysis;
    if (!isKnownBroadcastable(shapeComponentAnalysis, op.getShapes(),
                              op.getShapes().front())) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<shape::ConstWitnessOp>(op, true);
    return success();
  }
};

// Returns a shape tensor if the shapes can be broadcasted to a known shape.
// Will either return one of the shapes or a generated mix of the shapes.
std::optional<Value> simplifyBroadcast(ShapeComponentAnalysis &analysis,
                                       ValueRange shapes, Location loc,
                                       OpBuilder *builder) {
  // First find the input shape with the largest rank.
  SmallVector<ArrayRef<ShapeComponentAnalysis::SymbolicExpr>> shapesFound;
  size_t maxRank = 0;
  for (const auto &shape : llvm::enumerate(shapes)) {
    auto foundShape = analysis.GetValueInfo(shape.value());
    if (!foundShape)
      return {};
    shapesFound.push_back(*foundShape);
    maxRank = std::max(maxRank, foundShape->size());
  }
  if (maxRank == 0) {
    return Value(builder->create<tensor::FromElementsOp>(
        loc, shapes[0].getType(), SmallVector<Value>()));
  }

  SmallVector<const ShapeComponentAnalysis::SymbolicExpr *> joinedDimensions(
      maxRank);
  SmallVector<std::pair<Value, int64_t>> shapeAndRankForDim(maxRank);
  for (const auto &shape : llvm::enumerate(shapesFound)) {
    for (const auto &dim : llvm::enumerate(llvm::reverse(shape.value()))) {
      // 1 dimensions don't contribute to the final result.
      if (dim.value().isConstant(1))
        continue;
      // If it's not a 1 dimension it will be present in the result. Remember
      // where it came from.
      auto index = maxRank - dim.index() - 1;
      if (!joinedDimensions[index]) {
        joinedDimensions[index] = &dim.value();
        shapeAndRankForDim[index] =
            std::make_pair(shapes[shape.index()], shape.value().size());
        continue;
      }
      // Bail if the dimensions are neither equal nor 1.
      if (*joinedDimensions[index] != dim.value())
        return {};
    }
  }
  // If the output is the same as one of the inputs just return that.
  if (llvm::all_equal(shapeAndRankForDim) && shapeAndRankForDim[0].first) {
    return shapeAndRankForDim[0].first;
  }
  // Otherwise rematerialize the shape from the pieces we have.
  SmallVector<Value> elements;
  for (size_t i = 0; i != maxRank; ++i) {
    // 1 dimensions are filtered above, recreate the constant.
    if (!shapeAndRankForDim[i].first) {
      auto one = builder->getIntegerAttr(
          mlir::cast<RankedTensorType>(shapes[0].getType()).getElementType(),
          1);
      elements.push_back(builder->create<arith::ConstantOp>(loc, one));
      continue;
    }
    // Extract from one of the shapes, accounting for the reverse indexing
    // performed by broadcast.
    Value index = builder->create<arith::ConstantIndexOp>(
        loc, i - maxRank + shapeAndRankForDim[i].second);
    elements.push_back(builder->create<tensor::ExtractOp>(
        loc, shapeAndRankForDim[i].first, index));
  }
  return Value(builder->create<tensor::FromElementsOp>(loc, elements));
}

// Replace shape.broadcast with a shape if it's statically known.
struct BroadcastOpLowering final
    : public mlir::OpRewritePattern<shape::BroadcastOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult
  matchAndRewrite(shape::BroadcastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    ShapeComponentAnalysis shapeComponentAnalysis;
    auto newBroadcast = simplifyBroadcast(
        shapeComponentAnalysis, op.getShapes(), op.getLoc(), &rewriter);
    if (!newBroadcast)
      return failure();

    // Insert cast, if needed.
    Type expectedTy = op.getType();
    if (newBroadcast->getType() != expectedTy) {
      newBroadcast = rewriter.create<tensor::CastOp>(op.getLoc(), expectedTy,
                                                     *newBroadcast);
    }

    rewriter.replaceOp(op, {*newBroadcast});
    return success();
  }
};

class SymbolicShapeOptimizationPass final
    : public impl::SymbolicShapeOptimizationBase<
          SymbolicShapeOptimizationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    // clang-format off
    patterns.insert<
        AnnotateExpandingDimensionsInDynamicBroadcastInDim,
        BroadcastOpLowering,
        CstrBroadcastableOpLowering,
        SimplifyBroadcasts>(ctx);
    // clang-format on

    // Collect some relevant canonicalization patterns.
    shape::AssumingOp::getCanonicalizationPatterns(patterns, ctx);
    shape::ShapeOfOp::getCanonicalizationPatterns(patterns, ctx);

    if (failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createSymbolicShapeOptimizationPass() {
  return std::make_unique<SymbolicShapeOptimizationPass>();
}

} // namespace mlir::mhlo
