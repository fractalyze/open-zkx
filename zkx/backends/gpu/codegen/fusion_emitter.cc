/* Copyright 2023 The OpenXLA Authors.
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
#include "zkx/backends/gpu/codegen/fusion_emitter.h"

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

#include "zkx/layout_util.h"
#include "zkx/service/llvm_ir/llvm_util.h"
#include "zkx/shape_util.h"
#include "zkx/status_macros.h"
#include "zkx/util.h"

namespace zkx::gpu {
namespace {

void AnnotateWithInt32Value(std::string_view name, int64_t value,
                            std::string_view kernel_name,
                            llvm::Module* llvm_module) {
  llvm::NamedMDNode* nvvm_annotations_node =
      llvm_module->getOrInsertNamedMetadata("nvvm.annotations");
  llvm::Function* ir_kernel = llvm_module->getFunction(kernel_name.data());
  llvm::LLVMContext& llvm_context = llvm_module->getContext();

  nvvm_annotations_node->addOperand(llvm::MDNode::get(
      llvm_context,
      {llvm::ConstantAsMetadata::get(ir_kernel),
       llvm::MDString::get(llvm_context, name),
       llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
           llvm::IntegerType::get(llvm_context, /*NumBits=*/32), value))}));
}

}  // namespace

// Annotates the launch dimensions of the corresponding IR kernel in
// `llvm_module`.
absl::Status AnnotateKernelLaunchDimensions(
    const se::DeviceDescription& device_info,
    const LaunchDimensions& launch_dims, std::string_view kernel_name,
    llvm::Module* llvm_module) {
  TF_RET_CHECK(
      (device_info.block_dim_limit().x == 0 ||
       launch_dims.block_counts().x < device_info.block_dim_limit().x) &&
      (device_info.block_dim_limit().y == 0 ||
       launch_dims.block_counts().y < device_info.block_dim_limit().y))
      << "Kernel '" << kernel_name << "' launch needs more blocks ("
      << launch_dims.block_counts().x << ", " << launch_dims.block_counts().y
      << ") than allowed by hardware (" << device_info.block_dim_limit().x
      << ", " << device_info.block_dim_limit().y << ").";
  // Add __launch_bounds__ to metadata. This limits registers per thread to
  // avoid out-of-resources launching errors.

  // Our launch bounds are exact, so we can specify them as
  // reqntid[xyz] rather than maxntid[xyz].
  AnnotateWithInt32Value("reqntidx", launch_dims.thread_counts_per_block().x,
                         kernel_name, llvm_module);
  if (launch_dims.thread_counts_per_block().y > 1) {
    AnnotateWithInt32Value("reqntidy", launch_dims.thread_counts_per_block().y,
                           kernel_name, llvm_module);
  }
  if (launch_dims.thread_counts_per_block().z > 1) {
    AnnotateWithInt32Value("reqntidz", launch_dims.thread_counts_per_block().z,
                           kernel_name, llvm_module);
  }
  // Maybe we want to set "reqnctapercluster" here, but not sure if needed or if
  // LLVM supports that yet. Let's do that later when needed.
  return absl::OkStatus();
}

// static
IndexingMap KernelFusionInterface::GetDefaultThreadIdIndexingMap(
    const LaunchDimensions& launch_dims, int unroll_factor, const Shape& shape,
    mlir::MLIRContext* ctx) {
  std::vector<mlir::AffineExpr> output_dims(shape.rank());

  std::array<uint64_t, 3> thread_counts{
      launch_dims.thread_counts_per_block().x,
      launch_dims.thread_counts_per_block().y,
      launch_dims.thread_counts_per_block().z,
  };

  std::array<uint64_t, 3> total_sizes{
      launch_dims.thread_counts_per_block().x * launch_dims.block_counts().x,
      launch_dims.thread_counts_per_block().y * launch_dims.block_counts().y,
      launch_dims.thread_counts_per_block().z * launch_dims.block_counts().z,
  };

  // ParallelLoopEmitter makes some assumptions about launch dimensions and
  // computes the linear index using only the x and y components.
  //
  // We implement the general formula instead and rely on the simplifier to
  // fix it.
  //
  // This means that this code supports some launch grids that the parallel
  // loop emitter doesn't support. This is safe, since the latter CHECK fails
  // if its assumptions are not fulfilled.
  mlir::AffineExpr c0 = mlir::getAffineConstantExpr(0, ctx);
  mlir::AffineExpr linear_index = c0;
  uint64_t stride = 1;
  for (int i = 0; i < 3; ++i) {
    mlir::AffineExpr coord =
        mlir::getAffineDimExpr(kIndexingMapThreadIdxDims[i], ctx) +
        mlir::getAffineDimExpr(kIndexingMapBlockIdxDims[i], ctx) *
            thread_counts[i];
    mlir::AffineExpr linear_component = coord * stride;
    linear_index = linear_index + linear_component;
    stride *= total_sizes[i];
  }
  mlir::AffineExpr chunk_id = mlir::getAffineSymbolExpr(0, ctx);
  mlir::AffineExpr unroll_elem_id = mlir::getAffineSymbolExpr(1, ctx);

  linear_index = linear_index * unroll_factor +
                 chunk_id * unroll_factor * launch_dims.launch_bound() +
                 unroll_elem_id;

  // See IndexUtil::LinearIndexToMultidimensionalIndex.
  uint64_t divisor = 1;
  for (int64_t dimension : LayoutUtil::MinorToMajor(shape)) {
    output_dims[dimension] = (linear_index.floorDiv(divisor)) %
                             static_cast<uint64_t>(shape.dimensions(dimension));
    divisor *= shape.dimensions(dimension);
  }

  std::vector<IndexingMap::Variable> dim_vars = DimVarsFromGPUGrid(
      {static_cast<int64_t>(launch_dims.thread_counts_per_block().x),
       static_cast<int64_t>(launch_dims.thread_counts_per_block().y),
       static_cast<int64_t>(launch_dims.thread_counts_per_block().z),
       static_cast<int64_t>(launch_dims.block_counts().x),
       static_cast<int64_t>(launch_dims.block_counts().y),
       static_cast<int64_t>(launch_dims.block_counts().z)});
  std::vector<IndexingMap::Variable> range_vars;
  int64_t num_elements = ShapeUtil::ElementsIn(shape);
  range_vars.push_back(IndexingMap::Variable{
      {0, CeilOfRatio(num_elements,
                      static_cast<int64_t>(launch_dims.launch_bound()) *
                          unroll_factor) -
              1}});
  range_vars.push_back({0, unroll_factor - 1});
  IndexingMap indexing_map(
      mlir::AffineMap::get(/*dimCount=*/6,
                           /*symbolCount=*/2, output_dims, ctx),
      dim_vars, range_vars, /*rt_vars=*/{});
  indexing_map.AddConstraint(linear_index, Interval{0, num_elements - 1});
  indexing_map.Simplify();
  return indexing_map;
}

std::string GetSanitizedUniqueName(IrEmitterContext& ir_emitter_context,
                                   std::string_view suggested_name) {
  return ir_emitter_context.name_uniquer()->GetUniqueName(
      llvm_ir::SanitizeFunctionName(suggested_name));
}

}  // namespace zkx::gpu
