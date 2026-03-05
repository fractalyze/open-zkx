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

#ifndef ZKX_CODEGEN_EMITTERS_TRANSFORMS_PASSES_TD
#define ZKX_CODEGEN_EMITTERS_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertPureCallOpsPass
    : Pass<"zkx-convert-pure-call-ops", "mlir::func::FuncOp"> {
  let summary = "Converts zkx.pure_call to func.call";
  let description = [{
      We use zkx.pure_call ops for calls to enable CSE and other
      transformations (e.g. LICM). This pass rewrites our custom ops to standard
      ops.
  }];
  let dependentDialects = [
    "mlir::func::FuncDialect",
  ];
  let constructor = "CreateConvertPureCallOpsPass()";
}

def EraseDeadFunctionsPass : Pass<"zkx-erase-dead-functions", "mlir::ModuleOp"> {
  let summary = "Deletes unused functions";
  let description = [{
      Deletes functions that are not called.
  }];
  let dependentDialects = [
    "mlir::func::FuncDialect",
  ];

  let constructor = "CreateEraseDeadFunctionsPass()";
}

def FlattenTensorsPass : Pass<"zkx-flatten-tensors", "mlir::ModuleOp"> {
  let summary = "Flatten tensors.";
  let description = [{
    Linearizes all tensors loads and stores.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
    "mlir::vector::VectorDialect",
    "zkx::ZkxDialect",
    "zkx::gpu::ZkxGpuDialect",
  ];
  let constructor = "CreateFlattenTensorsPass()";
}

def LowerTensorsPass : Pass<"zkx-lower-tensors", "mlir::ModuleOp"> {
  let summary = "Lowers tensors to llvm pointers and loads/stores.";
  let description = [{
      Lowers tensors to LLVM. We cannot use the memref lowerings because they
      are not compatible with ZKX's ABI.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::gpu::GPUDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
    "mlir::vector::VectorDialect",
    "zkx::ZkxDialect",
    "zkx::gpu::ZkxGpuDialect",
  ];
  let options = [
    Option<"gpu_device_info_", "gpu_device_info", "std::string", /*default=*/"",
           "Serialized stream_executor::GPUDeviceInfo proto.">,
    Option<"target_type_", "target_type", "std::string", /*default=*/"\"gpu\"",
           "Whether the pass targets a 'cpu' or 'gpu'. If 'cpu', gpu_device_info_ must be empty.">,
  ];
  let constructor = "CreateLowerTensorsPass()";
}

def LowerToLLVMPass :
   Pass<"zkx-lower-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lowers to LLVM.";
  let description = [{
    Lowers the rest to LLVM
  }];
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::NVVM::NVVMDialect",
  ];
  let options = [
    Option<"gpu_device_info_", "gpu_device_info", "std::string", /*default=*/"",
           "Serialized stream_executor::GPUDeviceInfo proto.">,
    Option<"target_type_", "target_type", "std::string", /*default=*/"\"gpu\"",
           "Whether the pass targets a 'cpu' or 'gpu'. If 'cpu', gpu_device_info_ must be empty.">,
  ];
  let constructor = "CreateLowerToLLVMPass()";
}

def LowerZkxToScfPass :
   Pass<"zkx-lower-zkx-to-scf", "mlir::func::FuncOp"> {
  let summary = "Lowers zkx to SCF.";
  let dependentDialects = [
    "mlir::gpu::GPUDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
    "mlir::vector::VectorDialect",
    "zkx::ZkxDialect",
    "zkx::gpu::ZkxGpuDialect",
  ];
  let options = [
    Option<"warp_size", "warp_size", "int64_t", /*default=*/"32", "Warp size.">,
  ];
  let constructor = "CreateLowerZkxToScfPass()";
}

def LowerZkxLoopsToScfPass : Pass<
    "zkx-lower-zkx-loops-to-scf", "mlir::func::FuncOp"> {
  let summary = "Lowers zkx.loop to SCF.";
  let description = [{
    This pass is separate from zkx-lower-zkx-to-scf because
    zkx-lower-zkx-to-scf, inliner, peeling and zkx-lower-zkx-loops-to-scf
    have to run in that order.
  }];
  let dependentDialects = [
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
    "zkx::ZkxDialect",
    "zkx::gpu::ZkxGpuDialect",
  ];
  let constructor = "CreateLowerZkxLoopsToScfPass()";
}

def MergePointersToSameSlicePass :
   Pass<"zkx-merge-pointers", "mlir::ModuleOp"> {
  let summary = "Merges pointers that share slices.";
  let description = [{
      When a function has multiple pointer arguments with the same slice index,
      merges them.
  }];
  let dependentDialects = [
    "mlir::func::FuncDialect",
  ];
  let constructor = "CreateMergePointersToSameSlicePass()";
}

def PropagateSliceIndicesPass :
   Pass<"zkx-propagate-slice-indices", "mlir::ModuleOp"> {
  let summary = "Propagates slice indices from the entry function to all callees.";
  let description = [{
      Propagates zkx.slice_index attributes from the function with the zkx.entry
      attribute to all other functions.

      Note: This pass assumes that all functions share the same argument prefix
      as the entry function, and that tensor arguments are not interleaved with
      non-tensor arguments. If function signatures differ in this respect,
      attribute propagation may be incorrect. For more robust propagation,
      tracing arguments through call chains would be required.
  }];
  let dependentDialects = [
    "mlir::func::FuncDialect"
  ];
  let constructor = "CreatePropagateSliceIndicesPass()";
}

def SimplifyAffinePass : Pass<"zkx-simplify-affine", "mlir::ModuleOp"> {
  let summary = "Simplifies affine.apply using ZKX's range-aware simplifier.";
  let description = [{
      The standard affine canonicalizer cannot simplify all expressions, since
      it is unaware of range information. This pass uses `zkx.range` attributes
      on arguments and ops for simplification. It also lowers floordiv and mod
      to simpler expressions than lower-affine. This pass only works for
      expressions for which we can prove the LHS of mod and div is nonnegative.
  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "zkx::ZkxDialect",
  ];
  let constructor = "CreateSimplifyAffinePass()";
}

def SimplifyArithPass : Pass<"zkx-simplify-arith", "mlir::func::FuncOp"> {
  let summary = "Simplifies arith using ZKX's range-aware simplifier.";
  let description = [{
      We often emit bounds checks that are statically known to be satisfied.
      This pass removes them.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
  ];
  let constructor = "CreateSimplifyArithPass()";
}

def UnswitchLoopsPass :
   Pass<"zkx-unswitch-loops", "mlir::func::FuncOp"> {
  let summary = "Swaps scf.if and scf.for.";
  let description = [{
      Extracts `scf.if` ops with conditions that are independent of the loop
      variable from `scf.for` by doing the following rewrite:

      Before:

      %cond = some_cond() : i1
      %results = scf.for {
        %some_val = scf.if %cond  {
        } else {
        }
        scf.yield %some_val
      }

      After:

      %cond = some_cond() : i1
      %results = scf.if %cond {
         %results = scf.for {
            %some_val = scf.if %true  {
            } else {
            }
         }
         yield %results
      } else {
         %results = scf.for {
            %some_val = scf.if %false  {
            } else {
            }
         }
         yield %results
      }

      This only triggers if there is a single `scf.if` op in the loop body (and
      nothing else).
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect"
  ];
  let constructor = "CreateUnswitchLoopsPass()";
}

#endif  // ZKX_CODEGEN_EMITTERS_TRANSFORMS_PASSES_TD
