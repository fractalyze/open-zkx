/* Copyright 2024 The OpenXLA Authors.
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
#ifndef ZKX_BACKENDS_GPU_CODEGEN_EMITTERS_IR_ZKX_GPU_OPS_TD
#define ZKX_BACKENDS_GPU_CODEGEN_EMITTERS_IR_ZKX_GPU_OPS_TD

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/Interfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "zkx/backends/gpu/codegen/emitters/ir/zkx_gpu_dialect.td"
include "zkx/backends/gpu/codegen/emitters/ir/zkx_gpu_attrs.td"
include "zkx/backends/gpu/codegen/emitters/ir/zkx_gpu_types.td"

class ZKXGPU_Op<string mnemonic, list<Trait> traits = []> :
      Op<ZkxGpuDialect, mnemonic, traits> {
}

def ZKXGPU_AllocateSharedOp : ZKXGPU_Op<"allocate_shared", [
      DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = "Allocates a shared memory tile.";

  let description = [{
    Allocates a shared memory tensor. The tensor is shared among all threads in
    a block.

    ```mlir
    %shared = zkx_gpu.allocate_shared : tensor<32x32xf32>
    ```
  }];

  let results = (outs AnyStaticShapeTensor:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

def ZKXGPU_SyncThreadsOp : ZKXGPU_Op<"sync_threads", [
   TypesMatchWith<"result type matches type of dest",
                   "operands", "results", "$_self">,
   DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = "Synchronizes threads.";

  let description = [{
    Synchronizes threads, taking any number of distributed tensors and returning
    the synchronized state.
  }];

  let arguments = (ins Variadic<AnyRankedTensor>:$operands);
  let results = (outs Variadic<AnyRankedTensor>:$results);

  let assemblyFormat = "operands attr-dict `:` type($operands)";
}

def ZKXGPU_ShuffleReduceOp : ZKXGPU_Op<"shuffle_reduce",
    [Pure,  CallOpInterface,
     TypesMatchWith<"result type matches type of operands",
                     "operands", "results", "$_self">]> {
  let summary = "Performs a full warp shuffle and reduces the values";
  let description = [{
    This op performs a full warp shuffle and reduces the results using the given
    function. The function is invoked with the operands from the low lanes,
    followed by the operands from the high lanes. For example:

    ```
    %result:2 = zkx_gpu.shuffle_reduce (%in0, %in1) to 16 combiner=@argmax
    ```

    Will perform shuffles with distance 16, 8, 4, 2 and 1, and will invoke
    @argmax five times. The first invocations will be

    ```
    @argmax(%in0[i], %in1[i], %in0[16+i], %in1[16+i])
    ```
  }];
  let builders = [
    OpBuilder<(ins "mlir::func::FuncOp":$combiner, "mlir::ValueRange":$operands, "int64_t":$max_distance), [{
      $_state.addOperands(operands);
      $_state.addAttribute("combiner", mlir::SymbolRefAttr::get(combiner));
      $_state.addAttribute("max_distance",
          mlir::IntegerAttr::get(
              mlir::IntegerType::get(combiner.getContext(), 64),
              max_distance));
      $_state.addTypes(combiner.getFunctionType().getResults());
    }]>];
  let arguments = (ins FlatSymbolRefAttr:$combiner,
                   Variadic<AnyType>:$operands,
                   I64Attr:$max_distance);
  let results = (outs Variadic<AnyType>:$results);

  let extraClassDeclaration = [{
    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::SymbolRefAttr>("combiner");
    }
    operand_range getArgOperands() {
      return getOperands();
    }
    mlir::MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      (*this)->setAttr("combiner", mlir::cast<mlir::SymbolRefAttr>(callee));
    }
    mlir::Attribute removeArgAttrsAttr() { return nullptr; }
    mlir::Attribute removeResAttrsAttr() { return nullptr; }
    mlir::ArrayAttr getArgAttrsAttr() { return nullptr; }
    mlir::ArrayAttr getResAttrsAttr() { return nullptr; }
    void setArgAttrsAttr(mlir::ArrayAttr) { return; }
    void setResAttrsAttr(mlir::ArrayAttr) { return; }
  }];
  let hasCustomAssemblyFormat = 1;
}

def ZKXGPU_MaterializeOp : ZKXGPU_Op<"materialize",
    [AttrSizedOperandSegments, CallOpInterface]> {
  let summary = "Reads a tensor into registers";
  let arguments = (ins Variadic<AnyType>:$input,
                       Variadic<Index>:$indices,
                       FlatSymbolRefAttr:$callee,
                       ZKX_IndexingMapAttr:$map);
  let results = (outs ZKXGPU_IndexedVectorType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $callee `(` $input `)` `at` $map `(` $indices `)` attr-dict `:` functional-type($input, results)
  }];
  let extraClassDeclaration = [{
    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::SymbolRefAttr>("callee");
    }
    operand_range getArgOperands() {
      return getInput();
    }
    mlir::MutableOperandRange getArgOperandsMutable() {
      return getInputMutable();
    }
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      (*this)->setAttr("callee", mlir::cast<mlir::SymbolRefAttr>(callee));
    }
    mlir::Attribute removeArgAttrsAttr() { return nullptr; }
    mlir::Attribute removeResAttrsAttr() { return nullptr; }
    mlir::ArrayAttr getArgAttrsAttr() { return nullptr; }
    mlir::ArrayAttr getResAttrsAttr() { return nullptr; }
    void setArgAttrsAttr(mlir::ArrayAttr) { return; }
    void setResAttrsAttr(mlir::ArrayAttr) { return; }
  }];
}

def ZKXGPU_InsertOp : ZKXGPU_Op<"insert", [TypesMatchWith<
                   "expected result type to match dest type",
                   "result", "dest", "$_self">]> {
  let summary = "Inserts an indexed vector into a tensor";
  let arguments = (ins ZKXGPU_IndexedVectorType:$source,
                       Variadic<Index>:$indices,
                       AnyRankedTensor:$dest,
                       ZKX_IndexingMapAttr:$map);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $source `(` $indices `)` `into` $dest `at` $map attr-dict `:` type($source) `->` type($result)
  }];
}

def ZKXGPU_ReduceOp : ZKXGPU_Op<"reduce", [
    Pure,  CallOpInterface, SameVariadicOperandSize,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
    ]> {
  let summary = "Performs a reduction";
  let description = [{
    The `zkx_gpu.reduce` op performs a variadic reduction of the provided
    operands using the list of dimensions and a symbol for a combiner function.

    ```mlir
    func.func @add(%a_acc: f32, %b_acc: i32, %a: f32, %b: i32)
        -> (f32, i32) {
      %0 = arith.addf %a_acc, %a : f32
      %1 = arith.addi %b_acc, %b : i32
      func.return %0, %1 : f32, i32
    }
    %sum:2 = zkx_gpu.reduce (%in0, %in1) inits(%init0, %init1) dimensions=[0, 2]
      combiner=@add : tensor<16x8x4xf32>, tensor<16x8x4xi32>
        to tensor<8xf32>, tensor<8xi32>
    ```
  }];
  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    Variadic<AnyType>:$inits,
    ConfinedAttr<DenseI64ArrayAttr,
      [DenseArrayStrictlySorted<DenseI64ArrayAttr>]>:$dimensions,
    FlatSymbolRefAttr:$combiner);
  let results = (outs Variadic<AnyRankedTensor>:$results);

  let extraClassDeclaration = [{
    operand_range getArgOperands() {
      return getInputs();
    }
    mlir::MutableOperandRange getArgOperandsMutable() {
      return getInputsMutable();
    }
    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::SymbolRefAttr>("combiner");
    }
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      (*this)->setAttr("combiner", mlir::cast<mlir::SymbolRefAttr>(callee));
    }
    mlir::Attribute removeArgAttrsAttr() { return nullptr; }
    mlir::Attribute removeResAttrsAttr() { return nullptr; }
    mlir::ArrayAttr getArgAttrsAttr() { return nullptr; }
    mlir::ArrayAttr getResAttrsAttr() { return nullptr; }
    void setArgAttrsAttr(mlir::ArrayAttr) { return; }
    void setResAttrsAttr(mlir::ArrayAttr) { return; }
  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif // ZKX_BACKENDS_GPU_CODEGEN_EMITTERS_IR_ZKX_GPU_OPS_TD
