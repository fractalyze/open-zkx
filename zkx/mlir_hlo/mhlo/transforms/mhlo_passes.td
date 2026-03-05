/* Copyright 2020 The OpenXLA Authors.
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

include "mlir/Pass/PassBase.td"

def SinkConstantsToControlFlowPass : Pass<"mhlo-sink-constants-to-control-flow", "func::FuncOp"> {
  let summary = "Sink constants implicitly captured in control flow regions. This "
    "is necessary to export to ZKX.";
  let constructor = "createSinkConstantsToControlFlowPass()";
  let description = [{
    A pass that sinks constants implicitly captured in control flow regions. This
    is necessary to export to ZKX, because ZKX's representation of control flow
    doesn't have the notion of implicit capture.

    For example given this function:

    ```mlir
      func @sink_const_to_sort(%arg0: tensor<16xu32>) {
        %c0 = arith.constant dense<1.0> : tensor<u32>
        %0 = "mhlo.sort"(%arg0) ( {
        ^bb0(%arg1: tensor<u32>, %arg2: tensor<u32>):
          %1 = "mhlo.divide"(%arg1, %c0) : (tensor<u32>, tensor<u32>) -> tensor<u32>
          %2 = "mhlo.divide"(%arg2, %c0) : (tensor<u32>, tensor<u32>) -> tensor<u32>
          %3 = "mhlo.compare"(%1, %2) {comparison_direction = "GT"} : (tensor<u32>, tensor<u32>) -> tensor<i1>
          "mhlo.return"(%3) : (tensor<i1>) -> ()
        }) {is_stable = true} : (tensor<16xu32>) -> tensor<16xi32>
        return
      }
    ```

    Observe how the arith.constant is moved into the region it's used in:

    ```mlir
      module  {
        func @sink_const_to_sort(%arg0: tensor<16xu32>) {
          %0 = "mhlo.sort"(%arg0) ( {
          ^bb0(%arg1: tensor<u32>, %arg2: tensor<u32>):
            %cst = arith.constant dense<1> : tensor<u32>
            %1 = mhlo.divide %arg1, %cst : tensor<u32>
            %2 = mhlo.divide %arg2, %cst : tensor<u32>
            %3 = "mhlo.compare"(%1, %2) {comparison_direction = "GT"} : (tensor<u32>, tensor<u32>) -> tensor<i1>
            "mhlo.return"(%3) : (tensor<i1>) -> ()
          }) {is_stable = true} : (tensor<16xu32>) -> tensor<16xi32>
          return
        }
      }
    ```
  }];
}

def SymbolicShapeOptimization : Pass<"symbolic-shape-optimization", "func::FuncOp"> {
  let summary = "Analyzes shapes and performs shape-related optimizations";
  let constructor = "createSymbolicShapeOptimizationPass()";
}

def ConvertToSignlessPass : Pass<"convert-to-signless", "ModuleOp"> {
  let summary = "Pass to transform the IR to be on signless integers.";
  let constructor = "createConvertToSignlessPass()";
}

def StablehloLegalizeToHloPass : Pass<"stablehlo-legalize-to-hlo", "ModuleOp"> {
  let summary = "Legalize StableHLO to HLO.";
  let constructor = "createStablehloLegalizeToHloPass()";
  let dependentDialects = ["mhlo::MhloDialect"];
}

def PrepareForExportPass : Pass<"zkx-prepare-for-export", "mlir::func::FuncOp"> {
  let summary = "Prepare for ZKX export";

  let description = [{
    This pass transforms functions in preparation for exporting to ZKX. This

    * converts splat constants to constants and broadcasts to reduce size of
      and speedup the creation of the generated proto during export.

    Note: The result of this pass need not be a module in canonical form and
    canonicalization may undo transformations.
  }];
}

def ShapeLegalizeToHloPass : Pass<"shape-legalize-to-hlo", "func::FuncOp"> {
  let summary = "Legalize shape-related ops to HLO.";
  let constructor = "createShapeLegalizeToHloPass()";
  let description = [{
    An experimental pass that legalizes shape-related ops to MHLO ops.

    Bringing shape and data computations together via an optional pass will
    make it possible for the MHLO ecosystem to potentially leverage the
    compilation pipelines that use HLO operations to model dynamism.
  }];
  let dependentDialects = ["mhlo::MhloDialect"];
  let options = [
    Option<"legalize_constraints_", "legalize-constraints", "bool",
           /*default=*/"false", "Whether to legalize Cstr Ops to shape_assertion custom_call">
  ];
}
