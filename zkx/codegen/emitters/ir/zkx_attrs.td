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

#ifndef ZKX_CODEGEN_EMITTERS_IR_ZKX_ATTRS_TD
#define ZKX_CODEGEN_EMITTERS_IR_ZKX_ATTRS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "zkx/codegen/emitters/ir/zkx_dialect.td"

class ZKX_Attr<string name, list<Trait> traits = []> :
      AttrDef<ZkxDialect, name, traits> {
}

def ZKX_AffineMapParameter :
    AttrOrTypeParameter<"::mlir::AffineMap", ""> {
}

def ZKX_IndexingMapVariableParameter
  : ArrayRefParameter<"::zkx::IndexingMap::Variable",
                      "IndexingMapVariableArray"> {
}

def ZKX_ConstraintsParameter :
    ArrayRefParameter<"::std::pair<::mlir::AffineExpr, ::zkx::Interval>",
                      "ConstraintsArray"> {
}

def ZKX_IndexingMapAttr : ZKX_Attr<"IndexingMap"> {
  let summary = "An Attribute representing an indexing map.";
  let mnemonic = "indexing_map";
  let description = [{This attribute stores an indexing map. See
    https://openxla.org/xla/indexing for more details.
  }];
  let parameters = (ins ZKX_AffineMapParameter:$map,
                        ZKX_IndexingMapVariableParameter:$dim_vars,
                        ZKX_IndexingMapVariableParameter:$range_vars,
                        ZKX_ConstraintsParameter:$constraints);
  let hasCustomAssemblyFormat = 1;
  let builders =  [
    AttrBuilder<(ins "const ::zkx::IndexingMap&":$indexing_map)>,
  ];
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    // Returns the indexing map constructed from IndexingMapAttr.
    zkx::IndexingMap getIndexingMap() const;

    // Returns the number of indexing map results.
    int64_t getNumResults() const;
  }];
}

// Note: settle for BackendKind to avoid clashing with the existing
// zkx::Backend and zkx::BackendType types.
def ZKX_BackendKind : I32EnumAttr<"BackendKind", "ZKX Backend kind (or type)", [
    I32EnumAttrCase<"kCpu", 0, "cpu">,
    I32EnumAttrCase<"kGpu", 1, "gpu">,
    I32EnumAttrCase<"kTpu", 2, "tpu">
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::zkx";
}

def ZKX_BackendKindAttr :
    EnumAttr<ZkxDialect, ZKX_BackendKind, "backend_kind"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // ZKX_CODEGEN_EMITTERS_IR_ZKX_ATTRS_TD
