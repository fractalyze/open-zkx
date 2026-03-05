/* Copyright 2022 The OpenXLA Authors.
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

#ifndef ZKX_MLIR_HLO_MHLO_TRANSFORMS_MAP_STABLEHLO_TO_HLO_OP_H_
#define ZKX_MLIR_HLO_MHLO_TRANSFORMS_MAP_STABLEHLO_TO_HLO_OP_H_

#include "stablehlo/dialect/StablehloOps.h"

#include "zkx/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir::stablehlo {

template <typename HloOpTy>
struct HloToStablehloOpImpl;
template <typename HloOpTy>
using HloToStablehloOp = typename HloToStablehloOpImpl<HloOpTy>::Type;

template <typename StablehloOpTy>
struct StablehloToHloOpImpl;
template <typename StablehloOpTy>
using StablehloToHloOp = typename StablehloToHloOpImpl<StablehloOpTy>::Type;

#define MAP_STABLEHLO_TO_HLO(OpName)                                           \
  template <>                                                                  \
  struct HloToStablehloOpImpl<mhlo::OpName> {                                  \
    using Type = stablehlo::OpName;                                            \
  };                                                                           \
  template <>                                                                  \
  struct StablehloToHloOpImpl<stablehlo::OpName> {                             \
    using Type = mhlo::OpName;                                                 \
  };

MAP_STABLEHLO_TO_HLO(AbsOp)
MAP_STABLEHLO_TO_HLO(AddOp)
MAP_STABLEHLO_TO_HLO(AndOp)
MAP_STABLEHLO_TO_HLO(BitcastConvertOp)
MAP_STABLEHLO_TO_HLO(BitReverseOp)
MAP_STABLEHLO_TO_HLO(BroadcastInDimOp)
MAP_STABLEHLO_TO_HLO(BroadcastOp)
MAP_STABLEHLO_TO_HLO(CaseOp)
MAP_STABLEHLO_TO_HLO(ClampOp)
MAP_STABLEHLO_TO_HLO(ClzOp)
MAP_STABLEHLO_TO_HLO(CompareOp)
MAP_STABLEHLO_TO_HLO(ConcatenateOp)
MAP_STABLEHLO_TO_HLO(ConstantOp)
MAP_STABLEHLO_TO_HLO(ConvertOp)
MAP_STABLEHLO_TO_HLO(CreateTokenOp)
MAP_STABLEHLO_TO_HLO(DivOp)
MAP_STABLEHLO_TO_HLO(DotGeneralOp)
MAP_STABLEHLO_TO_HLO(DynamicBroadcastInDimOp)
MAP_STABLEHLO_TO_HLO(DynamicSliceOp)
MAP_STABLEHLO_TO_HLO(DynamicUpdateSliceOp)
MAP_STABLEHLO_TO_HLO(GatherOp)
MAP_STABLEHLO_TO_HLO(GetDimensionSizeOp)
MAP_STABLEHLO_TO_HLO(GetTupleElementOp)
MAP_STABLEHLO_TO_HLO(IfOp)
MAP_STABLEHLO_TO_HLO(IotaOp)
MAP_STABLEHLO_TO_HLO(MapOp)
MAP_STABLEHLO_TO_HLO(MaxOp)
MAP_STABLEHLO_TO_HLO(MinOp)
MAP_STABLEHLO_TO_HLO(MulOp)
MAP_STABLEHLO_TO_HLO(NegOp)
MAP_STABLEHLO_TO_HLO(NotOp)
MAP_STABLEHLO_TO_HLO(OrOp)
MAP_STABLEHLO_TO_HLO(PadOp)
MAP_STABLEHLO_TO_HLO(PopulationCountOp)
MAP_STABLEHLO_TO_HLO(PowOp)
MAP_STABLEHLO_TO_HLO(RealDynamicSliceOp)
MAP_STABLEHLO_TO_HLO(ReduceOp)
MAP_STABLEHLO_TO_HLO(ReduceWindowOp)
MAP_STABLEHLO_TO_HLO(RemOp)
MAP_STABLEHLO_TO_HLO(ReshapeOp)
MAP_STABLEHLO_TO_HLO(ReturnOp)
MAP_STABLEHLO_TO_HLO(ReverseOp)
MAP_STABLEHLO_TO_HLO(ScatterOp)
MAP_STABLEHLO_TO_HLO(SelectOp)
MAP_STABLEHLO_TO_HLO(SetDimensionSizeOp)
MAP_STABLEHLO_TO_HLO(ShiftLeftOp)
MAP_STABLEHLO_TO_HLO(ShiftRightArithmeticOp)
MAP_STABLEHLO_TO_HLO(ShiftRightLogicalOp)
MAP_STABLEHLO_TO_HLO(SignOp)
MAP_STABLEHLO_TO_HLO(SliceOp)
MAP_STABLEHLO_TO_HLO(SubtractOp)
MAP_STABLEHLO_TO_HLO(TransposeOp)
MAP_STABLEHLO_TO_HLO(TupleOp)
// (deprecated) MAP_STABLEHLO_TO_HLO(UnaryEinsumOp)
MAP_STABLEHLO_TO_HLO(WhileOp)
MAP_STABLEHLO_TO_HLO(XorOp)

#undef MAP_STABLEHLO_TO_HLO

} // namespace mlir::stablehlo

#endif // ZKX_MLIR_HLO_MHLO_TRANSFORMS_MAP_STABLEHLO_TO_HLO_OP_H_
