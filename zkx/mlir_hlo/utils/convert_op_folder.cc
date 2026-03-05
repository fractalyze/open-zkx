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

// This file defines helpers useful when creating or manipulating lhlo/hlo.

#include "zkx/mlir_hlo/utils/convert_op_folder.h"

#include "llvm/ADT/APSInt.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"

namespace mlir::hlo {

ElementsAttr convertElementsAttr(const ElementsAttr &elements, Type newType) {
  Type oldType = getElementTypeOrSelf(elements);
  if (!isa<IntegerType>(oldType) || !isa<IntegerType>(newType)) {
    return {};
  }

  size_t bitWidth = cast<IntegerType>(newType).getWidth();
  // Treat signless integers except i1 as signed.
  bool isOldTypeUnsigned = oldType.isInteger(1) || oldType.isUnsignedInteger();

  // Int -> Int
  return cast<DenseIntElementsAttr>(elements).mapValues(
      newType, [&](const APInt &intVal) -> APInt {
        return APSInt(intVal, isOldTypeUnsigned).extOrTrunc(bitWidth);
      });
}

} // namespace mlir::hlo
