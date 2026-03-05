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

#include "zkx/mlir_hlo/utils/hlo_utils.h"

#include <numeric>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"

namespace mlir::hlo {

DenseI64ArrayAttr getBroadcastDimensionsAttr(Builder *b, Value x, Value y,
                                             bool allowEmpty) {
  TensorType xType = dyn_cast<RankedTensorType>(x.getType());
  TensorType yType = dyn_cast<RankedTensorType>(y.getType());
  if (!xType || !yType)
    return {};
  if (allowEmpty && xType == yType)
    return {};

  // If the shapes have the same rank, then there is nothing to do.
  auto xRank = xType.getRank(), yRank = yType.getRank();
  if (allowEmpty && xRank == yRank)
    return {};

  // Otherwise if the ranks of the inputs don't match, TensorFlow automatically
  // reshapes the smaller by padding with dimensions of size 1 as a prefix. In
  // other words to pad a 5-vector to a 3-dimensional tensor it is reshaped to
  // have shape [1,1,5]. XLA's automatic broadcast code is able to broadcast
  // from lower to higher rank, but doesn't assume you want to pad as a prefix
  // of the dimensions, and instead needs to be told which dimensions of the
  // higher rank tensor to match to the lower rank tensor.
  auto maxRank = std::max(xRank, yRank);
  auto minRank = std::min(xRank, yRank);

  // Match the lower rank tensor along the larger-numbered dimensions of the
  // higher rank tensor.
  SmallVector<int64_t, 4> broadcastDimensions(minRank);
  std::iota(broadcastDimensions.begin(), broadcastDimensions.end(),
            maxRank - minRank);

  return b->getDenseI64ArrayAttr(broadcastDimensions);
}

bool isSequenceStartingWith0(Attribute attr) {
  DenseIntElementsAttr denseAttr = dyn_cast<DenseIntElementsAttr>(attr);
  for (int64_t i = 0, e = denseAttr.getNumElements(); i < e; ++i)
    if (denseAttr.getValues<APInt>()[i].getSExtValue() != i)
      return false;
  return true;
}

} // namespace mlir::hlo
