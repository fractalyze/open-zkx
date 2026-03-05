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

#ifndef ZKX_MLIR_HLO_UTILS_HLO_UTILS_H_
#define ZKX_MLIR_HLO_UTILS_HLO_UTILS_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"

namespace mlir::hlo {

// Computes the broadcast dimensions attr for an elementwise binary operator
// between two ranked tensors.
// If `allow_empty` is true, then null can be returned to mean that the
// broadcast is an "identity".lir
DenseI64ArrayAttr getBroadcastDimensionsAttr(Builder *b, Value x, Value y,
                                             bool allowEmpty = true);

// Return true if Attr has values [0, 1, ...].
bool isSequenceStartingWith0(Attribute attr);

} // namespace mlir::hlo

#endif // ZKX_MLIR_HLO_UTILS_HLO_UTILS_H_
