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

#ifndef ZKX_HLO_TRANSLATE_MHLO_TO_HLO_LITERAL_EXPORTER_H_
#define ZKX_HLO_TRANSLATE_MHLO_TO_HLO_LITERAL_EXPORTER_H_

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"

#include "zkx/layout.h"
#include "zkx/literal.h"

namespace mlir::mhlo {

absl::StatusOr<zkx::Literal> CreateLiteralFromAttribute(ElementsAttr attr,
                                                        zkx::Layout layout);

}  // namespace mlir::mhlo

#endif  // ZKX_HLO_TRANSLATE_MHLO_TO_HLO_LITERAL_EXPORTER_H_
