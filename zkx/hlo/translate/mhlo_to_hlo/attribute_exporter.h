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

#ifndef ZKX_HLO_TRANSLATE_MHLO_TO_HLO_ATTRIBUTE_EXPORTER_H_
#define ZKX_HLO_TRANSLATE_MHLO_TO_HLO_ATTRIBUTE_EXPORTER_H_

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"

#include "zkx/service/hlo.pb.h"
#include "zkx/zkx_data.pb.h"

namespace zkx {

absl::StatusOr<std::vector<ReplicaGroup>> ConvertReplicaGroups(
    mlir::DenseIntElementsAttr input);

// Convert a (N, 2) dense attribute to a list of tuples. This is the way padding
// and source-target pairs are defined in HLO.
absl::StatusOr<std::vector<std::pair<int64_t, int64_t>>> ConvertNx2Attribute(
    std::optional<mlir::DenseIntElementsAttr> optional_attr);

// Returns an OpSharding that represents the result of parsing the given string:
// first, as serialized protobuf, and then as prettyprinted representation.
// Will fail if both attempts at parsing failed.
std::optional<OpSharding> ConvertSharding(mlir::StringRef sharding);

std::optional<HloInputOutputAliasProto> ConvertInputOutputAlias(
    mlir::ArrayRef<mlir::Attribute> aliasing);

}  // namespace zkx

#endif  // ZKX_HLO_TRANSLATE_MHLO_TO_HLO_ATTRIBUTE_EXPORTER_H_
