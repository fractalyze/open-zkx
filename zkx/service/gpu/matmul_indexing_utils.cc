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

#include "zkx/service/gpu/matmul_indexing_utils.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/status_macros.h"
#include "zkx/util.h"

namespace zkx::gpu {

absl::StatusOr<std::vector<int64_t>> GetNonContractingDims(
    const Shape& shape, absl::Span<const int64_t> batch_dims,
    absl::Span<const int64_t> contracting_dims) {
  auto nc =
      ::zkx::GetNonContractingDims(shape.rank(), contracting_dims, batch_dims);

  TF_RET_CHECK(batch_dims.size() + contracting_dims.size() + nc.size() ==
               shape.rank());
  return std::vector<int64_t>(nc.begin(), nc.end());
}

const google::protobuf::RepeatedField<int64_t>& BatchDimensionsForOperand(
    const HloInstruction& dot, int operand_number) {
  const DotDimensionNumbers& dimension_numbers = dot.dot_dimension_numbers();
  if (operand_number == 0) {
    return dimension_numbers.lhs_batch_dimensions();
  }
  return dimension_numbers.rhs_batch_dimensions();
}

absl::StatusOr<int64_t> ContractingDimensionIndex(const HloInstruction& dot,
                                                  int operand_number) {
  const DotDimensionNumbers& dimension_numbers = dot.dot_dimension_numbers();
  if (operand_number == 0) {
    TF_RET_CHECK(dimension_numbers.lhs_contracting_dimensions().size() == 1);
    return dimension_numbers.lhs_contracting_dimensions(0);
  }
  TF_RET_CHECK(dimension_numbers.rhs_contracting_dimensions().size() == 1);
  return dimension_numbers.rhs_contracting_dimensions(0);
}

absl::StatusOr<int64_t> NonContractingDimensionIndex(const HloInstruction& dot,
                                                     int operand_number) {
  TF_ASSIGN_OR_RETURN(int64_t contracting_dim,
                      ContractingDimensionIndex(dot, operand_number));
  TF_ASSIGN_OR_RETURN(
      std::vector<int64_t> non_contracting_dims,
      GetNonContractingDims(dot.operand(operand_number)->shape(),
                            BatchDimensionsForOperand(dot, operand_number),
                            {contracting_dim}));
  TF_RET_CHECK(non_contracting_dims.size() == 1);
  return non_contracting_dims.front();
}

}  // namespace zkx::gpu
