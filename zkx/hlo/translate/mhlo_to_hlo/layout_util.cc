/* Copyright 2017 The OpenXLA Authors.
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

#include "zkx/hlo/translate/mhlo_to_hlo/layout_util.h"

#include <cstdint>
#include <vector>

#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/shape_util.h"

namespace mlir {

// There is a shape_representation_fn for an output, this function uses a
// reshape to fix the layout.
absl::StatusOr<zkx::ZkxOp> ReshapeWithCorrectRepresentationAndSharding(
    zkx::ZkxBuilder *builder, zkx::ZkxOp original, zkx::Shape original_shape,
    const LayoutPreferenceFn &layout_preference_fn,
    const ShapeRepresentationFn &shape_representation_fn, bool fast_mem) {
  if (original_shape.IsTuple()) {
    std::vector<zkx::ZkxOp> elements;
    for (int i = 0; i < original_shape.tuple_shapes_size(); ++i) {
      TF_ASSIGN_OR_RETURN(
          auto element,
          ReshapeWithCorrectRepresentationAndSharding(
              builder, zkx::GetTupleElement(original, i),
              original_shape.tuple_shapes(i), layout_preference_fn,
              shape_representation_fn, fast_mem));
      elements.push_back(element);
    }
    return zkx::Tuple(builder, elements);
  }
  if (!original_shape.IsArray()) return original;
  TF_ASSIGN_OR_RETURN(auto layout_preference,
                      layout_preference_fn
                          ? layout_preference_fn(original_shape)
                          : ZkxLayoutPreference::kNoPreference);
  TF_ASSIGN_OR_RETURN(
      auto to_shape,
      shape_representation_fn
          ? shape_representation_fn(original_shape, fast_mem, layout_preference)
          : original_shape);
  if (zkx::ShapeUtil::Compatible(original_shape, to_shape)) {
    for (int64_t i = 0; i < original_shape.rank(); ++i) {
      to_shape.set_dynamic_dimension(i, original_shape.is_dynamic_dimension(i));
    }
  }
  return zkx::Reshape(to_shape, original);
}

}  // namespace mlir
