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

// Utilities for working with ZKX layout and shapes.

#ifndef ZKX_HLO_TRANSLATE_MHLO_TO_HLO_LAYOUT_UTIL_H_
#define ZKX_HLO_TRANSLATE_MHLO_TO_HLO_LAYOUT_UTIL_H_

#include <functional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "zkx/hlo/builder/zkx_builder.h"
#include "zkx/shape.h"

namespace mlir {

// ZKX Layout preferences. Currently, when it comes to TPU, there are two
// primary layout choices for any ZKX arguments (parameter or resource): (1)
// CompactChunkPadded and (2) Linear. CompactChunkPadded is the native TPU
// layout while Linear is native host (CPU) layout.
// This enum allows the caller of ZKX to propagate layout preference to the ZKX
// compiler.
//   kNoPreference: the generic layout where the ZKX compiler has the freedom
//                  to assign any layout.
//   kTpuPreferCompactChunkPaddedLayout: use native TPU layout on TPU.
//   kTpuPreferLinearLayout: use native CPU layout on TPU. The compiler may
//                           insert transformation TPU kernels.
// As the layout of any argument will change from a native host layout to a
// native TPU layout either on host or on device, ZKX compiler and TPU runtime
// must be in coordination to transform the parameters in a consistent way.
enum class ZkxLayoutPreference {
  kNoPreference = 0,
  kTpuPreferCompactChunkPaddedLayout = 1,
  kTpuPreferLinearLayout = 2
};

// The following defines the layout preference of an zkx tensor.
// The return value of LayoutPreferenceFn can be used in
// ShapeRepresentationFn.
typedef std::function<absl::StatusOr<ZkxLayoutPreference>(
    const zkx::Shape &shape)>
    LayoutPreferenceFn;

typedef std::function<absl::StatusOr<zkx::Shape>(
    const zkx::Shape &shape, bool fast_mem,
    ZkxLayoutPreference layout_preference)>
    ShapeRepresentationFn;

// Return a LayoutPreferenceFn that always uses kNoPreference layout.
LayoutPreferenceFn UseNoPreferenceLayoutFn();

// Adds reshapes to fix the layout of an output, if a shape_representation_fn
// is present.
absl::StatusOr<zkx::ZkxOp> ReshapeWithCorrectRepresentationAndSharding(
    zkx::ZkxBuilder *builder, zkx::ZkxOp original, zkx::Shape original_shape,
    const LayoutPreferenceFn &layout_preference_fn,
    const ShapeRepresentationFn &shape_representation_fn, bool fast_mem);

}  // namespace mlir

#endif  // ZKX_HLO_TRANSLATE_MHLO_TO_HLO_LAYOUT_UTIL_H_
