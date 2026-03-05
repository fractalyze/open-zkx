/* Copyright 2017 The OpenXLA Authors.
Copyright 2026 The ZKX Authors.

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

#ifndef ZKX_WINDOW_UTIL_H_
#define ZKX_WINDOW_UTIL_H_

#include <cstdint>
#include <string>

#include "absl/types/span.h"

#include "zkx/zkx_data.pb.h"

namespace zkx {
namespace window_util {

// Creates a window with the given sizes in the dimensions and all strides set
// to 1.
Window MakeWindow(absl::Span<const int64_t> sizes);

// Creates a window with the given sizes in the dimensions and given strides.
Window MakeWindow(absl::Span<const int64_t> sizes,
                  absl::Span<const int64_t> strides);

std::string ToString(const Window& window);

// Returns true if the window overlaps.
bool HasOverlappingWindow(const Window& window);

// Returns the new bound after dilation.
//
// If a window with the given bound in some dimension is dilated with the given
// dilation factor in that dimension, then the value returned is the bound for
// the array in that dimension after dilation.
//
// For a 1D array with 3 entries 1, 2, 3, a dilation factor of 2 yields a new
// window with values 1, x, 2, x, 3, where x indicates holes left by the
// dilation. So DilatedBound(3, 2) == 5.
int64_t DilatedBound(int64_t bound, int64_t dilation);

// Returns the number of valid positions of a window with the given size and
// stride within an array with the given bound. This is the bound of an output
// array with one element per valid position of the window.
//
// For example, for arguments of (bound=5, window_size=2, stride=2), the
// returned value is 2. There are valid positions at offset 0 and offset 2,
// while offset 4 is not valid since the window's last entry would be at 5,
// which is beyond the bound of 5.
int64_t StridedBound(int64_t bound, int64_t window_size, int64_t stride);

}  // namespace window_util
}  // namespace zkx

#endif  // ZKX_WINDOW_UTIL_H_
