/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#ifndef XLA_TSL_UTIL_BYTE_SWAP_ARRAY_H_
#define XLA_TSL_UTIL_BYTE_SWAP_ARRAY_H_

#include "absl/status/status.h"

namespace tsl {

// Byte-swap an entire array of atomic C/C++ types in place.
//
// The input `array` must be aligned to at least `bytes_per_elem`.
//
// Args:
//  array: Pointer to the beginning of the array
//  bytes_per_elem: Number of bytes in each element of the array
//  array_len: Number of elements in the array
//
// Returns: absl::OkStatus() on success, UnimplementedError otherwise
//
absl::Status ByteSwapArray(char* array, size_t bytes_per_elem, int array_len);

}  // namespace tsl

#endif  // XLA_TSL_UTIL_BYTE_SWAP_ARRAY_H_
