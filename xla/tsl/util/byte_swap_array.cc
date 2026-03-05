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

#include "xla/tsl/util/byte_swap_array.h"

#include "absl/base/internal/endian.h"
#include "absl/strings/str_cat.h"

namespace tsl {
namespace {

template <typename T, typename SwapFunc>
absl::Status ByteSwapArrayHelper(char* array, int array_len,
                                 SwapFunc swap_func) {
  auto typed_array = reinterpret_cast<T*>(array);
  for (int i = 0; i < array_len; i++) {
    typed_array[i] = swap_func(typed_array[i]);
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status ByteSwapArray(char* array, size_t bytes_per_elem, int array_len) {
  switch (bytes_per_elem) {
    case 1: {
      // No-op
      return absl::OkStatus();
    }
    case 2: {
      return ByteSwapArrayHelper<uint16_t>(array, array_len, absl::gbswap_16);
    }
    case 4: {
      return ByteSwapArrayHelper<uint32_t>(array, array_len, absl::gbswap_32);
    }
    case 8: {
      return ByteSwapArrayHelper<uint64_t>(array, array_len, absl::gbswap_64);
    }
    default: {
      return absl::UnimplementedError(absl::StrCat(
          "Byte-swapping of ", bytes_per_elem, "-byte values not supported."));
    }
  }
}

}  // namespace tsl
