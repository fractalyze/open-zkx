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

#include "zkx/window_util.h"

#include <cstdint>
#include <string>

#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"

#include "zkx/zkx_data.pb.h"

namespace zkx::window_util {

Window MakeWindow(absl::Span<const int64_t> sizes) {
  Window window;
  for (int64_t size : sizes) {
    auto* dimension = window.add_dimensions();
    dimension->set_size(size);
    dimension->set_stride(1);
    dimension->set_base_dilation(1);
    dimension->set_window_dilation(1);
  }
  return window;
}

Window MakeWindow(absl::Span<const int64_t> sizes,
                  absl::Span<const int64_t> strides) {
  Window window;
  CHECK_EQ(sizes.size(), strides.size());
  for (auto nb = 0; nb < sizes.size(); ++nb) {
    auto* dimension = window.add_dimensions();
    dimension->set_size(sizes[nb]);
    dimension->set_stride(strides[nb]);
    dimension->set_base_dilation(1);
    dimension->set_window_dilation(1);
  }
  return window;
}

namespace {

bool HasStride(const Window& window) {
  for (const auto& dim : window.dimensions()) {
    if (dim.stride() != 1) {
      return true;
    }
  }
  return false;
}

bool HasPadding(const Window& window) {
  for (const auto& dim : window.dimensions()) {
    if (dim.padding_low() != 0 || dim.padding_high() != 0) {
      return true;
    }
  }
  return false;
}

bool HasBaseDilation(const Window& window) {
  for (const auto& dim : window.dimensions()) {
    if (dim.base_dilation() != 1) {
      return true;
    }
  }
  return false;
}

bool HasWindowDilation(const Window& window) {
  for (const auto& dim : window.dimensions()) {
    if (dim.window_dilation() != 1) {
      return true;
    }
  }
  return false;
}

bool HasWindowReversal(const Window& window) {
  for (const auto& dim : window.dimensions()) {
    if (dim.window_reversal()) {
      return true;
    }
  }
  return false;
}

}  // namespace

std::string ToString(const Window& window) {
  using absl::StrAppend;
  using absl::StrCat;

  std::string str;
  const auto add_field =
      [&](const char* heading,
          absl::FunctionRef<std::string(const WindowDimension&)> format) {
        StrAppend(&str, heading, "=");
        const char* prefix = "";
        for (const auto& window_dimension : window.dimensions()) {
          StrAppend(&str, prefix, format(window_dimension));
          prefix = "x";
        }
      };

  if (window.dimensions_size() > 0) {
    add_field("size",
              [](const WindowDimension& dim) { return StrCat(dim.size()); });
  }
  if (HasStride(window)) {
    add_field(" stride",
              [](const WindowDimension& dim) { return StrCat(dim.stride()); });
  }
  if (HasPadding(window)) {
    add_field(" pad", [](const WindowDimension& dim) {
      return StrCat(dim.padding_low(), "_", dim.padding_high());
    });
  }
  if (HasBaseDilation(window)) {
    add_field(" lhs_dilate", [](const WindowDimension& dim) {
      return StrCat(dim.base_dilation());
    });
  }
  if (HasWindowDilation(window)) {
    add_field(" rhs_dilate", [](const WindowDimension& dim) {
      return StrCat(dim.window_dilation());
    });
  }
  if (HasWindowReversal(window)) {
    add_field(" rhs_reversal", [](const WindowDimension& dim) {
      return StrCat(dim.window_reversal() ? 1 : 0);
    });
  }
  return str;
}

bool HasOverlappingWindow(const Window& window) {
  for (const auto& dim : window.dimensions()) {
    if (dim.size() > dim.stride()) {
      return true;
    }
  }
  return false;
}

int64_t DilatedBound(int64_t bound, int64_t dilation) {
  CHECK_GE(bound, 0);
  CHECK_GE(dilation, 1);
  if (bound == 0) {
    return 0;
  }

  // Suppose the array has three entries 123 and the dilation factor is 4. Then
  // the dilated array has 9 entries 1xxx2xxx3. Here, each original entry except
  // the last expands into 4 entries, so that is (bound - 1) * dilation. Then we
  // add 1 to account for the final input element.
  return (bound - 1) * dilation + 1;
}

int64_t StridedBound(int64_t bound, int64_t window_size, int64_t stride) {
  CHECK_GE(window_size, 0);
  CHECK_GE(bound, 0);
  CHECK_GE(stride, 1);

  if (bound == 0 || window_size > bound) {
    return 0;
  }

  // Without considering stride, the maximum valid offset is bound -
  // window_size. Taking stride into account, the valid offsets then have the
  // form q * stride for q = 0, ..., Q such that q * stride <= bound -
  // window_size. This implies that Q equals floor(bound - window_size /
  // stride). There are Q + 1 valid values of q, yielding the formula below.
  return (bound - window_size) / stride + 1;
}

}  // namespace zkx::window_util
