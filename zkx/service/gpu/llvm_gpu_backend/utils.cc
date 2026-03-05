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

#include "zkx/service/gpu/llvm_gpu_backend/utils.h"

#include <stddef.h>

#include "absl/strings/str_cat.h"

namespace zkx::gpu {

std::string ReplaceFilenameExtension(std::string_view filename,
                                     std::string_view new_extension) {
  size_t pos = filename.rfind('.');
  std::string_view stem = pos == std::string_view::npos
                              ? filename
                              : std::string_view(filename.data(), pos);
  return absl::StrCat(stem, ".", new_extension);
}

}  // namespace zkx::gpu
