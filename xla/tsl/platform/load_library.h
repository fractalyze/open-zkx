/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

#ifndef XLA_TSL_PLATFORM_LOAD_LIBRARY_H_
#define XLA_TSL_PLATFORM_LOAD_LIBRARY_H_

#include <string>

#include "absl/status/status.h"

namespace tsl {

namespace internal {

absl::Status LoadDynamicLibrary(const char* library_filename, void** handle);
absl::Status GetSymbolFromLibrary(void* handle, const char* symbol_name,
                                  void** symbol);
std::string FormatLibraryFileName(std::string_view name,
                                  std::string_view version);

}  // namespace internal

}  // namespace tsl

#endif  // XLA_TSL_PLATFORM_LOAD_LIBRARY_H_
