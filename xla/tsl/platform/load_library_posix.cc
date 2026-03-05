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

#include <dlfcn.h>

#include "absl/strings/substitute.h"

#include "xla/tsl/platform/load_library.h"

namespace tsl {

namespace internal {

absl::Status LoadDynamicLibrary(const char* library_filename, void** handle) {
  *handle = dlopen(library_filename, RTLD_NOW | RTLD_LOCAL);
  if (!*handle) {
    // Note that in C++17 std::string_view(nullptr) gives segfault!
    const char* error_msg = dlerror();
    return absl::NotFoundError(error_msg ? error_msg : "(null error message)");
  }
  return absl::OkStatus();
}

absl::Status GetSymbolFromLibrary(void* handle, const char* symbol_name,
                                  void** symbol) {
  // Check that the handle is not NULL to avoid dlsym's RTLD_DEFAULT behavior.
  if (!handle) {
    *symbol = nullptr;
  } else {
    *symbol = dlsym(handle, symbol_name);
  }
  if (!*symbol) {
    // Note that in C++17 std::string_view(nullptr) gives segfault!
    const char* error_msg = dlerror();
    return absl::NotFoundError(error_msg ? error_msg : "(null error message)");
  }
  return absl::OkStatus();
}

std::string FormatLibraryFileName(std::string_view name,
                                  std::string_view version) {
  std::string filename;
#if defined(__APPLE__)
  if (version.size() == 0) {
    filename = absl::Substitute("lib$0.dylib", name);
  } else {
    filename = absl::Substitute("lib$0.$1.dylib", name, version);
  }
#else
  if (version.empty()) {
    filename = absl::Substitute("lib$0.so", name);
  } else {
    filename = absl::Substitute("lib$0.so.$1", name, version);
  }
#endif
  return filename;
}

}  // namespace internal

}  // namespace tsl
