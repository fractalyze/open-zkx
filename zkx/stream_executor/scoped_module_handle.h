/* Copyright 2024 The OpenXLA Authors.
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

#ifndef ZKX_STREAM_EXECUTOR_SCOPED_MODULE_HANDLE_H_
#define ZKX_STREAM_EXECUTOR_SCOPED_MODULE_HANDLE_H_

#include <utility>

#include "absl/log/check.h"

#include "zkx/stream_executor/module_spec.h"
#include "zkx/stream_executor/stream_executor.h"

namespace stream_executor {

// A wrapper around ModuleHandle that uses RAII to manage its lifetime.
class ScopedModuleHandle {
 public:
  ScopedModuleHandle(StreamExecutor* executor, ModuleHandle module_handle)
      : executor_(executor), module_handle_(module_handle) {}

  ScopedModuleHandle(ScopedModuleHandle&& other)
      : executor_(std::exchange(other.executor_, nullptr)),
        module_handle_(std::exchange(other.module_handle_, ModuleHandle())) {}

  ScopedModuleHandle& operator=(ScopedModuleHandle&& other) {
    if (this != &other) {
      MaybeUnloadModule();
      executor_ = std::exchange(other.executor_, nullptr);
      module_handle_ = std::exchange(other.module_handle_, ModuleHandle());
    }
    return *this;
  }

  ~ScopedModuleHandle() { MaybeUnloadModule(); }

 private:
  void MaybeUnloadModule() {
    if (static_cast<bool>(module_handle_)) {
      CHECK(executor_->UnloadModule(module_handle_));
    }
  }

  StreamExecutor* executor_;  // not owned
  ModuleHandle module_handle_;

  ScopedModuleHandle(const ScopedModuleHandle&) = delete;
  void operator=(const ScopedModuleHandle&) = delete;
};

}  // namespace stream_executor

#endif  // ZKX_STREAM_EXECUTOR_SCOPED_MODULE_HANDLE_H_
