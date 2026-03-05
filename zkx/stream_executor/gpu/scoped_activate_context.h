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

#ifndef ZKX_STREAM_EXECUTOR_GPU_SCOPED_ACTIVATE_CONTEXT_H_
#define ZKX_STREAM_EXECUTOR_GPU_SCOPED_ACTIVATE_CONTEXT_H_

#include "zkx/stream_executor/activate_context.h"
#include "zkx/stream_executor/gpu/context.h"

namespace stream_executor::gpu {

// Ensures a context is activated within a scope.
class ScopedActivateContext : public ActivateContext {
 public:
  // Activates the context via Context::SetActive.
  explicit ScopedActivateContext(Context* gpu_context);

  // Checks that the context has remained activated for the duration of the
  // scope.
  ~ScopedActivateContext() override;

 private:
  Context* to_restore_ = nullptr;  // not owned
};

}  // namespace stream_executor::gpu

#endif  // ZKX_STREAM_EXECUTOR_GPU_SCOPED_ACTIVATE_CONTEXT_H_
