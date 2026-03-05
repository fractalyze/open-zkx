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

#ifndef ZKX_STREAM_EXECUTOR_GPU_CONTEXT_H_
#define ZKX_STREAM_EXECUTOR_GPU_CONTEXT_H_

#include "absl/status/status.h"

#include "zkx/stream_executor/namespace_alias.h"  // IWYU pragma: keep

namespace stream_executor::gpu {

// This defines a base class for interacting with any context-specific state
// residing in a GPU.
class Context {
 public:
  virtual ~Context() = default;

  // Sets this context to be the active GPU context.
  virtual void SetActive() = 0;

  // Returns true if this Context is the active GPU context.
  virtual bool IsActive() const = 0;

  // Returns the device ordinal associated with this context.
  virtual int device_ordinal() const = 0;

  // Synchronizes all activity on the GPU.
  virtual absl::Status Synchronize() = 0;
};
}  // namespace stream_executor::gpu

#endif  // ZKX_STREAM_EXECUTOR_GPU_CONTEXT_H_
