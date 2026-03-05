/* Copyright 2025 The OpenXLA Authors.
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

#ifndef ZKX_STREAM_EXECUTOR_MEMORY_ALLOCATOR_H_
#define ZKX_STREAM_EXECUTOR_MEMORY_ALLOCATOR_H_

#include <stdint.h>

#include <memory>

#include "absl/status/statusor.h"

#include "zkx/stream_executor/memory_allocation.h"

namespace stream_executor {

// This class defines the interface for memory allocators.
class MemoryAllocator {
 public:
  virtual ~MemoryAllocator() = default;

  // Allocates a region of memory, or returns an error if the allocation fails.
  virtual absl::StatusOr<std::unique_ptr<MemoryAllocation>> Allocate(
      uint64_t size) = 0;
};

}  // namespace stream_executor

#endif  // ZKX_STREAM_EXECUTOR_MEMORY_ALLOCATOR_H_
