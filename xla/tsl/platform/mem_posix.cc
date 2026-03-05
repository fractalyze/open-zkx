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

#include <stdlib.h>

#if defined(__linux__)
#include <sys/sysinfo.h>
#else
#include <sys/syscall.h>
#endif

#include "xla/tsl/platform/mem.h"

namespace tsl::port {

void* AlignedMalloc(size_t size, int minimum_alignment) {
#if defined(__ANDROID__)
  return memalign(minimum_alignment, size);
#else  // !defined(__ANDROID__)
  void* ptr = nullptr;
  // posix_memalign requires that the requested alignment be at least
  // sizeof(void*). In this case, fall back on malloc which should return
  // memory aligned to at least the size of a pointer.
  const int required_alignment = sizeof(void*);
  if (minimum_alignment < required_alignment) return Malloc(size);
  int err = posix_memalign(&ptr, minimum_alignment, size);
  if (err != 0) {
    return nullptr;
  } else {
    return ptr;
  }
#endif
}

void AlignedFree(void* aligned_memory) { Free(aligned_memory); }

void AlignedSizedFree(void* aligned_memory, size_t alignment, size_t size) {
  (void)alignment;
  (void)size;

  Free(aligned_memory);
}

void* Malloc(size_t size) { return malloc(size); }

void* Realloc(void* ptr, size_t size) { return realloc(ptr, size); }

void Free(void* ptr) { free(ptr); }

MemoryInfo GetMemoryInfo() {
  MemoryInfo mem_info = {INT64_MAX, INT64_MAX};
#if defined(__linux__)
  struct sysinfo info;
  int err = sysinfo(&info);
  if (err == 0) {
    mem_info.free = info.freeram;
    mem_info.total = info.totalram;
  }
#endif
  return mem_info;
}

MemoryBandwidthInfo GetMemoryBandwidthInfo() {
  MemoryBandwidthInfo membw_info = {INT64_MAX};
  return membw_info;
}

}  // namespace tsl::port
