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

#ifndef ZKX_STREAM_EXECUTOR_CUDA_DELAY_KERNEL_H_
#define ZKX_STREAM_EXECUTOR_CUDA_DELAY_KERNEL_H_

#include "absl/status/statusor.h"

#include "zkx/stream_executor/gpu/gpu_semaphore.h"
#include "zkx/stream_executor/stream.h"

namespace stream_executor::gpu {

// Launches the delay kernel on the given stream. The caller is responsible for
// keeping the returned semaphore alive until the kernel finished executing.
// Setting the semaphore to `kRelease` makes the kernel quit.
absl::StatusOr<GpuSemaphore> LaunchDelayKernel(Stream* stream);

}  // namespace stream_executor::gpu

#endif  // ZKX_STREAM_EXECUTOR_CUDA_DELAY_KERNEL_H_
