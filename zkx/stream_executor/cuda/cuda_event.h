/* Copyright 2015 The OpenXLA Authors.
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

#ifndef ZKX_STREAM_EXECUTOR_CUDA_CUDA_EVENT_H_
#define ZKX_STREAM_EXECUTOR_CUDA_CUDA_EVENT_H_

#include <stdint.h>

#include "absl/status/statusor.h"
#include "third_party/gpus/cuda/include/cuda.h"

#include "zkx/stream_executor/event.h"
#include "zkx/stream_executor/stream_executor.h"

namespace stream_executor::gpu {

class GpuContext;

// This class implements Event for CUDA devices.
class CudaEvent : public Event {
 public:
  Event::Status PollForStatus() override;
  absl::Status WaitForEventOnExternalStream(std::intptr_t stream) override;

  // Creates a new CudaEvent. If allow_timing is false, the event will not
  // support timing, which is cheaper to create.
  static absl::StatusOr<CudaEvent> Create(StreamExecutor* executor,
                                          bool allow_timing);

  CUevent GetHandle() const { return handle_; }

  ~CudaEvent() override;
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;
  CudaEvent(CudaEvent&& other);
  CudaEvent& operator=(CudaEvent&& other);

 private:
  explicit CudaEvent(StreamExecutor* executor, CUevent handle)
      : executor_(executor), handle_(handle) {}

  // The StreamExecutor to which this object and CUevent are bound.
  StreamExecutor* executor_;  // not owned

  // The underlying CUDA event handle.
  CUevent handle_;  // not owned
};

}  // namespace stream_executor::gpu

#endif  // ZKX_STREAM_EXECUTOR_CUDA_CUDA_EVENT_H_
