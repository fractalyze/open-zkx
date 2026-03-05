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

#ifndef ZKX_BACKENDS_GPU_RUNTIME_COPY_THUNK_H_
#define ZKX_BACKENDS_GPU_RUNTIME_COPY_THUNK_H_

#include <cstdint>

#include "absl/status/status.h"

#include "zkx/backends/gpu/runtime/thunk.h"
#include "zkx/service/buffer_assignment.h"

namespace zkx::gpu {

// A thunk that copies data from a device buffer to another device buffer.
class DeviceToDeviceCopyThunk : public Thunk {
 public:
  // Constructs a CopyThunk that copies host data from `source_buffer` to the
  // device buffer `destination_buffer`. `mem_size` is the size of the data in
  // bytes.
  DeviceToDeviceCopyThunk(ThunkInfo thunk_info,
                          const BufferAllocation::Slice &source_buffer,
                          const BufferAllocation::Slice &destination_buffer,
                          uint64_t mem_size);

  DeviceToDeviceCopyThunk(const DeviceToDeviceCopyThunk &) = delete;
  DeviceToDeviceCopyThunk &operator=(const DeviceToDeviceCopyThunk &) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams &params) override;

  const BufferAllocation::Slice &source() const { return source_buffer_; }
  const BufferAllocation::Slice &destination() const {
    return destination_buffer_;
  }
  uint64_t size_bytes() const { return mem_size_; }

 private:
  const BufferAllocation::Slice source_buffer_;
  const BufferAllocation::Slice destination_buffer_;
  const uint64_t mem_size_;
};

//===----------------------------------------------------------------------===//
// CopyThunk
//===----------------------------------------------------------------------===//
class CopyThunk : public Thunk {
 public:
  CopyThunk(ThunkInfo thunk_info, const BufferAllocation::Slice &source_buffer,
            const BufferAllocation::Slice &destination_buffer,
            uint64_t mem_size);
  absl::Status ExecuteOnStream(const ExecuteParams &params) override;
  const BufferAllocation::Slice &source() const { return source_buffer_; }
  const BufferAllocation::Slice &destination() const {
    return destination_buffer_;
  }
  uint64_t size_bytes() const { return mem_size_; }

 private:
  const BufferAllocation::Slice source_buffer_;
  const BufferAllocation::Slice destination_buffer_;
  const uint64_t mem_size_;
};

//===----------------------------------------------------------------------===//
// DeviceToHostCopyThunk
//===----------------------------------------------------------------------===//
// The memcpy between a host and a device

// A thunk that copies data from a device buffer to a host buffer.
class DeviceToHostCopyThunk : public CopyThunk {
 public:
  DeviceToHostCopyThunk(ThunkInfo thunk_info,
                        const BufferAllocation::Slice &source_buffer,
                        const BufferAllocation::Slice &destination_buffer,
                        uint64_t mem_size);
  absl::Status ExecuteOnStream(const ExecuteParams &params) override;
};

//===----------------------------------------------------------------------===//
// HostToDeviceCopyThunk
//===----------------------------------------------------------------------===//
// The memcpy between a host and a device

// A thunk that copies data from a host buffer to a device buffer.
class HostToDeviceCopyThunk : public CopyThunk {
 public:
  HostToDeviceCopyThunk(ThunkInfo thunk_info,
                        const BufferAllocation::Slice &source_buffer,
                        const BufferAllocation::Slice &destination_buffer,
                        uint64_t mem_size);
  absl::Status ExecuteOnStream(const ExecuteParams &params) override;
};

}  // namespace zkx::gpu

#endif  // ZKX_BACKENDS_GPU_RUNTIME_COPY_THUNK_H_
