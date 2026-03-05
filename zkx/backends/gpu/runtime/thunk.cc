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

#include "zkx/backends/gpu/runtime/thunk.h"

#include <algorithm>
#include <utility>

#include "xla/tsl/platform/statusor.h"
#include "zkx/service/gpu/backend_configs.pb.h"
#include "zkx/service/gpu/gpu_executable_run_options.h"

namespace zkx::gpu {

//===----------------------------------------------------------------------===//
// Thunk::ExecuteParams
//===----------------------------------------------------------------------===//

// static
Thunk::ExecuteParams Thunk::ExecuteParams::Create(
    const ServiceExecutableRunOptions& run_options,
    const BufferAllocations& buffer_allocations, se::Stream* stream,
    se::Stream* command_buffer_trace_stream,
    ExecutionStreamIdMap additional_compute_streams) {
  return ExecuteParams(&buffer_allocations, stream, command_buffer_trace_stream,
                       run_options.run_options().device_to_host_stream(),
                       run_options.run_options().host_to_device_stream(),
                       additional_compute_streams,
                       run_options.run_options().gpu_executable_run_options()
                           ? run_options.run_options()
                                 .gpu_executable_run_options()
                                 ->requires_exclusive_lock_on_gpu()
                           : false);
}

// static
Thunk::ExecuteParams Thunk::ExecuteParams::CloneWithNewAllocations(
    const Thunk::ExecuteParams& params,
    const BufferAllocations& buffer_allocations) {
  return ExecuteParams(
      &buffer_allocations, params.stream, params.command_buffer_trace_stream,
      params.device_to_host_stream, params.host_to_device_stream,
      params.additional_compute_streams);
}

Thunk::ExecuteParams::ExecuteParams(
    const BufferAllocations* buffer_allocations, se::Stream* stream,
    se::Stream* command_buffer_trace_stream, se::Stream* device_to_host_stream,
    se::Stream* host_to_device_stream,
    ExecutionStreamIdMap additional_compute_streams,
    bool requires_exclusive_lock_on_gpu)
    : buffer_allocations(buffer_allocations),
      stream(stream),
      command_buffer_trace_stream(command_buffer_trace_stream),
      device_to_host_stream(device_to_host_stream),
      host_to_device_stream(host_to_device_stream),
      additional_compute_streams(additional_compute_streams),
      requires_exclusive_lock_on_gpu(requires_exclusive_lock_on_gpu) {}

//===----------------------------------------------------------------------===//

// static
std::string_view Thunk::KindToString(Thunk::Kind kind) {
#define CASE(x)  \
  case Thunk::x: \
    return #x
  switch (kind) {
    CASE(kDynamicSlice);
    CASE(kCommandBuffer);
    CASE(kConditional);
    CASE(kCopy);
    CASE(kCubSort);
    CASE(kCublasLtMatmul);
    CASE(kInfeed);
    CASE(kKernel);
    CASE(kMemset32BitValue);
    CASE(kMemzero);
    CASE(kNorm);
    CASE(kOutfeed);
    CASE(kSequential);
    CASE(kWhile);
    CASE(kWaitForStreams);
  }
}

// static
absl::StatusOr<se::Stream*> Thunk::GetStreamForExecution(
    ExecutionStreamId stream_id, const ExecuteParams& params) {
  if (stream_id == kDefaultExecutionStreamId) {
    return params.stream;
  }
  auto iter = params.additional_compute_streams.find(stream_id);
  if (iter == params.additional_compute_streams.end()) {
    return absl::InvalidArgumentError("Invalid execution stream id.");
  }
  return iter->second;
}

std::ostream& operator<<(std::ostream& os, Thunk::Kind kind) {
  return os << Thunk::KindToString(kind);
}

// static
Thunk::ThunkInfo Thunk::ThunkInfo::WithProfileAnnotation(
    const HloInstruction* instr) {
  ThunkInfo thunk_info;
  thunk_info.profile_annotation = instr->name();
  auto gpu_backend_config = instr->backend_config<GpuBackendConfig>();
  if (gpu_backend_config.ok()) {
    thunk_info.execution_stream_id =
        std::max<uint64_t>(kDefaultExecutionStreamId.value(),
                           gpu_backend_config->operation_queue_id());
  }
  return thunk_info;
}

void Thunk::ForAllThunks(absl::FunctionRef<void(const Thunk*)> fn) const {
  fn(this);
}

}  // namespace zkx::gpu
