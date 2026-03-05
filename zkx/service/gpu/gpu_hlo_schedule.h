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

#ifndef ZKX_SERVICE_GPU_GPU_HLO_SCHEDULE_H_
#define ZKX_SERVICE_GPU_GPU_HLO_SCHEDULE_H_

#include <cstdint>

#include "absl/status/statusor.h"

#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/hlo/ir/hlo_schedule.h"
#include "zkx/stream_executor/device_description.h"

namespace zkx::gpu {

struct ScheduleMetadata {
  int64_t scheduler_mem_limit;
};

// Determines the schedule of HLO instructions for a module run on the GPU.
absl::StatusOr<ScheduleMetadata> ScheduleGpuModule(
    HloModule *module, int64_t pointer_size,
    const se::DeviceDescription &gpu_device_info);

// Schedules a GPU module with `DefaultMemoryScheduler` and
// `PostProcessSchedule` postprocessing. If `peak_memory_bytes` is not nullptr,
// then the it will be set to peak memory usage in bytes.
absl::StatusOr<HloSchedule> ScheduleGpuModuleWithMemoryScheduler(
    const HloModule *module, int64_t pointer_size,
    int64_t *peak_memory_bytes = nullptr);

HloInstructionSequence PostProcessSchedule(const HloInstructionSequence &input);

}  // namespace zkx::gpu

#endif  // ZKX_SERVICE_GPU_GPU_HLO_SCHEDULE_H_
