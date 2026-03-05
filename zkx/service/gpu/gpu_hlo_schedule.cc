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

#include "zkx/service/gpu/gpu_hlo_schedule.h"

#include "absl/log/log.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_input_output_alias_config.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/transforms/simplifiers/hlo_memory_scheduler.h"
#include "zkx/service/buffer_value.h"
#include "zkx/service/gpu/ir_emission_utils.h"
#include "zkx/service/hlo_module_config.h"
#include "zkx/shape.h"
#include "zkx/shape_util.h"
#include "zkx/util.h"

namespace zkx::gpu {
namespace {

// Compute the device memory limit to be used by passes like scheduler and
// HLO rematerialization.
int64_t GetSchedulerMemoryLimit(const HloModule &module,
                                const se::DeviceDescription &gpu_device_info,
                                int pointer_size) {
  constexpr int64_t kDefaultMemoryLimitPercentage = 80;

  // There is a "base" value which is either specified in HloModuleConfig
  // (this value should take into account the fact that we need to leave some
  // memory free for allocations that happen outside of XLA's allocator) or
  // obtained from GPU device info (we scale down this value to leave some
  // space for these outside XLA's allocator allocation).
  //
  // From that base value, subtract any input and output sizes (assuming they
  // are live throughout the execution) and then apply a slop factor.
  const int64_t base_limit = module.config().device_memory_size() != 0
                                 ? module.config().device_memory_size()
                                 : gpu_device_info.device_memory_size() *
                                       kDefaultMemoryLimitPercentage / 100;

  // Find the total size of inputs and outputs.
  int64_t total_io_size = 0;
  for (HloInstruction *param :
       module.entry_computation()->parameter_instructions()) {
    ShapeUtil::ForEachSubshape(
        param->shape(),
        [&](const Shape &subshape, const ShapeIndex & /*index*/) {
          total_io_size += GetSizeOfShape(subshape, pointer_size);
        });
  }
  ShapeUtil::ForEachSubshape(
      module.result_shape(),
      [&](const Shape &subshape, const ShapeIndex & /*index*/) {
        total_io_size += GetSizeOfShape(subshape, pointer_size);
      });

  // If any inputs and outputs are aliased, do not double count them.
  module.input_output_alias_config().ForEachAlias(
      [&](const ShapeIndex &output_index,
          const HloInputOutputAliasConfig::Alias &) {
        const Shape &subshape =
            ShapeUtil::GetSubshape(module.result_shape(), output_index);
        total_io_size -= GetSizeOfShape(subshape, pointer_size);
      });

  int64_t limit =
      (base_limit - total_io_size) *
      module.config().debug_options().zkx_gpu_memory_limit_slop_factor() / 100;
  return limit;
}

}  // namespace

absl::StatusOr<ScheduleMetadata> ScheduleGpuModule(
    HloModule *module, int64_t pointer_size,
    const se::DeviceDescription &gpu_device_info) {
  int64_t memory_limit =
      GetSchedulerMemoryLimit(*module, gpu_device_info, pointer_size);

  if (module->has_schedule()) {
    VLOG(1) << "Module already has a schedule, do nothing.";
    return ScheduleMetadata{memory_limit};
  }

  TF_ASSIGN_OR_RETURN(
      HloSchedule schedule,
      ScheduleGpuModuleWithMemoryScheduler(module, pointer_size));
  TF_RETURN_IF_ERROR(module->set_schedule(std::move(schedule)));

  return ScheduleMetadata{memory_limit};
}

absl::StatusOr<HloSchedule> ScheduleGpuModuleWithMemoryScheduler(
    const HloModule *module, int64_t pointer_size, int64_t *peak_memory_bytes) {
  BufferValue::SizeFunction size_func =
      [pointer_size](const BufferValue &buffer) -> int64_t {
    const Shape &shape = buffer.shape();
    if (shape.has_layout() &&
        shape.layout().memory_space() == Layout::kHostMemorySpace) {
      return int64_t{0};
    }
    return ShapeUtil::ByteSizeOf(shape, pointer_size);
  };
  ModuleSchedulerAlgorithm algorithm = ComputationSchedulerToModuleScheduler(
      DefaultMemoryScheduler, PostProcessSchedule);
  return ScheduleModule(module, size_func, algorithm,
                        /*execution_threads=*/{}, peak_memory_bytes);
}

HloInstructionSequence PostProcessSchedule(
    const HloInstructionSequence &input) {
  return input;
}

}  // namespace zkx::gpu
