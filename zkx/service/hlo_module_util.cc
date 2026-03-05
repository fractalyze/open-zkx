/* Copyright 2020 The OpenXLA Authors.
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

#include "zkx/service/hlo_module_util.h"

#include <stdint.h>

#include <utility>
#include <vector>

#include "absl/strings/str_format.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/debug_options_flags.h"
#include "zkx/service/computation_layout.h"
#include "zkx/shape_util.h"

namespace zkx {

namespace {

absl::Status ValidateResultShape(const Shape& client_shape,
                                 const Shape& result_shape) {
  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShapeWithOptionalLayout(client_shape));
  if (!ShapeUtil::Compatible(client_shape, result_shape)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Shape used to set computation result layout %s is not compatible "
        "with result shape %s",
        ShapeUtil::HumanStringWithLayout(client_shape),
        ShapeUtil::HumanString(result_shape)));
  }
  return absl::OkStatus();
}
}  // namespace

absl::StatusOr<std::unique_ptr<HloModuleConfig>> CreateModuleConfig(
    const ProgramShape& program_shape,
    absl::Span<const Shape* const> argument_shapes,
    const ExecutionOptions* execution_options, int default_num_replicas,
    std::optional<int> num_threads, const AotCompilationOptions* aot_options) {
  auto config = std::make_unique<HloModuleConfig>(program_shape);
  ComputationLayout* computation_layout =
      config->mutable_entry_computation_layout();
  const int64_t argument_shapes_size = argument_shapes.size();
  if (program_shape.parameters_size() != argument_shapes_size) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "computation takes %d parameters, but %u given",
        program_shape.parameters_size(), argument_shapes.size()));
  }
  for (int i = 0, end = argument_shapes.size(); i < end; ++i) {
    // Verify that shape of arguments matches the shape of the arguments in the
    // ProgramShape.
    if (!ShapeUtil::Compatible(*argument_shapes[i],
                               program_shape.parameters(i))) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Argument does not match shape of computation parameter %d: want "
          "%s, got %s",
          i, ShapeUtil::HumanString(program_shape.parameters(i)),
          ShapeUtil::HumanString(*argument_shapes[i])));
    }
    TF_RETURN_IF_ERROR(
        computation_layout->mutable_parameter_layout(i)->CopyLayoutFromShape(
            *argument_shapes[i]));
  }
  if (execution_options != nullptr &&
      execution_options->has_shape_with_output_layout()) {
    const Shape shape_with_output_layout(
        execution_options->shape_with_output_layout());
    TF_RETURN_IF_ERROR(
        ValidateResultShape(shape_with_output_layout, program_shape.result()));
    TF_RETURN_IF_ERROR(
        computation_layout->mutable_result_layout()->CopyLayoutFromShape(
            shape_with_output_layout));
  } else {
    // If the result layout is not set, then choose the default.
    computation_layout->mutable_result_layout()->SetToDefaultLayout();
  }

  if (execution_options != nullptr) {
    if (execution_options->num_replicas() > 0) {
      config->set_replica_count(execution_options->num_replicas());
    } else {
      config->set_replica_count(default_num_replicas);
    }
    if (execution_options->num_partitions() > 0) {
      config->set_num_partitions(execution_options->num_partitions());
    }
    config->set_exec_time_optimization_effort(
        execution_options->exec_time_optimization_effort());
    config->set_memory_fitting_effort(
        execution_options->memory_fitting_effort());
    config->set_deduplicate_hlo(execution_options->deduplicate_hlo());
    config->set_seed(execution_options->seed());
    config->set_launch_id(execution_options->launch_id());
    config->set_debug_options(execution_options->debug_options());
    config->set_alias_passthrough_params(
        execution_options->alias_passthrough_params());
    config->set_device_memory_size(execution_options->device_memory_size());
    config->set_use_shardy_partitioner(
        execution_options->use_shardy_partitioner());
  } else {
    config->set_replica_count(default_num_replicas);
    config->set_debug_options(GetDebugOptionsFromFlags());
  }

  if (num_threads.has_value()) {
    config->set_intra_op_parallelism_threads(*num_threads);
  }

  if (aot_options != nullptr) {
    if (aot_options->fusion_config_collection() !=
        FusionConfigCollection::kOff) {
      config->set_fusion_config_collection(
          aot_options->fusion_config_collection());
      config->set_fusion_config(aot_options->fusion_config());
    }
  }

  return std::move(config);
}

void UpdateEntryComputationLayout(
    HloModule* module, DeviceShapeRepresentationFn shape_representation_fn,
    bool empty_tiles_only) {
  CHECK(shape_representation_fn != nullptr);
  auto update_shape = [&shape_representation_fn,
                       empty_tiles_only](Shape* shape) {
    ShapeUtil::ForEachMutableSubshape(
        shape, [&shape_representation_fn, empty_tiles_only](
                   Shape* subshape, const ShapeIndex& index) {
          if (subshape->IsArray() && subshape->has_layout()) {
            if (!empty_tiles_only ||
                (empty_tiles_only && subshape->layout().tiles().empty())) {
              *subshape = shape_representation_fn(*subshape);
            }
          }
        });
  };

  for (int i = 0; i < module->entry_computation_layout().parameter_count();
       i++) {
    Shape shape =
        module->entry_computation_layout().parameter_layout(i).shape();
    update_shape(&shape);
    *module->mutable_entry_computation_layout()->mutable_parameter_layout(i) =
        ShapeLayout(shape);
  }
  Shape shape = module->entry_computation_layout().result_layout().shape();
  update_shape(&shape);
  *module->mutable_entry_computation_layout()->mutable_result_layout() =
      ShapeLayout(shape);
}

}  // namespace zkx
