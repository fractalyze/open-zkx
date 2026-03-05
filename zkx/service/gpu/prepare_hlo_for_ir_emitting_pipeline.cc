/* Copyright 2023 The OpenXLA Authors.
Copyright 2026 The ZKX Authors.

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

#include "zkx/service/gpu/prepare_hlo_for_ir_emitting_pipeline.h"

#include "zkx/hlo/transforms/simplifiers/hlo_dce.h"
#include "zkx/service/copy_insertion.h"

namespace zkx::gpu {
namespace {}  // namespace

HloPassPipeline PrepareHloModuleForIrEmittingPipeline(
    HloModule &hlo_module, HloDataflowAnalysis::CanShareBuffer can_share_buffer,
    const se::DeviceDescription &device_description) {
  (void)hlo_module;
  (void)device_description;

  // In some cases, we have to place the result of an instruction in a temporary
  // buffer. For instance, the buffer that holds an external parameter is
  // assumed immutable at this point, and should not be reused for output
  // (b/27180329). Therefore, in that case, we set the output to be a copy of
  // the parameter.
  HloPassPipeline pipeline("GPU-ir-emit-prepare");

  pipeline.AddPass<HloDCE>();

  pipeline.AddPass<CopyInsertion>(can_share_buffer);

  return pipeline;
}

}  // namespace zkx::gpu
