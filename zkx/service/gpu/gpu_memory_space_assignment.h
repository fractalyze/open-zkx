/* Copyright 2023 The OpenXLA Authors.
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

#ifndef ZKX_SERVICE_GPU_GPU_MEMORY_SPACE_ASSIGNMENT_H_
#define ZKX_SERVICE_GPU_GPU_MEMORY_SPACE_ASSIGNMENT_H_

#include <stdint.h>

#include "absl/status/status.h"

#include "zkx/hlo/analysis/hlo_alias_analysis.h"
#include "zkx/hlo/analysis/hlo_ordering.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/service/buffer_assignment.h"
#include "zkx/service/hlo_buffer.h"
#include "zkx/service/hlo_value.h"

namespace zkx::gpu {

inline constexpr int64_t kCollectiveMemorySpaceColor = 1;
inline constexpr int64_t kTempBufferMemorySpaceColor = 2;

// Assigns default color (0) to all buffer values.
inline BufferAssigner::Colorer CollectiveColorer() {
  return [](HloAliasAnalysis *alias_analysis, const HloOrdering &) {
    for (HloValue *value : alias_analysis->dataflow_analysis().values()) {
      if (!value->has_color()) {
        value->set_color(0);
      }
    }
    return absl::OkStatus();
  };
}

}  // namespace zkx::gpu

#endif  // ZKX_SERVICE_GPU_GPU_MEMORY_SPACE_ASSIGNMENT_H_
