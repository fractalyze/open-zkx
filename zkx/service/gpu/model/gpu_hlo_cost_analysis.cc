/* Copyright 2022 The OpenXLA Authors.
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

#include "zkx/service/gpu/model/gpu_hlo_cost_analysis.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"

#include "xla/tsl/platform/errors.h"
#include "zkx/hlo/ir/hlo_casting_utils.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_instructions.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/service/hlo_cost_analysis.h"
#include "zkx/shape.h"
#include "zkx/shape_util.h"

namespace zkx {
namespace gpu {

namespace {
// Use the "reserved" keys for these properties so lookups are fast.
static constexpr std::string_view kIRSizeKey = HloCostAnalysis::kReserved0Key;

}  // namespace

absl::Status GpuHloCostAnalysis::Preprocess(const HloInstruction *hlo) {
  TF_RETURN_IF_ERROR(HloCostAnalysis::Preprocess(hlo));

  current_properties_[kIRSizeKey] = 1;
  return absl::OkStatus();
}

int64_t GpuHloCostAnalysis::FusionParameterReadBytes(
    const HloInstruction *hlo) const {
  CHECK(hlo->IsFused() && (hlo->opcode() == HloOpcode::kParameter ||
                           hlo->opcode() == HloOpcode::kGetTupleElement));
  float utilization = hlo_properties_.at(hlo)[kUtilizationKey];
  if (!options_.count_multiple_input_accesses) {
    utilization = fmin(utilization, 1.0);
  }
  return std::llround(GetShapeSize(hlo->shape()) * utilization);
}

absl::Status GpuHloCostAnalysis::FusionCalculateUtilizations(
    const HloInstruction *fusion) {
  const HloInstruction *root = fusion->fused_expression_root();
  // Traverse through the computation from the root till parameters propagating
  // the utilization of operands; store utilization of each node in
  // hlo_properties_. All consumers of an instruction are processed before the
  // instruction itself.
  std::vector<HloInstruction *> instructions =
      fusion->fused_instructions_computation()->MakeInstructionPostOrder();
  absl::c_reverse(instructions);

  // Whenever we account a non-element-wise operation we forget about
  // element-wise roots encountered so far and provisionally set its operands
  // as new element-wise roots.
  absl::flat_hash_map<const HloInstruction *, int64_t> root_ir_sizes;

  for (const HloInstruction *instr : instructions) {
    hlo_properties_[instr][kUtilizationKey] = 0;
    hlo_properties_[instr][kIRSizeKey] = 0;
    elementwise_use_roots_[instr].clear();
    root_utilizations_[instr] = 0;
  }

  // For the purpose of operand utilization analysis, no matter how the fusion
  // outputs are used, we assume that fusion is always executed completely
  // producing 100% of its outputs.
  root_utilizations_[root] = 1.0;
  root_ir_sizes[root] = 1;
  elementwise_use_roots_[root].insert(root);

  current_properties_[kFlopsKey] = 0;
  current_properties_[kIRSizeKey] = 0;

  for (const HloInstruction *instr : instructions) {
    VLOG(8) << instr->name() << ":";
    VLOG(9) << "Elementwise use roots:";
    Properties &instr_props = hlo_properties_[instr];
    for (const HloInstruction *r : elementwise_use_roots_[instr]) {
      VLOG(9) << "\t" << r->name() << ": " << root_utilizations_[r];
      instr_props[kUtilizationKey] += root_utilizations_[r];
      instr_props[kIRSizeKey] += root_ir_sizes[r];
    }

    float cur_instr_utilization = instr_props[kUtilizationKey];
    VLOG(8) << "Total utilization: " << cur_instr_utilization;
    float cur_instr_times_emitted = instr_props[kIRSizeKey];
    VLOG(8) << "Times emitted: " << cur_instr_times_emitted;

    current_properties_[kFlopsKey] +=
        cur_instr_utilization * instr_props[kFlopsKey];
    current_properties_[kIRSizeKey] += cur_instr_times_emitted;

    for (int operand_idx = 0; operand_idx < instr->operand_count();
         ++operand_idx) {
      const HloInstruction *operand = instr->operand(operand_idx);
      if ((instr->IsElementwise()) || instr->opcode() == HloOpcode::kTuple ||
          instr->opcode() == HloOpcode::kGetTupleElement) {
        for (const HloInstruction *r : elementwise_use_roots_[instr]) {
          elementwise_use_roots_[operand].insert(r);
        }
      } else {
        elementwise_use_roots_[operand].insert(operand);
        float cur_operand_utilization =
            cur_instr_utilization * operand_utilization(*instr, operand_idx);
        // The utilization is always a best-effort estimate, but in some cases
        // cannot be precise due to dynamic nature of operations - dynamic
        // slice is one such example. We do an average estimate in these
        // cases and this can sometimes produce fractional utilizations which
        // should be at least rounded up to a whole number of produced elements
        // to be more realistic.
        int64_t operand_elements =
            ShapeUtil::ElementsInRecursive(operand->shape());

        if (operand_elements == 0) {
          // Element count should not be 0 in any production use case, but there
          // are valid HLO inputs that occur in tests.
          cur_operand_utilization = 0;
        } else {
          cur_operand_utilization =
              ceil(cur_operand_utilization * operand_elements) /
              operand_elements;
        }
        root_utilizations_[operand] += cur_operand_utilization;
        root_ir_sizes[operand] += cur_instr_times_emitted;
      }
    }
  }

  return absl::OkStatus();
}

float GpuHloCostAnalysis::CommonElementwiseUtilization(
    const HloInstruction *a, const HloInstruction *b) const {
  float ret = 0;
  for (auto r : elementwise_use_roots_.at(a)) {
    if (elementwise_use_roots_.at(b).count(r)) {
      ret += root_utilizations_.at(r);
    }
  }
  return ret;
}

bool GpuHloCostAnalysis::ProducerConsumerMergedTooLarge(
    const HloInstruction &producer, const HloInstruction &consumer) {
  int64_t producer_replication = 1;
  // Fusing 'producer' into 'consumer' fusion currently results in replicating
  // its IR the number of times the consumer replicates the access
  // to the parameter corresponding to the producer.
  if (consumer.opcode() == HloOpcode::kFusion) {
    producer_replication =
        IrSize(*consumer.fused_parameter(consumer.operand_index(&producer)));
  }
  VLOG(5) << producer.name() << " would be emitted by " << consumer.name()
          << " x" << producer_replication;
  int64_t merged_ir_size =
      (IrSize(producer) * producer_replication + IrSize(consumer));
  VLOG(5) << "IR sizes: " << IrSize(producer) << ", " << IrSize(consumer)
          << " -> " << merged_ir_size;
  return merged_ir_size > kMaxIRSize;
}

int64_t GpuHloCostAnalysis::GetFlopsPerElementwiseOpElement(
    const PrimitiveType type, const HloOpcode opcode) {
  // Elementwise instructions typically take at least a few clock cycles.
  constexpr int64_t kDefaultFlopsPerElement = 3;
  return kDefaultFlopsPerElement;
}

int64_t GpuHloCostAnalysis::GetFlopsForElementwiseOp(const HloOpcode op_code,
                                                     const Shape &shape) {
  int64_t flop_per_element =
      GetFlopsPerElementwiseOpElement(shape.element_type(), op_code);
  return flop_per_element * ShapeUtil::ElementsInRecursive(shape);
}

int64_t GpuHloCostAnalysis::GetFlopsForElementwiseOp(
    const HloInstruction *instr) {
  return GetFlopsForElementwiseOp(instr->opcode(), instr->shape());
}

absl::Status GpuHloCostAnalysis::HandleConcatenate(const HloInstruction *hlo) {
  // Concat turns into a compare plus branch instruction.
  int64_t flop_per_element = 6;
  // If a warp crosses the operands boundary, both branches are executed. This
  // depends on the tiling of the final fusion and is therefore hard to predict
  // at this level. Executing both branches drives up the flops, but not the
  // bandwidth. So it might seem like a good idea to fuse a concat into a
  // memory-bound consumer. However, the divergent warps increase the cost of
  // compute-heavy producers that might be fused later. We see this issue in
  // some important LLM models that fuse a concat into a column reduction (see
  // PriorityFusionTest.DontFuseConcat test). To prevent this particular fusion,
  // we add large number of flops to the concat. Both the condition and the flop
  // count are tuned to this particular case.
  // TODO(b/315776282): Model this more accurately once we can reason about
  // tiling patterns.
  int64_t dim = Cast<HloConcatenateInstruction>(hlo)->concatenate_dimension();
  if (dim > 0 && hlo->operand(0)->shape().dimensions()[dim] & 31) {
    flop_per_element = 400;
  }
  current_properties_[kFlopsKey] =
      flop_per_element * ShapeUtil::ElementsInRecursive(hlo->shape());
  return absl::OkStatus();
}

absl::Status GpuHloCostAnalysis::HandleReduce(const HloInstruction *hlo) {
  // HloCostAnalysis::HandleReduce computes FLOPs for the computation correctly,
  // but `bytes_accessed` estimates are different for GPU.
  TF_RETURN_IF_ERROR(HloCostAnalysis::HandleReduce(hlo));

  const HloReduceInstruction *reduce = DynCast<HloReduceInstruction>(hlo);
  auto output_shape = reduce->shape().IsArray()
                          ? reduce->shape()
                          : reduce->shape().tuple_shapes(0);

  int64_t output_bytes_accessed = 0;
  ShapeUtil::ForEachLeafShape(
      reduce->shape(), [&](const Shape &sub_shape, const ShapeIndex &index) {
        output_bytes_accessed += GetShapeSize(sub_shape);
      });

  current_properties_.set_output_bytes_accessed(output_bytes_accessed);

  int64_t bytes_accessed = output_bytes_accessed;
  for (int64_t input_operand_id = 0; input_operand_id < reduce->input_count();
       ++input_operand_id) {
    bytes_accessed +=
        current_properties_.operand_bytes_accessed(input_operand_id);
  }

  int64_t output_shape_size = ShapeUtil::ElementsIn(output_shape);
  for (int64_t init_operand_id = reduce->input_count();
       init_operand_id < reduce->operand_count(); ++init_operand_id) {
    auto init_operand = reduce->operand(init_operand_id);

    int64_t operand_bytes_accessed =
        output_shape_size * GetShapeSize(init_operand->shape());
    current_properties_.set_operand_bytes_accessed(init_operand_id,
                                                   operand_bytes_accessed);
    current_properties_.set_operand_utilization(init_operand_id,
                                                output_shape_size);

    bytes_accessed += operand_bytes_accessed;
  }

  current_properties_[kBytesAccessedKey] = bytes_accessed;

  return absl::OkStatus();
}

absl::Status GpuHloCostAnalysis::HandleElementwiseOp(
    const HloInstruction *hlo) {
  current_properties_[kFlopsKey] = GetFlopsForElementwiseOp(hlo);
  return absl::OkStatus();
}

std::unique_ptr<HloCostAnalysis>
GpuHloCostAnalysis::CreateNestedCostAnalysis() {
  return std::make_unique<GpuHloCostAnalysis>(options_);
}

bool GpuHloCostAnalysis::KeyToCopyFromSubcomputation(
    std::string_view key) const {
  return !absl::StartsWith(key, kBytesAccessedKey) &&
         !absl::StartsWith(key, kUtilizationKey) &&
         !absl::StartsWith(key, kIRSizeKey);
}

float GpuHloCostAnalysis::IrSize(const HloInstruction &hlo) const {
  return GetPropertyForHlo(hlo, kIRSizeKey, hlo_properties_);
}

}  // namespace gpu
}  // namespace zkx
