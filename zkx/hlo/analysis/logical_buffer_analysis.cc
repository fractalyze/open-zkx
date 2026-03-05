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

#include "zkx/hlo/analysis/logical_buffer_analysis.h"

#include "xla/tsl/platform/errors.h"
#include "zkx/hlo/ir/hlo_casting_utils.h"
#include "zkx/hlo/ir/hlo_instructions.h"

namespace zkx {

// Gather fusion instructions from `instruction` into `fusion_instructions`.
void GatherFusionInstructions(
    HloInstruction *instruction,
    std::vector<HloInstruction *> *fusion_instructions) {
  CHECK_EQ(HloOpcode::kFusion, instruction->opcode());
  for (auto *fused : instruction->fused_instructions()) {
    if (fused->opcode() == HloOpcode::kFusion) {
      GatherFusionInstructions(fused, fusion_instructions);
    }
  }
  fusion_instructions->push_back(instruction);
}

// static
absl::StatusOr<std::unique_ptr<LogicalBufferAnalysis>>
LogicalBufferAnalysis::Run(const HloModule *module) {
  std::unique_ptr<LogicalBufferAnalysis> analysis(
      new LogicalBufferAnalysis(module));
  TF_RETURN_IF_ERROR(analysis->Analyze());
  return std::move(analysis);
}

absl::Status LogicalBufferAnalysis::Analyze() {
  // Empirically we usually have a few more logical buffers than instructions,
  // so reserve 10% more than the number of instructions to avoid frequent
  // resizes.
  logical_buffers_.clear();
  logical_buffers_.reserve((module_->instruction_count() * 11) / 10);

  // We filter out fusion computations, and get to them through fusion
  // instructions. This is because it's possible to have orphaned (unreachable)
  // fusion computations, and we don't want to try to assign buffers to those.
  std::vector<HloInstruction *> fusion_instructions;
  for (auto *computation : module_->MakeNonfusionComputations()) {
    TF_RETURN_IF_ERROR(computation->Accept(this));
    for (auto *instruction : computation->instructions()) {
      if (instruction->opcode() != HloOpcode::kFusion) {
        continue;
      }
      GatherFusionInstructions(instruction, &fusion_instructions);
    }
  }
  for (auto *instruction : fusion_instructions) {
    TF_RETURN_IF_ERROR(instruction->fused_expression_root()->Accept(this));
  }
  return absl::OkStatus();
}

LogicalBuffer &LogicalBufferAnalysis::GetBuffer(LogicalBuffer::Id id) const {
  return *logical_buffers_[id];
}

LogicalBuffer &LogicalBufferAnalysis::GetBuffer(HloInstruction *instruction,
                                                const ShapeIndex &index) const {
  return *output_buffers_.at(std::make_pair(instruction, index));
}

void LogicalBufferAnalysis::NewLogicalBuffer(HloInstruction *instruction,
                                             const ShapeIndex &index) {
  LogicalBuffer::Id id = logical_buffers_.size();
  auto buffer = std::make_unique<LogicalBuffer>(instruction, index, id);
  auto position = std::make_pair(instruction, index);
  CHECK(output_buffers_.insert({position, buffer.get()}).second);
  logical_buffers_.push_back(std::move(buffer));
}

absl::Status LogicalBufferAnalysis::DefaultAction(
    HloInstruction *hlo_instruction) {
  // Create a logical buffer for each output of the instruction.
  ShapeUtil::ForEachSubshape(
      hlo_instruction->shape(),
      [this, hlo_instruction](const Shape &shape, const ShapeIndex &index) {
        NewLogicalBuffer(hlo_instruction, index);
      });

  return absl::OkStatus();
}

absl::Status LogicalBufferAnalysis::HandleGetTupleElement(HloInstruction *) {
  // GetTupleElement does not create buffers.
  return absl::OkStatus();
}

absl::Status LogicalBufferAnalysis::HandleCopy(HloInstruction *copy) {
  // The top-level buffer (index={}) for kCopy is newly created, but all other
  // buffers (in the case of a tuple shape) come from the operand
  NewLogicalBuffer(copy, /*index=*/{});
  return absl::OkStatus();
}

absl::Status LogicalBufferAnalysis::HandleBitcast(HloInstruction *) {
  // A kBitcast instruction aliases its operand. That is, the buffer of its
  // result *is* the buffer of its operand.
  return absl::OkStatus();
}

absl::Status LogicalBufferAnalysis::HandleTuple(HloInstruction *tuple) {
  // A Tuple instruction only creates the top-level buffer.
  NewLogicalBuffer(tuple, /*index=*/{});
  return absl::OkStatus();
}

// WARNING (b/259460539): output_to_operand_aliasing was moved from
// HloCustomCallInstruction to HloCallableInstruction so that fusions can
// also be annotated with this aliasing. This feature might not be complete.
absl::Status LogicalBufferAnalysis::HandleFusion(HloInstruction *fusion) {
  auto cfusion = Cast<HloFusionInstruction>(fusion);
  absl::flat_hash_set<ShapeIndex> aliased_outputs;
  for (const auto &pair : cfusion->output_to_operand_aliasing()) {
    aliased_outputs.insert(pair.first);
  }
  ShapeUtil::ForEachSubshape(cfusion->shape(),
                             [&](const Shape &shape, const ShapeIndex &index) {
                               if (!aliased_outputs.contains(index)) {
                                 NewLogicalBuffer(fusion, index);
                               }
                             });
  return absl::OkStatus();
}

}  // namespace zkx
