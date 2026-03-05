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

#include "zkx/hlo/utils/hlo_query.h"

#include <cstdint>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"

#include "zkx/service/pattern_matcher.h"
#include "zkx/shape_util.h"

namespace zkx::hlo_query {

bool AllOperandsAreParametersOrConstants(const HloInstruction &instruction) {
  for (const auto &operand : instruction.operands()) {
    if (operand->opcode() != HloOpcode::kParameter &&
        operand->opcode() != HloOpcode::kConstant) {
      return false;
    }
  }
  return true;
}

bool AllOperandsAreParametersOrConstantsWithSingleUser(
    const HloInstruction &instruction) {
  for (const auto &operand : instruction.operands()) {
    if (operand->opcode() != HloOpcode::kParameter &&
        operand->opcode() != HloOpcode::kConstant) {
      return false;
    }
    if (operand->user_count() > 1) {
      return false;
    }
  }
  return true;
}

bool AllOperandsAreParameters(const HloInstruction &instruction) {
  for (const auto &operand : instruction.operands()) {
    if (operand->opcode() != HloOpcode::kParameter) {
      return false;
    }
  }
  return true;
}

bool AllOperandsAreConstants(const HloInstruction &instruction) {
  for (const auto &operand : instruction.operands()) {
    if (operand->opcode() != HloOpcode::kConstant) {
      return false;
    }
  }
  return true;
}

HloInstruction *GetMatchingOperand(const HloPredicate &matcher,
                                   HloInstruction *instruction) {
  for (HloInstruction *op : instruction->operands()) {
    if (matcher(op)) {
      return op;
    }
  }
  return nullptr;
}

bool MatchBinaryInstructionOperand(const HloPredicate &matcher,
                                   HloInstruction *instruction,
                                   HloInstruction **matching_operand,
                                   HloInstruction **other_operand) {
  CHECK_EQ(instruction->operand_count(), 2);
  if (matcher(instruction->operand(0))) {
    *matching_operand = instruction->mutable_operand(0);
    *other_operand = instruction->mutable_operand(1);
    return true;
  }
  if (matcher(instruction->operand(1))) {
    *matching_operand = instruction->mutable_operand(1);
    *other_operand = instruction->mutable_operand(0);
    return true;
  }
  return false;
}

bool MatchBinaryInstructionOperandOpcode(HloOpcode opcode,
                                         HloInstruction *instruction,
                                         HloInstruction **matching_operand,
                                         HloInstruction **other_operand) {
  return MatchBinaryInstructionOperand(
      [opcode](const HloInstruction *instruction) {
        return instruction->opcode() == opcode;
      },
      instruction, matching_operand, other_operand);
}

bool IsScalarConstant(const HloInstruction *instruction) {
  return instruction->IsConstant() && ShapeUtil::IsScalar(instruction->shape());
}

bool IsBroadcastedConstantOrScalar(const HloInstruction &instr) {
  return instr.IsConstant() || ShapeUtil::IsScalar(instr.shape()) ||
         (HloOpcode::kBroadcast == instr.opcode() &&
          (instr.operand(0)->IsConstant() ||
           ShapeUtil::IsScalar(instr.operand(0)->shape())));
}

bool IsBroadcastOfScalarConstant(const HloInstruction &instr) {
  return instr.opcode() == HloOpcode::kBroadcast &&
         IsScalarConstant(instr.operand(0));
}

bool IsBroadcastOfParameter(const HloInstruction &instr) {
  return instr.opcode() == HloOpcode::kBroadcast &&
         instr.operand(0)->opcode() == HloOpcode::kParameter;
}

bool IsEffectiveParameter(const HloInstruction &instr) {
  return instr.opcode() == HloOpcode::kParameter ||
         ((instr.opcode() == HloOpcode::kBitcast ||
           instr.opcode() == HloOpcode::kGetTupleElement) &&
          IsEffectiveParameter(*instr.operand(0)));
}

HloInstruction *GetFirstInstructionWithOpcode(const HloComputation &computation,
                                              const HloOpcode opcode) {
  auto instructions = computation.instructions();
  auto it = absl::c_find_if(instructions, [&](HloInstruction *instr) {
    return instr->opcode() == opcode;
  });
  return it == instructions.end() ? nullptr : *it;
}

bool ContainsInstrWithOpcode(const HloComputation *comp,
                             const absl::flat_hash_set<HloOpcode> &opcodes) {
  for (const auto *instr : comp->instructions()) {
    if (opcodes.count(instr->opcode())) {
      return true;
    }
    for (const HloComputation *subcomp : instr->called_computations()) {
      if (ContainsInstrWithOpcode(subcomp, opcodes)) {
        return true;
      }
    }
  }
  return false;
}

HloInstruction *GetUniqueGteInstruction(const HloInstruction *operand,
                                        int64_t index) {
  HloInstruction *gte = nullptr;
  for (HloInstruction *instr : operand->parent()->MakeInstructionPostOrder()) {
    if (!Match(instr, match::GetTupleElement().WithTupleIndex(index))) {
      continue;
    }
    if (instr->operand(0) != operand) {
      continue;
    }
    // If gte is not unique, return nullptr.
    if (gte != nullptr) {
      return nullptr;
    }
    gte = instr;
  }
  return gte;
}

HloComputation *FindComputation(HloModule *module, std::string_view name) {
  auto computations = module->computations();
  auto it = absl::c_find_if(
      computations, [&](HloComputation *c) { return c->name() == name; });
  if (it == computations.end()) {
    return nullptr;
  }
  return *it;
}

HloInstruction *FindInstruction(const HloComputation *computation,
                                std::string_view name) {
  for (HloInstruction *instruction : computation->instructions()) {
    if (instruction->name() == name) return instruction;
  }
  return nullptr;
}

HloInstruction *FindInstruction(const HloComputation *computation,
                                HloOpcode opcode) {
  for (auto *instruction : computation->instructions()) {
    if (instruction->opcode() == opcode) return instruction;
  }
  return nullptr;
}

}  // namespace zkx::hlo_query
