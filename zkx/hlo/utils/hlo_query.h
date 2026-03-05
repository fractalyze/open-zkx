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

#ifndef ZKX_HLO_UTILS_HLO_QUERY_H_
#define ZKX_HLO_UTILS_HLO_QUERY_H_

#include <cstdint>

#include "absl/container/flat_hash_set.h"

#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/util.h"

// Helper interface for making queries about the HLO IR.
namespace zkx::hlo_query {

// Returns whether all of an instruction's operands are of the types constants
// and parameters.
bool AllOperandsAreParametersOrConstants(const HloInstruction &instruction);

// Returns whether all of an instruction's operands are of the type constant
// or parameter and the instruction is their only user.
bool AllOperandsAreParametersOrConstantsWithSingleUser(
    const HloInstruction &instruction);

// Returns whether all of an instruction's operands are parameters.
bool AllOperandsAreParameters(const HloInstruction &instruction);

// Returns whether all of an instruction's operands are constants.
bool AllOperandsAreConstants(const HloInstruction &instruction);

// Returns whether the instruction is a scalar constant.
bool IsScalarConstant(const HloInstruction *instruction);

// Returns whether the `instr` is either a constant, a scalar, or a
// broadcasted constant/scalar.
bool IsBroadcastedConstantOrScalar(const HloInstruction &instr);

// Returns whether the `instr` is a broadcast and its input is a
// scalar constant.
bool IsBroadcastOfScalarConstant(const HloInstruction &instr);

// Returns whether the `instr` is a broadcast and its input is a parameter.
bool IsBroadcastOfParameter(const HloInstruction &instr);

// Returns true for a parameter or a parameter followed by a chain of no-op
// instructions (bitcast, get-tuple-element).
bool IsEffectiveParameter(const HloInstruction &);

// Returns first HLO of the computation with the opcode, otherwise nullptr.
HloInstruction *GetFirstInstructionWithOpcode(const HloComputation &computation,
                                              HloOpcode opcode);

// Applies `fn` to a collection of instructions with `opcode` for a given
// `computation`.
template <typename Fn>
void ForEachInstructionWithOpcode(const HloComputation &computation,
                                  HloOpcode opcode, Fn &&fn) {
  for (HloInstruction *instr : computation.instructions()) {
    if (instr->opcode() == opcode) {
      fn(instr);
    }
  }
}

// Applies `fn` to a collection of instructions with `opcode` for a given
// `module`.
template <typename Fn>
void ForEachInstructionWithOpcode(const HloModule &module, HloOpcode opcode,
                                  Fn &&fn) {
  for (HloComputation *computation : module.computations()) {
    ForEachInstructionWithOpcode(*computation, opcode, fn);
  }
}

// Applies `fn` to a collection of instructions satisfying `pred` for a given
// `computation`.
template <typename Fn>
void ForEachInstructionWithPred(const HloComputation &computation,
                                HloPredicate pred, Fn &&fn) {
  for (HloInstruction *instr : computation.instructions()) {
    if (pred(instr)) {
      fn(instr);
    }
  }
}

// Applies `fn` to a collection of instructions satisfying `pred` for a given
// `module`.
template <typename Fn>
void ForEachInstructionWithPred(const HloModule &module, HloPredicate pred,
                                Fn &&fn) {
  for (HloComputation *computation : module.computations()) {
    ForEachInstructionWithPred(*computation, pred, fn);
  }
}

// Determines whether the given computation contains an instruction with one of
// the given opcodes. Checks both comp's instructions and the instructions of
// any computations nested within it.
bool ContainsInstrWithOpcode(const HloComputation *comp,
                             const absl::flat_hash_set<HloOpcode> &opcodes);

// Returns an operand of an instruction with the given opcode. If there are
// multiple matching operands, then the first matching operand is returned. If
// there are no matching operands then nullptr is returned.
HloInstruction *GetMatchingOperand(const HloPredicate &matcher,
                                   HloInstruction *instruction);

// Returns whether a binary instruction has a matching operand. Sets
// matching_operand to the matching operand and the other operand to
// other_operand. Note: in the case where both operands match, the first operand
// of the instruction is returned.
bool MatchBinaryInstructionOperand(const HloPredicate &matcher,
                                   HloInstruction *instruction,
                                   HloInstruction **matching_operand,
                                   HloInstruction **other_operand);

// Returns whether a binary instruction has an operand with a given opcode.
// This is a special case of MatchingBinaryInstructionOperand.
bool MatchBinaryInstructionOperandOpcode(HloOpcode opcode,
                                         HloInstruction *instruction,
                                         HloInstruction **matching_operand,
                                         HloInstruction **other_operand);

// Returns the unique GTE instruction with the given operand and index. Returns
// nullptr if no such instruction exists or is not unique.
HloInstruction *GetUniqueGteInstruction(const HloInstruction *operand,
                                        int64_t index);

// Gets the computation from the given module with the given name.
HloComputation *FindComputation(HloModule *module, std::string_view name);

// Gets the instruction from the given computation with the given instruction
// name. Returns nullptr if no such instruction can be found.
HloInstruction *FindInstruction(const HloComputation *computation,
                                std::string_view name);

// Gets any instruction from the given computation with the given opcode.
// Returns nullptr if no such instruction can be found.
HloInstruction *FindInstruction(const HloComputation *computation,
                                HloOpcode opcode);

}  // namespace zkx::hlo_query

#endif  // ZKX_HLO_UTILS_HLO_QUERY_H_
