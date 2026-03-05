/* Copyright 2026 The ZKX Authors.

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

#include "zkx/hlo/ir/hlo_module.h"

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/literal_util.h"
#include "zkx/service/hlo_module_config.h"

namespace zkx {
namespace {

TEST(HloModuleTest, OutlineExpressionFromComputationUsesConstantsAsArguments) {
  auto builder = HloComputation::Builder("entry");
  // (1.0 + 2.0) * (2.0 - 3.0)
  HloInstruction* const1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(1)));
  HloInstruction* const2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(2)));
  HloInstruction* const3 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(3)));
  HloInstruction* add = builder.AddInstruction(HloInstruction::CreateBinary(
      const1->shape(), HloOpcode::kAdd, const1, const2));
  HloInstruction* sub = builder.AddInstruction(HloInstruction::CreateBinary(
      const2->shape(), HloOpcode::kSubtract, const2, const3));
  HloInstruction* mul = builder.AddInstruction(HloInstruction::CreateBinary(
      add->shape(), HloOpcode::kMultiply, add, sub));

  auto module = std::make_unique<HloModule>("outline_expression_from_entry",
                                            HloModuleConfig());
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build(mul));
  TF_ASSERT_OK_AND_ASSIGN(HloInstruction * call,
                          module->OutlineExpressionFromComputation(
                              {add, sub, mul}, "entry", entry_computation));
  EXPECT_EQ(call, entry_computation->root_instruction());
  HloComputation* callee_computation = call->to_apply();
  // The arguments to the call should be const1, const2, and const3.
  EXPECT_THAT(call->operands(),
              ::testing::UnorderedElementsAre(const1, const2, const3));

  // The callee should contain 3 parameters and 3 binary operators.
  EXPECT_EQ(6, callee_computation->instruction_count());
}

}  // namespace
}  // namespace zkx
