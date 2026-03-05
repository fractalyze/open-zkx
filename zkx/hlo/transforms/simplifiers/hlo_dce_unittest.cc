/* Copyright 2017 The OpenXLA Authors.
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

#include "zkx/hlo/transforms/simplifiers/hlo_dce.h"

#include <memory>

#include "absl/algorithm/container.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "xla/tsl/platform/status.h"
#include "zkx/hlo/ir/hlo_casting_utils.h"
#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_instructions.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "zkx/hlo/testlib/pattern_matcher_gmock.h"
#include "zkx/layout_util.h"
#include "zkx/literal_util.h"
#include "zkx/service/pattern_matcher.h"
#include "zkx/shape_util.h"
#include "zkx/tests/literal_test_util.h"
#include "zkx/zkx_data.pb.h"

namespace zkx {
namespace {

namespace m = ::zkx::match;

class HloDceTest : public HloHardwareIndependentTestBase {
 protected:
  HloDceTest() {}

  // Returns whether the given instruction exists in the given computation.
  bool HasInstruction(const HloComputation &computation,
                      const HloInstruction *instruction) {
    return absl::c_linear_search(computation.instructions(), instruction);
  }
};

TEST_F(HloDceTest, NoDeadCode) {
  // Verify that no dead code is removed from a computation with no dead code.
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(42)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(123)));
  builder.AddInstruction(HloInstruction::CreateBinary(
      constant1->shape(), HloOpcode::kAdd, constant1, constant2));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(3, computation->instruction_count());

  HloDCE dce;
  EXPECT_FALSE(dce.Run(module.get()).value());

  EXPECT_EQ(3, computation->instruction_count());
}

TEST_F(HloDceTest, InstructionsWithSideEffect) {
  // Verify that side-effect instructions (Send in this test) are not removed.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(42)));
  auto token = builder.AddInstruction(HloInstruction::CreateToken());
  auto send = builder.AddInstruction(HloInstruction::CreateSend(
      constant, token, /*channel_id=*/0, /*is_host_transfer=*/false));
  builder.AddInstruction(HloInstruction::CreateSendDone(
      send, send->channel_id(), /*is_host_transfer=*/false));
  builder.AddInstruction(HloInstruction::CreateTuple({}));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(5, computation->instruction_count());

  HloDCE dce;
  EXPECT_FALSE(dce.Run(module.get()).value());

  EXPECT_EQ(5, computation->instruction_count());
}

TEST_F(HloDceTest, DeadParameters) {
  // Verify that dead parameters are not removed, but use of the dead parameters
  // is.
  auto builder = HloComputation::Builder(TestName());
  auto live_param = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(S32, {}), "live_param"));
  auto dead_param1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(S32, {}), "dead_param1"));
  builder.AddInstruction(HloInstruction::CreateParameter(
      2, ShapeUtil::MakeShape(S32, {}), "dead_param2"));

  // This is a dead negate instruction.
  builder.AddInstruction(HloInstruction::CreateUnary(
      dead_param1->shape(), HloOpcode::kNegate, dead_param1));

  // This negate is not dead because it is the root.
  builder.AddInstruction(HloInstruction::CreateUnary(
      live_param->shape(), HloOpcode::kNegate, live_param));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(5, computation->instruction_count());
  EXPECT_EQ(1, dead_param1->user_count());

  HloDCE dce;
  EXPECT_TRUE(dce.Run(module.get()).value());

  EXPECT_EQ(4, computation->instruction_count());
  EXPECT_EQ(0, dead_param1->user_count());
}

TEST_F(HloDceTest, ControlDependencies) {
  // Verify that instructions with control dependencies are not removed.
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(42)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(123)));

  // Create two dead instructions: a negate and an add.
  auto dead_negate = builder.AddInstruction(HloInstruction::CreateUnary(
      constant1->shape(), HloOpcode::kNegate, constant1));
  auto dead_add = builder.AddInstruction(HloInstruction::CreateBinary(
      constant1->shape(), HloOpcode::kAdd, constant1, constant2));

  // Create the same two instructions again, but these will have a control
  // dependency added.
  auto dead_negate_with_control_dep =
      builder.AddInstruction(HloInstruction::CreateUnary(
          constant1->shape(), HloOpcode::kNegate, constant1));
  auto dead_add_with_control_dep =
      builder.AddInstruction(HloInstruction::CreateBinary(
          constant1->shape(), HloOpcode::kAdd, constant1, constant2));

  // Create a root so the previously added instruction is dead.
  builder.AddInstruction(HloInstruction::CreateBinary(
      constant1->shape(), HloOpcode::kAdd, constant1, constant2));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  // Add a control dependency between two instructions.
  TF_ASSERT_OK(dead_negate_with_control_dep->AddControlDependencyTo(
      dead_add_with_control_dep));

  EXPECT_EQ(7, computation->instruction_count());
  EXPECT_TRUE(HasInstruction(*computation, dead_negate));
  EXPECT_TRUE(HasInstruction(*computation, dead_add));
  EXPECT_TRUE(HasInstruction(*computation, dead_negate_with_control_dep));
  EXPECT_TRUE(HasInstruction(*computation, dead_add_with_control_dep));

  HloDCE dce;
  EXPECT_TRUE(dce.Run(module.get()).value());

  EXPECT_EQ(5, computation->instruction_count());
  EXPECT_FALSE(HasInstruction(*computation, dead_negate));
  EXPECT_FALSE(HasInstruction(*computation, dead_add));
  EXPECT_TRUE(HasInstruction(*computation, dead_negate_with_control_dep));
  EXPECT_TRUE(HasInstruction(*computation, dead_add_with_control_dep));
}

// Tests that a dead call instruction is removed.
TEST_F(HloDceTest, DeadInstructionWithCalledComputation) {
  auto module = CreateNewVerifiedModule();
  Shape shape = ShapeUtil::MakeShape(S32, {});

  // Called computation for the call instruction.
  auto callee_builder = HloComputation::Builder(TestName() + "-callee");
  {
    auto param = callee_builder.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "param"));
    callee_builder.AddInstruction(
        HloInstruction::CreateUnary(shape, HloOpcode::kNegate, param));
  }
  auto called_computation =
      module->AddEmbeddedComputation(callee_builder.Build());

  // Entry computation with a call instruction.
  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));
  auto dead_call = builder.AddInstruction(
      HloInstruction::CreateCall(shape, {param}, called_computation));
  builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, param));
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(3, computation->instruction_count());
  EXPECT_EQ(2, param->user_count());
  EXPECT_EQ(0, dead_call->user_count());
  EXPECT_TRUE(HasInstruction(*computation, dead_call));

  HloDCE dce;
  EXPECT_TRUE(dce.Run(module.get()).value());

  EXPECT_EQ(2, computation->instruction_count());
  EXPECT_EQ(1, param->user_count());
  EXPECT_FALSE(HasInstruction(*computation, dead_call));
}

// Tests that a while instruction with an infeed (effectful instruction) in its
// body is not removed, even its user count is 0.
TEST_F(HloDceTest, CalledComputationWithSideEffect) {
  auto module = CreateNewVerifiedModule();
  Shape shape = ShapeUtil::MakeShape(S32, {});

  // Condition computation of a while instruction.
  auto cond_builder = HloComputation::Builder(TestName() + "-cond");
  {
    auto param = cond_builder.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "cond_param"));
    auto constant = cond_builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(42)));
    cond_builder.AddInstruction(
        HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), param,
                                      constant, ComparisonDirection::kLt));
  }
  auto cond_computation = module->AddEmbeddedComputation(cond_builder.Build());

  // Body computation of a while instruction.
  auto body_builder = HloComputation::Builder(TestName() + "-body");
  {
    auto param = body_builder.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "param"));
    auto token = body_builder.AddInstruction(HloInstruction::CreateToken());
    auto infeed = body_builder.AddInstruction(
        HloInstruction::CreateInfeed(shape, token, ""));
    auto infeed_data = body_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(shape, infeed, 0));
    body_builder.AddInstruction(HloInstruction::CreateBinary(
        shape, HloOpcode::kAdd, param, infeed_data));
  }
  auto body_computation = module->AddEmbeddedComputation(body_builder.Build());

  // Entry computation with a while instruction and a negate on the parameter.
  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));
  auto live_while = builder.AddInstruction(HloInstruction::CreateWhile(
      shape, cond_computation, body_computation, param));
  builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, param));
  auto computation = module->AddEntryComputation(builder.Build());

  // Check the while instruction is not removed even if its user count is 0.
  EXPECT_EQ(3, computation->instruction_count());
  EXPECT_EQ(2, param->user_count());
  EXPECT_EQ(0, live_while->user_count());
  EXPECT_TRUE(HasInstruction(*computation, live_while));

  HloDCE dce;
  EXPECT_FALSE(dce.Run(module.get()).value());

  EXPECT_EQ(3, computation->instruction_count());
  EXPECT_EQ(2, param->user_count());
  EXPECT_EQ(0, live_while->user_count());
  EXPECT_TRUE(HasInstruction(*computation, live_while));
}

// Tests that a nested call instruction with a side effect is not removed.
TEST_F(HloDceTest, CalledComputationWithNestedSideEffect) {
  auto module = CreateNewVerifiedModule();
  Shape shape = ShapeUtil::MakeShape(S32, {});

  // Nested called computation with a side effect.
  auto nested_callee_builder =
      HloComputation::Builder(TestName() + "-nested_callee");
  {
    auto param = nested_callee_builder.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "param"));
    auto token =
        nested_callee_builder.AddInstruction(HloInstruction::CreateToken());
    nested_callee_builder.AddInstruction(
        HloInstruction::CreateOutfeed(shape, param, token, ""));
  }
  auto nested_called_computation =
      module->AddEmbeddedComputation(nested_callee_builder.Build());

  // Outer called computation that calls the nested computation.
  auto callee_builder = HloComputation::Builder(TestName() + "-callee");
  {
    auto param = callee_builder.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "param"));
    callee_builder.AddInstruction(HloInstruction::CreateCall(
        ShapeUtil::MakeTokenShape(), {param}, nested_called_computation));
  }
  auto called_computation =
      module->AddEmbeddedComputation(callee_builder.Build());

  // Entry computation with a call instruction.
  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));
  auto live_call = builder.AddInstruction(HloInstruction::CreateCall(
      ShapeUtil::MakeTokenShape(), {param}, called_computation));
  builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, param));
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(3, computation->instruction_count());
  EXPECT_EQ(2, param->user_count());
  EXPECT_EQ(0, live_call->user_count());
  EXPECT_TRUE(HasInstruction(*computation, live_call));

  HloDCE dce;
  EXPECT_FALSE(dce.Run(module.get()).value());

  EXPECT_EQ(3, computation->instruction_count());
  EXPECT_EQ(2, param->user_count());
  EXPECT_EQ(0, live_call->user_count());
  EXPECT_TRUE(HasInstruction(*computation, live_call));
}

TEST_F(HloDceTest, RemoveDeadSubcomputation) {
  auto module = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  HloComputation::Builder subcomp_builder("reduction_subcomp");
  {
    auto *param0 =
        subcomp_builder.AddInstruction(HloInstruction::CreateParameter(
            /*parameter_number=*/0, ShapeUtil::MakeShape(S32, {}), "param0"));
    auto *param1 =
        subcomp_builder.AddInstruction(HloInstruction::CreateParameter(
            /*parameter_number=*/1, ShapeUtil::MakeShape(S32, {}), "param1"));
    subcomp_builder.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(S32, {}), HloOpcode::kAdd, param0, param1));
  }
  auto reduce_subcomp = module->AddEmbeddedComputation(subcomp_builder.Build());

  // Create a dead reduce instruction.
  builder.AddInstruction(HloInstruction::CreateReduce(
      ShapeUtil::MakeShape(S32, {1}),
      builder.AddInstruction(HloInstruction::CreateParameter(
          /*parameter_number=*/0, ShapeUtil::MakeShape(S32, {100}), "param0")),
      builder.AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(0))),
      /*dimensions_to_reduce=*/{0}, reduce_subcomp));

  // Add another instruction as the root of the computation.
  builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(0)));

  module->AddEntryComputation(builder.Build());
  EXPECT_EQ(module->MakeComputationPostOrder().size(), 2);

  HloDCE dce;
  EXPECT_TRUE(dce.Run(module.get()).value());

  // We should have DCE'ed the reduction computation along with the reduction
  // instruction.
  EXPECT_EQ(module->MakeComputationPostOrder().size(), 1);
}

TEST_F(HloDceTest, KeepUsedSubcomputation) {
  auto module = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  HloComputation::Builder subcomp_builder("reduction_subcomp");
  {
    auto *param0 =
        subcomp_builder.AddInstruction(HloInstruction::CreateParameter(
            /*parameter_number=*/0, ShapeUtil::MakeShape(S32, {}), "param0"));
    auto *param1 =
        subcomp_builder.AddInstruction(HloInstruction::CreateParameter(
            /*parameter_number=*/1, ShapeUtil::MakeShape(S32, {}), "param1"));
    subcomp_builder.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(S32, {}), HloOpcode::kAdd, param0, param1));
  }
  auto reduce_subcomp = module->AddEmbeddedComputation(subcomp_builder.Build());

  // Create a dead reduce instruction.
  builder.AddInstruction(HloInstruction::CreateReduce(
      ShapeUtil::MakeShape(S32, {}),
      builder.AddInstruction(HloInstruction::CreateParameter(
          /*parameter_number=*/0, ShapeUtil::MakeShape(S32, {100}), "param0")),
      builder.AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(0))),
      /*dimensions_to_reduce=*/{0}, reduce_subcomp));

  // Add another instruction as the root of the computation that also uses
  // reduce_subcomp.
  builder.AddInstruction(HloInstruction::CreateReduce(
      ShapeUtil::MakeShape(S32, {}),
      builder.AddInstruction(HloInstruction::CreateParameter(
          /*parameter_number=*/1, ShapeUtil::MakeShape(S32, {100}), "param1")),
      builder.AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(0))),
      /*dimensions_to_reduce=*/{0}, reduce_subcomp));

  module->AddEntryComputation(builder.Build());
  EXPECT_EQ(module->MakeComputationPostOrder().size(), 2);

  HloDCE dce;
  EXPECT_TRUE(dce.Run(module.get()).value());

  // We shouldn't have DCE'ed reduce_subcomp, even though we removed one of
  // its users.
  EXPECT_EQ(module->MakeComputationPostOrder().size(), 2);
}

TEST_F(HloDceTest, RemovedNestedDeadComputations) {
  auto module = CreateNewVerifiedModule();
  Shape shape = ShapeUtil::MakeShape(S32, {});

  HloComputation::Builder called_subcomp_builder("called_dead_add");
  {
    auto *param0 =
        called_subcomp_builder.AddInstruction(HloInstruction::CreateParameter(
            /*parameter_number=*/0, shape, "param0"));
    auto *param1 =
        called_subcomp_builder.AddInstruction(HloInstruction::CreateParameter(
            /*parameter_number=*/1, shape, "param1"));
    called_subcomp_builder.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(S32, {}), HloOpcode::kAdd, param0, param1));
  }
  auto called_subcomp =
      module->AddEmbeddedComputation(called_subcomp_builder.Build());

  // Creates a module with unflattened control flow with two dead computations
  // that both call the same subcomputation, which becomes dead after the two
  // callers are removed.
  {
    HloComputation::Builder dead_subcomp_builder("dead_caller0");
    auto *param0 = dead_subcomp_builder.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "param0"));
    auto *param1 = dead_subcomp_builder.AddInstruction(
        HloInstruction::CreateParameter(1, shape, "param1"));
    dead_subcomp_builder.AddInstruction(
        HloInstruction::CreateCall(shape, {param0, param1}, called_subcomp));
    module->AddEmbeddedComputation(dead_subcomp_builder.Build());
  }

  {
    HloComputation::Builder dead_subcomp_builder("dead_caller1");
    auto *param0 = dead_subcomp_builder.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "param0"));
    auto *param1 = dead_subcomp_builder.AddInstruction(
        HloInstruction::CreateParameter(1, shape, "param1"));
    dead_subcomp_builder.AddInstruction(
        HloInstruction::CreateCall(shape, {param0, param1}, called_subcomp));
    module->AddEmbeddedComputation(dead_subcomp_builder.Build());
  }

  HloComputation::Builder builder(TestName());

  // Adds a constant instruction as the root of the computation.
  builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(0)));

  module->AddEntryComputation(builder.Build());
  EXPECT_EQ(module->MakeComputationPostOrder().size(), 4);

  HloDCE dce;
  auto changed = dce.Run(module.get());
  ASSERT_TRUE(changed.ok());
  EXPECT_TRUE(*changed);

  // Only the entry computation should be left after eliminating the dead caller
  // and callee subcomputations.
  EXPECT_EQ(module->MakeComputationPostOrder().size(), 1);
}

TEST_F(HloDceTest, MultiOutputFusionRemoveUnusedTupleElementsRemoveTuple) {
  constexpr char kHloString[] = R"(
    HloModule test_module
    fused_add {
      p0 = s32[32,32]{1,0} parameter(0)
      p1 = s32[32,32]{1,0} parameter(1)
      p2 = s32[32,32]{1,0} parameter(2) // becomes dead
      add = s32[32,32]{1,0} add(p0, p1)
      ROOT res = (s32[32,32]{1,0}, s32[32,32]{1,0}) tuple(p2, add)
    }

    ENTRY reduce {
      param0 = s32[32,32]{1,0} parameter(0)
      param1 = s32[32,32]{1,0} parameter(1)
      param2 = s32[32,32]{1,0} parameter(2)
      fusion = (s32[32,32]{1,0}, s32[32,32]{1,0}) fusion(param0, param1, param2), kind=kLoop, calls=fused_add
      gte.0 = s32[32,32]{1,0} get-tuple-element(fusion), index=0  // dead
      ROOT gte.1 = s32[32,32]{1,0} get-tuple-element(fusion), index=1
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloString));
  HloDCE dce;
  auto changed = dce.Run(module.get());
  ASSERT_TRUE(changed.ok());
  EXPECT_TRUE(*changed);
  HloInstruction *root = module->entry_computation()->root_instruction();
  // We expect that the dead parameter and the dead tuple entry are removed.
  EXPECT_THAT(root, GmockMatch(m::Fusion(m::Parameter(0), m::Parameter(1))
                                   .WithShape(S32, {32, 32})));
  EXPECT_THAT(
      root->fused_expression_root(),
      GmockMatch(
          m::Add(m::Parameter(0), m::Parameter(1)).WithShape(S32, {32, 32})));
  EXPECT_EQ(module->MakeComputationPostOrder().size(), 2);
}

TEST_F(
    HloDceTest,
    MultiOutputFusionRemoveUnusedTupleElementsRemoveTupleMultiUsersPerOutput) {
  constexpr char kHloString[] = R"(
    HloModule test_module
    fused_add {
      p0 = s32[32,32]{1,0} parameter(0)
      p1 = s32[32,32]{1,0} parameter(1)
      p2 = s32[32,32]{1,0} parameter(2) // becomes dead
      add = s32[32,32]{1,0} add(p0, p1)
      ROOT res = (s32[32,32]{1,0}, s32[32,32]{1,0}, s32[32,32]{1,0}) tuple(p2, add, p2)
    }

    ENTRY reduce {
      param0 = s32[32,32]{1,0} parameter(0)
      param1 = s32[32,32]{1,0} parameter(1)
      param2 = s32[32,32]{1,0} parameter(2)
      fusion = (s32[32,32]{1,0}, s32[32,32]{1,0}, s32[32,32]{1,0}) fusion(param0, param1, param2), kind=kLoop, calls=fused_add
      gte.1 = s32[32,32]{1,0} get-tuple-element(fusion), index=1
      gte.1.again = s32[32,32]{1,0} get-tuple-element(fusion), index=1
      ROOT res = (s32[32,32]{1,0}, s32[32,32]{1,0}) tuple(gte.1, gte.1.again)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloString));
  HloDCE dce;
  auto changed = dce.Run(module.get());
  ASSERT_TRUE(changed.ok());
  EXPECT_TRUE(*changed);

  HloInstruction *gte_0 = FindInstruction(module.get(), "gte.0");
  EXPECT_EQ(gte_0, nullptr);
  HloInstruction *gte_1 = FindInstruction(module.get(), "gte.1");
  EXPECT_EQ(gte_1, nullptr);
  HloInstruction *gte_1_again = FindInstruction(module.get(), "gte.1.again");
  EXPECT_EQ(gte_1_again, nullptr);

  HloInstruction *fusion = FindInstruction(module.get(), "fusion");
  ASSERT_NE(fusion, nullptr);
  EXPECT_FALSE(fusion->shape().IsTuple());

  HloInstruction *root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(root->operand_count(), 2);
  EXPECT_EQ(root->operand(0), fusion);
  EXPECT_EQ(root->operand(1), fusion);
}

TEST_F(
    HloDceTest,
    MultiOutputFusionRemoveUnusedTupleElementsRemoveTupleNonContiguousRemoval) {
  constexpr char kHloString[] = R"(
    HloModule test_module
    fused_add {
      p0 = s32[32,32]{1,0} parameter(0)
      p1 = s32[32,32]{1,0} parameter(1)
      p2 = s32[32,32]{1,0} parameter(2) // becomes dead
      add = s32[32,32]{1,0} add(p0, p1)
      ROOT res = (s32[32,32]{1,0}, s32[32,32]{1,0}, s32[32,32]{1,0}, s32[32,32]{1,0}) tuple(p2, add, p2, p2)
    }

    ENTRY reduce {
      param0 = s32[32,32]{1,0} parameter(0)
      param1 = s32[32,32]{1,0} parameter(1)
      param2 = s32[32,32]{1,0} parameter(2)
      fusion = (s32[32,32]{1,0}, s32[32,32]{1,0}, s32[32,32]{1,0}, s32[32,32]{1,0}) fusion(param0, param1, param2), kind=kLoop, calls=fused_add
      gte.0 = s32[32,32]{1,0} get-tuple-element(fusion), index=0  // dead
      gte.1 = s32[32,32]{1,0} get-tuple-element(fusion), index=1
      gte.1.again = s32[32,32]{1,0} get-tuple-element(fusion), index=1
      gte.3 = s32[32,32]{1,0} get-tuple-element(fusion), index=3
      ROOT res = (s32[32,32]{1,0}, s32[32,32]{1,0}, s32[32,32]{1,0}) tuple(gte.1, gte.1.again, gte.3)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloString));
  HloDCE dce;
  auto changed = dce.Run(module.get());
  ASSERT_TRUE(changed.ok());
  EXPECT_TRUE(*changed);

  // We expect that the dead parameter and the dead tuple entry are removed.
  HloInstruction *gte_0 = FindInstruction(module.get(), "gte.0");
  EXPECT_EQ(gte_0, nullptr);
  HloInstruction *gte_1 = FindInstruction(module.get(), "gte.1");
  EXPECT_NE(gte_1, nullptr);
  EXPECT_EQ(static_cast<HloGetTupleElementInstruction *>(gte_1)->tuple_index(),
            0);
  HloInstruction *gte_1_again = FindInstruction(module.get(), "gte.1.again");
  EXPECT_EQ(
      static_cast<HloGetTupleElementInstruction *>(gte_1_again)->tuple_index(),
      0);
  EXPECT_NE(gte_1_again, nullptr);
  HloInstruction *gte_3 = FindInstruction(module.get(), "gte.3");
  EXPECT_NE(gte_3, nullptr);
  EXPECT_EQ(static_cast<HloGetTupleElementInstruction *>(gte_3)->tuple_index(),
            1);

  HloInstruction *fusion = FindInstruction(module.get(), "fusion");
  ASSERT_NE(fusion, nullptr);
  EXPECT_TRUE(fusion->shape().IsTuple());
  EXPECT_EQ(fusion->shape().tuple_shapes_size(), 2);

  HloInstruction *root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(root->operand_count(), 3);
  EXPECT_EQ(root->operand(0), gte_1);
  EXPECT_EQ(root->operand(1), gte_1_again);
  EXPECT_EQ(root->operand(2), gte_3);
}

TEST_F(HloDceTest, MultiOutputFusionRemoveUnusedTupleElementAdjustTuple) {
  constexpr char kHloString[] = R"(
    HloModule test_module
    fused_add {
      p0 = s32[32,32]{1,0} parameter(0)
      p1 = s32[32,32]{1,0} parameter(1)
      add = s32[32,32]{1,0} add(p0, p1)
      neg = s32[32,32]{1,0} negate(add)
      ROOT res = (s32[32,32]{1,0}, s32[32,32]{1,0}, s32[32,32]{1,0}) tuple(neg, p0, add)
    }

    ENTRY reduce {
      param0 = s32[32,32]{1,0} parameter(0)
      param1 = s32[32,32]{1,0} parameter(1)
      fusion = (s32[32,32]{1,0}, s32[32,32]{1,0}, s32[32,32]{1,0}) fusion(param0, param1), kind=kLoop, calls=fused_add
      gte.0 = s32[32,32]{1,0} get-tuple-element(fusion), index=0
      gte.1 = s32[32,32]{1,0} get-tuple-element(fusion), index=1
      gte.2 = s32[32,32]{1,0} get-tuple-element(fusion), index=2
      ROOT add = s32[32,32]{1,0} add(gte.0, gte.2)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloString));
  HloDCE dce;
  auto changed = dce.Run(module.get());
  ASSERT_TRUE(changed.ok());
  EXPECT_TRUE(*changed);
  Shape shape = ShapeUtil::MakeShape(S32, {32, 32});
  Shape expected_shape = ShapeUtil::MakeTupleShape({shape, shape});
  HloInstruction *fusion;
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Add(
                  m::GetTupleElement(
                      m::Fusion(&fusion).WithShapeEqualTo(&expected_shape), 0),
                  m::GetTupleElement(m::Fusion(), 1))));
  EXPECT_THAT(
      fusion->fused_expression_root(),
      GmockMatch(
          m::Tuple(m::Negate(), m::Add()).WithShapeEqualTo(&expected_shape)));
  EXPECT_EQ(module->MakeComputationPostOrder().size(), 2);
}

TEST_F(HloDceTest,
       MultiOutputFusionRemoveUnusedTupleElementWithControlAdjustTupleAndDep) {
  constexpr char kHloString[] = R"(
    HloModule test_module
    fused_add {
      p0 = s32[32,32]{1,0} parameter(0)
      p1 = s32[32,32]{1,0} parameter(1)
      add = s32[32,32]{1,0} add(p0, p1)
      ROOT res = (s32[32,32]{1,0}, s32[32,32]{1,0}) tuple(p0, add)
    }

    ENTRY reduce {
      param0 = s32[32,32]{1,0} parameter(0)
      param1 = s32[32,32]{1,0} parameter(1)
      fusion = (s32[32,32]{1,0}, s32[32,32]{1,0}) fusion(param0, param1), kind=kLoop, calls=fused_add
      gte.1 = s32[32,32]{1,0} get-tuple-element(fusion), index=1
      add.2 = s32[32,32]{1,0} add(param0, param1), control-predecessors={gte.1}
      ROOT add = s32[32,32]{1,0} add(add.2, gte.1)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloString));
  HloDCE dce;
  auto changed = dce.Run(module.get());
  ASSERT_TRUE(changed.ok());
  EXPECT_TRUE(*changed);
  HloInstruction *fusion;
  HloInstruction *add2;
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Add(m::Add(&add2, m::Parameter(), m::Parameter()),
                                m::Fusion(&fusion))));
  EXPECT_EQ(add2->control_predecessors().size(), 1);
  EXPECT_EQ(add2->control_predecessors()[0], fusion);
}

}  // namespace
}  // namespace zkx
