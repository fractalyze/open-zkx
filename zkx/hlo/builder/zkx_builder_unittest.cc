/* Copyright 2018 The OpenXLA Authors.
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

#include "zkx/hlo/builder/zkx_builder.h"

#include <array>
#include <functional>

#include "absl/status/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/debug_options_flags.h"
#include "zkx/hlo/builder/zkx_computation.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/hlo/parser/hlo_parser.h"
#include "zkx/hlo/testlib/pattern_matcher_gmock.h"
#include "zkx/service/pattern_matcher.h"

namespace zkx {

namespace {

namespace m = ::zkx::match;

using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::Test;

HloInstruction* GetRoot(HloModule& module) {
  return module.entry_computation()->root_instruction();
}

// TODO(b/74197823): Move the tests to service/.
absl::StatusOr<std::unique_ptr<HloModule>> BuildHloModule(ZkxBuilder& b) {
  TF_ASSIGN_OR_RETURN(ZkxComputation computation,
                      b.Build(/*remove_dynamic_dimensions=*/false));
  const HloModuleProto& proto = computation.proto();
  TF_ASSIGN_OR_RETURN(const auto& config,
                      HloModule::CreateModuleConfigFromProto(
                          proto, GetDebugOptionsFromFlags()));
  return HloModule::CreateFromProto(proto, config);
}

// Overload which explicitly specifies the root instruction.
absl::StatusOr<std::unique_ptr<HloModule>> BuildHloModule(ZkxBuilder& b,
                                                          ZkxOp root) {
  TF_ASSIGN_OR_RETURN(ZkxComputation computation,
                      b.Build(root, /*remove_dynamic_dimensions=*/false));
  const HloModuleProto& proto = computation.proto();
  TF_ASSIGN_OR_RETURN(const auto& config,
                      HloModule::CreateModuleConfigFromProto(
                          proto, GetDebugOptionsFromFlags()));
  return HloModule::CreateFromProto(proto, config);
}

// Returns the name of the test currently being run.
std::string TestName() {
  return ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

TEST(ZkxBuilderTest, OnePlusTwo) {
  ZkxBuilder b(TestName());
  Add(ConstantR0<uint32_t>(&b, 1), ConstantR0<uint32_t>(&b, 2));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Add(m::Constant(), m::Constant())));
}

TEST(ZkxBuilderTest, UnaryOperatorsBuildExpectedHLO) {
  auto test_unary_operator = [&](std::function<ZkxOp(ZkxOp)> op,
                                 auto matches_pattern) {
    ZkxBuilder b(TestName());
    op(ConstantR0<int32_t>(&b, 1));
    TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
    EXPECT_THAT(GetRoot(*module), matches_pattern);
  };
  test_unary_operator([](ZkxOp x) { return -x; },
                      GmockMatch(m::Negate(m::Constant())));
  test_unary_operator([](ZkxOp x) { return ~x; },
                      GmockMatch(m::Not(m::Constant())));
}

TEST(ZkxBuilderTest, BinaryOperatorsBuildExpectedHLO) {
  auto test_binary_operator = [&](std::function<ZkxOp(ZkxOp, ZkxOp)> op,
                                  auto matches_pattern) {
    ZkxBuilder b(TestName());
    op(ConstantR0<int32_t>(&b, 1), ConstantR0<int32_t>(&b, 2));
    TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
    EXPECT_THAT(GetRoot(*module), matches_pattern);
  };

  test_binary_operator([](ZkxOp x, ZkxOp y) { return x + y; },
                       GmockMatch(m::Add(m::Constant(), m::Constant())));
  test_binary_operator([](ZkxOp x, ZkxOp y) { return x - y; },
                       GmockMatch(m::Subtract(m::Constant(), m::Constant())));
  test_binary_operator([](ZkxOp x, ZkxOp y) { return x * y; },
                       GmockMatch(m::Multiply(m::Constant(), m::Constant())));
  test_binary_operator([](ZkxOp x, ZkxOp y) { return x / y; },
                       GmockMatch(m::Divide(m::Constant(), m::Constant())));

  test_binary_operator([](ZkxOp x, ZkxOp y) { return x & y; },
                       GmockMatch(m::And(m::Constant(), m::Constant())));
  test_binary_operator([](ZkxOp x, ZkxOp y) { return x | y; },
                       GmockMatch(m::Or(m::Constant(), m::Constant())));
  test_binary_operator([](ZkxOp x, ZkxOp y) { return x ^ y; },
                       GmockMatch(m::Xor(m::Constant(), m::Constant())));
  test_binary_operator([](ZkxOp x, ZkxOp y) { return x << y; },
                       GmockMatch(m::ShiftLeft(m::Constant(), m::Constant())));
  test_binary_operator(
      [](ZkxOp x, ZkxOp y) { return x >> y; },
      GmockMatch(m::ShiftRightArithmetic(m::Constant(), m::Constant())));

  auto test_unsigned_binary_operator =
      [&](std::function<ZkxOp(ZkxOp, ZkxOp)> op, auto matches_pattern) {
        ZkxBuilder b(TestName());
        op(ConstantR0<uint32_t>(&b, 1), ConstantR0<uint32_t>(&b, 2));
        TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
        EXPECT_THAT(GetRoot(*module), matches_pattern);
      };
  test_unsigned_binary_operator(
      [](ZkxOp x, ZkxOp y) { return x >> y; },
      GmockMatch(m::ShiftRightLogical(m::Constant(), m::Constant())));
}

TEST(ZkxBuilderTest, VariadicAnd) {
  ZkxBuilder b(TestName());
  const Shape s = ShapeUtil::MakeShape(PRED, {});
  And(Parameter(&b, 0, s, "p0"), Parameter(&b, 1, s, "p1"),
      Parameter(&b, 2, s, "p2"));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  // Don't specify in the test whether And(x, y, z) is right- or
  // left-associative; accept either one.
  EXPECT_THAT(GetRoot(*module),
              ::testing::AnyOf(
                  GmockMatch(m::And(m::Parameter(0),
                                    m::And(m::Parameter(1), m::Parameter(2)))),
                  GmockMatch(m::And(m::And(m::Parameter(0), m::Parameter(1)),
                                    m::Parameter(2)))));
}

TEST(ZkxBuilderTest, VariadicOr) {
  ZkxBuilder b(TestName());
  const Shape s = ShapeUtil::MakeShape(PRED, {});
  Or(Parameter(&b, 0, s, "p0"), Parameter(&b, 1, s, "p1"),
     Parameter(&b, 2, s, "p2"));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  // Don't specify in the test whether Or(x, y, z) is right- or
  // left-associative; accept either one.
  EXPECT_THAT(GetRoot(*module),
              ::testing::AnyOf(
                  GmockMatch(m::Or(m::Parameter(0),
                                   m::Or(m::Parameter(1), m::Parameter(2)))),
                  GmockMatch(m::Or(m::Or(m::Parameter(0), m::Parameter(1)),
                                   m::Parameter(2)))));
}

TEST(ZkxBuilderTest, ParamPlusConstantHasScalarBroadcast) {
  ZkxBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(U32, {3, 5}), "x");
  Add(x, ConstantR0<uint32_t>(&b, 1));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Add(m::Parameter(), m::Broadcast(m::Constant()))));
}

TEST(ZkxBuilderTest, ParamPlusConstantHasScalarBroadcastReversed) {
  ZkxBuilder b(TestName());
  const ZkxOp x = Parameter(&b, 0, ShapeUtil::MakeShape(U32, {3, 5}), "x");
  Add(ConstantR0<uint32_t>(&b, 1), x);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Add(m::Broadcast(m::Constant()), m::Parameter())));
}

TEST(ZkxBuilderTest, ParamPlusParamHasBroadcast) {
  ZkxBuilder b(TestName());
  const auto& x_shape = ShapeUtil::MakeShape(S32, {2, 4, 6});
  const auto& y_shape = ShapeUtil::MakeShape(S32, {2, 4});
  auto x = Parameter(&b, 0, x_shape, "x");
  auto y = Parameter(&b, 1, y_shape, "y");
  auto add = Add(x, y, /*broadcast_dimensions=*/{0, 1});

  TF_ASSERT_OK_AND_ASSIGN(const auto add_shape, b.GetShape(add));
  EXPECT_TRUE(ShapeUtil::Equal(add_shape, x_shape));

  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(
      GetRoot(*module),
      GmockMatch(m::Add(m::Parameter(0), m::Broadcast(m::Parameter(1)))));
}

TEST(ZkxBuilderTest, XPlusX) {
  ZkxBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(S32, {1, 3, 5, 7}), "x");
  Add(x, x);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Add(m::Parameter(0), m::Parameter(0))));
}

TEST(ZkxBuilderTest, ShapeInferenceError) {
  ZkxBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(U32, {2, 4, 6}), "x");
  auto y = Parameter(&b, 1, ShapeUtil::MakeShape(U32, {2, 4}), "y");
  Add(x, y);
  auto statusor = BuildHloModule(b);
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("Shapes must be equal rank"));
}

TEST(ZkxBuilderTest, ParameterAlreadyRegistered) {
  ZkxBuilder b_call("add");
  Parameter(&b_call, 0, ShapeUtil::MakeShape(PRED, {}), "x");

  ZkxBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(PRED, {}), "x");
  auto y = Parameter(&b, 0, ShapeUtil::MakeShape(PRED, {}), "y");
  Add(x, y);
  auto statusor = BuildHloModule(b);
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("parameter 0 already registered"));
}

TEST(ZkxBuilderTest, BroadcastInDim) {
  ZkxBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(U32, {2, 3}), "x");
  BroadcastInDim(x, {2, 4, 3},
                 /*broadcast_dimensions=*/{0, 2});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module), GmockMatch(m::Broadcast()));
}

TEST(ZkxBuilderTest, BroadcastInDimWithBoundedDim) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape shape, ParseShape("u32[2, <=3]"));
  auto x = Parameter(&b, 0, shape, "x");
  BroadcastInDim(x, {1, 2, 3},
                 /*broadcast_dimensions=*/{1, 2});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module), GmockMatch(m::Broadcast()));
}

TEST(ZkxBuilderTest, BroadcastInDimWithNegativeSize) {
  ZkxBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(U32, {2, 1, 4}), "x");
  BroadcastInDim(x, {-3, 3, 4},
                 /*broadcast_dimensions=*/{0, 1, 2});
  auto statusor = BuildHloModule(b);
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(), HasSubstr("invalid shape"));
}

TEST(ZkxBuilderTest, OperandFromWrongBuilder) {
  ZkxBuilder b1("b1");
  auto p0 = Parameter(&b1, 0, ShapeUtil::MakeShape(U32, {}), "p0");
  ZkxBuilder builder("main");
  auto p = Parameter(&builder, 0, ShapeUtil::MakeShape(U32, {}), "p");
  Add(p, p0);
  auto statusor = builder.Build();
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().message(),
      HasSubstr(
          "built by builder 'b1', but is trying to use it in builder 'main'"));
}

// Test the special case where split_dimension is the same as concat_dimension.

TEST(ZkxBuilderTest, GetDimensionSizeConstant) {
  ZkxBuilder b(TestName());
  auto x =
      Parameter(&b, 0, ShapeUtil::MakeShape(U32, {5, 7}, {false, true}), "x");
  // Get dimension size from a constant dimension gives us a constant.
  GetDimensionSize(x, 0);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_EQ(GetRoot(*module)->opcode(), HloOpcode::kConstant);
}

TEST(ZkxBuilderTest, ReportError) {
  ZkxBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(U32, {5, 7}), "x");
  Add(b.ReportError(absl::InvalidArgumentError("a test error")), x);
  auto statusor = b.Build();
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(), HasSubstr("a test error"));
}

TEST(ZkxBuilderTest, ReportErrorOrReturnHandlesNonErrors) {
  ZkxBuilder b(TestName());
  absl::StatusOr<ZkxOp> op(ConstantR0<uint32_t>(&b, 1));
  Add(b.ReportErrorOrReturn(op), ConstantR0<uint32_t>(&b, 2));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Add(m::Constant(), m::Constant())));
}

TEST(ZkxBuilderTest, ReportErrorOrReturnHandlesErrors) {
  ZkxBuilder b(TestName());
  absl::StatusOr<ZkxOp> op(absl::InvalidArgumentError("a test error"));
  Add(b.ReportErrorOrReturn(op), ConstantR0<uint32_t>(&b, 2));
  auto statusor = b.Build();
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(), HasSubstr("a test error"));
}

TEST(ZkxBuilderTest, BuildWithSpecificRoot) {
  ZkxBuilder b(TestName());
  const ZkxOp constant = ConstantR0<uint32_t>(&b, 1);
  Add(constant, ConstantR0<uint32_t>(&b, 2));
  TF_ASSERT_OK_AND_ASSIGN(const auto module,
                          BuildHloModule(b, /*root=*/constant));
  EXPECT_THAT(GetRoot(*module), GmockMatch(m::Constant()));
}

TEST(ZkxBuilderTest, BuildWithSpecificRootAndMultipleParameters) {
  // Specifying a particular root in Build should still include all entry
  // parameters.
  ZkxBuilder b(TestName());
  const Shape shape = ShapeUtil::MakeShape(U32, {42, 123});
  const ZkxOp x = Parameter(&b, 0, shape, "x");
  const ZkxOp y = Parameter(&b, 1, shape, "y");
  const ZkxOp z = Parameter(&b, 2, shape, "z");
  Add(x, Sub(y, z));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b, /*root=*/x));
  EXPECT_THAT(GetRoot(*module), GmockMatch(m::Parameter()));
  EXPECT_EQ(module->entry_computation()->num_parameters(), 3);
  EXPECT_EQ(module->entry_computation()->instruction_count(), 5);
}

TEST(ZkxBuilderTest, BuildWithSpecificRootWithWrongBuilder) {
  ZkxBuilder b(TestName());
  ZkxBuilder other_b(TestName());
  const Shape shape = ShapeUtil::MakeShape(U32, {42, 123});

  Parameter(&b, 0, shape, "param");
  const ZkxOp other_param = Parameter(&other_b, 0, shape, "other_param");

  absl::Status status = b.Build(other_param).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("root operation is not in this computation"));
}

TEST(ZkxBuilderTest, ProtoMatches) {
  std::vector<ZkxComputation> computations;
  const int n = 2;
  computations.reserve(n);
  for (int i = 0; i < n; ++i) {
    ZkxBuilder b_call("the_only_to_apply");
    auto p0 = Parameter(&b_call, 0, ShapeUtil::MakeShape(U32, {}), "p0");
    auto p1 = Parameter(&b_call, 1, ShapeUtil::MakeShape(U32, {}), "p1");
    Add(p0, Add(p1, p0));
    TF_ASSERT_OK_AND_ASSIGN(const auto call, b_call.Build());
    ZkxBuilder b(TestName());
    auto x = Parameter(&b, 0, ShapeUtil::MakeShape(U32, {}), "x");
    auto y = Parameter(&b, 1, ShapeUtil::MakeShape(U32, {}), "y");
    auto one = ConstantR0<uint32_t>(&b, 1);
    auto two = ConstantR0<uint32_t>(&b, 2);
    Add(Call(&b, call, {x, y}), Call(&b, call, {one, two}));
    computations.push_back(b.Build().value());
  }
  auto c0_string = computations[0].proto().SerializeAsString();
  auto c1_string = computations[1].proto().SerializeAsString();
  EXPECT_EQ(c0_string, c1_string);
}

TEST(ZkxBuilderTest, DynamicParameter) {
  ZkxBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(U32, {5}), ShapeUtil::MakeShape(U32, {6}, {true})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  Parameter(&b, 1, ShapeUtil::MakeShape(U32, {}), "p1");
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b, /*root=*/p0));
  const Shape& param_shape = module->entry_computation()
                                 ->parameter_instruction(0)
                                 ->shape()
                                 .tuple_shapes(1);
  EXPECT_TRUE(param_shape.is_dynamic_dimension(0));
}

TEST(ZkxBuilderTest, DynamicUnary) {
  ZkxBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(U32, {5}, {true}), ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto gte = GetTupleElement(p0, 0);
  Neg(gte);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(result_shape.is_dynamic_dimension(0));
}

TEST(ZkxBuilderTest, DynamicBinary) {
  ZkxBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(U32, {5}, {true}),
       ShapeUtil::MakeShape(U32, {5}, {true}), ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto gte0 = GetTupleElement(p0, 0);
  auto gte1 = GetTupleElement(p0, 1);
  Add(gte0, gte1);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(result_shape.is_dynamic_dimension(0));
}

TEST(ZkxBuilderTest, DynamicBinaryHasBroadcast) {
  ZkxBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(U32, {5, 4}, {true, false}),
       ShapeUtil::MakeShape(U32, {5}, {true}), ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto gte0 = GetTupleElement(p0, 0);
  auto gte1 = GetTupleElement(p0, 1);
  Add(gte0, gte1, {0});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_THAT(result_shape.dynamic_dimensions(), ElementsAre(true, false))
      << result_shape;
}

TEST(ZkxBuilderTest, DynamicBroadcast) {
  ZkxBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(U32, {5, 4}, {true, false}),
       ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto gte = GetTupleElement(p0, 0);
  BroadcastInDim(gte, /*out_dim_size=*/{3, 5, 4},
                 /*broadcast_dimensions=*/{1, 2});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_THAT(result_shape.dynamic_dimensions(),
              ElementsAre(false, true, false))
      << result_shape;
}

TEST(ZkxBuilderTest, DynamicSelectOnlyPredDynamic) {
  ZkxBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(PRED, {10}, {true}),
       ShapeUtil::MakeShape(U32, {10}), ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto gte0 = GetTupleElement(p0, 0);
  auto gte1 = GetTupleElement(p0, 1);

  Select(gte0, gte1, gte1);

  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_THAT(result_shape.dynamic_dimensions(), ElementsAre(true))
      << result_shape;
}

TEST(ZkxBuilderTest, DynamicSelect) {
  ZkxBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(U32, {4, 5, 6}, {false, true, false}),
       ShapeUtil::MakeShape(U32, {4, 5, 6}, {false, true, false}),
       ShapeUtil::MakeShape(U32, {}), ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto pred = Parameter(&b, 1, ShapeUtil::MakeShape(PRED, {}), "pred");
  auto gte0 = GetTupleElement(p0, 0);
  auto gte1 = GetTupleElement(p0, 1);
  Select(pred, gte0, gte1);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(result_shape.is_dynamic_dimension(1));
  EXPECT_FALSE(result_shape.is_dynamic_dimension(2));
  EXPECT_THAT(result_shape.dynamic_dimensions(),
              ElementsAre(false, true, false))
      << result_shape;
}

TEST(ZkxBuilderTest, DynamicSelectNotCompatible) {
  ZkxBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(U32, {4, 5, 6}, {false, true, false}),
       ShapeUtil::MakeShape(U32, {4, 5, 6}, {false, false, true}),
       ShapeUtil::MakeShape(U32, {}), ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto pred = Parameter(&b, 1, ShapeUtil::MakeShape(PRED, {}), "pred");
  auto gte0 = GetTupleElement(p0, 0);  // u32[4,<=5,6]
  auto gte1 = GetTupleElement(p0, 1);  // u32[4,5,<=6]
  Select(pred, gte0, gte1);
  absl::Status status = BuildHloModule(b).status();
  ASSERT_TRUE(status.ok());
}

TEST(ZkxBuilderTest, CheckInputOutputAlias) {
  ZkxBuilder b(TestName());
  auto p0 = Parameter(&b, 0, ShapeUtil::MakeShape(U32, {8, 4}), "p0");
  auto p1 = Parameter(&b, 1, ShapeUtil::MakeShape(U32, {8, 4}), "p1");
  auto add = Add(p0, p1);
  auto sub = Sub(p0, p1);
  auto root = Tuple(&b, {add, sub});

  b.SetUpAlias({1}, 0, {});
  b.SetUpAlias({0}, 1, {});

  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b, root));

  const HloInputOutputAliasConfig& config = module->input_output_alias_config();
  EXPECT_TRUE(config.ParameterHasAlias(0, {}));
  EXPECT_TRUE(config.ParameterHasAlias(1, {}));

  auto alias_p0 = config.GetAliasedOutput(0, {});
  ASSERT_TRUE(alias_p0.has_value());
  EXPECT_EQ(*alias_p0, ShapeIndex({1}));

  auto alias_p1 = config.GetAliasedOutput(1, {});
  ASSERT_TRUE(alias_p1.has_value());
  EXPECT_EQ(*alias_p1, ShapeIndex({0}));
}

TEST(ZkxBuilderTest, CheckBufferDonor) {
  ZkxBuilder b(TestName());
  auto p0 = Parameter(&b, 0, ShapeUtil::MakeShape(U32, {8, 4}), "p0");
  auto p1 = Parameter(&b, 1, ShapeUtil::MakeShape(U32, {8, 4}), "p1");
  auto add = Add(p0, p1);
  auto sub = Sub(p0, p1);
  auto root = Tuple(&b, {add, sub});

  b.AddBufferDonor(0, {});

  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b, root));

  const HloBufferDonorConfig& config = module->buffer_donor_config();
  EXPECT_TRUE(config.ParameterIsBufferDonor(0, {}));
  EXPECT_FALSE(config.ParameterIsBufferDonor(1, {}));
}

TEST(ZkxBuilderTest, ConstantLiteral) {
  ZkxBuilder b(TestName());
  ConstantR1<uint32_t>(&b, {0, 1});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const HloInstruction* root = GetRoot(*module);
  ASSERT_THAT(root, GmockMatch(m::Constant()));
}

TEST(ZkxBuilderTest, InvalidInputOutputAliasBufferDonor) {
  ZkxBuilder b(TestName());

  auto p0 = Parameter(&b, 0, ShapeUtil::MakeShape(U32, {8, 4}), "p0");
  auto p1 = Parameter(&b, 1, ShapeUtil::MakeShape(U32, {8, 4}), "p1");
  auto add = Add(p0, p1);
  auto sub = Sub(p0, p1);
  auto root = Tuple(&b, {add, sub});

  b.SetUpAlias({1}, 0, {});
  b.AddBufferDonor(0, {});

  auto statusor = BuildHloModule(b, root);
  EXPECT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("is already aliased with one output, thus it cannot be "
                        "added as a buffer donor for any output."));
}

TEST(ZkxBuilderTest, ValidInputOutputAliasBufferDonor) {
  ZkxBuilder b(TestName());

  auto p0 = Parameter(&b, 0, ShapeUtil::MakeShape(U32, {8, 4}), "p0");
  auto p1 = Parameter(&b, 1, ShapeUtil::MakeShape(U32, {8, 4}), "p1");
  auto add = Add(p0, p1);
  auto sub = Sub(p0, p1);
  auto root = Tuple(&b, {add, sub});

  b.SetUpAlias({1}, 0, {});
  b.AddBufferDonor(1, {});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b, root));

  const HloInputOutputAliasConfig& io_alias_config =
      module->input_output_alias_config();
  const HloBufferDonorConfig& buffer_donor_config =
      module->buffer_donor_config();

  EXPECT_TRUE(io_alias_config.ParameterHasAlias(0, {}));
  EXPECT_FALSE(io_alias_config.ParameterHasAlias(1, {}));
  EXPECT_FALSE(buffer_donor_config.ParameterIsBufferDonor(0, {}));
  EXPECT_TRUE(buffer_donor_config.ParameterIsBufferDonor(1, {}));

  auto alias_p0 = io_alias_config.GetAliasedOutput(0, {});
  ASSERT_TRUE(alias_p0.has_value());
  EXPECT_EQ(*alias_p0, ShapeIndex({1}));
}

void ExpectAttributesMatch(const FrontendAttributes& attr,
                           const FrontendAttributes& ref) {
  EXPECT_EQ(ref.map_size(), attr.map_size());
  for (auto reference : ref.map()) {
    auto other = attr.map().find(reference.first);
    EXPECT_NE(other, attr.map().end());
    EXPECT_EQ(other->second, reference.second);
  }
}

void ExpectInstructionsAttributesMatch(
    const HloModule& module, const std::vector<FrontendAttributes>& expected) {
  ASSERT_EQ(module.computation_count(), 1);
  auto expected_it = expected.begin();
  for (auto inst : module.entry_computation()->instructions()) {
    ASSERT_NE(expected_it, expected.end());
    ExpectAttributesMatch(inst->frontend_attributes(), *expected_it);
    expected_it++;
  }
  EXPECT_EQ(expected_it, expected.end());
}

TEST(ZkxBuilderTest, SimpleSetFrontendAttributes) {
  ZkxBuilder b(TestName());
  FrontendAttributes attributes;

  ConstantR0(&b, 0);  // No attribute set

  (*attributes.mutable_map())["attr_a"] = "a";
  b.SetFrontendAttributes(attributes);
  ConstantR0(&b, 0);  // One attribute: { "attr_a": "a" }

  b.ClearFrontendAttributes();
  ConstantR0(&b, 0);  // No attribute set

  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));

  std::vector<FrontendAttributes> expected{FrontendAttributes(), attributes,
                                           FrontendAttributes()};
  ExpectInstructionsAttributesMatch(*module, expected);
}

TEST(ZkxBuilderTest, ComplexSetFrontendAttributes) {
  ZkxBuilder b(TestName());

  ConstantR0(&b, 0);  // No attribute set.
  std::vector<FrontendAttributes> expected{FrontendAttributes()};

  {
    FrontendAttributes attributes;
    (*attributes.mutable_map())["attr_a"] = "a";
    b.SetFrontendAttributes(attributes);
    ConstantR0(&b, 0);  // One attribute: { "attr_a": "a" }
    expected.push_back(attributes);
  }

  {
    FrontendAttributes attributes;
    (*attributes.mutable_map())["attr_b"] = "b";
    b.SetFrontendAttributes(attributes);
    ConstantR0(&b, 0);  // One attribute: { "attr_b": "b" }
    expected.push_back(attributes);
  }

  {
    FrontendAttributes attributes;
    (*attributes.mutable_map())["attr_b"] = "b";
    (*attributes.mutable_map())["attr_c"] = "c";
    b.SetFrontendAttributes(attributes);
    ConstantR0(&b, 0);  // Two attributes: { "attr_b": "b", "attr_c": "c" }
    expected.push_back(attributes);
  }

  b.ClearFrontendAttributes();
  ConstantR0(&b, 0);  // No attribute set
  expected.push_back(FrontendAttributes());

  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  ExpectInstructionsAttributesMatch(*module, expected);
}

TEST(ZkxBuilderTest, AddFrontendAttribute) {
  ZkxBuilder b(TestName());

  ConstantR0(&b, 0);
  std::vector<FrontendAttributes> expected{FrontendAttributes()};

  // One attribute: { "attr_a": "a" }
  {
    FrontendAttributes attributes;
    (*attributes.mutable_map())["attr_a"] = "a";
    b.SetFrontendAttributes(attributes);
    ConstantR0(&b, 0);
    expected.push_back(attributes);
  }

  // Two attributes: {"attra": "a", "attr_c": "c"}
  {
    auto op = ConstantR0(&b, 0);
    EXPECT_TRUE(b.SetInstructionFrontendAttribute(op, "attr_c", "c").ok());

    FrontendAttributes attributes;
    (*attributes.mutable_map())["attr_a"] = "a";
    (*attributes.mutable_map())["attr_c"] = "c";
    expected.push_back(attributes);
  }

  // Override value of existing "attr_a"
  // One attribute: { "attr_a", "a2"}
  {
    auto op = ConstantR0(&b, 0);
    EXPECT_TRUE(b.SetInstructionFrontendAttribute(op, "attr_a", "a2").ok());
    FrontendAttributes attributes;
    (*attributes.mutable_map())["attr_a"] = "a2";
    expected.push_back(attributes);
  }

  // Check "attr_a" is back to its original value
  // One attribute: { "attr_a", "a"}
  {
    auto op = ConstantR0(&b, 0);
    (void)op;
    FrontendAttributes attributes;
    (*attributes.mutable_map())["attr_a"] = "a";
    expected.push_back(attributes);
  }

  b.ClearFrontendAttributes();
  ConstantR0(&b, 0);  // No attribute set
  expected.push_back(FrontendAttributes());

  // One attribute: { "attr_d", "d"}
  {
    auto op = ConstantR0(&b, 0);
    EXPECT_TRUE(b.SetInstructionFrontendAttribute(op, "attr_d", "d").ok());
    FrontendAttributes attributes;
    (*attributes.mutable_map())["attr_d"] = "d";
    expected.push_back(attributes);
  }

  ConstantR0(&b, 0);  // No attribute set
  expected.push_back(FrontendAttributes());

  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  ExpectInstructionsAttributesMatch(*module, expected);
}

TEST(ZkxBuilderTest, Comparison) {
  ZkxBuilder b(TestName());
  (void)Le(ConstantR0<int32_t>(&b, 1), ConstantR0<int32_t>(&b, 2));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const HloInstruction* root = GetRoot(*module);
  ASSERT_THAT(root, GmockMatch(m::Compare(m::Constant(), m::Constant())));
}

TEST(ZkxBuilderTest, StableLookUpInstructionByHandle) {
  ZkxBuilder b(TestName());
  internal::ZkxBuilderFriend builder_friend;
  const ZkxOp le = Le(ConstantR0<int32_t>(&b, 1), ConstantR0<int32_t>(&b, 2));
  HloInstructionProto* first_op = builder_friend.GetInstruction(le);
  // Create some more instructions.
  for (int i = 0; i < 100; ++i) {
    (void)Le(ConstantR0<int32_t>(&b, 1), ConstantR0<int32_t>(&b, 2));
  }
  // Make sure first_op hasn't changed.
  HloInstructionProto* first_op_now = builder_friend.GetInstruction(le);
  EXPECT_EQ(first_op, first_op_now);
}

//============================================================================//
// Experimental Test
//============================================================================//

//============================================================================//
// Unbounded Dynamism Test
//============================================================================//

struct UnaryOpTestCase {
  std::string operand;
  std::string expected;
  std::function<ZkxOp(ZkxOp)> unary_op;
};

constexpr std::string_view kBroadcastDimensionMismatch =
    "Broadcast dimension 0 mismatch: 2 != -9223372036854775808; u32[2] and "
    "u32[?,10].";
std::array<const int64_t, 1> zero_array = {0};

class ZkxBuilderUnboundedUnaryOpTest
    : public ::testing::TestWithParam<UnaryOpTestCase> {};

TEST_P(ZkxBuilderUnboundedUnaryOpTest, UnboundedUnaryOpTest) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape(GetParam().operand));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape(GetParam().expected));
  GetParam().unary_op(Parameter(&b, 0, operand, "operand"));
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, UnboundedAddUnsupportedImplicitBroadcast) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("u32[2]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));
  Add(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
      /*broadcast_dimensions=*/zero_array);
  EXPECT_THAT(BuildHloModule(b),
              StatusIs(_, HasSubstr(kBroadcastDimensionMismatch)));
}

TEST(ZkxBuilderTest, UnboundedBitcastConvert) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u16[?, 10, 2]"));
  BitcastConvertType(Parameter(&b, 0, operand, "operand"), PrimitiveType::U16);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, UnboundedBroadcastUnsupportedOperand) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[<=3, ?]"));
  Broadcast(Parameter(&b, 0, operand, "operand"), /*broadcast_sizes=*/{1});
  EXPECT_THAT(BuildHloModule(b),
              StatusIs(_, HasSubstr("is_unbounded_dynamic")));
}

TEST(ZkxBuilderTest, UnboundedBroadcastUnsupportedBroadcastSize) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[1]"));
  Broadcast(Parameter(&b, 0, operand, "operand"),
            /*broadcast_sizes=*/{Shape::kUnboundedSize});
  EXPECT_THAT(
      BuildHloModule(b),
      StatusIs(_, HasSubstr("Non-broadcast dimensions must not be dynamic.")));
}

TEST(ZkxBuilderTest, UnboundedBroadcastInDim) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[<=2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[<=2, 3, 4]"));
  BroadcastInDim(Parameter(&b, 0, operand, "operand"),
                 /*out_dim_size=*/{2, 3, 4},
                 /*broadcast_dimensions=*/{0, 2});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, UnboundedBroadcastInDimUnsupported) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[<=3, ?]"));
  BroadcastInDim(Parameter(&b, 0, operand, "operand"),
                 /*out_dim_size=*/{2, 3, Shape::kUnboundedSize},
                 /*broadcast_dimensions=*/{0, 2});
  EXPECT_THAT(BuildHloModule(b),
              StatusIs(_, HasSubstr("BroadcastInDim output must shape be "
                                    "static or bounded dynamic")));
}

TEST(ZkxBuilderTest, UnboundedClamp) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs,
                          ParseShape("u32[1, ?, 2, ?, <=2, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs,
                          ParseShape("u32[?, 1, ?, 2, ?, <=2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs,
                          ParseShape("u32[1, ?, 2, ?, <=2, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape("u32[?, 1, ?, 2, ?, <=2, ?]"));
  Clamp(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
        Parameter(&b, 2, ehs, "ehs"));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest,
     UnboundedClampUnsupportedDegenerateOperandImplicitBroadcast) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("u32[1]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs, ParseShape("u32[?, 10]"));
  Clamp(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
        Parameter(&b, 2, ehs, "ehs"));
  EXPECT_THAT(BuildHloModule(b),
              StatusIs(_, HasSubstr("Unimplemented implicit broadcast.")));
}

TEST(ZkxBuilderTest, UnboundedConvert) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("s32[?]"));
  ConvertElementType(Parameter(&b, 0, operand, "operand"), PrimitiveType::S32);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, UnboundedGetTupleElement) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));
  GetTupleElement(Tuple(&b, {Parameter(&b, 0, operand, "operand")}), 0);
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, UnboundedReshapeUnsupportedOutputShape) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[6]"));
  Reshape(Parameter(&b, 0, operand, "operand"), /*dimensions=*/{0},
          /*new_sizes=*/{Shape::kUnboundedSize, Shape::kUnboundedSize});
  EXPECT_THAT(
      BuildHloModule(b),
      StatusIs(_,
               HasSubstr(
                   "Reshaping with unbounded result shape is not supported.")));
}

TEST(ZkxBuilderTest, UnboundedReshapeUnsupportedInferredShape) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?]"));
  Reshape(operand, Parameter(&b, 0, operand, "operand"));
  EXPECT_THAT(
      BuildHloModule(b),
      StatusIs(_,
               HasSubstr(
                   "Reshaping with unbounded result shape is not supported.")));
}

TEST(ZkxBuilderTest, UnboundedSelect) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs,
                          ParseShape("pred[1, ?, 2, ?, <=2, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs,
                          ParseShape("u32[?, 1, ?, 2, ?, <=2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs,
                          ParseShape("u32[1, ?, 2, ?, <=2, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape("u32[1, 1, 2, 2, <=2, <=2, ?]"));
  Select(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
         Parameter(&b, 2, ehs, "ehs"));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest,
     UnboundedSelectUnsupportedDegenerateOperandImplicitBroadcast) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("pred[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("u32[1]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs, ParseShape("u32[?, 10]"));
  Select(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
         Parameter(&b, 2, ehs, "ehs"));
  EXPECT_THAT(BuildHloModule(b),
              StatusIs(_, HasSubstr("Unimplemented implicit broadcast.")));
}

TEST(ZkxBuilderTest, UnboundedSlice) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[1, <=3, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[1, <=2, 3]"));
  Slice(Parameter(&b, 0, operand, "operand"),
        /*start_indices=*/{0, 1, 2},
        /*limit_indices=*/{1, 3, 5},
        /*strides=*/{1, 1, 1});
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, UnboundedTuple) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?, 10]"));
  const Shape expected = ShapeUtil::MakeTupleShape({operand});
  Tuple(&b, {Parameter(&b, 0, operand, "operand")});
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

INSTANTIATE_TEST_SUITE_P(UnboundedDynamism, ZkxBuilderUnboundedUnaryOpTest,
                         ::testing::ValuesIn<UnaryOpTestCase>(
                             {{"s32[?]", "s32[?]", &Abs},
                              {"u32[?]", "u32[?]", &Clz},
                              {"u32[?]", "u32[?]", &Neg},
                              {"s32[?]", "s32[?]", &Not},
                              {"u32[?]", "u32[?]", &PopulationCount},
                              {"s32[?]", "s32[?]", &Sign}}));

}  // namespace
}  // namespace zkx
