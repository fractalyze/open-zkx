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

#include "zkx/mlir/codegen_utils.h"

#include "gtest/gtest.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"

namespace zkx::mlir_utils {
namespace {

using ::mlir::arith::TruncIOp;
using ::mlir::prime_ir::field::BitcastOp;
using ::mlir::prime_ir::field::FromMontOp;
using ::mlir::prime_ir::field::PrimeFieldType;

class ConvertFieldTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context_.loadDialect<mlir::arith::ArithDialect,
                         mlir::prime_ir::field::FieldDialect>();
  }

  PrimeFieldType MakeFieldType(unsigned storage_bits, bool montgomery) {
    auto modulus =
        mlir::IntegerAttr::get(mlir::IntegerType::get(&context_, storage_bits),
                               (1u << 4) + 1);  // small odd prime stand-in
    return PrimeFieldType::get(&context_, modulus, montgomery);
  }

  mlir::MLIRContext context_;
};

// Regression: ConvertField(felt → smaller integer) used to fall through to
// `return args[0]`, leaking the felt-typed Value into a position the caller
// declared as integer. AES-256-encrypt's GPU JIT then crashed inside
// ClampIndex with `arith.index_cast operand must be signless-integer-like,
// but got !field.pf<...>`.
TEST_F(ConvertFieldTest, FeltStandardFormToSmallerIntegerProducesTrunc) {
  mlir::OpBuilder b(&context_);
  mlir::ImplicitLocOpBuilder lb(b.getUnknownLoc(), b);

  auto block = std::make_unique<mlir::Block>();
  lb.setInsertionPointToStart(block.get());

  PrimeFieldType felt = MakeFieldType(/*storage_bits=*/256, /*montgomery=*/false);
  mlir::Type i32 = lb.getIntegerType(32);

  mlir::Value placeholder = block->addArgument(felt, lb.getUnknownLoc());
  mlir::Value result = ConvertField(lb, /*result_types=*/{i32},
                                    /*source_type=*/felt,
                                    /*target_type=*/i32, {placeholder});

  ASSERT_TRUE(result);
  EXPECT_EQ(result.getType(), i32);
  auto trunc = result.getDefiningOp<TruncIOp>();
  ASSERT_TRUE(trunc) << "expected trailing arith.trunci";
  auto bitcast = trunc.getIn().getDefiningOp<BitcastOp>();
  ASSERT_TRUE(bitcast) << "expected field.bitcast feeding the trunc";
  EXPECT_EQ(bitcast.getInput(), placeholder);
}

TEST_F(ConvertFieldTest, FeltMontgomeryToIntegerInsertsFromMontFirst) {
  mlir::OpBuilder b(&context_);
  mlir::ImplicitLocOpBuilder lb(b.getUnknownLoc(), b);

  auto block = std::make_unique<mlir::Block>();
  lb.setInsertionPointToStart(block.get());

  PrimeFieldType mont_felt = MakeFieldType(/*storage_bits=*/256,
                                           /*montgomery=*/true);
  mlir::Type i32 = lb.getIntegerType(32);

  mlir::Value placeholder = block->addArgument(mont_felt, lb.getUnknownLoc());
  mlir::Value result = ConvertField(lb, /*result_types=*/{i32},
                                    /*source_type=*/mont_felt,
                                    /*target_type=*/i32, {placeholder});

  ASSERT_TRUE(result);
  EXPECT_EQ(result.getType(), i32);
  auto trunc = result.getDefiningOp<TruncIOp>();
  ASSERT_TRUE(trunc);
  auto bitcast = trunc.getIn().getDefiningOp<BitcastOp>();
  ASSERT_TRUE(bitcast);
  auto from_mont = bitcast.getInput().getDefiningOp<FromMontOp>();
  ASSERT_TRUE(from_mont) << "Montgomery source must be normalized via from_mont";
  EXPECT_EQ(from_mont.getInput(), placeholder);
}

}  // namespace
}  // namespace zkx::mlir_utils
