/* Copyright 2019 The OpenXLA Authors.
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

#include "zkx/comparison_util.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace zkx {

using ::testing::Eq;

TEST(Comparison, IntegersDefaultToTotalOrder) {
  EXPECT_EQ(
      Comparison(Comparison::Direction::kGe, PrimitiveType::S32).GetOrder(),
      Comparison::Order::kTotal);
  EXPECT_EQ(
      Comparison(Comparison::Direction::kGe, PrimitiveType::U8).GetOrder(),
      Comparison::Order::kTotal);
  EXPECT_EQ(
      Comparison(Comparison::Direction::kGe, PrimitiveType::PRED).GetOrder(),
      Comparison::Order::kTotal);
}

TEST(Comparison, TotalOrderReflexivity) {
  EXPECT_TRUE(Comparison(Comparison::Direction::kLe, PrimitiveType::U16,
                         Comparison::Order::kTotal)
                  .IsReflexive());
  EXPECT_TRUE(Comparison(Comparison::Direction::kGe, PrimitiveType::U32,
                         Comparison::Order::kTotal)
                  .IsReflexive());
  EXPECT_TRUE(
      Comparison(Comparison::Direction::kEq, PrimitiveType::S32).IsReflexive());

  EXPECT_FALSE(Comparison(Comparison::Direction::kNe, PrimitiveType::U32,
                          Comparison::Order::kTotal)
                   .IsReflexive());
  EXPECT_FALSE(Comparison(Comparison::Direction::kLt, PrimitiveType::U64,
                          Comparison::Order::kTotal)
                   .IsReflexive());
}

TEST(Comparison, TotalOrderAntiReflexivity) {
  EXPECT_TRUE(Comparison(Comparison::Direction::kNe, PrimitiveType::U16,
                         Comparison::Order::kTotal)
                  .IsAntireflexive());
  EXPECT_TRUE(Comparison(Comparison::Direction::kNe, PrimitiveType::S32)
                  .IsAntireflexive());

  EXPECT_FALSE(Comparison(Comparison::Direction::kEq, PrimitiveType::U32,
                          Comparison::Order::kTotal)
                   .IsAntireflexive());
  EXPECT_FALSE(Comparison(Comparison::Direction::kLe, PrimitiveType::U64,
                          Comparison::Order::kTotal)
                   .IsAntireflexive());
  EXPECT_FALSE(Comparison(Comparison::Direction::kLe, PrimitiveType::S8)
                   .IsAntireflexive());
}

TEST(Comparison, Converse) {
  EXPECT_THAT(
      Comparison(Comparison::Direction::kLe, PrimitiveType::S8).Converse(),
      Eq(Comparison(Comparison::Direction::kGe, PrimitiveType::S8)));

  EXPECT_THAT(
      Comparison(Comparison::Direction::kEq, PrimitiveType::U16).Converse(),
      Eq(Comparison(Comparison::Direction::kEq, PrimitiveType::U16)));

  EXPECT_THAT(
      Comparison(Comparison::Direction::kGt, PrimitiveType::U32).Converse(),
      Eq(Comparison(Comparison::Direction::kLt, PrimitiveType::U32)));
}

TEST(Comparison, Inverse) {
  EXPECT_THAT(
      *Comparison(Comparison::Direction::kLe, PrimitiveType::S64).Inverse(),
      Eq(Comparison(Comparison::Direction::kGt, PrimitiveType::S64)));

  EXPECT_THAT(
      *Comparison(Comparison::Direction::kEq, PrimitiveType::U16).Inverse(),
      Eq(Comparison(Comparison::Direction::kNe, PrimitiveType::U16)));

  EXPECT_THAT(*Comparison(Comparison::Direction::kGt, PrimitiveType::U32,
                          Comparison::Order::kTotal)
                   .Inverse(),
              Eq(Comparison(Comparison::Direction::kLe, PrimitiveType::U32,
                            Comparison::Order::kTotal)));
}

TEST(Comparison, ToString) {
  EXPECT_EQ(
      Comparison(Comparison::Direction::kLt, PrimitiveType::U32).ToString(),
      ".LT.U32.TOTALORDER");
  EXPECT_EQ(
      Comparison(Comparison::Direction::kEq, PrimitiveType::S8).ToString(),
      ".EQ.S8.TOTALORDER");

  EXPECT_EQ(Comparison(Comparison::Direction::kGe, PrimitiveType::U64)
                .ToString("_1_", "_2_", "_3_"),
            "_1_GE_2_U64_3_TOTALORDER");
}

TEST(Comparison, Compare) {
  EXPECT_TRUE(Comparison(Comparison::Direction::kLt, PrimitiveType::U32)
                  .Compare<uint32_t>(1, 2));

  EXPECT_TRUE(Comparison(Comparison::Direction::kGe, PrimitiveType::U16)
                  .Compare<uint16_t>(2, 1));

  EXPECT_FALSE(Comparison(Comparison::Direction::kNe, PrimitiveType::S64)
                   .Compare<int64_t>(1'000'000, 1'000'000));

  EXPECT_TRUE(Comparison(Comparison::Direction::kEq, PrimitiveType::U8)
                  .Compare<uint8_t>(63, 63));
}

}  // namespace zkx
