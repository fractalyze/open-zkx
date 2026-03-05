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

#include "zkx/util.h"

#include <vector>

#include "gtest/gtest.h"

namespace zkx {

TEST(UtilTest, CommonFactors) {
  struct {
    std::vector<int64_t> a, b;
    absl::InlinedVector<std::pair<int64_t, int64_t>, 8> expected;
  } test_cases[] = {
      {
          .a = {0},
          .b = {0},
          .expected = {{0, 0}, {1, 1}},
      },
      {
          .a = {1},
          .b = {},
          .expected = {{0, 0}, {1, 0}},
      },
      {
          .a = {},
          .b = {1},
          .expected = {{0, 0}, {0, 1}},
      },
      {
          .a = {0, 10},
          .b = {0, 10, 3},
          .expected = {{0, 0}, {2, 3}},
      },
      {
          .a = {1, 0},
          .b = {1, 0, 1},
          .expected = {{0, 0}, {1, 1}, {2, 2}, {2, 3}},
      },
      {
          .a = {0, 1},
          .b = {0, 1},
          .expected = {{0, 0}, {1, 1}, {2, 2}},
      },
      {
          .a = {},
          .b = {},
          .expected = {{0, 0}},
      },
      {
          .a = {2, 5, 1, 3},
          .b = {1, 10, 3, 1},
          .expected = {{0, 0}, {0, 1}, {2, 2}, {3, 2}, {4, 3}, {4, 4}},
      },
      {
          .a = {1, 1, 3},
          .b = {1, 1, 3},
          .expected = {{0, 0}, {1, 1}, {2, 2}, {3, 3}},
      },
      // Splitting and combining dimensions.
      {
          .a = {2, 6},
          .b = {4, 3},
          .expected = {{0, 0}, {2, 2}},
      },
      {
          .a = {1, 2, 6},
          .b = {4, 1, 3, 1},
          .expected = {{0, 0}, {1, 0}, {3, 3}, {3, 4}},
      },
      // Extra degenerated dimension (second and third dims in the output) forms
      // single common factor group.
      {
          .a = {1, 2, 1},
          .b = {1, 1, 1, 2},
          .expected = {{0, 0}, {1, 1}, {1, 2}, {1, 3}, {2, 4}, {3, 4}},
      },
  };
  for (const auto& test_case : test_cases) {
    EXPECT_EQ(test_case.expected, CommonFactors(test_case.a, test_case.b));
  }
}

TEST(UtilTest, SanitizeFileName) {
  EXPECT_EQ(SanitizeFileName(""), "");
  EXPECT_EQ(SanitizeFileName("abc"), "abc");
  EXPECT_EQ(SanitizeFileName("/\\[]"), "____");
  EXPECT_EQ(SanitizeFileName("/A\\B[C]"), "_A_B_C_");
}

namespace {

void PackInt4(absl::Span<const char> input, absl::Span<char> output) {
  ASSERT_EQ(output.size(), CeilOfRatio(input.size(), size_t{2}));
  for (size_t i = 0; i < input.size(); ++i) {
    // Mask out the high-order 4 bits in case they have extraneous data.
    char val = input[i] & 0xf;
    if (i % 2 == 0) {
      output[i / 2] = val << 4;
    } else {
      output[i / 2] |= val;
    }
  }
}

}  // namespace

TEST(UtilTest, PackInt4) {
  std::vector<char> input(7);
  std::iota(input.begin(), input.end(), 0);

  std::vector<char> output_ref(CeilOfRatio<int64_t>(input.size(), 2));
  PackInt4(input, absl::MakeSpan(output_ref));

  std::vector<char> output_dut(CeilOfRatio<int64_t>(input.size(), 2));
  PackIntN(4, input, absl::MakeSpan(output_dut));
  for (size_t i = 0; i < output_dut.size(); ++i) {
    EXPECT_EQ(output_ref[i], output_dut[i]) << i;
  }

  std::vector<char> unpacked(input.size());
  UnpackIntN(4, output_ref, absl::MakeSpan(unpacked));
  for (size_t i = 0; i < input.size(); ++i) {
    EXPECT_EQ(unpacked[i], input[i]) << i;
  }
}

}  // namespace zkx
