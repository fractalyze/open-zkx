/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

#include "xla/tsl/platform/str_util.h"

#include "gtest/gtest.h"

namespace tsl {

void TestConsumeLeadingDigits(std::string_view s, int64_t expected,
                              std::string_view remaining) {
  uint64_t v;
  std::string_view input(s);
  if (str_util::ConsumeLeadingDigits(&input, &v)) {
    EXPECT_EQ(v, static_cast<uint64_t>(expected));
    EXPECT_EQ(input, remaining);
  } else {
    EXPECT_LT(expected, 0);
    EXPECT_EQ(input, remaining);
  }
}

TEST(ConsumeLeadingDigits, Basic) {
  TestConsumeLeadingDigits("123", 123, "");
  TestConsumeLeadingDigits("a123", -1, "a123");
  TestConsumeLeadingDigits("9_", 9, "_");
  TestConsumeLeadingDigits("11111111111xyz", 11111111111ll, "xyz");

  // Overflow case
  TestConsumeLeadingDigits("1111111111111111111111111111111xyz", -1,
                           "1111111111111111111111111111111xyz");

  // 2^64
  TestConsumeLeadingDigits("18446744073709551616xyz", -1,
                           "18446744073709551616xyz");
  // 2^64-1
  TestConsumeLeadingDigits("18446744073709551615xyz", 18446744073709551615ull,
                           "xyz");
  // (2^64-1)*10+9
  TestConsumeLeadingDigits("184467440737095516159yz", -1,
                           "184467440737095516159yz");
}

}  // namespace tsl
