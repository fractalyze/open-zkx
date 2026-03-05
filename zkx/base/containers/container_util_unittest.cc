/* Copyright 2025 The ZKX Authors.

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

#include "zkx/base/containers/container_util.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace zkx::base {

TEST(ContainerUtilTest, CreateVector) {
  auto result1 = CreateVector(3, []() { return 42; });
  EXPECT_THAT(result1, testing::ContainerEq(std::vector<int>{42, 42, 42}));

  auto result2 =
      CreateVector(4, [](size_t idx) { return static_cast<int>(idx * 2); });
  EXPECT_THAT(result2, testing::ContainerEq(std::vector<int>{0, 2, 4, 6}));
}

TEST(ContainerUtilTest, Map) {
  std::vector<int> arr({1, 2, 3});
  EXPECT_THAT(Map(arr.begin(), arr.end(),
                  [](int v) { return static_cast<double>(v * 2); }),
              testing::ContainerEq(std::vector<double>{2.0, 4.0, 6.0}));
  EXPECT_THAT(Map(arr, [](int v) { return static_cast<double>(v * 2); }),
              testing::ContainerEq(std::vector<double>{2.0, 4.0, 6.0}));
}

TEST(ContainerUtilTest, MapWithIdx) {
  std::vector<int> arr({1, 2, 3});
  EXPECT_THAT(
      Map(arr.begin(), arr.end(),
          [](size_t idx, int v) { return static_cast<double>(v * 2 + idx); }),
      testing::ContainerEq(std::vector<double>{2.0, 5.0, 8.0}));
  EXPECT_THAT(Map(arr, [](size_t idx,
                          int v) { return static_cast<double>(v * 2 + idx); }),
              testing::ContainerEq(std::vector<double>{2.0, 5.0, 8.0}));
}

}  // namespace zkx::base
