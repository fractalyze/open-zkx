// Copyright 2019 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE.chromium file.

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

#include "zkx/base/auto_reset.h"

#include "gtest/gtest.h"

namespace zkx::base {

TEST(AutoReset, Move) {
  int value = 10;
  {
    AutoReset<int> resetter1{&value, 20};
    EXPECT_EQ(20, value);
    {
      value = 15;
      AutoReset<int> resetter2 = std::move(resetter1);
      // Moving to a new resetter does not change the value;
      EXPECT_EQ(15, value);
    }
    // Moved-to `resetter2` is out of scoped, and resets to the original value
    // that was in moved-from `resetter1`.
    EXPECT_EQ(10, value);
    value = 105;
  }
  // Moved-from `resetter1` does not reset to anything.
  EXPECT_EQ(105, value);
}

}  // namespace zkx::base
