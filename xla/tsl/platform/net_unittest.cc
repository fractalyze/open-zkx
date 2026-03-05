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

#include "xla/tsl/platform/net.h"

#include "gtest/gtest.h"

namespace tsl::internal {

TEST(Net, PickUnusedPortOrDie) {
  int port0 = PickUnusedPortOrDie();
  int port1 = PickUnusedPortOrDie();
  ASSERT_GE(port0, 0);
  ASSERT_LT(port0, 65536);
  ASSERT_GE(port1, 0);
  ASSERT_LT(port1, 65536);
  ASSERT_NE(port0, port1);
}

}  // namespace tsl::internal
