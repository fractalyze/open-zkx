/* Copyright 2024 The OpenXLA Authors.
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

#include "zkx/hlo/ir/backend_config.h"

#include <memory>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "absl/synchronization/notification.h"
#include "gtest/gtest.h"

namespace zkx {

const int kNumThreads = 100;
const int kNumRepetitions = 100;

// This string has to be in a canonical form (without spaces and new lines)
// since the == operator does not canonicalize the raw strings before comparing
// them.
constexpr std::string_view kRawString =
    R"({"operation_queue_id":"0","wait_on_operation_queues":[],"fusion_backend_config":{"kind":"__triton_gemm","triton_gemm_config":{"block_m":"256","block_n":"256","block_k":"32","split_k":"1","num_stages":"1","num_warps":"16","num_ctas":"1"}},"force_earliest_schedule":false})";

template <typename Input, typename CheckFn>
void RunThreaded(Input input, CheckFn check_fn) {
  for (int i = 0; i < kNumRepetitions; ++i) {
    BackendConfigWrapper source(input);

    absl::Notification all_threads_created;
    std::vector<std::unique_ptr<std::thread>> threads;

    for (int i = 0; i < kNumThreads; ++i) {
      threads.emplace_back(std::make_unique<std::thread>([&] {
        all_threads_created.WaitForNotification();
        check_fn(source);
      }));
    }
    all_threads_created.Notify();

    for (int i = 0; i < kNumThreads; ++i) {
      threads[i]->join();
    }
  }
}

TEST(BackendConfigWrapperTest, AssignmentToNonEmptyIsOK) {
  BackendConfigWrapper a(std::string{kRawString});
  BackendConfigWrapper b(std::string{kRawString});
  a = std::move(b);
  EXPECT_TRUE(a == BackendConfigWrapper(std::string{kRawString}));
}

TEST(BackendConfigWrapperTest, AssignmentDoesNotDeadlock) {
  BackendConfigWrapper source;
  BackendConfigWrapper& ref = source;
  source = std::move(ref);
}

TEST(BackendConfigWrapperTest, SelfComparisonDoesNotDeadlock) {
  BackendConfigWrapper source(std::string{kRawString});
  EXPECT_TRUE(source == source);
}

}  // namespace zkx
