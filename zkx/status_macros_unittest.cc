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

#include "zkx/status_macros.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace zkx {
namespace {

absl::Status RetCheckFail() {
  TF_RET_CHECK(2 > 3);
  return absl::OkStatus();
}

absl::Status RetCheckFailWithExtraMessage() {
  TF_RET_CHECK(2 > 3) << "extra message";
  return absl::OkStatus();
}

absl::Status RetCheckSuccess() {
  TF_RET_CHECK(3 > 2);
  return absl::OkStatus();
}

absl::StatusOr<int> CreateIntSuccessfully() { return 42; }

absl::StatusOr<int> CreateIntUnsuccessfully() {
  return absl::InternalError("foobar");
}

absl::Status ReturnStatusOK() { return absl::OkStatus(); }

absl::Status ReturnStatusError() { return absl::InternalError("foobar"); }

using StatusReturningFunction = std::function<absl::Status()>;

absl::StatusOr<int> CallStatusReturningFunction(
    const StatusReturningFunction& func) {
  TF_RETURN_IF_ERROR(func());
  return 42;
}

}  // namespace

TEST(StatusMacros, RetCheckFailing) {
  absl::Status status = RetCheckFail();
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(status.message(),
              ::testing::ContainsRegex("RET_CHECK failure.*2 > 3"));
}

TEST(StatusMacros, RetCheckFailingWithExtraMessage) {
  absl::Status status = RetCheckFailWithExtraMessage();
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(status.message(),
              ::testing::ContainsRegex("RET_CHECK.*2 > 3 extra message"));
}

TEST(StatusMacros, RetCheckSucceeding) {
  absl::Status status = RetCheckSuccess();
  EXPECT_TRUE(status.ok());
}

TEST(StatusMacros, AssignOrAssertOnOK) {
  TF_ASSERT_OK_AND_ASSIGN(int result, CreateIntSuccessfully());
  EXPECT_EQ(42, result);
}

TEST(StatusMacros, ReturnIfErrorOnOK) {
  absl::StatusOr<int> rc = CallStatusReturningFunction(ReturnStatusOK);
  EXPECT_TRUE(rc.ok());
  EXPECT_EQ(42, std::move(rc).value());
}

TEST(StatusMacros, ReturnIfErrorOnError) {
  absl::StatusOr<int> rc = CallStatusReturningFunction(ReturnStatusError);
  EXPECT_FALSE(rc.ok());
  EXPECT_EQ(rc.status().code(), absl::StatusCode::kInternal);
}

TEST(StatusMacros, AssignOrReturnSuccessfully) {
  absl::Status status = []() {
    TF_ASSIGN_OR_RETURN(int value, CreateIntSuccessfully());
    EXPECT_EQ(value, 42);
    return absl::OkStatus();
  }();
  EXPECT_TRUE(status.ok());
}

TEST(StatusMacros, AssignOrReturnUnsuccessfully) {
  absl::Status status = []() {
    TF_ASSIGN_OR_RETURN(int value, CreateIntUnsuccessfully());
    (void)value;
    return absl::OkStatus();
  }();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
}

}  // namespace zkx
