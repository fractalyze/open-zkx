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

#ifndef XLA_TSL_PLATFORM_STATUS_H_
#define XLA_TSL_PLATFORM_STATUS_H_

#include <stddef.h>

#include <functional>
#include <initializer_list>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/cord.h"

namespace tsl {

// Helper class to manage multiple child status values.
class StatusGroup {
 public:
  StatusGroup() = default;
  // Constructor to form a StatusGroup from any N set of Status arguments.
  // Usage: StatusGroup({status_a, status_b, status_c});
  StatusGroup(std::initializer_list<absl::Status> statuses);

  // Utility function to mark a Status as derived. By marking derived status,
  // Derived status messages are ignored when reporting errors to end users.
  static absl::Status MakeDerived(const absl::Status& s);
  static bool IsDerived(const absl::Status& s);

  // Enable warning and error log collection for appending to the aggregated
  // status. This function may be called more than once.
  static void ConfigureLogHistory();

  // Returns merged payloads of all statuses. In case multiple statuses have the
  // same payload key, non-derived statuses have priority over derived ones,
  // otherwise one payload value will be chosen in an unspecified but
  // deterministic order.
  // NOTE: The payload marking derived statuses as derived will not be returned.
  std::unordered_map<std::string, absl::Cord> GetPayloads() const;

  // Return a merged status with combined child status messages with a summary.
  absl::Status as_summary_status() const;
  // Return a merged status with combined child status messages with
  // concatenation.
  absl::Status as_concatenated_status() const;

  bool ok() const { return ok_; }

  // Augment this group with the child status `status`.
  void Update(const absl::Status& status);

  // Attach recent warning and error log messages
  void AttachLogMessages();
  bool HasLogMessages() const { return !recent_logs_.empty(); }

 private:
  bool ok_ = true;
  size_t num_ok_ = 0;

  // Maintain a sorted collection of statuses.
  struct CompareStatus {
    bool operator()(const absl::Status& a, const absl::Status& b) const {
      return a.ToString() > b.ToString();
    }
  };
  // Using std::set instead of absl::btree_set to keep size for certain
  // dependent libraries under the limit.
  std::set<absl::Status, CompareStatus> derived_;
  std::set<absl::Status, CompareStatus> non_derived_;

  std::vector<std::string> recent_logs_;  // recent warning and error logs
};

typedef std::function<void(const absl::Status&)> StatusCallback;

#define TF_ASSERT_OK(expr) ASSERT_TRUE(expr.ok())
#define TF_EXPECT_OK(expr) EXPECT_TRUE(expr.ok())

}  // namespace tsl

#endif  // XLA_TSL_PLATFORM_STATUS_H_
