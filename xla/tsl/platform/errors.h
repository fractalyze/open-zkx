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

#ifndef XLA_TSL_PLATFORM_ERRORS_H_
#define XLA_TSL_PLATFORM_ERRORS_H_

#include <string>
#include <unordered_map>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace tsl::errors {
namespace internal {

// Returns all payloads from a Status as a key-value map.
std::unordered_map<std::string, std::string> GetPayloads(
    const absl::Status& status);

// Inserts all given payloads into the given status. Will overwrite existing
// payloads if they exist with the same key.
void InsertPayloads(
    absl::Status& status,
    const std::unordered_map<std::string, std::string>& payloads);

// Copies all payloads from one Status to another. Will overwrite existing
// payloads in the destination if they exist with the same key.
void CopyPayloads(const absl::Status& from, absl::Status& to);

absl::Status Create(
    absl::StatusCode code, std::string_view message,
    const std::unordered_map<std::string, std::string>& payloads);

// Returns a new Status, replacing its message with the given.
absl::Status CreateWithUpdatedMessage(const absl::Status& status,
                                      std::string_view message);

}  // namespace internal

// Maps UNIX errors into a Status.
absl::Status IOError(std::string_view context, int err_number);

// Append some context to an error message.  Each time we append
// context put it on a new line, since it is possible for there
// to be several layers of additional context.
template <typename... Args>
void AppendToMessage(absl::Status* status, Args... args) {
  auto new_status = internal::CreateWithUpdatedMessage(
      *status, absl::StrCat(status->message(), "\n\t", args...));
  internal::CopyPayloads(*status, new_status);
  *status = std::move(new_status);
}

}  // namespace tsl::errors

// For propagating errors when calling a function.
#define TF_RETURN_IF_ERROR(...)              \
  do {                                       \
    absl::Status _status = (__VA_ARGS__);    \
    if (ABSL_PREDICT_FALSE(!_status.ok())) { \
      return _status;                        \
    }                                        \
  } while (0)

#define TF_RETURN_WITH_CONTEXT_IF_ERROR(expr, ...)           \
  do {                                                       \
    absl::Status _status = (expr);                           \
    if (ABSL_PREDICT_FALSE(!_status.ok())) {                 \
      ::tsl::errors::AppendToMessage(&_status, __VA_ARGS__); \
      return _status;                                        \
    }                                                        \
  } while (0)

#endif  // XLA_TSL_PLATFORM_ERRORS_H_
