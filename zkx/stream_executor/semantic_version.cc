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

#include "zkx/stream_executor/semantic_version.h"

#include <vector>

#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"

#include "xla/tsl/platform/statusor.h"

namespace stream_executor {

namespace {

absl::StatusOr<unsigned> ParseUnsignedNumber(std::string_view component) {
  unsigned number;
  if (!absl::SimpleAtoi(component, &number)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("'%s' is not an unsigned number.", component));
  }
  return number;
}

}  // namespace

std::string SemanticVersion::ToString() const {
  return absl::StrFormat("%d.%d.%d", major_, minor_, patch_);
}

absl::StatusOr<SemanticVersion> SemanticVersion::ParseFromString(
    std::string_view str) {
  std::vector<std::string_view> components = absl::StrSplit(str, '.');

  if (components.size() != 3) {
    return absl::InvalidArgumentError(
        "Version does not match the format X.Y.Z");
  }

  uint32_t major, minor, patch;
  TF_ASSIGN_OR_RETURN(major, ParseUnsignedNumber(components[0]));
  TF_ASSIGN_OR_RETURN(minor, ParseUnsignedNumber(components[1]));
  TF_ASSIGN_OR_RETURN(patch, ParseUnsignedNumber(components[2]));
  return SemanticVersion(major, minor, patch);
}

}  // namespace stream_executor
