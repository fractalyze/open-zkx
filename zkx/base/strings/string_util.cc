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

#include "zkx/base/strings/string_util.h"

#include "absl/log/check.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/strip.h"

namespace zkx::base {
namespace {

constexpr const char* k0x = "0x";

}  // namespace

std::string_view BoolToString(bool b) { return b ? "true" : "false"; }

std::string ToHexStringWithLeadingZero(const std::string& str, size_t num) {
  CHECK_GE(num, str.size());
  std::string_view sv = str;
  if (absl::ConsumePrefix(&sv, k0x)) {
    return absl::StrCat(k0x, std::string(num - sv.size(), '0'), sv);
  } else {
    return absl::StrCat(std::string(num - sv.size(), '0'), sv);
  }
}

std::string MaybePrepend0x(std::string_view str) {
  if (absl::StartsWith(str, k0x)) {
    return std::string(str);
  }
  return absl::StrCat(k0x, str);
}

}  // namespace zkx::base
