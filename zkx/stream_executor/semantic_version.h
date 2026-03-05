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

#ifndef ZKX_STREAM_EXECUTOR_SEMANTIC_VERSION_H_
#define ZKX_STREAM_EXECUTOR_SEMANTIC_VERSION_H_

#include <stdint.h>

#include <array>
#include <ostream>
#include <string>
#include <tuple>

#include "absl/hash/hash.h"
#include "absl/status/statusor.h"

#include "zkx/stream_executor/namespace_alias.h"

namespace stream_executor {

// `SemanticVersion` represents a version number of the form X.Y.Z with
// - X being called the major version,
// - Y being called the minor version,
// - Z being called the patch version.
//
// The type is lexicographically ordered and supports printing and parsing.
class SemanticVersion {
 public:
  constexpr SemanticVersion(uint32_t major, uint32_t minor, uint32_t patch)
      : major_(major), minor_(minor), patch_(patch) {}
  explicit SemanticVersion(std::array<uint32_t, 3> other)
      : major_(other[0]), minor_(other[1]), patch_(other[2]) {}

  static absl::StatusOr<SemanticVersion> ParseFromString(std::string_view str);

  uint32_t major() const { return major_; }

  uint32_t minor() const { return minor_; }

  uint32_t patch() const { return patch_; }

  std::string ToString() const;

  friend bool operator==(SemanticVersion lhs, SemanticVersion rhs) {
    return std::tie(lhs.major_, lhs.minor_, lhs.patch_) ==
           std::tie(rhs.major_, rhs.minor_, rhs.patch_);
  }
  friend bool operator<(SemanticVersion lhs, SemanticVersion rhs) {
    return std::tie(lhs.major_, lhs.minor_, lhs.patch_) <
           std::tie(rhs.major_, rhs.minor_, rhs.patch_);
  }
  friend bool operator!=(SemanticVersion lhs, SemanticVersion rhs) {
    return !(lhs == rhs);
  }
  friend bool operator>(SemanticVersion lhs, SemanticVersion rhs) {
    return rhs < lhs;
  }
  friend bool operator>=(SemanticVersion lhs, SemanticVersion rhs) {
    return !(lhs < rhs);
  }
  friend bool operator<=(SemanticVersion lhs, SemanticVersion rhs) {
    return !(lhs > rhs);
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, SemanticVersion version) {
    sink.Append(version.ToString());
  }

  template <typename H>
  friend H AbslHashValue(H h, SemanticVersion v) {
    return H::combine(std::move(h), v.major_, v.minor_, v.patch_);
  }

  friend std::ostream& operator<<(std::ostream& os, SemanticVersion version) {
    return os << version.ToString();
  }

 private:
  uint32_t major_;
  uint32_t minor_;
  uint32_t patch_;
};

}  // namespace stream_executor

#endif  // ZKX_STREAM_EXECUTOR_SEMANTIC_VERSION_H_
