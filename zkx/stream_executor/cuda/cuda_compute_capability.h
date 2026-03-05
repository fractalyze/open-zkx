/* Copyright 2025 The OpenXLA Authors.
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

#ifndef ZKX_STREAM_EXECUTOR_CUDA_CUDA_COMPUTE_CAPABILITY_H_
#define ZKX_STREAM_EXECUTOR_CUDA_CUDA_COMPUTE_CAPABILITY_H_

#include <string>
#include <tuple>
#include <utility>

#include "absl/hash/hash.h"  // IWYU pragma: keep
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

#include "zkx/stream_executor/cuda/cuda_compute_capability.pb.h"

namespace stream_executor {

// CUDA compute capability, as reported by the device description.
struct CudaComputeCapability {
  int major = 0;
  int minor = 0;

  // MSVC does not like "PASCAL" symbol.
  enum CudaComputeCapabilities {
    PASCAL_ = 6,
    VOLTA = 7,
    AMPERE = 8,
    HOPPER = 9,
    BLACKWELL = 10
  };

  constexpr CudaComputeCapability() = default;
  constexpr CudaComputeCapability(int major, int minor)
      : major(major), minor(minor) {}

  // Parses the architecture name in the format "major.minor", example: "8.6".
  static absl::StatusOr<CudaComputeCapability> FromString(
      std::string_view cuda_arch_name);

  explicit CudaComputeCapability(const CudaComputeCapabilityProto& proto)
      : major(proto.major()), minor(proto.minor()) {}

  static CudaComputeCapability Volta() {
    return CudaComputeCapability{VOLTA, 0};
  }

  static CudaComputeCapability Ampere() {
    return CudaComputeCapability{AMPERE, 0};
  }

  static CudaComputeCapability Hopper() {
    return CudaComputeCapability{HOPPER, 0};
  }

  static CudaComputeCapability Blackwell() {
    return CudaComputeCapability{BLACKWELL, 0};
  }

  bool IsAtLeast(int other_major, int other_minor = 0) const {
    return IsAtLeast(CudaComputeCapability{other_major, other_minor});
  }

  bool IsAtLeast(const CudaComputeCapability& cc) const {
    return !(*this < cc);
  }

  bool IsAtLeastVolta() const {
    return major >= CudaComputeCapabilities::VOLTA;
  }

  bool IsAtLeastAmpere() const {
    return major >= CudaComputeCapabilities::AMPERE;
  }

  bool IsAtLeastHopper() const {
    return major >= CudaComputeCapabilities::HOPPER;
  }

  bool IsAtLeastBlackwell() const {
    return major >= CudaComputeCapabilities::BLACKWELL;
  }

  bool IsHopper() const { return major == CudaComputeCapabilities::HOPPER; }

  bool operator<(const CudaComputeCapability& other) const {
    return std::tie(major, minor) < std::tie(other.major, other.minor);
  }

  bool operator==(const CudaComputeCapability& other) const {
    return std::tie(major, minor) == std::tie(other.major, other.minor);
  }

  bool operator!=(const CudaComputeCapability& other) const {
    return !(*this == other);
  }

  bool operator>(const CudaComputeCapability& other) const {
    return other < *this;
  }

  bool operator>=(const CudaComputeCapability& other) const {
    return !(*this < other);
  }

  bool operator<=(const CudaComputeCapability& other) const {
    return !(other < *this);
  }

  std::string ToString() const { return absl::StrCat(major, ".", minor); }

  CudaComputeCapabilityProto ToProto() const {
    CudaComputeCapabilityProto proto;
    proto.set_major(major);
    proto.set_minor(minor);
    return proto;
  }

  template <typename H>
  friend H AbslHashValue(H state, const CudaComputeCapability& cc) {
    return H::combine(std::move(state), cc.major, cc.minor);
  }
};

}  // namespace stream_executor

#endif  // ZKX_STREAM_EXECUTOR_CUDA_CUDA_COMPUTE_CAPABILITY_H_
