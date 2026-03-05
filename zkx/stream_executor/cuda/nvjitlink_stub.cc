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

#include "zkx/stream_executor/cuda/nvjitlink.h"

namespace stream_executor {

absl::StatusOr<NvJitLinkVersion> GetNvJitLinkVersion() {
  return absl::UnimplementedError("libnvjitlink is not supported");
}

absl::StatusOr<std::vector<uint8_t>> CompileAndLinkUsingLibNvJitLink(
    const CudaComputeCapability& cc, absl::Span<const NvJitLinkInput> inputs,
    const GpuAsmOpts& options, bool cancel_if_reg_spill) {
  return absl::UnimplementedError("libnvjitlink is not supported");
}

}  // namespace stream_executor
