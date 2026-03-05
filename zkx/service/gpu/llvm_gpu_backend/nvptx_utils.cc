/* Copyright 2024 The OpenXLA Authors.
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
#include "zkx/service/gpu/llvm_gpu_backend/nvptx_utils.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

#include "xla/tsl/platform/cuda_root_path.h"

namespace zkx::gpu::nvptx {

std::string CantFindCudaMessage(std::string_view msg,
                                std::string_view zkx_gpu_cuda_data_dir) {
  return absl::StrCat(
      msg, "\nSearched for CUDA in the following directories:\n  ",
      absl::StrJoin(tsl::CandidateCudaRoots(std::string{zkx_gpu_cuda_data_dir}),
                    "\n  "),
      "\nYou can choose the search directory by setting zkx_gpu_cuda_data_dir "
      "in HloModule's DebugOptions.  For most apps, setting the environment "
      "variable ZKX_FLAGS=--zkx_gpu_cuda_data_dir=/path/to/cuda will work.");
}

}  // namespace zkx::gpu::nvptx
