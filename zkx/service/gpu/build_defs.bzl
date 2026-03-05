# Copyright The OpenXLA Authors.
# Copyright 2025 The ZKX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""" GPU-specific build macros.
"""

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library", "if_cuda_is_configured")

def gpu_kernel_library(name, copts = [], local_defines = [], tags = [], **kwargs):
    cuda_library(
        name = name + "_cuda",
        local_defines = local_defines + if_cuda_is_configured(["GOOGLE_CUDA=1"]),
        copts = copts,
        tags = ["manual"] + tags,
        **kwargs
    )

    native.alias(
        name = name,
        actual = "%s_cuda" % name,
        tags = ["gpu"] + tags,
    )
