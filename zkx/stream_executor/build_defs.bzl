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

"""Build defs for stream_executor."""

load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured")
load("@rules_cc//cc:defs.bzl", "cc_library")

def gpu_only_cc_library(name, tags = [], **kwargs):
    """A library that only gets compiled when GPU is configured, otherwise it's an empty target.

    Args:
      name: Name of the target
      tags: Tags being applied to the implementation target
      **kwargs: Accepts all arguments that a `cc_library` would also accept
    """
    if not native.package_name().startswith("zkx/stream_executor"):
        fail("gpu_only_cc_library may only be used in `zkx/stream_executor/...`.")

    cc_library(
        name = "%s_non_gpu" % name,
        tags = ["manual"],
    )
    cc_library(
        name = "%s_gpu_only" % name,
        tags = tags + ["manual"],
        **kwargs
    )
    native.alias(
        name = name,
        actual = if_cuda_is_configured(":%s_gpu_only" % name, ":%s_non_gpu" % name),
        visibility = kwargs.get("visibility"),
        compatible_with = kwargs.get("compatible_with"),
        restricted_to = kwargs.get("restricted_to"),
        target_compatible_with = kwargs.get("target_compatible_with"),
    )
