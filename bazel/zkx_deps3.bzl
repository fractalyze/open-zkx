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

"""ZKX dependencies 3."""

load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")
load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
load("//third_party/llvm:setup.bzl", "llvm_setup")

def zkx_deps3():
    """ZKX dependencies."""

    # Load the raw llvm-project. llvm does not have build rules set up by default,
    # but provides a script for setting up build rules via overlays.
    llvm_setup(name = "llvm-project")

    grpc_extra_deps()
    protobuf_deps()
