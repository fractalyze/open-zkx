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

"""ZKX dependencies 2."""

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
)
load("@prime_ir//bazel:prime_ir_deps.bzl", "prime_ir_deps")
load("@version//:lastchange.bzl", "lastchange_setup")
load("//third_party/gpus/cuda/hermetic:cuda_configure.bzl", "cuda_configure")
load(
    "//third_party/gpus/cuda/hermetic:cuda_redist_init_repositories.bzl",
    "cuda_redist_init_repositories",
)
load("//third_party/llvm:workspace.bzl", llvm = "repo")

def zkx_deps2():
    """ZKX dependencies."""

    llvm("llvm-raw")

    lastchange_setup(name = "lastchange")

    grpc_deps()
    prime_ir_deps()

    cuda_redist_init_repositories(
        cuda_redistributions = CUDA_REDISTRIBUTIONS,
    )

    cuda_configure(name = "local_config_cuda")
