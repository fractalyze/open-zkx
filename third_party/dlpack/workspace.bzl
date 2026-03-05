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

"""DLPack is a protocol for sharing arrays between deep learning frameworks."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "dlpack",
        strip_prefix = "dlpack-2a7e9f1256ddc48186c86dff7a00e189b47e5310",
        sha256 = "044d2f5738e677c5f0f1ff9fb616a0245af67d09e42ae3514c73ba50cea0e4a5",
        urls = tf_mirror_urls("https://github.com/dmlc/dlpack/archive/2a7e9f1256ddc48186c86dff7a00e189b47e5310.tar.gz"),
        build_file = "//third_party/dlpack:dlpack.BUILD",
    )
