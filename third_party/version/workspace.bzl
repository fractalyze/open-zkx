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

"""Provides the repo macro to import version.

version generates c++ header file for the version number.
"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    VERSION_COMMIT = "022bd9f7851643966bc3ee0bf2c2fe8795d3488f"
    VERSION_SHA256 = "2e65daa275198d4fa6e2081746a48b70999f5a6c0b533478b5ce77752d5ad54c"
    tf_http_archive(
        name = "version",
        sha256 = VERSION_SHA256,
        strip_prefix = "version-{commit}".format(commit = VERSION_COMMIT),
        urls = tf_mirror_urls("https://github.com/fractalyze/version/archive/{commit}.tar.gz".format(commit = VERSION_COMMIT)),
    )
