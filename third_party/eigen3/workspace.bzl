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

"""Provides the repository macro to import Eigen."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Eigen."""

    # Attention: tools parse and update these lines.
    # LINT.IfChange
    EIGEN_COMMIT = "4c38131a16803130b66266a912029504f2cf23cd"
    EIGEN_SHA256 = "1a432ccbd597ea7b9faa1557b1752328d6adc1a3db8969f6fe793ff704be3bf0"
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/eigen.cmake)

    tf_http_archive(
        name = "eigen_archive",
        build_file = "//third_party/eigen3:eigen_archive.BUILD",
        sha256 = EIGEN_SHA256,
        strip_prefix = "eigen-{commit}".format(commit = EIGEN_COMMIT),
        urls = tf_mirror_urls("https://gitlab.com/libeigen/eigen/-/archive/{commit}/eigen-{commit}.tar.gz".format(commit = EIGEN_COMMIT)),
    )
