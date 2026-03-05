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

load("@rules_cc//cc:defs.bzl", "cc_library")
load("//bazel:zkx_cc.bzl", "zkx_cc_unittest")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "compactptrset",
    hdrs = ["compactptrset.h"],
    deps = [
        ":flatset",
        "@com_google_absl//absl/log:check",
    ],
)

cc_library(
    name = "flatmap",
    hdrs = ["flatmap.h"],
    deps = [
        ":flatrep",
        "//xla/tsl/platform:hash",
        "@com_google_absl//absl/log:check",
    ],
)

cc_library(
    name = "flatrep",
    hdrs = ["flatrep.h"],
    deps = ["@com_google_absl//absl/base:prefetch"],
)

cc_library(
    name = "flatset",
    hdrs = ["flatset.h"],
    deps = [
        ":flatrep",
        "//xla/tsl/platform:hash",
        "@com_google_absl//absl/log:check",
    ],
)

cc_library(
    name = "int_type",
    hdrs = ["int_type.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/hash",
    ],
)

cc_library(
    name = "iterator_range",
    hdrs = ["iterator_range.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/hash",
    ],
)

cc_library(
    name = "map_util",
    hdrs = [
        "map_util.h",
        "//xla/tsl/lib/gtl/subtle:map_traits",
    ],
)

zkx_cc_unittest(
    name = "tsl_lib_gtl_tests",
    srcs = [
        "compactptrset_test.cc",
        "flatmap_test.cc",
        "flatset_test.cc",
        "int_type_test.cc",
        "iterator_range_test.cc",
        "map_util_test.cc",
    ],
    deps = [
        ":compactptrset",
        ":flatmap",
        ":flatset",
        ":int_type",
        ":iterator_range",
        ":map_util",
        "@com_google_absl//absl/log",
    ],
)
