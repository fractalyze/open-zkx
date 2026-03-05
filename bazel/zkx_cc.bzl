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

"""ZKX cc rules."""

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load(
    "//bazel:zkx.bzl",
    "if_has_exception",
    "if_has_openmp",
)

def zkx_safe_code():
    return [
        "-Wall",
        "-Werror",
        "-Wno-invalid-offsetof",
        "-Wno-deprecated-declarations",
        "-Wno-nullability-completeness",
    ]

def zkx_warnings(safe_code):
    warnings = []
    if safe_code:
        warnings.extend(zkx_safe_code())
    return warnings

def zkx_exceptions(force_exceptions):
    return if_has_exception(["-fexceptions"], (["-fexceptions"] if force_exceptions else ["-fno-exceptions"]))

def zkx_copts(safe_code = True):
    return zkx_warnings(safe_code)

def zkx_cxxopts(safe_code = True, force_exceptions = False):
    return zkx_copts(safe_code) + zkx_exceptions(force_exceptions)

def zkx_openmp_defines():
    return if_has_openmp(["ZKX_HAS_OPENMP"])

def zkx_defines():
    return []

def zkx_local_defines():
    return []

def zkx_openmp_linkopts():
    return select({
        "//:zkx_has_openmp_on_macos": ["-Xclang -fopenmp"],
        "//:zkx_has_openmp": ["-fopenmp"],
        "//:zkx_has_intel_openmp": ["-liomp5"],
        "//conditions:default": [],
    })

def zkx_linkopts():
    return []

def zkx_openmp_num_threads_env(n):
    return if_has_openmp({
        "OMP_NUM_THREADS": "{}".format(n),
    }, {})

def zkx_cc_library(
        name,
        copts = [],
        defines = [],
        local_defines = [],
        linkopts = [],
        alwayslink = True,
        safe_code = True,
        force_exceptions = False,
        **kwargs):
    cc_library(
        name = name,
        copts = copts + zkx_cxxopts(safe_code = safe_code, force_exceptions = force_exceptions),
        defines = defines + zkx_defines(),
        local_defines = local_defines + zkx_local_defines(),
        linkopts = linkopts + zkx_linkopts(),
        alwayslink = alwayslink,
        **kwargs
    )

def zkx_cc_binary(
        name,
        copts = [],
        defines = [],
        local_defines = [],
        linkopts = [],
        safe_code = True,
        force_exceptions = False,
        **kwargs):
    cc_binary(
        name = name,
        copts = copts + zkx_cxxopts(safe_code = safe_code, force_exceptions = force_exceptions),
        defines = defines + zkx_defines(),
        local_defines = local_defines + zkx_local_defines(),
        linkopts = linkopts + zkx_linkopts(),
        **kwargs
    )

def zkx_cc_test(
        name,
        copts = [],
        defines = [],
        local_defines = [],
        linkopts = [],
        linkstatic = True,
        deps = [],
        safe_code = True,
        force_exceptions = False,
        **kwargs):
    cc_test(
        name = name,
        copts = copts + zkx_cxxopts(safe_code = safe_code, force_exceptions = force_exceptions),
        defines = defines + zkx_defines(),
        local_defines = local_defines + zkx_local_defines(),
        linkopts = linkopts + zkx_linkopts(),
        linkstatic = linkstatic,
        deps = deps + ["@com_google_googletest//:gtest_main"],
        **kwargs
    )

def zkx_cc_unittest(
        name,
        size = "small",
        **kwargs):
    zkx_cc_test(
        name = name,
        size = size,
        **kwargs
    )
