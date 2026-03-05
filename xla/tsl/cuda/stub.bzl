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

"""Macros to generate CUDA library stubs from a list of symbols."""

def cuda_stub(name, srcs):
    """Generates a CUDA stub from a list of symbols.

    Generates two files:
    * library.inc, which contains a list of symbols suitable for inclusion by
        C++, and
    * library.tramp.S, which contains assembly-language trampolines for each
      symbol.
    """
    native.genrule(
        name = "{}_stub_gen".format(name),
        srcs = srcs,
        tools = ["//third_party/implib_so:make_stub"],
        outs = [
            "{}.inc".format(name),
            "{}.tramp.S".format(name),
        ],
        tags = ["gpu"],
        cmd = select({
            "//:linux_aarch64": "$(location //third_party/implib_so:make_stub) $< --outdir $(RULEDIR) --target aarch64",
            "//:linux_x86_64": "$(location //third_party/implib_so:make_stub) $< --outdir $(RULEDIR) --target x86_64",
            "//:linux_ppc64le": "$(location //third_party/implib_so:make_stub) $< --outdir $(RULEDIR) --target powerpc64le",
            "//conditions:default": "NOT_IMPLEMENTED_FOR_THIS_PLATFORM_OR_ARCHITECTURE",
        }),
    )
