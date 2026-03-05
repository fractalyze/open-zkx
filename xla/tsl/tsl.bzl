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

"""TSL rules."""

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_shared_library")
load("@rules_python//python:py_library.bzl", "py_library")

# These configs are used to determine whether we should use CUDA tools and libs in cc_libraries.
# They are intended for the OSS builds only.
def if_cuda_tools(if_true, if_false = []):  # buildifier: disable=unused-variable
    """Shorthand for select()'ing on whether we're building with hCUDA tools."""
    return select({"@local_config_cuda//cuda:cuda_tools": if_true, "//conditions:default": if_false})  # copybara:comment_replace return if_false

def if_cuda_libs(if_true, if_false = []):  # buildifier: disable=unused-variable
    """Shorthand for select()'ing on whether we need to include hermetic CUDA libraries."""
    return select({"@local_config_cuda//cuda:cuda_tools_and_libs": if_true, "//conditions:default": if_false})  # copybara:comment_replace return if_false

# Bazel rule for collecting the header files that a target depends on.
def _transitive_hdrs_impl(ctx):
    outputs = _get_transitive_headers([], ctx.attr.deps)
    return DefaultInfo(files = outputs)

_transitive_hdrs = rule(
    attrs = {
        "deps": attr.label_list(
            allow_files = True,
            providers = [CcInfo],
        ),
    },
    implementation = _transitive_hdrs_impl,
)

def transitive_hdrs(name, deps = [], **kwargs):
    _transitive_hdrs(name = name + "_gather", deps = deps)
    native.filegroup(name = name, srcs = [":" + name + "_gather"], **kwargs)

def _get_transitive_headers(hdrs, deps):
    """Obtain the header files for a target and its transitive dependencies.

      Args:
        hdrs: a list of header files
        deps: a list of targets that are direct dependencies

      Returns:
        a collection of the transitive headers
      """
    return depset(
        hdrs,
        transitive = [dep[CcInfo].compilation_context.headers for dep in deps],
    )

# buildozer: disable=function-docstring-args
def tsl_pybind_extension_opensource(
        name,
        srcs,
        module_name = None,  # @unused
        hdrs = [],
        dynamic_deps = [],
        static_deps = [],
        deps = [],
        additional_exported_symbols = [],
        compatible_with = None,
        copts = [],
        data = [],
        defines = [],
        deprecation = None,
        enable_stub_generation = False,  # @unused
        features = [],
        licenses = None,
        linkopts = [],
        pytype_deps = [],
        pytype_srcs = [],
        restricted_to = None,
        srcs_version = "PY3",
        testonly = None,
        visibility = None,
        win_def_file = None):  # @unused
    """Builds a generic Python extension module."""
    p = name.rfind("/")
    if p == -1:
        sname = name
        prefix = ""
    else:
        sname = name[p + 1:]
        prefix = name[:p + 1]
    so_file = "%s%s.so" % (prefix, sname)
    filegroup_name = "%s_filegroup" % name
    pyd_file = "%s%s.pyd" % (prefix, sname)
    exported_symbols = [
        "PyInit_%s" % sname,
    ] + additional_exported_symbols

    exported_symbols_file = "%s-exported-symbols.lds" % name
    version_script_file = "%s-version-script.lds" % name

    exported_symbols_output = "\n".join(["_%s" % symbol for symbol in exported_symbols])
    version_script_output = "\n".join([" %s;" % symbol for symbol in exported_symbols])

    native.genrule(
        name = name + "_exported_symbols",
        outs = [exported_symbols_file],
        cmd = "echo '%s' >$@" % exported_symbols_output,
        output_licenses = ["unencumbered"],
        visibility = ["//visibility:private"],
        testonly = testonly,
    )

    native.genrule(
        name = name + "_version_script",
        outs = [version_script_file],
        cmd = "echo '{global:\n%s\n local: *;};' >$@" % version_script_output,
        output_licenses = ["unencumbered"],
        visibility = ["//visibility:private"],
        testonly = testonly,
    )

    if static_deps:
        cc_library_name = so_file + "_cclib"
        cc_library(
            name = cc_library_name,
            hdrs = hdrs,
            srcs = srcs + hdrs,
            data = data,
            deps = deps,
            compatible_with = compatible_with,
            copts = copts + [
                "-fno-strict-aliasing",
                "-fexceptions",
            ] + select({
                "@platforms//os:windows": [],
                "//conditions:default": [
                    "-fvisibility=hidden",
                ],
            }),
            defines = defines,
            features = features + ["-use_header_modules"],
            restricted_to = restricted_to,
            testonly = testonly,
            visibility = visibility,
        )

        cc_shared_library(
            name = so_file,
            roots = [cc_library_name],
            dynamic_deps = dynamic_deps,
            static_deps = static_deps,
            additional_linker_inputs = [exported_symbols_file, version_script_file],
            compatible_with = compatible_with,
            deprecation = deprecation,
            features = features + ["-use_header_modules"],
            licenses = licenses,
            restricted_to = restricted_to,
            shared_lib_name = so_file,
            testonly = testonly,
            user_link_flags = linkopts + select({
                "@platforms//os:macos": [
                    # TODO: the -w suppresses a wall of harmless warnings about hidden typeinfo symbols
                    # not being exported.  There should be a better way to deal with this.
                    "-Wl,-w",
                    "-Wl,-exported_symbols_list,$(location %s)" % exported_symbols_file,
                ],
                "@platforms//os:windows": [],
                "//conditions:default": [
                    "-Wl,--version-script",
                    "$(location %s)" % version_script_file,
                ],
            }),
            visibility = visibility,
        )

        # cc_shared_library can generate more than one file.
        # Solution to avoid the error "variable '$<' : more than one input file."
        native.filegroup(
            name = filegroup_name,
            srcs = [so_file],
            output_group = "main_shared_library_output",
            testonly = testonly,
        )

    else:
        cc_binary(
            name = so_file,
            srcs = srcs + hdrs,
            data = data,
            copts = copts + [
                "-fno-strict-aliasing",
                "-fexceptions",
            ] + select({
                "@platforms//os:windows": [],
                "//conditions:default": [
                    "-fvisibility=hidden",
                ],
            }),
            linkopts = linkopts + select({
                "@platforms//os:macos": [
                    # TODO: the -w suppresses a wall of harmless warnings about hidden typeinfo symbols
                    # not being exported.  There should be a better way to deal with this.
                    "-Wl,-w",
                    "-Wl,-exported_symbols_list,$(location %s)" % exported_symbols_file,
                ],
                "@platforms//os:windows": [],
                "//conditions:default": [
                    "-Wl,--version-script",
                    "$(location %s)" % version_script_file,
                ],
            }),
            deps = deps + [
                exported_symbols_file,
                version_script_file,
            ],
            defines = defines,
            features = features + ["-use_header_modules"],
            linkshared = 1,
            testonly = testonly,
            licenses = licenses,
            visibility = visibility,
            deprecation = deprecation,
            restricted_to = restricted_to,
            compatible_with = compatible_with,
        )

        # For Windows, emulate the above filegroup with the shared object.
        native.alias(
            name = filegroup_name,
            actual = so_file,
        )

    # For Windows only.
    native.genrule(
        name = name + "_pyd_copy",
        srcs = [filegroup_name],
        outs = [pyd_file],
        cmd = "cp $< $@",
        output_to_bindir = True,
        visibility = visibility,
        deprecation = deprecation,
        restricted_to = restricted_to,
        compatible_with = compatible_with,
        testonly = testonly,
    )

    py_library(
        name = name,
        data = select({
            "@platforms//os:windows": [pyd_file],
            "//conditions:default": [so_file],
        }) + pytype_srcs,
        deps = pytype_deps,
        srcs_version = srcs_version,
        licenses = licenses,
        testonly = testonly,
        visibility = visibility,
        deprecation = deprecation,
        restricted_to = restricted_to,
        compatible_with = compatible_with,
    )
