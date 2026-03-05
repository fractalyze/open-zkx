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

# Each platform creates a constraint @<platform>//:platform_constraint that
# is listed in its constraint_values; rule that want to select a specific
# platform to run on can put @<platform>//:platform_constraing into their
# exec_compatible_with attribute.
# Toolchains can similarly be marked with target_compatible_with or
# exec_compatible_with to bind them to this platform.
constraint_setting(
    name = "platform_setting"
)

constraint_value(
    name = "platform_constraint",
    constraint_setting = ":platform_setting",
    visibility = ["//visibility:public"],
)

platform(
    name = "platform",
    visibility = ["//visibility:public"],
    constraint_values = [
        "@platforms//cpu:%{cpu}",
        "@platforms//os:%{platform}",
        ":platform_constraint",
    ],
    exec_properties = %{exec_properties},
)
