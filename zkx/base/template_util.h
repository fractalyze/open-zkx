/* Copyright 2025 The ZKX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef ZKX_BASE_TEMPLATE_UTIL_H_
#define ZKX_BASE_TEMPLATE_UTIL_H_

#include <type_traits>
#include <utility>

namespace zkx::base::internal {

template <typename, typename = std::void_t<>>
struct has_resize : std::false_type {};

template <typename T>
struct has_resize<T, std::void_t<decltype(std::declval<T&>().resize(
                         std::declval<std::size_t>()))>> : std::true_type {};

template <typename T>
constexpr bool has_resize_v = has_resize<T>::value;

}  // namespace zkx::base::internal

#endif  // ZKX_BASE_TEMPLATE_UTIL_H_
