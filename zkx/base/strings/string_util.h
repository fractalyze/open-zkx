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

#ifndef ZKX_BASE_STRINGS_STRING_UTIL_H_
#define ZKX_BASE_STRINGS_STRING_UTIL_H_

#include <sstream>
#include <string>

namespace zkx::base {

std::string_view BoolToString(bool b);

// Return a string that is left padded with zero so that the size of a hex
// number is a given `num`. For example,
//
//   - `str` = "0x1234" and `num` = 6 => "0x001234"
//   - `str` = "1234" and `num` = 6 => "001234"
std::string ToHexStringWithLeadingZero(std::string_view str, size_t num);

std::string MaybePrepend0x(std::string_view str);

template <typename Container, typename Callback>
std::string StrJoin(const Container& container, Callback&& callback,
                    std::string_view delim = ",", std::string_view prefix = "[",
                    std::string_view suffix = "]") {
  size_t size = std::size(container);

  if (size == 0) return "[]";

  std::stringstream ss;
  ss << prefix;
  for (size_t i = 0; i < size - 1; ++i) {
    callback(ss, container[i]);
    ss << delim;
  }
  callback(ss, container[size - 1]);
  ss << suffix;
  return ss.str();
}

}  // namespace zkx::base

#endif  // ZKX_BASE_STRINGS_STRING_UTIL_H_
