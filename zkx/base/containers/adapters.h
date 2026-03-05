// Copyright 2014 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE.chromium file.

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

#ifndef ZKX_BASE_CONTAINERS_ADAPTERS_H_
#define ZKX_BASE_CONTAINERS_ADAPTERS_H_

#include <iterator>
#include <type_traits>

namespace zkx::base {
namespace internal {

// Internal adapter class for implementing zkx::base::Reversed.
template <typename T>
class ReversedAdapter {
 public:
  explicit ReversedAdapter(T& t) : t_(t) {}
  ReversedAdapter(const ReversedAdapter& ra) : t_(ra.t_) {}

  ReversedAdapter& operator=(const ReversedAdapter& ra) = delete;

  auto begin() const {
    if constexpr (std::is_const_v<T>) {
      return std::crbegin(t_);
    } else {
      return std::rbegin(t_);
    }
  }
  auto end() const {
    if constexpr (std::is_const_v<T>) {
      return std::crend(t_);
    } else {
      return std::rend(t_);
    }
  }

 private:
  T& t_;
};

}  // namespace internal

// Reversed returns a container adapter usable in a range-based "for" statement
// for iterating a reversible container in reverse order.
//
// Example:
//
//   std::vector<int> v = {1, 2, 3, 4, 5};
//   for (int i : zkx::base::Reversed(v)) {
//     // iterates through v from back to front
//     // 5, 4, 3, 2, 1
//   }
template <typename T>
internal::ReversedAdapter<T> Reversed(T& t) {
  return internal::ReversedAdapter<T>(t);
}

}  // namespace zkx::base

#endif  // ZKX_BASE_CONTAINERS_ADAPTERS_H_
