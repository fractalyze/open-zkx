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

#ifndef ZKX_BASE_CONTAINERS_CONTAINER_UTIL_H_
#define ZKX_BASE_CONTAINERS_CONTAINER_UTIL_H_

#include <algorithm>
#include <iterator>
#include <vector>

#include "zkx/base/functional/functor_traits.h"

namespace zkx::base {

template <typename Generator,
          typename FunctorTraits = internal::MakeFunctorTraits<Generator>,
          typename ReturnType = typename FunctorTraits::ReturnType,
          typename RunType = typename FunctorTraits::RunType,
          typename ArgList = internal::ExtractArgs<RunType>,
          size_t ArgNum = internal::GetSize<ArgList>,
          std::enable_if_t<ArgNum == 0>* = nullptr>
std::vector<ReturnType> CreateVector(size_t size, Generator&& generator) {
  std::vector<ReturnType> ret;
  ret.reserve(size);
  std::generate_n(std::back_inserter(ret), size,
                  std::forward<Generator>(generator));
  return ret;
}

template <typename Generator,
          typename FunctorTraits = internal::MakeFunctorTraits<Generator>,
          typename ReturnType = typename FunctorTraits::ReturnType,
          typename RunType = typename FunctorTraits::RunType,
          typename ArgList = internal::ExtractArgs<RunType>,
          size_t ArgNum = internal::GetSize<ArgList>,
          std::enable_if_t<ArgNum == 1>* = nullptr>
std::vector<ReturnType> CreateVector(size_t size, Generator&& generator) {
  std::vector<ReturnType> ret;
  ret.reserve(size);
  size_t idx = 0;
  std::generate_n(
      std::back_inserter(ret), size,
      [&idx, generator = std::forward<Generator>(generator)]() mutable {
        return generator(idx++);
      });
  return ret;
}

template <typename Iterator, typename UnaryOp,
          typename FunctorTraits = internal::MakeFunctorTraits<UnaryOp>,
          typename ReturnType = typename FunctorTraits::ReturnType,
          typename RunType = typename FunctorTraits::RunType,
          typename ArgList = internal::ExtractArgs<RunType>,
          size_t ArgNum = internal::GetSize<ArgList>,
          std::enable_if_t<ArgNum == 1>* = nullptr>
std::vector<ReturnType> Map(Iterator begin, Iterator end, UnaryOp&& op) {
  std::vector<ReturnType> ret;
  ret.reserve(std::distance(begin, end));
  std::transform(begin, end, std::back_inserter(ret),
                 std::forward<UnaryOp>(op));
  return ret;
}

template <typename Iterator, typename UnaryOp,
          typename FunctorTraits = internal::MakeFunctorTraits<UnaryOp>,
          typename ReturnType = typename FunctorTraits::ReturnType,
          typename RunType = typename FunctorTraits::RunType,
          typename ArgList = internal::ExtractArgs<RunType>,
          size_t ArgNum = internal::GetSize<ArgList>,
          std::enable_if_t<ArgNum == 2>* = nullptr>
std::vector<ReturnType> Map(Iterator begin, Iterator end, UnaryOp&& op) {
  std::vector<ReturnType> ret;
  ret.reserve(std::distance(begin, end));
  size_t idx = 0;
  std::transform(begin, end, std::back_inserter(ret),
                 [&idx, op = std::forward<UnaryOp>(op)](auto& item) mutable {
                   return op(idx++, item);
                 });
  return ret;
}

template <typename Container, typename UnaryOp>
auto Map(Container&& container, UnaryOp&& op) {
  return Map(std::begin(container), std::end(container),
             std::forward<UnaryOp>(op));
}

}  // namespace zkx::base

#endif  // ZKX_BASE_CONTAINERS_CONTAINER_UTIL_H_
