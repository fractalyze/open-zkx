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

#ifndef ZKX_BASE_RANDOM_H_
#define ZKX_BASE_RANDOM_H_

#include "absl/algorithm/container.h"
#include "absl/random/random.h"

namespace zkx::base {

absl::BitGen& GetAbslBitGen();

template <typename T>
T Uniform() {
  return absl::Uniform<T>(GetAbslBitGen());
}

template <typename T>
T Uniform(T low, T high) {
  return absl::Uniform<T>(GetAbslBitGen(), low, high);
}

template <typename Iterator>
void Shuffle(Iterator first, Iterator last) {
  absl::c_shuffle(absl::MakeSpan(first, last), GetAbslBitGen());
}

template <typename Container>
void Shuffle(Container& container) {
  absl::c_shuffle(container, GetAbslBitGen());
}

}  // namespace zkx::base

#endif  // ZKX_BASE_RANDOM_H_
