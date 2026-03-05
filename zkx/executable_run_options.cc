/* Copyright 2017 The OpenXLA Authors.
Copyright 2025 The ZKX Authors.

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

#include "zkx/executable_run_options.h"

#include <atomic>

namespace zkx {

RunId::RunId() {
  static std::atomic<int64_t> counter{0};
  data_ = counter.fetch_add(1);
}

std::string RunId::ToString() const {
  return "RunId: " + std::to_string(data_);
}

int64_t RunId::ToInt() const { return data_; }

}  // namespace zkx
