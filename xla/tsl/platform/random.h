/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

#ifndef XLA_TSL_PLATFORM_RANDOM_H_
#define XLA_TSL_PLATFORM_RANDOM_H_

#include <stdint.h>

namespace tsl::random {

// Return a 64-bit random value.  Different sequences are generated
// in different processes.
uint64_t New64();

// Same as previous method, but uses a different RNG for each thread.
uint64_t ThreadLocalNew64();

// Return a 64-bit random value. Uses
// std::mersenne_twister_engine::default_seed as seed value.
uint64_t New64DefaultSeed();

}  // namespace tsl::random

#endif  // XLA_TSL_PLATFORM_RANDOM_H_
