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

#ifndef XLA_TSL_PLATFORM_TEST_UTIL_H_
#define XLA_TSL_PLATFORM_TEST_UTIL_H_

#include <string>

namespace tsl::testing {

// Return a temporary directory suitable for temporary testing files.
//
// Where possible, consider using Env::LocalTempFilename over this function.
std::string TmpDir();

// Returns the path to ZKX in the directory containing data dependencies.
std::string ZkxSrcRoot();

// Returns the path to XLA in the directory containing data dependencies.
std::string XlaSrcRoot();

// Returns the path to TSL in the directory containing data dependencies.
std::string TslSrcRoot();

// Small helper library to access "data" dependencies defined in BUILD files.
std::string GetDataDependencyFilepath(std::string_view relative_path);

// Return a random number generator seed to use in randomized tests.
// Returns the same value for the lifetime of the process.
int RandomSeed();

}  // namespace tsl::testing

#endif  // XLA_TSL_PLATFORM_TEST_UTIL_H_
