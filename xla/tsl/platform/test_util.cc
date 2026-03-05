/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

#include "xla/tsl/platform/test_util.h"

#include <stdio.h>
#include <stdlib.h>

#include "absl/log/log.h"

#include "xla/tsl/platform/path.h"

namespace tsl {
namespace testing {
namespace {

std::string GetEnvVarOrDie(const char* env_var) {
  const char* value = std::getenv(env_var);
  if (!value) {
    LOG(FATAL) << "Failed to find environment variable:" << env_var;
  }
  return value;
}

}  // namespace

std::string TmpDir() {
  const char* tmp_dir = std::getenv("TEST_TMPDIR");
  if (!tmp_dir) {
    tmp_dir = std::getenv("TMPDIR");
  }
  if (tmp_dir) {
    return tmp_dir;
  }
  LOG(FATAL)  // Crash OK
      << "Failed to find environment variables: TEST_TMPDIR, TMPDIR";

  return tmp_dir;
}

std::string ZkxSrcRoot() {
  std::string workspace = GetEnvVarOrDie("TEST_WORKSPACE");
  std::string srcdir = GetEnvVarOrDie("TEST_SRCDIR");
  return io::JoinPath(srcdir, workspace, "zkx");
}

std::string XlaSrcRoot() {
  std::string workspace = GetEnvVarOrDie("TEST_WORKSPACE");
  std::string srcdir = GetEnvVarOrDie("TEST_SRCDIR");
  return io::JoinPath(srcdir, workspace, "xla");
}

std::string TslSrcRoot() {
  std::string workspace = GetEnvVarOrDie("TEST_WORKSPACE");
  std::string srcdir = GetEnvVarOrDie("TEST_SRCDIR");
  return io::JoinPath(srcdir, workspace, "xla", "tsl");
}

std::string GetDataDependencyFilepath(std::string_view relative_path) {
  std::string workspace = GetEnvVarOrDie("TEST_WORKSPACE");
  std::string srcdir = GetEnvVarOrDie("TEST_SRCDIR");
  return io::JoinPath(srcdir, workspace, relative_path);
}

int RandomSeed() {
  const char* random_seed_str = std::getenv("TEST_RANDOM_SEED");
  int seed;
  if (random_seed_str && std::sscanf(random_seed_str, "%d", &seed) == 1) {
    return seed;
  }
  return 301;
}

}  // namespace testing
}  // namespace tsl
