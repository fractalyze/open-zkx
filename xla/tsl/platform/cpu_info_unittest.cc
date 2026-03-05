/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

#include "xla/tsl/platform/cpu_info.h"

#include "gtest/gtest.h"

namespace tsl::port {

TEST(CPUInfo, GetCurrentCPU) {
  const int cpu = GetCurrentCPU();
#if !defined(__APPLE__)
  // GetCurrentCPU does not currently work on MacOS.
  EXPECT_GE(cpu, 0);
  EXPECT_LT(cpu, NumTotalCPUs());
#else
  static_cast<void>(cpu);
#endif
}

TEST(CPUInfo, TestFeature) {
  // We don't know what the result should be on this platform, so just make
  // sure it's callable.
  const bool has_avx = TestCPUFeature(CPUFeature::AVX);
  static_cast<void>(has_avx);
  const bool has_avx2 = TestCPUFeature(CPUFeature::AVX2);
  static_cast<void>(has_avx2);
}

TEST(CPUInfo, CommonX86CPU) {
  // CPUs from 1999 onwards support SSE.
  if (TestCPUFeature(CPUFeature::SSE)) {
    EXPECT_TRUE(IsX86CPU());
  }
}

TEST(CPUInfo, Aarch64NeoverseV1CPU) {
  if (TestAarch64CPU(Aarch64CPU::ARM_NEOVERSE_V1)) {
    EXPECT_TRUE(IsAarch64CPU());
  }
}

}  // namespace tsl::port
