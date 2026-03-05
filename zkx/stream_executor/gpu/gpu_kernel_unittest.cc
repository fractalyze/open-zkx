/* Copyright 2023 The OpenXLA Authors.
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

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "gtest/gtest.h"

#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/service/platform_util.h"
#include "zkx/stream_executor/device_memory.h"
#include "zkx/stream_executor/gpu/gpu_test_kernels.h"
#include "zkx/stream_executor/gpu/gpu_test_kernels_fatbin.h"
#include "zkx/stream_executor/kernel_spec.h"
#include "zkx/stream_executor/launch_dim.h"
#include "zkx/stream_executor/platform.h"
#include "zkx/stream_executor/platform_manager.h"
#include "zkx/stream_executor/stream.h"
#include "zkx/stream_executor/stream_executor.h"
#include "zkx/stream_executor/typed_kernel_factory.h"

namespace stream_executor::gpu {
namespace {

using AddI32Kernel =
    TypedKernelFactory<DeviceMemory<int32_t>, DeviceMemory<int32_t>,
                       DeviceMemory<int32_t>>;

class GpuKernelTest : public ::testing::Test {
 public:
  void SetUp() override {
    auto name = absl::AsciiStrToUpper(
        zkx::PlatformUtil::CanonicalPlatformName("gpu").value());
    Platform *platform = PlatformManager::PlatformWithName(name).value();
    executor_ = platform->ExecutorForDevice(0).value();
  }

  void RunAddI32Kernel(const MultiKernelLoaderSpec &spec) {
    TF_ASSERT_OK_AND_ASSIGN(auto stream, executor_->CreateStream());
    TF_ASSERT_OK_AND_ASSIGN(auto add, AddI32Kernel::Create(executor_, spec));

    int64_t length = 4;
    int64_t byte_length = sizeof(int32_t) * length;

    // Prepare arguments: a=1, b=2, c=0
    DeviceMemory<int32_t> a = executor_->AllocateArray<int32_t>(length, 0);
    DeviceMemory<int32_t> b = executor_->AllocateArray<int32_t>(length, 0);
    DeviceMemory<int32_t> c = executor_->AllocateArray<int32_t>(length, 0);

    TF_ASSERT_OK(stream->Memset32(&a, 1, byte_length));
    TF_ASSERT_OK(stream->Memset32(&b, 2, byte_length));
    TF_ASSERT_OK(stream->MemZero(&c, byte_length));

    // Launch kernel.
    ASSERT_TRUE(
        add.Launch(ThreadDim(), BlockDim(4), stream.get(), a, b, c).ok());

    // Copy data back to host.
    std::vector<int32_t> dst(4, 42);
    TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));

    std::vector<int32_t> expected = {3, 3, 3, 3};
    ASSERT_EQ(dst, expected);
  }

  StreamExecutor *executor_;
};

TEST_F(GpuKernelTest, LoadAndRunKernelFromPtx) {
  RunAddI32Kernel(GetAddI32PtxKernelSpec());
}

TEST_F(GpuKernelTest, LoadAndRunKernelFromCubin) {
  MultiKernelLoaderSpec spec(/*arity=*/3);
  TF_ASSERT_OK_AND_ASSIGN(auto binary, GetGpuTestKernelsFatbin());
  spec.AddCudaCubinInMemory(binary, "AddI32");
  RunAddI32Kernel(spec);
}

TEST_F(GpuKernelTest, LoadAndRunKernelFromSymbol) {
  RunAddI32Kernel(GetAddI32KernelSpec());
}

}  // namespace
}  // namespace stream_executor::gpu
