/* Copyright 2015 The OpenXLA Authors.
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

// CUDA userspace driver library wrapper functionality.

#ifndef ZKX_STREAM_EXECUTOR_CUDA_CUDA_CONTEXT_H_
#define ZKX_STREAM_EXECUTOR_CUDA_CUDA_CONTEXT_H_

#include "absl/status/statusor.h"
#include "third_party/gpus/cuda/include/cuda.h"

#include "zkx/stream_executor/gpu/context.h"
#include "zkx/stream_executor/gpu/context_map.h"

namespace stream_executor::gpu {

// CudaContext implements the Context class for CUDA GPUs.
class CudaContext : public Context {
 public:
  CudaContext(CUcontext context, int device_ordinal)
      : context_(context), device_ordinal_(device_ordinal) {}
  ~CudaContext() override;

  void SetActive() override;
  bool IsActive() const override;
  CUcontext context() const { return context_; }
  int device_ordinal() const override { return device_ordinal_; }
  absl::Status Synchronize() override;

  // Disallow copying and moving.
  CudaContext(CudaContext&&) = delete;
  CudaContext(const CudaContext&) = delete;
  CudaContext& operator=(CudaContext&&) = delete;
  CudaContext& operator=(const CudaContext&) = delete;

  // Returns a new context for the given device.
  static absl::StatusOr<CudaContext*> Create(int device_ordinal,
                                             CUdevice device);

  // Returns the context map for all ZKX-known CUDA contexts.
  static ContextMap<CUcontext, CudaContext>* GetContextMap();

 private:
  CUcontext const context_;  // not owned
  const int device_ordinal_;
};

}  // namespace stream_executor::gpu

#endif  // ZKX_STREAM_EXECUTOR_CUDA_CUDA_CONTEXT_H_
