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
#include "xla/tsl/platform/dso_loader.h"

#include <stdlib.h>

#include <string>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "third_party/gpus/cuda/cuda_config.h"

#include "xla/tsl/platform/load_library.h"
#include "xla/tsl/platform/path.h"
#include "xla/tsl/platform/platform.h"

#if TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#endif

namespace tsl {
namespace internal {
namespace {

std::string GetCudaRtVersion() { return TF_CUDART_VERSION; }
std::string GetCuptiVersion() { return TF_CUPTI_VERSION; }
std::string GetCublasVersion() { return TF_CUBLAS_VERSION; }
std::string GetCusparseVersion() { return TF_CUSPARSE_VERSION; }
std::string GetHipVersion() {
#if TENSORFLOW_USE_ROCM
  return TF_HIPRUNTIME_SOVERSION;
#else   // TENSORFLOW_USE_ROCM
  return "";
#endif  // TENSORFLOW_USE_ROCM
}

std::string GetRocBlasVersion() {
#if TENSORFLOW_USE_ROCM
  return TF_ROCBLAS_SOVERSION;
#else   // TENSORFLOW_USE_ROCM
  return "";
#endif  // TENSORFLOW_USE_ROCM
}

absl::StatusOr<void*> GetDsoHandle(const std::string& name,
                                   const std::string& version) {
  auto filename = tsl::internal::FormatLibraryFileName(name, version);
  void* dso_handle;
  absl::Status status =
      tsl::internal::LoadDynamicLibrary(filename.c_str(), &dso_handle);
  if (status.ok()) {
    VLOG(1) << "Successfully opened dynamic library " << filename;
    return dso_handle;
  }

  auto message = absl::StrCat("Could not load dynamic library '", filename,
                              "'; dlerror: ", status.message());
#if !defined(PLATFORM_WINDOWS)
  if (const char* ld_library_path = getenv("LD_LIBRARY_PATH")) {
    message += absl::StrCat("; LD_LIBRARY_PATH: ", ld_library_path);
  }
#endif
  VLOG(1) << message;
  return absl::Status(absl::StatusCode::kFailedPrecondition, message);
}
}  // namespace

namespace DsoLoader {
absl::StatusOr<void*> GetCudaDriverDsoHandle() {
#if defined(PLATFORM_WINDOWS)
  return GetDsoHandle("nvcuda", "");
#elif defined(__APPLE__)
  // On Mac OS X, CUDA sometimes installs libcuda.dylib instead of
  // libcuda.1.dylib.
  auto handle_or = GetDsoHandle("cuda", "");
  if (handle_or.ok()) {
    return handle_or;
  }
#endif
  return GetDsoHandle("cuda", "1");
}

absl::StatusOr<void*> GetCudaRuntimeDsoHandle() {
  return GetDsoHandle("cudart", GetCudaRtVersion());
}

absl::StatusOr<void*> GetCublasDsoHandle() {
  return GetDsoHandle("cublas", GetCublasVersion());
}

absl::StatusOr<void*> GetCublasLtDsoHandle() {
  return GetDsoHandle("cublasLt", GetCublasVersion());
}

absl::StatusOr<void*> GetCusparseDsoHandle() {
  return GetDsoHandle("cusparse", GetCusparseVersion());
}

absl::StatusOr<void*> GetCuptiDsoHandle() {
  // Load specific version of CUPTI this is built.
  auto status_or_handle = GetDsoHandle("cupti", GetCuptiVersion());
  if (status_or_handle.ok()) return status_or_handle;
  // Load whatever libcupti.so user specified.
  return GetDsoHandle("cupti", "");
}

absl::StatusOr<void*> GetRocblasDsoHandle() {
  return GetDsoHandle("rocblas", GetRocBlasVersion());
}

absl::StatusOr<void*> GetMiopenDsoHandle() {
  return GetDsoHandle("MIOpen", "");
}

absl::StatusOr<void*> GetRocrandDsoHandle() {
  return GetDsoHandle("rocrand", "");
}

absl::StatusOr<void*> GetRoctracerDsoHandle() {
  return GetDsoHandle("roctracer64", "");
}

absl::StatusOr<void*> GetHipsparseDsoHandle() {
  return GetDsoHandle("hipsparse", "");
}

absl::StatusOr<void*> GetHipblasltDsoHandle() {
  return GetDsoHandle("hipblaslt", "");
}

absl::StatusOr<void*> GetHipDsoHandle() {
  return GetDsoHandle("amdhip64", GetHipVersion());
}

}  // namespace DsoLoader

namespace CachedDsoLoader {

absl::StatusOr<void*> GetCudaDriverDsoHandle() {
  static absl::StatusOr<void*> result = DsoLoader::GetCudaDriverDsoHandle();
  return result;
}

absl::StatusOr<void*> GetCudaRuntimeDsoHandle() {
  static absl::StatusOr<void*> result = DsoLoader::GetCudaRuntimeDsoHandle();
  return result;
}

absl::StatusOr<void*> GetCublasDsoHandle() {
  static absl::StatusOr<void*> result = DsoLoader::GetCublasDsoHandle();
  return result;
}

absl::StatusOr<void*> GetCublasLtDsoHandle() {
  static absl::StatusOr<void*> result = DsoLoader::GetCublasLtDsoHandle();
  return result;
}

absl::StatusOr<void*> GetCusparseDsoHandle() {
  static absl::StatusOr<void*> result = DsoLoader::GetCusparseDsoHandle();
  return result;
}

absl::StatusOr<void*> GetCuptiDsoHandle() {
  static absl::StatusOr<void*> result = DsoLoader::GetCuptiDsoHandle();
  return result;
}

absl::StatusOr<void*> GetRocblasDsoHandle() {
  static absl::StatusOr<void*> result = DsoLoader::GetRocblasDsoHandle();
  return result;
}

absl::StatusOr<void*> GetMiopenDsoHandle() {
  static absl::StatusOr<void*> result = DsoLoader::GetMiopenDsoHandle();
  return result;
}

absl::StatusOr<void*> GetRocrandDsoHandle() {
  static absl::StatusOr<void*> result = DsoLoader::GetRocrandDsoHandle();
  return result;
}

absl::StatusOr<void*> GetRoctracerDsoHandle() {
  static absl::StatusOr<void*> result = DsoLoader::GetRoctracerDsoHandle();
  return result;
}

absl::StatusOr<void*> GetHipsparseDsoHandle() {
  static absl::StatusOr<void*> result = DsoLoader::GetHipsparseDsoHandle();
  return result;
}

absl::StatusOr<void*> GetHipblasltDsoHandle() {
  static absl::StatusOr<void*> result = DsoLoader::GetHipblasltDsoHandle();
  return result;
}

absl::StatusOr<void*> GetHipDsoHandle() {
  static absl::StatusOr<void*> result = DsoLoader::GetHipDsoHandle();
  return result;
}

}  // namespace CachedDsoLoader
}  // namespace internal
}  // namespace tsl
