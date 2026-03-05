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

#ifndef XLA_TSL_PLATFORM_FILE_SYSTEM_POSIX_H_
#define XLA_TSL_PLATFORM_FILE_SYSTEM_POSIX_H_

#include "xla/tsl/platform/file_system.h"

namespace tsl {

class FileSystemPosix : public FileSystem {
 public:
  FileSystemPosix() {}

  ~FileSystemPosix() override {}

  TF_USE_FILESYSTEM_METHODS_WITH_NO_TRANSACTION_SUPPORT;

  absl::Status NewRandomAccessFile(
      std::string_view filename, TransactionToken* token,
      std::unique_ptr<RandomAccessFile>* result) override;

  absl::Status NewWritableFile(std::string_view fname, TransactionToken* token,
                               std::unique_ptr<WritableFile>* result) override;

  absl::Status NewAppendableFile(
      std::string_view fname, TransactionToken* token,
      std::unique_ptr<WritableFile>* result) override;

  absl::Status NewReadOnlyMemoryRegionFromFile(
      std::string_view filename, TransactionToken* token,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) override;

  absl::Status FileExists(std::string_view fname,
                          TransactionToken* token) override;

  absl::Status GetChildren(std::string_view dir, TransactionToken* token,
                           std::vector<std::string>* result) override;

  absl::Status Stat(std::string_view fname, TransactionToken* token,
                    FileStatistics* stats) override;

  absl::Status GetMatchingPaths(std::string_view pattern,
                                TransactionToken* token,
                                std::vector<std::string>* results) override;

  absl::Status DeleteFile(std::string_view fname,
                          TransactionToken* token) override;

  absl::Status CreateDir(std::string_view name,
                         TransactionToken* token) override;

  absl::Status DeleteDir(std::string_view name,
                         TransactionToken* token) override;

  absl::Status GetFileSize(std::string_view fname, TransactionToken* token,
                           uint64_t* size) override;

  absl::Status RenameFile(std::string_view src, std::string_view target,
                          TransactionToken* token) override;

  absl::Status CopyFile(std::string_view src, std::string_view target,
                        TransactionToken* token) override;
};

class LocalFileSystemPosix : public FileSystemPosix {
 public:
  std::string TranslateName(std::string_view name) const override;
};

}  // namespace tsl

#endif  // XLA_TSL_PLATFORM_FILE_SYSTEM_POSIX_H_
