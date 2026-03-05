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

#ifndef XLA_TSL_PLATFORM_ENV_H_
#define XLA_TSL_PLATFORM_ENV_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <string>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/time/time.h"
#include "google/protobuf/message.h"

#include "xla/tsl/platform/env_time.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/numa.h"

namespace tsl {

class Thread;
struct ThreadOptions;

class Env {
 public:
  Env();
  virtual ~Env() = default;

  /// \brief Returns a default environment suitable for the current operating
  /// system.
  ///
  /// Sophisticated users may wish to provide their own Env
  /// implementation instead of relying on this default environment.
  ///
  /// The result of Default() belongs to this library and must never be deleted.
  static Env* Default();

  /// \brief Returns the FileSystem object to handle operations on the file
  /// specified by 'fname'. The FileSystem object is used as the implementation
  /// for the file system related (non-virtual) functions that follow.
  /// Returned FileSystem object is still owned by the Env object and will
  // (might) be destroyed when the environment is destroyed.
  virtual absl::Status GetFileSystemForFile(std::string_view fname,
                                            FileSystem** result);

  /// \brief Returns the file system schemes registered for this Env.
  virtual absl::Status GetRegisteredFileSystemSchemes(
      std::vector<std::string_view>* schemes);

  /// \brief Register a file system for a scheme.
  virtual absl::Status RegisterFileSystem(std::string_view scheme,
                                          FileSystemRegistry::Factory factory);

  /// \brief Register a modular file system for a scheme.
  ///
  /// Same as `RegisterFileSystem` but for filesystems provided by plugins.
  ///
  /// TODO(b/139060984): After all filesystems are converted, make this be the
  /// canonical registration function.
  virtual absl::Status RegisterFileSystem(
      std::string_view scheme, std::unique_ptr<FileSystem> filesystem);

  absl::Status SetOption(std::string_view scheme, std::string_view key,
                         std::string_view value);

  absl::Status SetOption(std::string_view scheme, std::string_view key,
                         const std::vector<std::string>& values);

  absl::Status SetOption(std::string_view scheme, std::string_view key,
                         const std::vector<int64_t>& values);

  absl::Status SetOption(std::string_view scheme, std::string_view key,
                         const std::vector<double>& values);

  /// \brief Flush filesystem caches for all registered filesystems.
  absl::Status FlushFileSystemCaches();

  /// \brief Creates a brand new random access read-only file with the
  /// specified name.

  /// On success, stores a pointer to the new file in
  /// *result and returns OK.  On failure stores NULL in *result and
  /// returns non-OK.  If the file does not exist, returns a non-OK
  /// status.
  ///
  /// The returned file may be concurrently accessed by multiple threads.
  ///
  /// The ownership of the returned RandomAccessFile is passed to the caller
  /// and the object should be deleted when is not used. The file object
  /// shouldn't live longer than the Env object.
  absl::Status NewRandomAccessFile(std::string_view fname,
                                   std::unique_ptr<RandomAccessFile>* result);

  absl::Status NewRandomAccessFile(std::string_view fname,
                                   TransactionToken* token,
                                   std::unique_ptr<RandomAccessFile>* result) {
    // We duplicate these methods due to Google internal coding style prevents
    // virtual functions with default arguments. See PR #41615.
    return absl::OkStatus();
  }

  /// \brief Creates an object that writes to a new file with the specified
  /// name.
  ///
  /// Deletes any existing file with the same name and creates a
  /// new file.  On success, stores a pointer to the new file in
  /// *result and returns OK.  On failure stores NULL in *result and
  /// returns non-OK.
  ///
  /// The returned file will only be accessed by one thread at a time.
  ///
  /// The ownership of the returned WritableFile is passed to the caller
  /// and the object should be deleted when is not used. The file object
  /// shouldn't live longer than the Env object.
  absl::Status NewWritableFile(std::string_view fname,
                               std::unique_ptr<WritableFile>* result);

  absl::Status NewWritableFile(std::string_view fname, TransactionToken* token,
                               std::unique_ptr<WritableFile>* result) {
    return absl::OkStatus();
  }

  /// \brief Creates an object that either appends to an existing file, or
  /// writes to a new file (if the file does not exist to begin with).
  ///
  /// On success, stores a pointer to the new file in *result and
  /// returns OK.  On failure stores NULL in *result and returns
  /// non-OK.
  ///
  /// The returned file will only be accessed by one thread at a time.
  ///
  /// The ownership of the returned WritableFile is passed to the caller
  /// and the object should be deleted when is not used. The file object
  /// shouldn't live longer than the Env object.
  absl::Status NewAppendableFile(std::string_view fname,
                                 std::unique_ptr<WritableFile>* result);

  absl::Status NewAppendableFile(std::string_view fname,
                                 TransactionToken* token,
                                 std::unique_ptr<WritableFile>* result) {
    return absl::OkStatus();
  }
  /// \brief Creates a readonly region of memory with the file context.
  ///
  /// On success, it returns a pointer to read-only memory region
  /// from the content of file fname. The ownership of the region is passed to
  /// the caller. On failure stores nullptr in *result and returns non-OK.
  ///
  /// The returned memory region can be accessed from many threads in parallel.
  ///
  /// The ownership of the returned ReadOnlyMemoryRegion is passed to the caller
  /// and the object should be deleted when is not used. The memory region
  /// object shouldn't live longer than the Env object.
  absl::Status NewReadOnlyMemoryRegionFromFile(
      std::string_view fname, std::unique_ptr<ReadOnlyMemoryRegion>* result);

  absl::Status NewReadOnlyMemoryRegionFromFile(
      std::string_view fname, TransactionToken* token,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) {
    return absl::OkStatus();
  }

  /// Returns OK if the named path exists and NOT_FOUND otherwise.
  absl::Status FileExists(std::string_view fname);

  absl::Status FileExists(std::string_view fname, TransactionToken* token) {
    return absl::OkStatus();
  }

  /// Returns true if all the listed files exist, false otherwise.
  /// if status is not null, populate the vector with a detailed status
  /// for each file.
  bool FilesExist(const std::vector<std::string>& files,
                  std::vector<absl::Status>* status);

  bool FilesExist(const std::vector<std::string>& files,
                  TransactionToken* token, std::vector<absl::Status>* status) {
    return true;
  }

  /// \brief Stores in *result the names of the children of the specified
  /// directory. The names are relative to "dir".
  ///
  /// Original contents of *results are dropped.
  absl::Status GetChildren(std::string_view dir,
                           std::vector<std::string>* result);

  absl::Status GetChildren(std::string_view dir, TransactionToken* token,
                           std::vector<std::string>* result) {
    return absl::OkStatus();
  }

  /// \brief Returns true if the path matches the given pattern. The wildcards
  /// allowed in pattern are described in FileSystem::GetMatchingPaths.
  virtual bool MatchPath(std::string_view path, std::string_view pattern) = 0;

  /// \brief Given a pattern, stores in *results the set of paths that matches
  /// that pattern. *results is cleared.
  ///
  /// More details about `pattern` in FileSystem::GetMatchingPaths.
  virtual absl::Status GetMatchingPaths(std::string_view pattern,
                                        std::vector<std::string>* results);

  absl::Status GetMatchingPaths(std::string_view pattern,
                                TransactionToken* token,
                                std::vector<std::string>* results) {
    return absl::OkStatus();
  }

  /// Deletes the named file.
  absl::Status DeleteFile(std::string_view fname);

  absl::Status DeleteFile(std::string_view fname, TransactionToken* token) {
    return absl::OkStatus();
  }

  /// \brief Deletes the specified directory and all subdirectories and files
  /// underneath it. This is accomplished by traversing the directory tree
  /// rooted at dirname and deleting entries as they are encountered.
  ///
  /// If dirname itself is not readable or does not exist, *undeleted_dir_count
  /// is set to 1, *undeleted_file_count is set to 0 and an appropriate status
  /// (e.g. NOT_FOUND) is returned.
  ///
  /// If dirname and all its descendants were successfully deleted, TF_OK is
  /// returned and both error counters are set to zero.
  ///
  /// Otherwise, while traversing the tree, undeleted_file_count and
  /// undeleted_dir_count are updated if an entry of the corresponding type
  /// could not be deleted. The returned error status represents the reason that
  /// any one of these entries could not be deleted.
  ///
  /// REQUIRES: undeleted_files, undeleted_dirs to be not null.
  ///
  /// Typical return codes:
  ///  * OK - dirname exists and we were able to delete everything underneath.
  ///  * NOT_FOUND - dirname doesn't exist
  ///  * PERMISSION_DENIED - dirname or some descendant is not writable
  ///  * UNIMPLEMENTED - Some underlying functions (like Delete) are not
  ///                    implemented
  absl::Status DeleteRecursively(std::string_view dirname,
                                 int64_t* undeleted_files,
                                 int64_t* undeleted_dirs);

  absl::Status DeleteRecursively(std::string_view dirname,
                                 TransactionToken* token,
                                 int64_t* undeleted_files,
                                 int64_t* undeleted_dirs) {
    return absl::OkStatus();
  }

  /// \brief Creates the specified directory and all the necessary
  /// subdirectories. Typical return codes.
  ///  * OK - successfully created the directory and sub directories, even if
  ///         they were already created.
  ///  * PERMISSION_DENIED - dirname or some subdirectory is not writable.
  absl::Status RecursivelyCreateDir(std::string_view dirname);

  absl::Status RecursivelyCreateDir(std::string_view dirname,
                                    TransactionToken* token) {
    return absl::OkStatus();
  }
  /// \brief Creates the specified directory. Typical return codes
  ///  * OK - successfully created the directory.
  ///  * ALREADY_EXISTS - directory already exists.
  ///  * PERMISSION_DENIED - dirname is not writable.
  absl::Status CreateDir(std::string_view dirname);

  absl::Status CreateDir(std::string_view dirname, TransactionToken* token) {
    return absl::OkStatus();
  }

  /// Deletes the specified directory.
  absl::Status DeleteDir(std::string_view dirname);

  absl::Status DeleteDir(std::string_view dirname, TransactionToken* token) {
    return absl::OkStatus();
  }

  /// Obtains statistics for the given path.
  absl::Status Stat(std::string_view fname, FileStatistics* stat);

  absl::Status Stat(std::string_view fname, TransactionToken* token,
                    FileStatistics* stat) {
    return absl::OkStatus();
  }

  /// \brief Returns whether the given path is a directory or not.
  /// Typical return codes (not guaranteed exhaustive):
  ///  * OK - The path exists and is a directory.
  ///  * FAILED_PRECONDITION - The path exists and is not a directory.
  ///  * NOT_FOUND - The path entry does not exist.
  ///  * PERMISSION_DENIED - Insufficient permissions.
  ///  * UNIMPLEMENTED - The file factory doesn't support directories.
  absl::Status IsDirectory(std::string_view fname);

  /// \brief Returns whether the given path is on a file system
  /// that has atomic move capabilities. This can be used
  /// to determine if there needs to be a temp location to safely write objects.
  /// The second boolean argument has_atomic_move contains this information.
  ///
  /// Returns one of the following status codes (not guaranteed exhaustive):
  ///  * OK - The path is on a recognized file system,
  ///         so has_atomic_move holds the above information.
  ///  * UNIMPLEMENTED - The file system of the path hasn't been implemented in
  ///  TF
  absl::Status HasAtomicMove(std::string_view path, bool* has_atomic_move);

  /// Returns whether the give path is on a file system
  /// that has ability to create a new temp file. This can be used
  /// to determine if there needs to be a temp location to safely write objects.
  /// If this returns false, TensorFlow will write directly to output files
  /// instead of creating a temporary file and swapping it in. This may mean
  /// that incomplete writes are visible to consumers.
  absl::Status CanCreateTempFile(std::string_view fname,
                                 bool* can_create_temp_file);

  /// Stores the size of `fname` in `*file_size`.
  absl::Status GetFileSize(std::string_view fname, uint64_t* file_size);

  absl::Status GetFileSize(std::string_view fname, TransactionToken* token,
                           uint64_t* file_size) {
    return absl::OkStatus();
  }

  /// \brief Renames file src to target. If target already exists, it will be
  /// replaced.
  absl::Status RenameFile(std::string_view src, std::string_view target);

  absl::Status RenameFile(std::string_view src, std::string_view target,
                          TransactionToken* token) {
    return absl::OkStatus();
  }

  /// \brief Copy the src to target.
  absl::Status CopyFile(std::string_view src, std::string_view target);

  absl::Status CopyFile(std::string_view src, std::string_view target,
                        TransactionToken* token) {
    return absl::OkStatus();
  }

  /// \brief starts a new transaction on the filesystem that handles filename
  absl::Status StartTransaction(std::string_view filename,
                                TransactionToken** token) {
    *token = nullptr;
    return absl::OkStatus();
  }

  /// \brief Adds `path` to transaction in `token` if token belongs to
  /// filesystem that handles the path.
  absl::Status AddToTransaction(std::string_view path,
                                TransactionToken* token) {
    return absl::OkStatus();
  }

  /// \brief Get token for `path` or start a new transaction and add `path` to
  /// it.
  absl::Status GetTokenOrStartTransaction(std::string_view path,
                                          TransactionToken** token) {
    *token = nullptr;
    return absl::OkStatus();
  }

  /// \brief Returns the transaction for `path` or nullptr in `token`
  absl::Status GetTransactionForPath(std::string_view path,
                                     TransactionToken** token) {
    *token = nullptr;
    return absl::OkStatus();
  }

  /// \brief Finalizes the transaction
  absl::Status EndTransaction(TransactionToken* token) {
    return absl::OkStatus();
  }

  /// \brief Returns the absolute path of the current executable. It resolves
  /// symlinks if there is any.
  std::string GetExecutablePath();

  /// Creates a local unique temporary file name. Returns true if success.
  bool LocalTempFilename(std::string* filename);

  /// Creates a local unique file name that starts with |prefix| and ends with
  /// |suffix|. Returns true if success.
  bool CreateUniqueFileName(std::string* prefix, std::string_view suffix);

  /// \brief Return the runfiles directory if running under bazel. Returns
  /// the directory the executable is located in if not running under bazel.
  virtual std::string GetRunfilesDir() = 0;

  /// \brief Returns the number of nano-seconds since the Unix epoch.
  virtual uint64_t NowNanos() const { return EnvTime::NowNanos(); }

  /// \brief Returns the number of micro-seconds since the Unix epoch.
  virtual uint64_t NowMicros() const { return EnvTime::NowMicros(); }

  /// \brief Returns the number of seconds since the Unix epoch.
  virtual uint64_t NowSeconds() const { return EnvTime::NowSeconds(); }

  /// Sleeps/delays the thread for the prescribed duration.
  virtual void Sleep(absl::Duration duration) = 0;

  /// Returns the process ID of the calling process.
  int32_t GetProcessId();

  /// \brief Returns a new thread that is running fn() and is identified
  /// (for debugging/performance-analysis) by "name".
  ///
  /// Caller takes ownership of the result and must delete it eventually
  /// (the deletion will block until fn() stops running).
  virtual Thread* StartThread(
      const ThreadOptions& thread_options, std::string_view name,
      absl::AnyInvocable<void()> fn) ABSL_MUST_USE_RESULT = 0;

  // Returns the thread id of calling thread.
  // Posix: Returns pthread id which is only guaranteed to be unique within a
  //        process.
  // Windows: Returns thread id which is unique.
  virtual int64_t GetCurrentThreadId() = 0;

  // Copies current thread name to "name". Returns true if success.
  virtual bool GetCurrentThreadName(std::string* name) = 0;

  // \brief Schedules the given closure on a thread-pool.
  //
  // NOTE(mrry): This closure may block.
  virtual void SchedClosure(absl::AnyInvocable<void()> closure) = 0;

  // \brief Schedules the given closure on a thread-pool after the given number
  // of duration.
  //
  // NOTE(mrry): This closure must not block.
  virtual void SchedClosureAfter(absl::Duration duration,
                                 absl::AnyInvocable<void()> closure) = 0;

  // Returns a possible list of local temporary directories.
  virtual void GetLocalTempDirectories(std::vector<std::string>* list) = 0;

 private:
  std::unique_ptr<FileSystemRegistry> file_system_registry_;
  Env(const Env&) = delete;
  void operator=(const Env&) = delete;
};

/// Represents a thread used to run a TSL function.
class Thread {
 public:
  Thread() {}

  /// Blocks until the thread of control stops running.
  virtual ~Thread();

 private:
  Thread(const Thread&) = delete;
  void operator=(const Thread&) = delete;
};

/// \brief Cross-platform setenv.
///
/// Since setenv() is not available on windows, we provide an
/// alternative with platform specific implementations here.
int setenv(const char* name, const char* value, int overwrite);

/// Cross-platform unsetenv.
int unsetenv(const char* name);

/// \brief Options to configure a Thread.
///
/// Note that the options are all hints, and the
/// underlying implementation may choose to ignore it.
struct ThreadOptions {
  /// Thread stack size to use (in bytes).
  size_t stack_size = 0;  // 0: use system default value
  /// Guard area size to use near thread stacks to use (in bytes)
  size_t guard_size = 0;  // 0: use system default value
  int numa_node = port::kNUMANoAffinity;
};

/// A utility routine: copy contents of `src` in file system `src_fs`
/// to `target` in file system `target_fs`.
absl::Status FileSystemCopyFile(FileSystem* src_fs, std::string_view src,
                                FileSystem* target_fs, std::string_view target);

/// A utility routine: reads contents of named file into `*data`
absl::Status ReadFileToString(Env* env, std::string_view fname,
                              std::string* data);

/// A utility routine: write contents of `data` to file named `fname`
/// (overwriting existing contents, if any).
absl::Status WriteStringToFile(Env* env, std::string_view fname,
                               std::string_view data);

/// Write binary representation of "proto" to the named file.
absl::Status WriteBinaryProto(Env* env, std::string_view fname,
                              const google::protobuf::MessageLite& proto);

/// Reads contents of named file and parse as binary encoded proto data
/// and store into `*proto`.
absl::Status ReadBinaryProto(Env* env, std::string_view fname,
                             google::protobuf::MessageLite* proto);

/// Write the text representation of "proto" to the named file.
inline absl::Status WriteTextProto(
    Env* /* env */, std::string_view /* fname */,
    const google::protobuf::MessageLite& /* proto */) {
  return absl::UnimplementedError("Can't write text protos with protolite.");
}
absl::Status WriteTextProto(Env* env, std::string_view fname,
                            const google::protobuf::Message& proto);

/// Read contents of named file and parse as text encoded proto data
/// and store into `*proto`.
inline absl::Status ReadTextProto(Env* /* env */, std::string_view /* fname */,
                                  google::protobuf::MessageLite* /* proto */) {
  return absl::UnimplementedError("Can't parse text protos with protolite.");
}
absl::Status ReadTextProto(Env* env, std::string_view fname,
                           google::protobuf::Message* proto);

/// Read contents of named file and parse as either text or binary encoded proto
/// data and store into `*proto`.
absl::Status ReadTextOrBinaryProto(Env* env, std::string_view fname,
                                   google::protobuf::Message* proto);
absl::Status ReadTextOrBinaryProto(Env* env, std::string_view fname,
                                   google::protobuf::MessageLite* proto);

// START_SKIP_DOXYGEN

// The following approach to register filesystems is deprecated and will be
// replaced with modular filesystem plugins registration.
// TODO(b/139060984): After all filesystems are converted, remove this.
namespace register_file_system {

template <typename Factory>
struct Register {
  Register(Env* env, std::string_view scheme, bool try_modular_filesystems) {
    // TODO(yongtang): Remove legacy file system registration for hdfs/s3/gcs
    if (try_modular_filesystems) {
      const char* env_value = getenv("ZKX_USE_MODULAR_FILESYSTEM");
      std::string load_plugin =
          env_value ? absl::AsciiStrToLower(env_value) : "";
      if (load_plugin == "true" || load_plugin == "1") {
        // We don't register the static filesystem and wait for SIG IO one
        LOG(WARNING) << "Using modular file system for '" << scheme << "'.";
        return;
      }
      // If the envvar is missing or not "true"/"1", then fall back to legacy
      // implementation to be backwards compatible.
    }
    // TODO(b/32704451): Don't just ignore the ::tensorflow::Status object!
    env->RegisterFileSystem(scheme, []() -> FileSystem* { return new Factory; })
        .IgnoreError();
  }
};

}  // namespace register_file_system

// END_SKIP_DOXYGEN

}  // namespace tsl

// Register a FileSystem implementation for a scheme. Files with names that have
// "scheme://" prefixes are routed to use this implementation.
#define REGISTER_FILE_SYSTEM_ENV(env, scheme, factory, modular) \
  REGISTER_FILE_SYSTEM_UNIQ_HELPER(__COUNTER__, env, scheme, factory, modular)
#define REGISTER_FILE_SYSTEM_UNIQ_HELPER(ctr, env, scheme, factory, modular) \
  REGISTER_FILE_SYSTEM_UNIQ(ctr, env, scheme, factory, modular)
#define REGISTER_FILE_SYSTEM_UNIQ(ctr, env, scheme, factory, modular)    \
  static ::tsl::register_file_system::Register<factory> register_ff##ctr \
      ABSL_ATTRIBUTE_UNUSED =                                            \
          ::tsl::register_file_system::Register<factory>(env, scheme, modular)

#define REGISTER_FILE_SYSTEM(scheme, factory) \
  REGISTER_FILE_SYSTEM_ENV(::tsl::Env::Default(), scheme, factory, false);

#define REGISTER_LEGACY_FILE_SYSTEM(scheme, factory) \
  REGISTER_FILE_SYSTEM_ENV(::tsl::Env::Default(), scheme, factory, true);

#endif  // XLA_TSL_PLATFORM_ENV_H_
