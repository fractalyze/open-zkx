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

#include <fnmatch.h>
#include <stdlib.h>
#include <sys/stat.h>

#include <map>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>

#ifdef __FreeBSD__
#include <pthread_np.h>
#endif

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/debugging/leak_check.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"

#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system_posix.h"

namespace tsl {

ABSL_CONST_INIT absl::Mutex g_name_mutex(absl::kConstInit);

std::map<std::thread::id, std::string>& GetThreadNameRegistry()
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(g_name_mutex) {
  static auto* thread_name_registry =
      absl::IgnoreLeak(new std::map<std::thread::id, std::string>());
  return *thread_name_registry;
}

// We use the pthread API instead of std::thread so we can control stack sizes.
class PThread : public Thread {
 public:
  PThread(const ThreadOptions& thread_options, std::string_view name,
          absl::AnyInvocable<void()> fn) {
    ThreadParams* params = new ThreadParams;
    params->name = std::move(name);
    params->fn = std::move(fn);
    pthread_attr_t attributes;
    pthread_attr_init(&attributes);
    if (thread_options.stack_size != 0) {
      pthread_attr_setstacksize(&attributes, thread_options.stack_size);
    }
    int ret = pthread_create(&thread_, &attributes, &ThreadFn, params);
    // There is no mechanism for the thread creation API to fail, so we CHECK.
    CHECK_EQ(ret, 0) << "Thread " << name
                     << " creation via pthread_create() failed.";
    pthread_attr_destroy(&attributes);
  }

  ~PThread() override { pthread_join(thread_, nullptr); }

 private:
  struct ThreadParams {
    std::string name;
    absl::AnyInvocable<void()> fn;
  };
  static void* ThreadFn(void* params_arg) {
    std::unique_ptr<ThreadParams> params(
        reinterpret_cast<ThreadParams*>(params_arg));
    {
      absl::MutexLock l(&g_name_mutex);
      GetThreadNameRegistry().emplace(std::this_thread::get_id(), params->name);
    }
    params->fn();
    {
      absl::MutexLock l(&g_name_mutex);
      GetThreadNameRegistry().erase(std::this_thread::get_id());
    }
    return nullptr;
  }

  pthread_t thread_;
};

class EnvPosix : public Env {
 public:
  EnvPosix() {}

  ~EnvPosix() override { LOG(FATAL) << "Env::Default() must not be destroyed"; }

  bool MatchPath(std::string_view path, std::string_view pattern) override {
    // NOTE(chokobole): Using string_view::data() with fnmatch is unsafe, as
    // fnmatch expects null-terminated strings and string_view does not provide
    // this guarantee. This can lead to out-of-bounds memory access. To fix
    // this, you should convert the string_views to std::strings before calling
    // c_str().
    return fnmatch(std::string(pattern).c_str(), std::string(path).c_str(),
                   FNM_PATHNAME) == 0;
  }

  Thread* StartThread(const ThreadOptions& thread_options,
                      std::string_view name,
                      absl::AnyInvocable<void()> fn) override {
    return new PThread(thread_options, name, std::move(fn));
  }

  void Sleep(absl::Duration duration) override {
    int64_t micros = absl::ToInt64Microseconds(duration);
    while (micros > 0) {
      timespec sleep_time;
      sleep_time.tv_sec = 0;
      sleep_time.tv_nsec = 0;

      if (micros >= 1e6) {
        sleep_time.tv_sec =
            std::min<int64_t>(micros / 1e6, std::numeric_limits<time_t>::max());
        micros -= static_cast<int64_t>(sleep_time.tv_sec) * 1e6;
      }
      if (micros < 1e6) {
        sleep_time.tv_nsec = 1000 * micros;
        micros = 0;
      }
      while (nanosleep(&sleep_time, &sleep_time) != 0 && errno == EINTR) {
        // Ignore signals and wait for the full interval to elapse.
      }
    }
  }

  int64_t GetCurrentThreadId() override {
    static thread_local int64_t current_thread_id =
        GetCurrentThreadIdInternal();
    return current_thread_id;
  }

  bool GetCurrentThreadName(std::string* name) override {
    {
      absl::MutexLock l(&g_name_mutex);
      auto thread_name =
          GetThreadNameRegistry().find(std::this_thread::get_id());
      if (thread_name != GetThreadNameRegistry().end()) {
        *name = absl::StrCat(thread_name->second, "/", GetCurrentThreadId());
        return true;
      }
    }
#if defined(__GLIBC__) || defined(__FreeBSD__)
    char buf[100];
#ifdef __FreeBSD__
    int res = 0;
    pthread_get_name_np(pthread_self(), buf, static_cast<size_t>(100));
#else
    int res = pthread_getname_np(pthread_self(), buf, static_cast<size_t>(100));
#endif
    if (res != 0) {
      return false;
    }
    *name = buf;
    return true;
#else
    return false;
#endif
  }

  void SchedClosure(absl::AnyInvocable<void()> closure) override {
    // TODO(b/27290852): Spawning a new thread here is wasteful, but
    // needed to deal with the fact that many `closure` functions are
    // blocking in the current codebase.
    std::thread closure_thread(std::move(closure));
    closure_thread.detach();
  }

  void SchedClosureAfter(absl::Duration duration,
                         absl::AnyInvocable<void()> closure) override {
    // TODO(b/27290852): Consuming a thread here is wasteful, but this
    // code is (currently) only used in the case where a step fails
    // (AbortStep). This could be replaced by a timer thread
    SchedClosure([this, duration, closure = std::move(closure)]() mutable {
      Sleep(duration);
      closure();
    });
  }

  std::string GetRunfilesDir() override {
    std::string bin_path = this->GetExecutablePath();
    std::string runfiles_suffix = ".runfiles/org_tensorflow";
    std::size_t pos = bin_path.find(runfiles_suffix);

    // Sometimes (when executing under python) bin_path returns the full path to
    // the python scripts under runfiles. Get the substring.
    if (pos != std::string::npos) {
      return bin_path.substr(0, pos + runfiles_suffix.length());
    }

    // See if we have the executable path. if executable.runfiles exists, return
    // that folder.
    std::string runfiles_path = bin_path + runfiles_suffix;
    absl::Status s = this->IsDirectory(runfiles_path);
    if (s.ok()) {
      return runfiles_path;
    }

    // If nothing can be found, return something close.
    return bin_path.substr(0, bin_path.find_last_of("/\\"));
  }

 private:
  void GetLocalTempDirectories(std::vector<std::string>* list) override;

  int64_t GetCurrentThreadIdInternal() {
#ifdef __APPLE__
    uint64_t tid64;
    pthread_threadid_np(nullptr, &tid64);
    return static_cast<int64_t>(tid64);
#elif defined(__FreeBSD__)
    return pthread_getthreadid_np();
#elif defined(__NR_gettid)
    return static_cast<int64_t>(syscall(__NR_gettid));
#else
    return std::hash<std::thread::id>()(std::this_thread::get_id());
#endif
  }
};

REGISTER_FILE_SYSTEM("", FileSystemPosix);
REGISTER_FILE_SYSTEM("file", LocalFileSystemPosix);

Env* Env::Default() {
  static Env* default_env = absl::IgnoreLeak(new EnvPosix);
  return default_env;
}

void EnvPosix::GetLocalTempDirectories(std::vector<std::string>* list) {
  list->clear();
  // Directories, in order of preference. If we find a dir that
  // exists, we stop adding other less-preferred dirs
  const char* candidates[] = {
      // Non-null only during unittest/regtest
      getenv("TEST_TMPDIR"),

      // Explicitly-supplied temp dirs
      getenv("TMPDIR"),
      getenv("TMP"),

#if defined(__ANDROID__)
      "/data/local/tmp",
#endif

      // If all else fails
      "/tmp",
  };

  std::vector<std::string> paths;  // Only in case of errors.
  for (const char* d : candidates) {
    if (!d || d[0] == '\0') continue;  // Empty env var
    paths.push_back(d);
    // Make sure we don't surprise anyone who's expecting a '/'
    std::string dstr = d;
    if (dstr[dstr.size() - 1] != '/') {
      dstr += "/";
    }

    struct stat statbuf;
    if (!stat(d, &statbuf) && S_ISDIR(statbuf.st_mode) &&
        !access(dstr.c_str(), 0)) {
      // We found a dir that exists and is accessible - we're done.
      list->push_back(dstr);
      return;
    }
  }
  LOG(WARNING) << "We are not able to find a directory for temporary files.\n"
               << "Verify the directory access and available space under: "
               << absl::StrJoin(paths, ",") << ". "
               << "You can also provide a directory for temporary files with"
               << " the environment variable TMP or TMPDIR. "
               << "Example under bash: `export TMP=/my_new_temp_directory;`";
}

int setenv(const char* name, const char* value, int overwrite) {
  return ::setenv(name, value, overwrite);
}

int unsetenv(const char* name) { return ::unsetenv(name); }

}  // namespace tsl
