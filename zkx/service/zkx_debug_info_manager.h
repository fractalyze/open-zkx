/* Copyright 2019 The OpenXLA Authors.
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

#ifndef ZKX_SERVICE_ZKX_DEBUG_INFO_MANAGER_H_
#define ZKX_SERVICE_ZKX_DEBUG_INFO_MANAGER_H_

#include <memory>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"

#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/service/hlo.pb.h"

namespace zkx {

using ModuleIdentifier = int;

// ZkxDebugInfoManager tracks all ZKX programs (Executables) throughout their
// lifetime. Because the tracing period can start during an Executable's
// execution, we need to track Executables even when tracing is off.
// This class is thread-safe.
class ZkxDebugInfoManager {
 public:
  static ZkxDebugInfoManager* Get() {
    static absl::NoDestructor<ZkxDebugInfoManager> singleton;
    return singleton.get();
  }

  // Registers an active module to ZkxDebugInfoManager.
  // The module_id of the module is expected to be unique per process.
  void RegisterModule(std::shared_ptr<const HloModule> hlo_module,
                      BufferAssignmentProto buffer_assignment);

  // Unregisters an active module.
  void UnregisterModule(ModuleIdentifier module_id);

  // Start tracing, begins to collect debug information for all the running
  // modules during the tracing period.
  void StartTracing();

  // Stops tracing.
  // If `module_debug_info` is not null, returns debug information for all the
  // modules that were alive since StartTracing().
  void StopTracing(
      std::vector<std::unique_ptr<HloProto>>* module_debug_info = nullptr);

  // Returns whether `module_id` is tracked by ZkxDebugInfoManager.
  bool TracksModule(ModuleIdentifier module_id) const;

  friend class ZkxDebugInfoManagerTestPeer;

 private:
  friend class absl::NoDestructor<ZkxDebugInfoManager>;

  ZkxDebugInfoManager() = default;

  struct ZkxModuleEntry {
    std::shared_ptr<const HloModule> hlo_module;
    BufferAssignmentProto buffer_assignment;
    bool active = false;
  };

  mutable absl::Mutex mutex_;
  bool tracing_active_ ABSL_GUARDED_BY(mutex_) = false;
  // Active modules are those still tracked by us. There could be much more
  // active modules than running modules, we will try to reduce the trace size
  // by only transfer those modules that were running during tracing period.
  absl::flat_hash_map<ModuleIdentifier, ZkxModuleEntry> modules_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace zkx

#endif  // ZKX_SERVICE_ZKX_DEBUG_INFO_MANAGER_H_
