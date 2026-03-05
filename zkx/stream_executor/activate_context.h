/* Copyright 2024 The OpenXLA Authors.
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

#ifndef ZKX_STREAM_EXECUTOR_ACTIVATE_CONTEXT_H_
#define ZKX_STREAM_EXECUTOR_ACTIVATE_CONTEXT_H_

#include "zkx/stream_executor/namespace_alias.h"

namespace stream_executor {

// An RAII handle for ensuring a context is activated for the duration of the
// ActivateContext's scope. The creation of an ActivateContext ensures that any
// necessary state changes are done to make the requested context active.  When
// the ActivateContext is destroyed, it will enable any previous context that
// was active.
class ActivateContext {
 public:
  virtual ~ActivateContext() = default;
};

}  // namespace stream_executor

#endif  // ZKX_STREAM_EXECUTOR_ACTIVATE_CONTEXT_H_
