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

#include "zkx/service/gpu/execution_stream_assignment.h"

#include <deque>
#include <memory>
#include <optional>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"

#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/service/call_graph.h"

namespace zkx::gpu {

ExecutionStreamAssignment::ExecutionStreamAssignment(
    const HloModule *module, ExecutionStreamAssignmentOptions options) {
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);

  // We'll walk the `CallGraph` starting from the entrypoint. The instructions
  // on the entrypoint computation will be assigned `ExecutionStreamId(0)`.

  // Each `Pending` item represents an `HloComputation` that needs to be
  // processed. We start with the entrypoint and add callees as we discover
  // them.
  struct Pending {
    Pending(HloComputation *node, ExecutionStreamId stream_id)
        : node(node), stream_id(stream_id) {}
    HloComputation *node;
    ExecutionStreamId stream_id;
  };
  std::deque<Pending> queue;
  queue.emplace_back(module->entry_computation(), ExecutionStreamId(0));

  // Enqueues called computations of a given `callsite` unless the callees are
  // only invoked in an embedded context, in which case children nodes will all
  // be executed in a single kernel.
  auto enqueue_called_computations = [&](const CallSite &callsite,
                                         ExecutionStreamId stream) {
    if (GetInstructionCallContext(callsite.instruction()->opcode()) ==
        CallContext::kEmbedded) {
      return;
    }
    for (HloComputation *computation : callsite.called_computations()) {
      queue.emplace_back(computation, stream);
    }
  };

  while (!queue.empty()) {
    Pending pending = queue.front();
    queue.pop_front();

    // Assign the current `ExecutionStreamId` to all instructions.
    for (HloInstruction *instruction : pending.node->instructions()) {
      CHECK(sync_instructions_.try_emplace(instruction, pending.stream_id)
                .second);
    }

    // Process all callsites in the current computation.
    for (const CallSite &callsite :
         call_graph->GetNode(pending.node).callsites()) {
      // Synchronous calls will result in the called computations being
      // invoked using the same `ExecutionStreamId`.
      enqueue_called_computations(callsite, pending.stream_id);
    }
  }
}

namespace {

absl::Status StreamNotFoundError(const HloInstruction *instruction) {
  return absl::NotFoundError(absl::StrCat(
      "No ExecutionStreamId found for ", instruction->ToString(),
      "; this may happen if the Computation is not reachable from the module's "
      "entrypoint, or if it's only reachable through embedded calls."));
}

}  // namespace

absl::StatusOr<ExecutionStreamId>
ExecutionStreamAssignment::GetSyncExecutionStreamId(
    const HloInstruction *instruction) const {
  auto stream = sync_instructions_.find(instruction);
  if (stream == sync_instructions_.end()) {
    return StreamNotFoundError(instruction);
  }
  return stream->second;
}

}  // namespace zkx::gpu
