/* Copyright 2017 The OpenXLA Authors.
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

#include "zkx/service/gpu/ir_emitter.h"

#include "absl/log/log.h"

namespace zkx::gpu {

IrEmitter::IrEmitter(IrEmitterContext* ir_emitter_context, bool is_nested)
    : ir_emitter_context_(ir_emitter_context),
      module_(ir_emitter_context->llvm_module()),
      b_(module_->getContext()) {}

absl::Status IrEmitter::DefaultAction(HloInstruction* hlo) {
  return absl::UnimplementedError("DefaultAction is not implemented on GPU");
}

absl::Status IrEmitter::HandleConstant(HloInstruction* constant) {
  return absl::OkStatus();
}

absl::Status IrEmitter::HandleGetTupleElement(
    HloInstruction* get_tuple_element) {
  return absl::UnimplementedError("GetTupleElement is not implemented on GPU");
}

absl::Status IrEmitter::HandleScatter(HloInstruction*) {
  return absl::UnimplementedError("Scatter is not implemented on GPUs.");
}

absl::Status IrEmitter::HandleTuple(HloInstruction* tuple) {
  return absl::UnimplementedError("Tuple is not implemented on GPUs.");
}

absl::Status IrEmitter::HandleParameter(HloInstruction* parameter) {
  return absl::OkStatus();
}

absl::Status IrEmitter::HandleCall(HloInstruction* call) {
  return absl::UnimplementedError("Call is not implemented on GPU");
}

}  // namespace zkx::gpu
