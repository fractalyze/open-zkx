/* Copyright 2018 The OpenXLA Authors.
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

#ifndef ZKX_SERVICE_GPU_IR_EMITTER_UNNESTED_H_
#define ZKX_SERVICE_GPU_IR_EMITTER_UNNESTED_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "zkx/backends/gpu/runtime/copy_thunk.h"
#include "zkx/backends/gpu/runtime/sequential_thunk.h"
#include "zkx/backends/gpu/runtime/thunk.h"
#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_instructions.h"
#include "zkx/service/buffer_assignment.h"
#include "zkx/service/gpu/ir_emitter.h"
#include "zkx/service/gpu/ir_emitter_context.h"
#include "zkx/shape_util.h"

namespace zkx::gpu {

// Emits LLVM IR for an "unnested computation".
//
// An unnested computation is an HloComputation which you run by executing one
// or more kernels for each HloInstruction it contains.  Examples of unnested
// computations:
//
//  - An HloModule's root computation,
//  - The body of an HLO while loop,
//  - The true/false computation of an HLO conditional.
//
// Note the opportunity for confusion -- the while loop's computation is nested
// within the root computation, but it's emitted using IrEmitterUnnested! Don't
// think about it too hard.
//
// Examples of things that are not unnested computations:
//
//  - The body of a fusion node.  IrEmitterUnnested emits the relevant code
//    within a kernel function using FusedIrEmitter. (FusedIrEmitter is not
//    really an IrEmitter, but is more an "IR generator generator".)
//
class IrEmitterUnnested : public IrEmitter {
 public:
  std::string_view platform_name() const {
    return ir_emitter_context_->platform_name();
  }

  IrEmitterUnnested(const IrEmitterUnnested&) = delete;
  IrEmitterUnnested& operator=(const IrEmitterUnnested&) = delete;

  static std::unique_ptr<IrEmitterUnnested> Create(
      IrEmitterContext* ir_emitter_context);

  // Transfers the ownership of thunk_sequence_ out.
  std::unique_ptr<SequentialThunk> ConsumeThunkSequence(
      Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo{}) {
    return std::make_unique<SequentialThunk>(thunk_info,
                                             std::move(thunk_sequence_));
  }

  // Emits code for the given HLO computation.
  //
  // Also populates related information to 'ir_emitter_context_' for
  // large-constant initializations. Large constants don't get initializers in
  // the generated code and so must be initialized by ZKX. The value of these
  // constants will be stored in 'content'. Constants with initializers in the
  // generated code will have empty 'content'.
  absl::Status EmitHloComputation(const HloComputation* computation);

 private:
  explicit IrEmitterUnnested(IrEmitterContext* ir_emitter_context);

  absl::Status EmitCommandBufferThunk(const HloInstruction* instr);

  // IrEmitterUnnested handles the following instructions differently from
  // IrEmitter. It also mixes in some special handling for custom kernels
  // via the ThunkEmitter.
  absl::Status EmitConstant(const HloConstantInstruction* instr);

  absl::Status EmitConditional(const HloInstruction* instr);
  absl::Status EmitFusion(const HloFusionInstruction* instr);
  absl::Status EmitCopy(const HloInstruction* instr);
  absl::Status EmitSlice(const HloInstruction* instr);
  absl::Status EmitWhile(const HloInstruction* instr);

  absl::Status EmitHloInstruction(const HloInstruction* instr);

  void AddThunkToThunkSequence(std::unique_ptr<Thunk> thunk) {
    if (emit_group_thunks_) {
      scoped_thunk_sequence_.emplace_back(std::move(thunk));
      return;
    }
    thunk_sequence_.emplace_back(std::move(thunk));
  }

  // Load data from potentially unaligned address. If address is offset by
  // `alignment_bytes`, data is read in the unit of `alignment_bytes` to avoid
  // memory read misalignment in CUDA; otherwise, the entire data are loaded
  // from the given memory address.
  //
  //   address: the memory address to load data from.
  //   data_type: the type of data to load.
  //   alignment_bytes: the number of bytes required to align. The number of
  //     bytes of the data_type must be divisible by alignment_bytes.

  absl::StatusOr<std::unique_ptr<Thunk>> BuildWhileThunk(
      const HloInstruction* instr, const Thunk::ThunkInfo& thunk_info,
      std::optional<int64_t> trip_count);

  absl::StatusOr<BufferAllocation::Slice> GetAllocationSliceForHlo(
      const HloInstruction* instr, const ShapeIndex& index = {}) const;

  ThunkSequence thunk_sequence_;
  ThunkSequence scoped_thunk_sequence_;
  bool emit_group_thunks_ = false;
};

}  // namespace zkx::gpu

#endif  // ZKX_SERVICE_GPU_IR_EMITTER_UNNESTED_H_
