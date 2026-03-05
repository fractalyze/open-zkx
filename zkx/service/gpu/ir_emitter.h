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

#ifndef ZKX_SERVICE_GPU_IR_EMITTER_H_
#define ZKX_SERVICE_GPU_IR_EMITTER_H_

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"

#include "zkx/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/service/gpu/ir_emitter_context.h"

namespace zkx::gpu {

// Abstract base class for translating HLO graphs to LLVM IR for a GPU.
//
// There are two concrete subclasses of IrEmitter: IrEmitterNested and
// IrEmitterUnnested. In the unnested variety, each HLO gets its own kernel
// function, whereas in the nested version the whole computation is emitted as
// one *non-kernel* function.
//
// In ZKX, kernel functions never call other kernel functions. This means that
// if we have a kernel -- e.g. implementing a kReduce HLO -- that wants to use
// an HLO computation as a "subroutine" -- e.g. the HLO computation that
// specifies how to reduce two elements -- then the subroutine computation must
// be emitted using IrEmitterNested.
//
// Fusion nodes are a special case. A fusion node is emitted using
// IrEmitterUnnested, but the code is generated using FusedIrEmitter, which is
// not a subclass of gpu::IrEmitter, and in fact is better understood as an IR
// generator generator.  See comments on that class.
class IrEmitter : public DfsHloVisitorWithDefault {
 public:
  IrEmitter(const IrEmitter&) = delete;
  IrEmitter& operator=(const IrEmitter&) = delete;

  absl::Status DefaultAction(HloInstruction* hlo) override;
  absl::Status HandleConstant(HloInstruction* constant) override;
  absl::Status HandleGetTupleElement(
      HloInstruction* get_tuple_element) override;
  absl::Status HandleParameter(HloInstruction* parameter) override;
  absl::Status HandleTuple(HloInstruction* tuple) override;
  absl::Status HandleScatter(HloInstruction* scatter) override;
  absl::Status HandleCall(HloInstruction* call) override;
  absl::Status FinishVisit(HloInstruction* root) override {
    return absl::OkStatus();
  }

 protected:
  // Constructs an IrEmitter with the given IrEmitter context.
  // ir_emitter_context is owned by the caller and should outlive the IrEmitter
  // object.
  explicit IrEmitter(IrEmitterContext* ir_emitter_context, bool is_nested);

  // Helper for calling HloToIrBindings::GetIrArray.
  //
  // Gets the IrArray which contains inst.  This array has metadata that makes
  // it valid only within the IR that implements consumer.  If you are
  // implementing an HLO and want to get its own output buffer, call
  // GetIrArray(hlo, hlo).

  IrEmitterContext* ir_emitter_context_;  // not owned
  llvm::Module* module_;                  // not owned

  // The following fields track the IR emission state. According to LLVM memory
  // management rules, their memory is owned by the module.
  llvm::IRBuilder<> b_;

  // Mapping from HLO to its underlying LLVM value.
};

}  // namespace zkx::gpu

#endif  // ZKX_SERVICE_GPU_IR_EMITTER_H_
