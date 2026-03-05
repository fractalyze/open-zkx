/*Copyright 2022 The OpenXLA Authors.

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

#include "zkx/service/gpu/ir_emitter_unnested.h"

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/backends/gpu/codegen/fusion_emitter.h"
#include "zkx/backends/gpu/codegen/fusions.h"
#include "zkx/backends/gpu/runtime/command_buffer_cmd.h"
#include "zkx/backends/gpu/runtime/command_buffer_cmd_emitter.h"
#include "zkx/backends/gpu/runtime/command_buffer_thunk.h"
#include "zkx/backends/gpu/runtime/wait_for_streams_thunk.h"
#include "zkx/backends/gpu/runtime/while_thunk.h"
#include "zkx/hlo/ir/hlo_casting_utils.h"
#include "zkx/service/gpu/hlo_fusion_analysis.h"
#include "zkx/service/gpu/ir_emission_utils.h"
#include "zkx/service/gpu/stream_executor_util.h"
#include "zkx/service/llvm_ir/buffer_assignment_util.h"
#include "zkx/status_macros.h"

namespace zkx::gpu {

IrEmitterUnnested::IrEmitterUnnested(IrEmitterContext* ir_emitter_context)
    : IrEmitter(ir_emitter_context, /*is_nested=*/false) {}

std::unique_ptr<IrEmitterUnnested> IrEmitterUnnested::Create(
    IrEmitterContext* ir_emitter_context) {
  return std::unique_ptr<IrEmitterUnnested>(
      new IrEmitterUnnested(ir_emitter_context));
}

absl::Status IrEmitterUnnested::EmitConstant(
    const HloConstantInstruction* instr) {
  TF_ASSIGN_OR_RETURN(DenseDataIntermediate content,
                      LiteralToZkxFormat(instr->literal()));

  int element_bytes =
      primitive_util::ByteWidth(instr->literal().shape().element_type());
  TF_RET_CHECK(content.span().size() % element_bytes == 0);
  // Treat packed constants as a byte constant.
  int num_elements = content.span().size() / element_bytes;

  std::string global_name = llvm_ir::ConstantHloToGlobalName(*instr);
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                      GetAllocationSliceForHlo(instr, {}));

  ir_emitter_context_->emit_constant(num_elements, element_bytes, global_name,
                                     slice.index(), std::move(content), &b_);
  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitConditional(const HloInstruction* instr) {
  return absl::UnimplementedError("Not implemented for EmitConditional");
}

absl::Status IrEmitterUnnested::EmitCommandBufferThunk(
    const HloInstruction* instr) {
  // Spawn a new IrEmitterUnnested to emit thunks for the command buffer
  // computation. Then convert emitted thunks to a sequence of CommandBufferCmd.
  // The resulting thunk added to the thunk sequence is a CommandBufferThunk.
  // Thunks emitted from the command buffer computation are discarded.
  DCHECK_EQ(instr->called_computations().size(), 1);
  const HloComputation* command_buffer = instr->called_computations().front();
  auto ir_emitter = IrEmitterUnnested::Create(ir_emitter_context_);
  TF_RETURN_IF_ERROR(ir_emitter->EmitHloComputation(command_buffer));
  std::unique_ptr<SequentialThunk> thunk_sequence =
      ir_emitter->ConsumeThunkSequence();

  // Maybe serialize all commands in a sequence by forcing barriers between all
  // recorded commands. This guarantees that we execute all device operations
  // in the exact same order as a thunk sequence.
  CommandBufferCmdSequence::SynchronizationMode synchronization_mode =
      ir_emitter_context_->debug_options()
              .zkx_gpu_graph_enable_concurrent_region()
          ? CommandBufferCmdSequence::SynchronizationMode::kAutomatic
          : CommandBufferCmdSequence::SynchronizationMode::kSerialize;

  TF_ASSIGN_OR_RETURN(
      CommandBufferCmdSequence cmd_sequence,
      ConvertToCommands(thunk_sequence->thunks(), synchronization_mode));

  AddThunkToThunkSequence(std::make_unique<CommandBufferThunk>(
      std::move(cmd_sequence), Thunk::ThunkInfo::WithProfileAnnotation(instr),
      std::move(thunk_sequence),
      ir_emitter_context_->debug_options()
          .zkx_enable_command_buffers_during_profiling()));

  return absl::OkStatus();
}

absl::StatusOr<BufferAllocation::Slice>
IrEmitterUnnested::GetAllocationSliceForHlo(const HloInstruction* instr,
                                            const ShapeIndex& index) const {
  return GetAllocationSlice(ir_emitter_context_->buffer_assignment(), instr,
                            index);
}

absl::Status IrEmitterUnnested::EmitFusion(const HloFusionInstruction* instr) {
  const se::DeviceDescription& device_info =
      ir_emitter_context_->gpu_device_info();
  const HloFusionAnalysis fusion_analysis =
      HloFusionAnalysis::Create(*instr, device_info);
  VLOG(3) << "IrEmitterUnnested::EmitFusion:start";
  std::unique_ptr<FusionInterface> emitter = GetFusionEmitter(HloFusionInfo(
      fusion_analysis, instr, &ir_emitter_context_->buffer_assignment()));
  TF_ASSIGN_OR_RETURN(auto result, emitter->Emit(*ir_emitter_context_, *instr));

  const ExecutionStreamAssignment& stream_assignment =
      ir_emitter_context_->execution_stream_assignment();
  for (std::unique_ptr<Thunk>& thunk : result.thunks) {
    TF_ASSIGN_OR_RETURN(ExecutionStreamId execution_stream_id,
                        stream_assignment.GetSyncExecutionStreamId(instr));
    thunk->set_execution_stream_id(execution_stream_id);
    AddThunkToThunkSequence(std::move(thunk));
  }
  VLOG(3) << "IrEmitterUnnested::EmitFusion:complete";
  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitCopy(const HloInstruction* instr) {
  TF_RET_CHECK(LayoutUtil::LayoutsInShapesEqual(
      instr->operand(0)->shape(), instr->shape(),
      Layout::Equal().MinorToMajorOnly()));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice src_buffer,
                      GetAllocationSliceForHlo(instr->operand(0)));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice dst_buffer,
                      GetAllocationSliceForHlo(instr));
  AddThunkToThunkSequence(std::make_unique<DeviceToDeviceCopyThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(instr),
      /*source_buffer=*/src_buffer,
      /*destination_buffer=*/dst_buffer,
      /*mem_size=*/src_buffer.size()));
  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitSlice(const HloInstruction* instr) {
  const Shape& src_shape = instr->operand(0)->shape();

  // Only support unit strides.
  for (int64_t i = 0; i < src_shape.rank(); ++i) {
    if (instr->slice_strides(i) != 1) {
      return absl::UnimplementedError(
          "Non-unit stride slice not supported as standalone instruction");
    }
  }

  // Check contiguity: for all but the most-major dimension, the slice must
  // cover the full extent. Otherwise, the slice is non-contiguous in memory
  // and cannot be emitted as a single memcpy.
  const auto& minor_to_major = src_shape.layout().minor_to_major();
  for (int64_t i = 0; i < src_shape.rank() - 1; ++i) {
    int64_t dim = minor_to_major[i];
    if (instr->slice_starts(dim) != 0 ||
        instr->slice_limits(dim) != src_shape.dimensions(dim)) {
      return absl::UnimplementedError(
          "Non-contiguous slice not supported as standalone instruction");
    }
  }

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice src_buffer,
                      GetAllocationSliceForHlo(instr->operand(0)));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice dst_buffer,
                      GetAllocationSliceForHlo(instr));

  // Compute byte offset. Only the most-major dimension can have non-zero start
  // (verified by the contiguity check above).
  int64_t byte_offset = 0;
  if (src_shape.rank() > 0) {
    int64_t most_major_dim = minor_to_major.back();
    int64_t element_size =
        ShapeUtil::ByteSizeOfPrimitiveType(src_shape.element_type());
    int64_t stride = 1;
    for (int64_t i = 0; i < src_shape.rank() - 1; ++i) {
      stride *= src_shape.dimensions(minor_to_major[i]);
    }
    byte_offset = instr->slice_starts(most_major_dim) * stride * element_size;
  }

  BufferAllocation::Slice adjusted_src(src_buffer.allocation(),
                                       src_buffer.offset() + byte_offset,
                                       dst_buffer.size());

  AddThunkToThunkSequence(std::make_unique<DeviceToDeviceCopyThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(instr),
      /*source_buffer=*/adjusted_src,
      /*destination_buffer=*/dst_buffer,
      /*mem_size=*/dst_buffer.size()));
  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitWhile(const HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto config,
                      instr->backend_config<WhileLoopBackendConfig>());

  std::optional<int64_t> trip_count = std::nullopt;
  if (config.has_known_trip_count()) trip_count = config.known_trip_count().n();

  TF_ASSIGN_OR_RETURN(
      auto thunk,
      BuildWhileThunk(instr, Thunk::ThunkInfo::WithProfileAnnotation(instr),
                      trip_count));

  AddThunkToThunkSequence(std::move(thunk));
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Thunk>> IrEmitterUnnested::BuildWhileThunk(
    const HloInstruction* instr, const Thunk::ThunkInfo& thunk_info,
    std::optional<int64_t> trip_count) {
  HloComputation* condition = instr->while_condition();
  HloComputation* body = instr->while_body();

  // Generate thunk sequence for while 'condition'.
  auto ir_emitter_condition = IrEmitterUnnested::Create(ir_emitter_context_);
  TF_RETURN_IF_ERROR(ir_emitter_condition->EmitHloComputation(condition));

  // Generate thunk sequence for while 'body'.
  auto ir_emitter_body = IrEmitterUnnested::Create(ir_emitter_context_);
  TF_RETURN_IF_ERROR(ir_emitter_body->EmitHloComputation(body));

  // Buffer slice holding while loop predicate.
  TF_ASSIGN_OR_RETURN(
      auto pred, GetAllocationSliceForHlo(condition->root_instruction(), {}));

  Thunk::ThunkInfo cond_thunk_info =
      Thunk::ThunkInfo::WithProfileAnnotation(instr);
  cond_thunk_info.profile_annotation += "_condition";
  Thunk::ThunkInfo body_thunk_info =
      Thunk::ThunkInfo::WithProfileAnnotation(instr);
  body_thunk_info.profile_annotation += "_body";

  return std::unique_ptr<Thunk>(new WhileThunk(
      thunk_info, pred,
      ir_emitter_condition->ConsumeThunkSequence(cond_thunk_info),
      ir_emitter_body->ConsumeThunkSequence(body_thunk_info), trip_count));
}

absl::Status IrEmitterUnnested::EmitHloInstruction(
    const HloInstruction* instr) {
  switch (instr->opcode()) {
    case HloOpcode::kCall:
      return EmitCommandBufferThunk(instr);
    case HloOpcode::kConditional:
      return EmitConditional(instr);
    case HloOpcode::kConstant:
      return EmitConstant(Cast<HloConstantInstruction>(instr));
    case HloOpcode::kFusion:
      return EmitFusion(Cast<HloFusionInstruction>(instr));
    case HloOpcode::kCopy:
      return EmitCopy(instr);
    case HloOpcode::kSlice:
      return EmitSlice(instr);
    case HloOpcode::kWhile:
      return EmitWhile(instr);

    // We don't need to emit thunks for these operations because their
    // semantics are encoded by buffers.
    case HloOpcode::kBitcast:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kParameter:
    case HloOpcode::kTuple:
      return absl::OkStatus();
    default:
      LOG(ERROR) << "Unsupported instruction opcode: "
                 << HloOpcodeString(instr->opcode()) << "\nHLO module:\n"
                 << instr->parent()->parent()->ToString();
      return absl::InternalError(
          absl::StrFormat("Unsupported instruction opcode: %s",
                          HloOpcodeString(instr->opcode())));
  }

  return absl::InternalError("Unhandled HLO instruction");
}

absl::Status IrEmitterUnnested::EmitHloComputation(
    const HloComputation* computation) {
  const HloSchedule& schedule = computation->parent()->schedule();
  if (!schedule.is_computation_scheduled(computation))
    return absl::InternalError(absl::StrFormat(
        "Sequence not found for computation: %s", computation->name()));

  const HloInstructionSequence& sequence = schedule.sequence(computation);
  for (HloInstruction* instr : sequence.instructions()) {
    TF_RETURN_IF_ERROR(EmitHloInstruction(instr));
  }
  return absl::OkStatus();
}

}  // namespace zkx::gpu
