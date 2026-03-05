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

#include "zkx/service/hlo_verifier.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"

#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/comparison_util.h"
#include "zkx/hlo/ir/hlo_casting_utils.h"
#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_input_output_alias_config.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_instructions.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/hlo/ir/hlo_schedule.h"
#include "zkx/layout.h"
#include "zkx/layout_util.h"
#include "zkx/permutation_util.h"
#include "zkx/primitive_util.h"
#include "zkx/service/hlo_module_config.h"
#include "zkx/service/shape_inference.h"
#include "zkx/shape.h"
#include "zkx/shape_layout.h"
#include "zkx/shape_util.h"
#include "zkx/status_macros.h"
#include "zkx/util.h"

namespace zkx {
namespace {

bool IsCallerInstruction(HloInstruction *hlo) {
  return HloInstruction::MightHaveCalledComputations(hlo->opcode());
}

absl::Status CheckOperandCount(const HloInstruction *hlo, int expected) {
  if (hlo->operand_count() != expected) {
    return absl::InternalError(
        absl::StrFormat("Expected %d operands for %s instruction: %s", expected,
                        HloOpcodeString(hlo->opcode()), hlo->ToString()));
  }
  return absl::OkStatus();
}

absl::Status CheckNestedComputationThreadNameEqual(
    const HloComputation *comp, bool skip_nested_async_op_check) {
  for (const HloInstruction *instr : comp->instructions()) {
    for (const HloComputation *called_cmp : instr->called_computations()) {
      if (called_cmp->execution_thread() != comp->execution_thread()) {
        return absl::InternalError(absl::StrFormat(
            "Nested computations expects same computation's thread name: %s vs "
            "%s, in called computation `%s` vs caller computation `%s`",
            called_cmp->execution_thread(), comp->execution_thread(),
            called_cmp->name(), comp->name()));
      }
      TF_RETURN_IF_ERROR(CheckNestedComputationThreadNameEqual(
          called_cmp, skip_nested_async_op_check));
    }
  }
  return absl::OkStatus();
}

}  // namespace

/*static*/ absl::Status ShapeVerifier::CheckParameterCount(
    const HloInstruction *calling_instruction,
    const HloComputation *computation, int expected) {
  if (computation->num_parameters() != expected) {
    return absl::InternalError(absl::StrFormat(
        "Expected computation %s called from %s to have %d parameters, has %d",
        computation->name(), calling_instruction->name(), expected,
        computation->num_parameters()));
  }
  return absl::OkStatus();
}

absl::Status ShapeVerifier::Preprocess(HloInstruction *hlo) {
  if (!hlo->called_computations().empty() && !IsCallerInstruction(hlo)) {
    return absl::InternalError(absl::StrFormat(
        "Called computations specified for non-caller instruction %s",
        hlo->ToString()));
  }
  std::optional<int> arity = HloOpcodeArity(hlo->opcode());
  if (arity) {
    TF_RETURN_IF_ERROR(CheckOperandCount(hlo, *arity));
  }
  if (!opts_.allow_unbounded_dynamism && hlo->shape().is_unbounded_dynamic()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Unbounded dynamism is disabled for instruction: %s", hlo->ToString()));
  }
  if (hlo->shape().has_layout()) {
    if (hlo->shape().layout().minor_to_major_size() !=
        hlo->shape().dimensions_size()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Instruction has mismatched minor-to-major size and dimension size: "
          "%s",
          hlo->ToString()));
    }
  }
  return absl::OkStatus();
}

absl::Status ShapeVerifier::HandleElementwiseUnary(HloInstruction *hlo) {
  return CheckUnaryShape(hlo);
}

absl::Status ShapeVerifier::HandleElementwiseBinary(HloInstruction *hlo) {
  return CheckBinaryShape(hlo);
}

absl::Status ShapeVerifier::HandleClamp(HloInstruction *clamp) {
  return CheckTernaryShape(clamp);
}

absl::Status ShapeVerifier::HandleSelect(HloInstruction *select) {
  return CheckTernaryShape(select);
}

absl::Status ShapeVerifier::HandleConcatenate(HloInstruction *concatenate) {
  std::vector<const Shape *> operand_shapes;
  for (const HloInstruction *operand : concatenate->operands()) {
    operand_shapes.push_back(&operand->shape());
  }
  return CheckShape(concatenate,
                    ShapeInference::InferConcatOpShape(
                        operand_shapes, concatenate->concatenate_dimension()));
}

absl::Status ShapeVerifier::HandleConvert(HloInstruction *convert) {
  return CheckShape(convert, ShapeInference::InferConvertShape(
                                 convert->operand(0)->shape(),
                                 convert->shape().element_type()));
}

absl::Status ShapeVerifier::HandleBitcastConvert(HloInstruction *convert) {
  return CheckShape(convert, ShapeInference::InferBitcastConvertShape(
                                 convert->operand(0)->shape(),
                                 convert->shape().element_type()));
}

absl::Status ShapeVerifier::HandleCopy(HloInstruction *copy) {
  return CheckUnaryShape(copy);
}

absl::Status ShapeVerifier::HandleDot(HloInstruction *dot) {
  auto sparsity = Cast<HloDotInstruction>(dot)->sparsity();
  TF_RETURN_IF_ERROR(
      CheckOperandCount(dot, HloDotInstruction::kOperands + sparsity.size()));
  TF_ASSIGN_OR_RETURN(
      const Shape expected,
      ShapeInference::InferDotOpShape(
          dot->operand(0)->shape(), dot->operand(1)->shape(),
          dot->dot_dimension_numbers(),
          /*preferred_element_type=*/dot->shape().element_type(), sparsity));
  for (int i = 0; i < sparsity.size(); ++i) {
    const SparsityDescriptor &descriptor = sparsity[i];
    TF_RET_CHECK(descriptor.index() == 0 || descriptor.index() == 1);
    // clang-format off
    return absl::UnimplementedError("HandleDot with sparsity not supported");
  }
  return CheckShape(dot, expected);
}

bool ShapeVerifier::ShapesSame(const Shape &a, const Shape &b,
                               Shape::Equal equal) {
  if (!opts_.layout_sensitive) {
    return ShapeUtil::Compatible(a, b);
  }
  return equal(a, b);
}

absl::Status
ShapeVerifier::CheckIsTokenOperand(const HloInstruction *instruction,
                                   int64_t operand_no) {
  const HloInstruction *token = instruction->operand(operand_no);
  if (!ShapeUtil::Equal(token->shape(), ShapeUtil::MakeTokenShape())) {
    return absl::InternalError(absl::StrFormat(
        "Expected operand %d to be token-shaped, actual shape is "
        "%s:\n%s",
        operand_no, StringifyShape(token->shape()), instruction->ToString()));
  }
  return absl::OkStatus();
}

absl::Status ShapeVerifier::CheckOperandAndParameter(
    const HloInstruction *instruction, int64_t operand_number,
    const HloComputation *computation, int64_t parameter_number) {
  const HloInstruction *operand = instruction->operand(operand_number);
  const HloInstruction *parameter =
      computation->parameter_instruction(parameter_number);
  if (!ShapesSame(operand->shape(), parameter->shape())) {
    return absl::InternalError(absl::StrFormat(
        "Operand %s shape does not match parameter's %s in %s",
        operand->ToString(), parameter->ToString(), instruction->ToString()));
  }
  return absl::OkStatus();
}

bool ShapeVerifier::HasCompatibleElementTypes(const Shape &shape_0,
                                              const Shape &shape_1,
                                              const Shape &result_shape) {
  return ShapeUtil::SameElementType(shape_0, shape_1) &&
         ShapeUtil::SameElementType(shape_0, result_shape);
}

absl::Status ShapeVerifier::HandleReverse(HloInstruction *reverse) {
  return CheckShape(
      reverse, ShapeInference::InferReverseShape(reverse->operand(0)->shape(),
                                                 reverse->dimensions()));
}

absl::Status ShapeVerifier::HandleConstant(HloInstruction *constant) {
  if (!Cast<HloConstantInstruction>(constant)->HasLiteral()) {
    return absl::InternalError(
        absl::StrFormat("Constant is required to have a valid literal: %s",
                        constant->ToString()));
  }
  return CheckShape(constant, constant->literal().shape(),
                    /*only_compare_minor_to_major_in_layout=*/true);
}

absl::Status ShapeVerifier::HandleIota(HloInstruction *hlo) {
  auto *iota = Cast<HloIotaInstruction>(hlo);
  if (!iota->shape().IsArray()) {
    return absl::InternalError("Iota does not support non-array result.");
  }
  const int64_t rank = iota->shape().rank();
  if (rank == 0) {
    return absl::InternalError("Iota does not support scalars.");
  }
  int64_t iota_dimension = iota->iota_dimension();
  if (iota_dimension >= rank || iota_dimension < 0) {
    return absl::InternalError(
        "The iota dimension cannot go beyond the operation rank or be "
        "negative.");
  }

  PrimitiveType primitive_type = iota->shape().element_type();
  if (!primitive_util::IsIntegralType(primitive_type)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Only support iota of integral primitive types, got %s",
                        PrimitiveType_Name(primitive_type)));
  }

  return absl::OkStatus();
}

absl::Status
ShapeVerifier::HandleGetTupleElement(HloInstruction *get_tuple_element) {
  return CheckShape(get_tuple_element,
                    ShapeInference::InferGetTupleElementShape(
                        get_tuple_element->operand(0)->shape(),
                        get_tuple_element->tuple_index()));
}

namespace {

absl::Status SameElementTypesForOperandsAndToApplyParameters(
    const HloInstruction &instruction, int64_t num_operands_to_check) {
  const ProgramShape &to_apply = instruction.to_apply()->ComputeProgramShape();
  for (int i = 0; i < num_operands_to_check; ++i) {
    const Shape &parameter_shape = to_apply.parameters(i);
    const Shape &operand_shape = instruction.operands()[i]->shape();
    if (!ShapeUtil::SameElementType(parameter_shape, operand_shape)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Shape mismatch between to_apply computation"
                          " parameter and operand %d in %s.",
                          i, instruction.ToString()));
    }
  }
  return absl::OkStatus();
}

} // namespace

absl::Status ShapeVerifier::HandleReduce(HloInstruction *reduce) {
  if (reduce->operand_count() % 2 != 0) {
    return absl::InternalError(absl::StrFormat(
        "Expected an even number of operands for %s instruction: %s",
        HloOpcodeString(reduce->opcode()), reduce->ToString()));
  }

  std::vector<const Shape *> operand_shapes;
  for (const HloInstruction *operand : reduce->operands()) {
    operand_shapes.push_back(&operand->shape());
  }
  TF_RETURN_IF_ERROR(
      CheckShape(reduce, ShapeInference::InferReduceShape(
                             operand_shapes, reduce->dimensions(),
                             reduce->to_apply()->ComputeProgramShape())));

  return SameElementTypesForOperandsAndToApplyParameters(
      *reduce, reduce->operand_count());
}

absl::Status ShapeVerifier::HandleBitcast(HloInstruction *bitcast) {
  const Shape &output_shape = bitcast->shape();
  const Shape &operand_shape = bitcast->operand(0)->shape();
  if (opts_.layout_sensitive &&
      opts_.shape_size(output_shape) != opts_.shape_size(operand_shape)) {
    // Allow bitcast that has the same data size but different trailing
    // paddings.
    if (!opts_.allow_bitcast_to_have_different_size ||
        !(output_shape.is_static() && operand_shape.is_static() &&
          (ShapeUtil::ArrayDataSize(output_shape) ==
           ShapeUtil::ArrayDataSize(operand_shape)))) {
      return absl::InternalError(absl::StrFormat(
          "%s: Bitcast cannot have different shape sizes of output (%d) and "
          "operand "
          "(%d) (%s) (%s)",
          bitcast->ToString(), opts_.shape_size(output_shape),
          opts_.shape_size(operand_shape), output_shape.ToString(true),
          operand_shape.ToString(true)));
    }
  }
  return absl::OkStatus();
}

absl::Status ShapeVerifier::HandleBitReverse(HloInstruction *bit_reverse) {
  return CheckShape(bit_reverse, ShapeInference::InferBitReverseShape(
                                     bit_reverse->operand(0)->shape(),
                                     bit_reverse->dimensions()));
}

absl::Status ShapeVerifier::HandleBroadcast(HloInstruction *broadcast) {
  // HLO broadcast has no exact analog at the client level so there is no
  // ShapeInference method. Check the output shape explicitly.
  const Shape &operand_shape = broadcast->operand(0)->shape();
  TF_RET_CHECK(SameElementType(broadcast->shape(), operand_shape))
      << broadcast->ToString();
  TF_RET_CHECK(operand_shape.rank() == broadcast->dimensions().size())
      << broadcast->ToString();
  for (int64_t operand_dimension = 0; operand_dimension < operand_shape.rank();
       ++operand_dimension) {
    int64_t output_dimension = broadcast->dimensions()[operand_dimension];
    TF_RET_CHECK((output_dimension < broadcast->shape().rank()) &&
                 output_dimension >= 0 &&
                 (broadcast->shape().dimensions(output_dimension) ==
                  operand_shape.dimensions(operand_dimension)))
        << broadcast->ToString() << " operand shape " << operand_shape;
  }
  return absl::OkStatus();
}

absl::Status
ShapeVerifier::HandleDynamicReshape(HloInstruction *dynamic_reshape) {
  const Shape &operand_shape = dynamic_reshape->operand(0)->shape();
  TF_RET_CHECK(SameElementType(dynamic_reshape->shape(), operand_shape));
  TF_RET_CHECK(ShapeUtil::ElementsIn(dynamic_reshape->shape()) ==
               ShapeUtil::ElementsIn(operand_shape));
  TF_RET_CHECK(dynamic_reshape->shape().rank() + 1 ==
               dynamic_reshape->operand_count());
  for (int64_t i = 1; i < dynamic_reshape->operand_count(); ++i) {
    TF_RET_CHECK(dynamic_reshape->operand(i)->shape().element_type() == S32);
  }
  return absl::OkStatus();
}

absl::Status ShapeVerifier::HandleReshape(HloInstruction *reshape) {
  const Shape &operand_shape = reshape->operand(0)->shape();
  TF_RET_CHECK(SameElementType(reshape->shape(), operand_shape));
  TF_RET_CHECK(ShapeUtil::ElementsIn(reshape->shape()) ==
               ShapeUtil::ElementsIn(operand_shape));
  return absl::OkStatus();
}

absl::Status ShapeVerifier::HandleTranspose(HloInstruction *transpose) {
  return CheckShape(
      transpose, ShapeInference::InferTransposeShape(
                     transpose->operand(0)->shape(), transpose->dimensions()));
}

absl::Status ShapeVerifier::HandleParameter(HloInstruction *hlo) {
  return absl::OkStatus();
}

absl::Status ShapeVerifier::HandleFusion(HloInstruction *fusion) {
  if (fusion->called_computations().size() != 1) {
    return absl::InternalError(absl::StrFormat(
        "Fusion has a non-unary number of called computations (%s)",
        fusion->ToString()));
  }
  const Shape &root_computation_shape =
      fusion->called_computations()[0]->root_instruction()->shape();
  if (!ShapesSame(fusion->shape(), root_computation_shape)) {
    return absl::InternalError(absl::StrFormat(
        "Fused computation shape (%s) is not equal to the fusion shape (%s)",
        root_computation_shape.ToString(true), fusion->shape().ToString(true)));
  }

  auto &fused_parameters = fusion->fused_parameters();
  if (fused_parameters.size() > fusion->operand_count()) {
    return absl::InternalError(absl::StrFormat(
        "Fused parameter count (%d) is greater than the number of operands (%d)"
        " passed to the fusion instruction in: %s.",
        fused_parameters.size(), fusion->operand_count(), fusion->ToString()));
  }
  for (HloInstruction *fused_param : fused_parameters) {
    int64_t param_no = fused_param->parameter_number();
    if (!ShapesSame(fused_param->shape(), fusion->operand(param_no)->shape())) {
      return absl::InternalError(absl::StrFormat(
          "Shape mismatch between parameter number %d and its operand in "
          "%s.",
          param_no, fusion->ToString()));
    }
  }
  const HloFusionInstruction *casted_fusion =
      DynCast<const HloFusionInstruction>(fusion);
  for (const auto &pair : casted_fusion->output_to_operand_aliasing()) {
    TF_RET_CHECK(pair.second.first < casted_fusion->operand_count())
        << "Invalid aliasing operand index.";
    TF_RET_CHECK(ShapeUtil::IndexIsValid(
        casted_fusion->operand(pair.second.first)->shape(), pair.second.second))
        << "Invalid aliasing operand shape index.";
    TF_RET_CHECK(ShapeUtil::IndexIsValid(casted_fusion->shape(), pair.first))
        << "Invalid aliasing output shape index.";
    const Shape &output_subshape =
        ShapeUtil::GetSubshape(casted_fusion->shape(), pair.first);
    const Shape &operand_subshape = ShapeUtil::GetSubshape(
        casted_fusion->operand(pair.second.first)->shape(), pair.second.second);
    if (opts_.layout_sensitive) {
      if (casted_fusion->IsFused()) {
        // Nested fusions can have aliasing that does not require the
        // tiling/memory space assignment to be the same in order to alias.
        TF_RET_CHECK(
            Shape::Equal().IgnoreTilesInLayout().IgnoreMemorySpaceInLayout()(
                operand_subshape, output_subshape))
            << "Different aliasing shapes: "
            << operand_subshape.ToString(/*print_layout=*/true) << " vs "
            << output_subshape.ToString(/*print_layout=*/true);
      } else {
        TF_RET_CHECK(Shape::Equal()(operand_subshape, output_subshape))
            << "Different aliasing shapes: "
            << operand_subshape.ToString(/*print_layout=*/true) << " vs "
            << output_subshape.ToString(/*print_layout=*/true);
      }
    } else {
      TF_RET_CHECK(ShapeUtil::Compatible(output_subshape, operand_subshape))
          << "Different aliasing shapes: "
          << operand_subshape.ToString(/*print_layout=*/true) << " vs "
          << output_subshape.ToString(/*print_layout=*/true);
    }
  }
  return absl::OkStatus();
}

absl::Status ShapeVerifier::HandleCall(HloInstruction *call) {
  TF_RETURN_IF_ERROR(
      CheckParameterCount(call, call->to_apply(), call->operand_count()));
  for (int64_t i = 0; i < call->to_apply()->num_parameters(); ++i) {
    TF_RETURN_IF_ERROR(CheckOperandAndParameter(call, i, call->to_apply(), i));
  }
  if (call->is_composite()) {
    TF_RET_CHECK(call->has_frontend_attributes())
        << "A composite call op must have frontend attributes";
    auto map = call->frontend_attributes().map();
    if (auto name = map.find("composite.name");
        name == map.end() || name->second.empty()) {
      return absl::InvalidArgumentError(
          "A composite call op must have frontend attributes with key "
          "composite.name whose value is non-empty");
    }
    if (auto attributes = map.find("composite.attributes");
        attributes != map.end() && attributes->second.empty()) {
      return absl::InvalidArgumentError(
          "A composite call op must have frontend attributes with key "
          "composite.attributes whose value is default: {} or non-empty");
    }
    if (auto version_str = map.find("composite.version");
        version_str != map.end()) {
      int64_t version = 0;
      if (!absl::SimpleAtoi(version_str->second, &version) || version < 0) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "A composite call op must have frontend attributes with a "
            "composite.version whose value is a non-negative integer but got: "
            "%s",
            version_str->second));
      }
    }
  }
  // The shape of kCall should match the shape of the computation it calls.
  return CheckShape(call, call->to_apply()->root_instruction()->shape());
}

absl::Status ShapeVerifier::HandleSlice(HloInstruction *slice) {
  return CheckShape(slice,
                    ShapeInference::InferSliceShape(
                        slice->operand(0)->shape(), slice->slice_starts(),
                        slice->slice_limits(), slice->slice_strides()));
}

absl::Status ShapeVerifier::HandleDynamicSlice(HloInstruction *dynamic_slice) {
  return CheckShape(
      dynamic_slice,
      ShapeInference::InferDynamicSliceShape(
          dynamic_slice->operand(0)->shape(),
          Cast<HloDynamicSliceInstruction>(dynamic_slice)->index_shapes(),
          dynamic_slice->dynamic_slice_sizes()));
}

absl::Status
ShapeVerifier::HandleDynamicUpdateSlice(HloInstruction *dynamic_update_slice) {
  return CheckShape(
      dynamic_update_slice,
      ShapeInference::InferDynamicUpdateSliceShape(
          dynamic_update_slice->operand(0)->shape(),
          dynamic_update_slice->operand(1)->shape(),
          Cast<HloDynamicUpdateSliceInstruction>(dynamic_update_slice)
              ->index_shapes()));
}

absl::Status ShapeVerifier::HandleTuple(HloInstruction *tuple) {
  return CheckVariadicShape(tuple);
}

absl::Status ShapeVerifier::HandleMap(HloInstruction *map) {
  std::vector<const Shape *> operand_shapes;
  int64_t max_operand_rank = 0;
  for (const HloInstruction *operand : map->operands()) {
    operand_shapes.push_back(&operand->shape());
    max_operand_rank = std::max(max_operand_rank, operand->shape().rank());
  }
  // TODO(b/65689298) Remove code below once Map is generalized to accept
  // arbitrary map dimensions.
  std::vector<int64_t> map_dims(max_operand_rank);
  std::iota(map_dims.begin(), map_dims.end(), 0);

  TF_RETURN_IF_ERROR(CheckShape(
      map, ShapeInference::InferMapShape(operand_shapes,
                                         map->to_apply()->ComputeProgramShape(),
                                         map_dims)));

  return SameElementTypesForOperandsAndToApplyParameters(*map,
                                                         map->operand_count());
}

absl::Status ShapeVerifier::HandleReduceWindow(HloInstruction *reduce_window) {
  auto reduce_window_instr = Cast<HloReduceWindowInstruction>(reduce_window);
  auto input_shapes = reduce_window_instr->input_shapes();
  auto init_shapes = reduce_window_instr->init_value_shapes();
  TF_RETURN_IF_ERROR(CheckShape(
      reduce_window, ShapeInference::InferReduceWindowShape(
                         input_shapes, init_shapes, reduce_window->window(),
                         reduce_window->to_apply()->ComputeProgramShape())));

  return SameElementTypesForOperandsAndToApplyParameters(
      *reduce_window, reduce_window->operand_count());
}

absl::Status ShapeVerifier::HandleWhile(HloInstruction *zkx_while) {
  TF_RETURN_IF_ERROR(
      CheckParameterCount(zkx_while, zkx_while->while_body(), 1));
  TF_RETURN_IF_ERROR(
      CheckParameterCount(zkx_while, zkx_while->while_condition(), 1));
  TF_RETURN_IF_ERROR(
      CheckOperandAndParameter(zkx_while, 0, zkx_while->while_body(), 0));
  TF_RETURN_IF_ERROR(
      CheckOperandAndParameter(zkx_while, 0, zkx_while->while_condition(), 0));
  const Shape &conditional_shape =
      zkx_while->while_condition()->root_instruction()->shape();
  if (!ShapeUtil::Compatible(conditional_shape,
                             ShapeUtil::MakeShape(PRED, {}))) {
    return absl::InternalError(absl::StrFormat(
        "Conditional computation shape does not lead to a scalar predicate "
        "shape: %s",
        StringifyShape(conditional_shape)));
  }
  // The shape of kWhile should match the shape of the body computation it
  // calls.
  return CheckShape(zkx_while,
                    zkx_while->while_body()->root_instruction()->shape());
}

absl::Status ShapeVerifier::HandleConditional(HloInstruction *conditional) {
  if (!ShapeUtil::IsScalar(conditional->operand(0)->shape())) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "The first operand of conditional must be a scalar. Got %s",
        conditional->operand(0)->shape().DebugString()));
  }
  const int num_branches = conditional->branch_count();
  PrimitiveType operand0_type = conditional->operand(0)->shape().element_type();
  if (operand0_type == PRED) {
    TF_RET_CHECK(num_branches == 2);
  } else {
    if (operand0_type != S32) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "The first operand of indexed conditional must be a scalar of S32. "
          "Got type %s.",
          PrimitiveType_Name(operand0_type)));
    }
    TF_RET_CHECK(num_branches >= 1);
  }
  TF_RETURN_IF_ERROR(CheckOperandCount(conditional, num_branches + 1));
  for (int j = 0; j < num_branches; ++j) {
    TF_RETURN_IF_ERROR(CheckParameterCount(
        conditional, conditional->branch_computation(j), 1));
    TF_RETURN_IF_ERROR(CheckOperandAndParameter(
        conditional, j + 1, conditional->branch_computation(j), 0));
    TF_RETURN_IF_ERROR(CheckShape(
        conditional,
        conditional->branch_computation(j)->root_instruction()->shape()));
  }
  return absl::OkStatus();
}

absl::Status ShapeVerifier::HandlePad(HloInstruction *pad) {
  return CheckShape(pad, ShapeInference::InferPadShape(pad->operand(0)->shape(),
                                                       pad->operand(1)->shape(),
                                                       pad->padding_config()));
}

namespace {
absl::Status
CheckCallableInstructionThreadName(const HloInstruction *instruction,
                                   bool skip_nested_async_op_check) {
  for (const HloComputation *computation : instruction->called_computations()) {
    if (instruction->parent() != nullptr) {
      if (instruction->parent()->execution_thread() !=
          computation->execution_thread()) {
        return absl::InternalError(absl::StrFormat(
            "callable instruction %s expects parent computation thread name "
            "same as called computation's thread name (%s vs %s).",
            instruction->ToString(), instruction->parent()->execution_thread(),
            computation->execution_thread()));
      }
    }
    TF_RETURN_IF_ERROR(CheckNestedComputationThreadNameEqual(
        computation, skip_nested_async_op_check));
  }
  return absl::OkStatus();
}
} // namespace

absl::Status ShapeVerifier::HandleGather(HloInstruction *gather) {
  return CheckShape(gather, ShapeInference::InferGatherShape(
                                gather->operand(0)->shape(),
                                gather->operand(1)->shape(),
                                gather->gather_dimension_numbers(),
                                gather->gather_slice_sizes()));
}

absl::Status ShapeVerifier::HandleScatter(HloInstruction *scatter) {
  absl::InlinedVector<const Shape *, 3> arg_shapes;
  arg_shapes.reserve(scatter->operand_count());
  for (const HloInstruction *operand : scatter->operands()) {
    arg_shapes.push_back(&operand->shape());
  }
  return CheckShape(scatter,
                    ShapeInference::InferScatterShape(
                        arg_shapes, scatter->to_apply()->ComputeProgramShape(),
                        scatter->scatter_dimension_numbers()));
}

absl::Status ShapeVerifier::HandleGetDimensionSize(HloInstruction *get_size) {
  return CheckShape(get_size,
                    ShapeInference::InferGetDimensionSizeShape(
                        get_size->operand(0)->shape(), get_size->dimension()));
}

absl::Status ShapeVerifier::HandleSetDimensionSize(HloInstruction *set_size) {
  return CheckShape(set_size,
                    ShapeInference::InferSetDimensionSizeShape(
                        set_size->operand(0)->shape(),
                        set_size->operand(1)->shape(), set_size->dimension()));
}

absl::Status
ShapeVerifier::CheckShape(const HloInstruction *instruction,
                          const Shape &inferred_shape,
                          bool only_compare_minor_to_major_in_layout) {
  // Check if the output shape matches the expected shape.
  //
  // We treat BF16 and F32 as compatible types if mixed precision is allowed,
  // but only when the instruction defines the BF16/F32 buffer.
  bool equal = [&] {
    switch (instruction->opcode()) {
    // The opcodes below can't have implicit layout conversions, nor can they
    // implicitly transform f32 -> bf16.  Fundamentally these are either
    // reinterpreting existing data (e.g. kBitcast) or shuffling data around
    // without modifying it (e.g. kGetTupleElement).
    case HloOpcode::kBitcast:
    case HloOpcode::kCall:
    case HloOpcode::kConditional:
    case HloOpcode::kConstant:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kParameter:
    case HloOpcode::kTuple:
    case HloOpcode::kWhile: {
      Shape::Equal equal;
      if (only_compare_minor_to_major_in_layout) {
        equal.MinorToMajorOnlyInLayout();
      }
      return ShapesSame(instruction->shape(), inferred_shape, equal);
    }
    case HloOpcode::kDynamicUpdateSlice: {
      Shape::Equal equal;
      if (only_compare_minor_to_major_in_layout) {
        equal.MinorToMajorOnlyInLayout();
      }
      if (instruction->parent()->IsFusionComputation()) {
        // For DynamicUpdateSlice it has an "in-place" update semantics, but
        // inside of fusions memory space propagation doesn't propagate the
        // memory spaces all the way, causing possible mismatches. Relax the
        // constraint in that condition. Tiling also is not necessarily
        // meaningful within fusions, so we can relax this as well.
        equal.IgnoreMemorySpaceInLayout().IgnoreTilesInLayout();
      }
      return ShapesSame(instruction->shape(), inferred_shape, equal);
    }
    case HloOpcode::kCopy: {
      // Disallow host offloading copies which change FpPrecision.
      if (opts_.IsLayoutSensitive()) {
        if (instruction->shape().has_layout() && inferred_shape.has_layout()) {
          int64_t instruction_memory_space =
              instruction->shape().layout().memory_space();
          int64_t operand_memory_space = inferred_shape.layout().memory_space();
          if (instruction_memory_space != operand_memory_space &&
              (instruction_memory_space == Layout::kHostMemorySpace ||
               operand_memory_space == Layout::kHostMemorySpace)) {
            if (instruction_memory_space == Layout::kHostMemorySpace) {
              // Unfortunately it might still be a host->host copy before
              // memory space is propagated. A transpose is allowed in that
              // case.
              return instruction->shape().element_type() ==
                     inferred_shape.element_type();
            }
            // A host->device copy or a device->host copy cannot do a
            // transpose.
            return Shape::Equal().IgnoreMemorySpaceInLayout()(
                instruction->shape(), inferred_shape);
          }
        }
      }
      [[fallthrough]];
    }

    // We allow arbitrary layout and f32->bf16 transformations on all other
    // instructions, although this may be made more strict pending discussion
    // in b/112709536.
    default:
      return ShapeUtil::Compatible(instruction->shape(), inferred_shape);
    }
  }();
  if (!equal) {
    return absl::InternalError(absl::StrFormat(
        "Expected instruction to have shape equal to %s, actual "
        "shape is %s:\n%s",
        StringifyShape(inferred_shape), StringifyShape(instruction->shape()),
        instruction->ToString()));
  }
  return absl::OkStatus();
}

absl::Status
ShapeVerifier::CheckShape(const HloInstruction *instruction,
                          const absl::StatusOr<Shape> &inferred_shape_status) {
  if (!inferred_shape_status.ok()) {
    absl::Status s = inferred_shape_status.status();
    tsl::errors::AppendToMessage(&s, ", for instruction ",
                                 instruction->ToString());
    return s;
  }
  return CheckShape(instruction, inferred_shape_status.value());
}

absl::Status ShapeVerifier::CheckUnaryShape(const HloInstruction *instruction) {
  return CheckShape(instruction,
                    ShapeInference::InferUnaryOpShape(instruction->opcode(),
                                                      instruction->operand(0)));
}

absl::Status
ShapeVerifier::CheckBinaryShape(const HloInstruction *instruction) {
  return CheckShape(
      instruction, ShapeInference::InferBinaryOpShape(instruction->opcode(),
                                                      instruction->operand(0),
                                                      instruction->operand(1)));
}

absl::Status
ShapeVerifier::CheckTernaryShape(const HloInstruction *instruction) {
  return CheckShape(instruction,
                    ShapeInference::InferTernaryOpShape(
                        instruction->opcode(), instruction->operand(0),
                        instruction->operand(1), instruction->operand(2)));
}

absl::Status
ShapeVerifier::CheckVariadicShape(const HloInstruction *instruction) {
  return CheckShape(instruction,
                    ShapeInference::InferVariadicOpShape(
                        instruction->opcode(), instruction->operands()));
}

absl::Status
ShapeVerifier::VerifyEntryComputationLayout(const HloModule &module) {
  const HloComputation *computation = module.entry_computation();
  const auto &layout = module.entry_computation_layout();
  const ShapeLayout &result_layout = layout.result_layout();

  TF_RETURN_IF_ERROR(
      ShapeUtil::ValidateShapeWithOptionalLayout(result_layout.shape()));

  // TPU layout assignment doesn't set the tiles on entry_computation_layout, so
  // let's not check that.
  if (!ShapesSame(computation->root_instruction()->shape(),
                  result_layout.shape(),
                  Shape::Equal()
                      .IgnoreTilesInLayout()
                      .IgnoreTailPaddingAlignmentInElements()
                      .IgnoreMemorySpaceInLayout())) {
    return absl::InternalError(absl::StrFormat(
        "Shape of the root instruction of entry computation (%s) should be "
        "compatible to one specified in module's entry computation layout (%s)",
        StringifyShape(computation->root_instruction()->shape()),
        StringifyShape(result_layout.shape())));
  }

  if (computation->num_parameters() != layout.parameter_count()) {
    return absl::InternalError(absl::StrFormat(
        "Number of parameters in entry computation layout (%d) must be same "
        "as number of parameters of entry computation (%d)",
        layout.parameter_count(), computation->num_parameters()));
  }

  for (int i = 0; i < computation->num_parameters(); ++i) {
    const HloInstruction *parameter = computation->parameter_instruction(i);
    TF_RETURN_IF_ERROR(
        ShapeUtil::ValidateShapeWithOptionalLayout(layout.parameter_shape(i)));
    // TPU layout assignment doesn't set the tiles on entry_computation_layout,
    // so let's not check that.
    if (!ShapesSame(parameter->shape(), layout.parameter_shape(i),
                    Shape::Equal()
                        .IgnoreTilesInLayout()
                        .IgnoreTailPaddingAlignmentInElements()
                        .IgnoreMemorySpaceInLayout())) {
      return absl::InternalError(absl::StrFormat(
          "Shape of the entry computation parameter %d is %s should be "
          "compatible to the one specified in module's entry computation "
          "layout %s",
          i, StringifyShape(parameter->shape()),
          StringifyShape(layout.parameter_shape(i))));
    }
  }

  // If result is aliased with a parameter, entry computation layout must have
  // same shape, layout and memory space for them (for example we can't alias
  // parameter and result if they have different memory spaces).
  const auto &alias_config = module.input_output_alias_config();
  TF_RETURN_IF_ERROR(alias_config.ForEachAliasWithStatus(
      [&](ShapeIndex result_index,
          HloInputOutputAliasConfig::Alias alias) -> absl::Status {
        // We skip may-alias buffers as they do not force aliasing.
        if (!alias.must_alias()) {
          return absl::OkStatus();
        }

        const Shape &result_shape =
            ShapeUtil::GetSubshape(result_layout.shape(), result_index);
        const Shape &parameter_shape = ShapeUtil::GetSubshape(
            layout.parameter_layout(alias.parameter_number).shape(),
            alias.parameter_index);

        if (result_shape != parameter_shape) {
          return absl::InternalError(absl::StrFormat(
              "Shape and memory space of the result at index %s (%s) "
              "must be the same as the shape and memory spaceof aliased "
              "parameter %d at index %s (%s)",
              result_index.ToString(), StringifyShape(result_shape),
              alias.parameter_number, alias.parameter_index.ToString(),
              StringifyShape(parameter_shape)));
        }

        return absl::OkStatus();
      }));

  return absl::OkStatus();
}

std::string
ComputationsToString(absl::Span<HloComputation *const> computations) {
  return absl::StrJoin(computations, ",",
                       [](std::string *s, const HloComputation *computation) {
                         absl::StrAppend(s, computation->name());
                       });
}

absl::Status VerifyInstructionNameUnchanged(const HloModule &module,
                                            const HloVerifierOpts &opts) {
  if (!opts.verify_instruction_name_unchanged) {
    return absl::OkStatus();
  }
  for (auto *comp : module.computations()) {
    for (auto *inst : comp->instructions()) {
      if (inst->metadata().scheduling_name().empty()) {
        continue;
      }
      // We do not enforce the invariant when the instruction has been cloned
      // explicitly via .clone or .remat suffix.
      if (inst->metadata().scheduling_name() != inst->name() &&
          (!absl::StrContains(inst->name(), ".remat") &&
           !absl::StrContains(inst->name(), ".clone"))) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Expected instruction name to remain the same. Was '",
            inst->metadata().scheduling_name(), "' is '", inst->name(), "'."));
      }
    }
  }
  return absl::OkStatus();
}

// Verifies various invariants about the structure of the HLO:
//
// (1) each instruction is non-null and has a non-null parent() set to the
// HloComputation which contains it.
//
// (2) each computation is non-null and has a non-null parent() set to the
// HloModule which contains it.
//
// (3) the operands of each instruction are non-null and are in the same
// computation as the instruction.
absl::Status VerifyHloStructure(HloModule *module) {
  for (const HloComputation *computation : module->computations()) {
    if (computation == nullptr) {
      return absl::InternalError(absl::StrFormat(
          "Computation in module %s is a null pointer", module->name()));
    }

    if (computation->parent() == nullptr) {
      return absl::InternalError(absl::StrFormat(
          "Computation %s has a null parent pointer", computation->name()));
    }
    if (computation->parent() != module) {
      return absl::InternalError(absl::StrFormat(
          "Computation %s parent() does not point to parent module",
          computation->name()));
    }

    for (const HloInstruction *instruction : computation->instructions()) {
      if (instruction == nullptr) {
        return absl::InternalError(
            absl::StrFormat("Instruction in computation %s is a null pointer",
                            computation->name()));
      }
      if (instruction->parent() == nullptr) {
        return absl::InternalError(absl::StrFormat(
            "Instruction %s has a null parent pointer", instruction->name()));
      }
      if (instruction->parent() != computation) {
        return absl::InternalError(absl::StrFormat(
            "Instruction %s parent() does not point to parent computation",
            instruction->name()));
      }
    }
  }

  // Check that operands are in the same computation separately from verifying
  // parent() correctness so conditions like a null HloInstruction::parent()
  // are identified and reported explicitly above rather than reporting a
  // mismatched operand.
  for (const HloComputation *computation : module->computations()) {
    for (const HloInstruction *instruction : computation->instructions()) {
      for (int i = 0; i < instruction->operand_count(); ++i) {
        const HloInstruction *operand = instruction->operand(i);
        if (operand == nullptr) {
          return absl::InternalError(absl::StrFormat(
              "Operand %d (out of %d) of instruction: %s is a null pointer", i,
              instruction->operand_count(), instruction->name()));
        }
        if (operand->parent() == nullptr) {
          return absl::InternalError(absl::StrFormat(
              "Operand %d (out of %d) of instruction: %s has a null pointer "
              "parent",
              i, instruction->operand_count(), instruction->name()));
        }
        if (operand->parent() != instruction->parent()) {
          return absl::InternalError(absl::StrFormat(
              "Operand %d (%s) of instruction %s is in a different "
              "computation: %s vs %s",
              i, operand->name(), instruction->name(),
              operand->parent() ? operand->parent()->name() : "(null)",
              instruction->parent()->name()));
        }
      }
    }
  }
  return absl::OkStatus();
}

namespace {

// Verifies that leaf nodes in an original value contain values.
absl::Status VerifyOriginalValue(const HloModule &module) {
  for (const HloComputation *computation : module.computations()) {
    for (const HloInstruction *instruction : computation->instructions()) {
      if (auto original_value = instruction->original_value()) {
        // An original value is expected to have intermediate nodes that are
        // always nullopt and leaves with actual values.
        for (const auto &leaf : original_value->leaves()) {
          if (!leaf.second.has_value()) {
            return absl::InternalError(absl::StrFormat(
                "Leaf nodes in an original value is expected to contain values."
                " Instruction: %s.",
                instruction->ToString()));
          }
        }
      }
    }
  }
  return absl::OkStatus();
}

// CHECKs various invariants of a fusion instruction.
absl::Status CheckFusionInstruction(HloInstruction *fusion) {
  // The parent fusion instruction of the fusion computation must be 'fusion'.
  HloComputation *fused_computation = fusion->fused_instructions_computation();
  if (fusion != fused_computation->FusionInstruction()) {
    return absl::InternalError(absl::StrFormat(
        "Instruction of fused computation does not match expected "
        "instruction "
        "%s.",
        fusion->ToString()));
  }

  // Fused root instruction and fused parameters must all be owned by the
  // fusion computation.
  bool root_owned = false;
  const auto &fused_parameters = fusion->fused_parameters();
  const HloInstruction *fused_root = fusion->fused_expression_root();
  std::vector<bool> parameter_owned(fused_parameters.size(), false);
  for (auto *instruction : fused_computation->instructions()) {
    if (fused_root == instruction) {
      if (root_owned) {
        return absl::InternalError(absl::StrFormat(
            "Root appears more than once in %s.", fusion->ToString()));
      }
      root_owned = true;
    }
    for (int i = 0; i < fused_parameters.size(); ++i) {
      if (fused_parameters[i] == instruction) {
        if (parameter_owned[i]) {
          return absl::InternalError(absl::StrFormat(
              "Parameter appears more than once in %s.", fusion->ToString()));
        }
        parameter_owned[i] = true;
      }
    }
  }
  if (!root_owned) {
    return absl::InternalError(absl::StrFormat(
        "Root not found in computation of %s.", fusion->ToString()));
  }
  // Make sure all the parameter_owned entries are set
  for (int i = 0; i < parameter_owned.size(); i++) {
    if (!parameter_owned[i]) {
      return absl::InternalError(
          absl::StrFormat("Parameter %d not found in computation of %s.", i,
                          fusion->ToString()));
    }
  }

  // Fused root must have no users.
  if (fused_root->user_count() != 0) {
    return absl::InternalError(
        absl::StrFormat("Root of %s may not have users.", fusion->ToString()));
  }

  // All uses of fused instructions must be in the fusion computation, and
  // every non-root instruction must have at least one use.
  for (auto *instruction :
       fusion->fused_instructions_computation()->instructions()) {
    if (instruction != fused_root) {
      if (instruction->user_count() == 0) {
        return absl::InternalError(
            absl::StrFormat("Non-root instruction %s in %s must have users.",
                            instruction->ToString(), fusion->ToString()));
      }
      for (auto &user : instruction->users()) {
        if (fused_computation != user->parent()) {
          return absl::InternalError(absl::StrFormat(
              "Non-root instruction %s in %s may not have external users.",
              instruction->ToString(), fusion->ToString()));
        }
      }
    }
  }

  // Fused parameter instructions must be numbered contiguously and match up
  // (shapes equal) with their respective operand.
  CHECK_GE(fusion->operands().size(), fused_parameters.size());
  std::vector<bool> parameter_numbers(fused_parameters.size(), false);
  for (auto fused_param : fused_parameters) {
    int64_t param_no = fused_param->parameter_number();
    if (param_no < 0) {
      return absl::InternalError(
          absl::StrFormat("Unexpected negative parameter number %d in %s.",
                          param_no, fusion->ToString()));
    }
    if (param_no >= fused_parameters.size()) {
      return absl::InternalError(absl::StrFormat(
          "Unexpected parameter number %d in %s: higher then number of "
          "parameters %lu.",
          param_no, fusion->ToString(), fused_parameters.size()));
    }
    if (parameter_numbers[param_no]) {
      return absl::InternalError(absl::StrFormat(
          "Did not expect parameter number %d more than once in %s.", param_no,
          fusion->ToString()));
    }
    parameter_numbers[param_no] = true;
  }
  // Make sure all the parameter_numbers entries were seen.
  for (int i = 0; i < parameter_numbers.size(); i++) {
    if (!parameter_numbers[i]) {
      return absl::InternalError(absl::StrFormat(
          "Did not see parameter number %d in %s.", i, fusion->ToString()));
    }
  }

  TF_RET_CHECK(fusion->called_computations() ==
               absl::Span<HloComputation *const>(
                   {fusion->fused_instructions_computation()}))
      << "Fusion HLO calls computations other than the "
         "fused_instructions_computation: "
      << fusion->ToString() << " fusion->fused_instructions_computation(): "
      << fusion->fused_instructions_computation()->ToString()
      << " fusion->called_computations(): "
      << ComputationsToString(fusion->called_computations());

  for (const auto &fused : fusion->fused_instructions()) {
    TF_RET_CHECK(fused->parent() == fusion->fused_instructions_computation())
        << "Fused HLO was missing a parent: " << fused->ToString()
        << " parent: " << fused->parent()
        << " computation: " << fusion->parent();
  }

  // TODO(b/65423525): We'd like to check that all operands are distinct.
  // This is currently disabled due to the invariant being violated by
  // multi-output fusion.
  return absl::OkStatus();
}

// Checks that the operand shapes are compatible to the output shape, i.e.,
// that there are no implicit broadcasts.
absl::Status CheckElementwiseInstruction(HloInstruction *instruction) {
  const Shape &out_shape = instruction->shape();
  for (HloInstruction *operand : instruction->operands()) {
    const Shape &operand_shape = operand->shape();
    if (!ShapeUtil::CompatibleIgnoringElementType(operand_shape, out_shape)) {
      return absl::FailedPreconditionError(
          absl::StrFormat("Implicit broadcast is not allowed in HLO."
                          "Found different shapes for instruction %s.\n"
                          "output: %s\noperand: %s\n",
                          HloOpcodeString(instruction->opcode()),
                          ShapeUtil::HumanString(out_shape),
                          ShapeUtil::HumanString(operand_shape)));
    }
  }
  return absl::OkStatus();
}

// Visitor which verifies various fields on the HLO instruction. This class does
// not check result shape as that is checked in the ShapeVerifier.
class InstructionVerifier : public DfsHloVisitorWithDefault {
public:
  InstructionVerifier(const HloModule *module, const HloVerifierOpts &opts)
      : opts_(opts) {
    (void)module;
  }

  absl::Status DefaultAction(HloInstruction *) override {
    return absl::OkStatus();
  }

  absl::Status HandleFusion(HloInstruction *fusion) override {
    TF_RETURN_IF_ERROR(CheckCallableInstructionThreadName(
        fusion, /*skip_nested_async_op_check*/ false));
    return CheckFusionInstruction(fusion);
  }

  absl::Status HandleBroadcast(HloInstruction *broadcast) override {
    // If you see this failure then someone has confused the difference
    // between the HLO broadcast op, and the UserComputation broadcast
    // op. See https://groups.google.com/forum/#!topic/xla-dev/9LqijHmTt_I
    // or ComputationLowerer::Visit()
    TF_RET_CHECK(broadcast->dimensions().size() ==
                 broadcast->operand(0)->shape().rank())
        << "Broadcast HLO (" << broadcast->ToShortString()
        << ") has invalid number of dimensions: "
        << broadcast->dimensions().size()
        << " != " << broadcast->operand(0)->shape().rank();
    if (opts_.verify_broadcast_dimensions_order) {
      TF_RET_CHECK(absl::c_is_sorted(broadcast->dimensions()))
          << "Broadcast dimensions should be ordered, got: "
          << broadcast->ToString();
    }
    return absl::OkStatus();
  }

  absl::Status HandleBitcastConvert(HloInstruction *c) override {
    // Shape verifier will check all we need.
    return absl::OkStatus();
  }

  absl::Status HandleWhile(HloInstruction *zkx_while) override {
    auto *while_cond = zkx_while->while_condition();
    auto *while_body = zkx_while->while_body();
    if (while_cond->num_parameters() != 1) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "While condition must have exactly 1 parameter; had %d : %s",
          while_cond->num_parameters(), while_cond->ToString()));
    }
    if (while_body->num_parameters() != 1) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "While body must have exactly 1 parameter; had %d : %s",
          while_body->num_parameters(), while_body->ToString()));
    }
    if (zkx_while->operand_count() != 1) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "While loop must have exactly one operand; had %d : %s",
          zkx_while->operand_count(), zkx_while->ToString()));
    }
    // Allow kWhile to contain computations on separate thread.
    TF_RETURN_IF_ERROR(CheckCallableInstructionThreadName(
        zkx_while, /*skip_nested_async_op_check=*/true));

    // Verify consistency of sharding of while instructions and related
    // instructions (parameters, root) in its called computations.
    TF_RETURN_IF_ERROR(VerifyConsistentSharding(
        zkx_while, {zkx_while, zkx_while->while_body()->root_instruction(),
                    zkx_while->while_body()->parameter_instruction(0),
                    zkx_while->while_condition()->parameter_instruction(0)}));

    return absl::OkStatus();
  }

  absl::Status HandleCall(HloInstruction *call) override {
    if (opts_.verify_call_nested_computation_thread_name) {
      return CheckCallableInstructionThreadName(
          call, /*skip_nested_async_op_check=*/true);
    }
    return absl::OkStatus();
  }

  absl::Status HandleConditional(HloInstruction *conditional) override {
    const std::vector<HloComputation *> branch_computations =
        conditional->branch_computations();
    std::vector<const HloInstruction *> sharding_check_instructions;
    sharding_check_instructions.reserve(branch_computations.size() + 1);
    sharding_check_instructions.push_back(conditional);

    for (const HloComputation *branch_computation : branch_computations) {
      if (branch_computation->num_parameters() != 1) {
        return absl::FailedPreconditionError(absl::StrFormat(
            "Branch computation %s of %s must have 1 parameter instead of %d",
            branch_computation->name(), conditional->ToString(),
            branch_computation->num_parameters()));
      }
      sharding_check_instructions.push_back(
          branch_computation->root_instruction());
    }
    // Allow kConditional to contain computations on separate thread.
    TF_RETURN_IF_ERROR(CheckCallableInstructionThreadName(
        conditional, /*skip_nested_async_op_check=*/true));

    // Verify consistency of sharding of conditional instructions and roots of
    // its branches.
    TF_RETURN_IF_ERROR(
        VerifyConsistentSharding(conditional, sharding_check_instructions));

    return absl::OkStatus();
  }

  absl::Status HandleElementwiseUnary(HloInstruction *instruction) override {
    return CheckElementwiseInstruction(instruction);
  }

  absl::Status HandleElementwiseBinary(HloInstruction *instruction) override {
    return CheckElementwiseInstruction(instruction);
  }

  absl::Status HandleGetTupleElement(HloInstruction *gte) override {
    TF_RET_CHECK(gte->operand(0)->shape().IsTuple());
    return absl::OkStatus();
  }

  absl::Status HandleTranspose(HloInstruction *transpose) override {
    const Shape &shape = transpose->shape();
    const HloInstruction *operand = transpose->operand(0);
    TF_RET_CHECK(shape.dimensions().size() == transpose->dimensions().size());
    TF_RET_CHECK(shape.dimensions().size() ==
                 transpose->operand(0)->shape().dimensions().size());
    TF_RET_CHECK(std::equal(
        shape.dimensions().begin(), shape.dimensions().end(),
        Permute(operand->shape().dimensions(), transpose->dimensions())
            .begin()))
        << "shape: " << shape << ", operand->shape(): " << shape
        << ", dimensions: {" << absl::StrJoin(transpose->dimensions(), ", ")
        << "}";
    return absl::OkStatus();
  }

  absl::Status HandleReshape(HloInstruction *hlo) override {
    if (opts_.verify_reshape_is_bitcast && !hlo->IsFused()) {
      TF_RET_CHECK(
          ShapeUtil::ReshapeIsBitcast(hlo->operand(0)->shape(), hlo->shape()))
          << "Reshape should be a physical bitcast, got: " << hlo->ToString();
    }
    return absl::OkStatus();
  }

  absl::Status HandleScatter(HloInstruction *scatter) override {
    int64_t rank = scatter->operand(0)->shape().rank();
    for (int64_t operand_dim :
         scatter->scatter_dimension_numbers().scatter_dims_to_operand_dims()) {
      if (operand_dim > rank) {
        return absl::OutOfRangeError(absl::StrCat(
            "The provided scatter_dims_to_operand_dim was out of range.",
            " (operand_dim: ", operand_dim, ", rank: ", rank, ")"));
      }
    }
    return absl::OkStatus();
  }

  absl::Status Preprocess(HloInstruction *instruction) override {
    auto [it, inserted] =
        instructions_by_name_.emplace(instruction->name(), instruction);
    TF_RET_CHECK(inserted) << "HLO has name that is not unique within module:\n"
                           << instruction->ToString() << " in computation: "
                           << instruction->parent()->name()
                           << "\nPrevious HLO with same name:\n"
                           << it->second->ToString() << " in computation: "
                           << it->second->parent()->name();

    if (opts_.verify_call_nested_computation_thread_name &&
        instruction->has_to_apply() &&
        instruction->to_apply()->execution_thread() !=
            instruction->parent()->execution_thread()) {
      return absl::InternalError(absl::StrFormat(
          "%s top_apply computation execution thread does not match (%s vs %s)",
          instruction->name(), instruction->to_apply()->execution_thread(),
          instruction->parent()->execution_thread()));
    }

    return absl::OkStatus();
  }

  absl::Status Postprocess(HloInstruction *instruction) override {
    if (!opts_.InstructionCanChangeLayout(instruction) &&
        LayoutUtil::IsDenseArray(instruction->shape()) &&
        instruction->shape().has_layout()) {
      const Shape &result_shape = instruction->shape();
      const Layout &result_layout = result_shape.layout();
      for (HloInstruction *operand : instruction->operands()) {
        const Shape &operand_shape = operand->shape();
        if (LayoutUtil::IsDenseArray(operand_shape) &&
            operand_shape.rank() == result_shape.rank() &&
            operand_shape.has_layout()) {
          const Layout &operand_layout = operand_shape.layout();
          Layout::Equal equal_predicate =
              Layout::Equal().IgnoreTiles().IgnoreMemorySpace();
          if (instruction->opcode() == HloOpcode::kConvert ||
              instruction->opcode() == HloOpcode::kCompare ||
              (instruction->opcode() == HloOpcode::kSelect &&
               operand_shape.element_type() == PRED)) {
            // Some instructions can change element_size_in_bits
            // Select instructions ignore element_size_in_bits for predicate
            equal_predicate.IgnoreElementSize();
          } else if (instruction->opcode() == HloOpcode::kDynamicSlice ||
                     instruction->opcode() == HloOpcode::kDynamicUpdateSlice ||
                     instruction->opcode() == HloOpcode::kCopy) {
            TF_RETURN_IF_ERROR(HostOffloadInstructionCanChangeMemorySpace(
                instruction, operand_layout.memory_space(),
                result_layout.memory_space()));
            equal_predicate.IgnoreMemorySpace();
          }
          TF_RET_CHECK(equal_predicate(result_layout, operand_layout))
              << "Instruction shouldn't change layouts "
              << instruction->ToString() << " From " << result_shape << " To "
              << operand_shape;
        }
      }
    }
    return absl::OkStatus();
  }

private:
  static absl::Status VerifyConsistentSharding(
      const HloInstruction * /*parent*/,
      absl::Span<const HloInstruction *const> /*instructions*/) {
    return absl::OkStatus();
  }

  // Verifies whether a given `instruction` is permitted to change the layout
  // memory space from `operand_memory_space` to `result_memory_space`.
  // Returns absl::OkStatus() if the instruction's layout changes are valid;
  // otherwise, returns an appropriate error status.
  static absl::Status HostOffloadInstructionCanChangeMemorySpace(
      const HloInstruction *instruction, const int64_t operand_memory_space,
      const int64_t result_memory_space) {
    TF_RET_CHECK(!(operand_memory_space == Layout::kGenericFastMemorySpace &&
                   result_memory_space != Layout::kGenericFastMemorySpace) ||
                 (operand_memory_space != Layout::kGenericFastMemorySpace &&
                  result_memory_space == Layout::kGenericFastMemorySpace))
        << "Instruction shouldn't change layout memory space between generic "
           "fast memory space and others for instruction: "
        << instruction->ToString();

    if (instruction->opcode() == HloOpcode::kDynamicSlice) {
      TF_RET_CHECK(!(operand_memory_space == Layout::kDefaultMemorySpace &&
                     result_memory_space == Layout::kHostMemorySpace))
          << "DynamicSlice instruction shouldn't change layout memory "
          << "space from device to host: " << instruction->ToString();
    } else if (instruction->opcode() == HloOpcode::kDynamicUpdateSlice) {
      TF_RET_CHECK(!(operand_memory_space == Layout::kHostMemorySpace &&
                     result_memory_space == Layout::kDefaultMemorySpace))
          << "DynamicUpdateSlice instruction shouldn't change layout "
          << "memory space from host to device: " << instruction->ToString();
    } else if (instruction->opcode() != HloOpcode::kCopy) {
      return absl::InvalidArgumentError(
          absl::StrCat("Instruction shouldn't change layout memory space: ",
                       instruction->ToString()));
    }
    return absl::OkStatus();
  }

  absl::flat_hash_map<std::string, const HloInstruction *>
      instructions_by_name_;
  const HloVerifierOpts &opts_;
  std::optional<int64_t> num_devices_;
};

} // namespace

absl::StatusOr<bool> HloVerifier::Run(
    HloModule *module,
    const absl::flat_hash_set<std::string_view> &execution_threads) {
  auto disabled = module->config().debug_options().zkx_disable_hlo_passes();
  if (std::find(disabled.begin(), disabled.end(), name()) != disabled.end()) {
    return false;
  }
  auto status_or_changed = [&]() -> absl::StatusOr<bool> {
    TF_RET_CHECK(!module->name().empty());

    if (module->entry_computation()->IsFusionComputation()) {
      return absl::InvalidArgumentError(
          "Module entry computation cannot be a fusion computation");
    }

    TF_RETURN_IF_ERROR(VerifyHloStructure(module));
    TF_RETURN_IF_ERROR(VerifyInstructionNameUnchanged(
        *module, target_metadata_->GetVerifierOpts()));

    std::unique_ptr<ShapeVerifier> shape_verifier =
        target_metadata_->GetVerifier();
    InstructionVerifier instruction_verifier(
        module, target_metadata_->GetVerifierOpts());
    for (auto *computation : module->computations(execution_threads)) {
      TF_RETURN_IF_ERROR(computation->Accept(shape_verifier.get()));
      TF_RETURN_IF_ERROR(computation->Accept(&instruction_verifier));
    }

    TF_RETURN_IF_ERROR(shape_verifier->VerifyEntryComputationLayout(*module));

    // If the module has a schedule, it must be valid.
    if (module->has_schedule()) {
      TF_RETURN_IF_ERROR(module->schedule().Verify());
    }

    if (HloInstruction::IsThreadIncluded(
            module->entry_computation()->execution_thread(),
            execution_threads)) {
      TF_RETURN_IF_ERROR(module->input_output_alias_config().Verify(
          *module, [this](const Shape &shape) -> int64_t {
            if (target_metadata_->GetVerifierOpts().IsLayoutSensitive()) {
              return target_metadata_->GetVerifierOpts().ShapeSize(shape);
            } else {
              return 0;
            }
          }));
    }

    TF_RETURN_IF_ERROR(module->buffer_donor_config().Verify(*module));
    TF_RETURN_IF_ERROR(VerifyOriginalValue(*module));
    return false;
  }();
  if (status_or_changed.ok()) {
    return status_or_changed.value();
  }
  return absl::Status(status_or_changed.status().code(),
                      absl::StrCat("during context [", context_, "]: ",
                                   status_or_changed.status().message()));
}

MetadataTracker::MetadataTracker(std::string_view prefix) : prefix_(prefix) {}

MetadataTracker::~MetadataTracker() {
  if (instruction_count_ == 0) {
    return;
  }
  const std::map<std::string, double> values = {
      {"instruction_count", 1.0 * instruction_count_},
      {"op_type_coverage", 1.0 * has_op_type_count_ / instruction_count_},
      {"op_name_coverage", 1.0 * has_op_name_count_ / instruction_count_},
      {"source_file_coverage",
       1.0 * has_source_file_count_ / instruction_count_},
      {"dummy_source_file_coverage",
       1.0 * has_dummy_source_file_count_ / instruction_count_},
      {"source_line_coverage",
       1.0 * has_source_line_count_ / instruction_count_},
      {"creation_pass_coverage",
       1.0 * has_creation_pass_id_count_ / instruction_count_},
      {"logical_creation_pass_coverage",
       1.0 * has_logical_creation_pass_id_count_ / instruction_count_},
      {"size_of_generated_code_in_bytes_coverage",
       1.0 * has_size_of_generated_code_in_bytes_count_ / instruction_count_},
      {"size_of_memory_working_set_in_bytes_coverage",
       1.0 * has_size_of_memory_working_set_in_bytes_count_ /
           instruction_count_},
      {"profile_info_coverage",
       1.0 * has_profile_info_count_ / instruction_count_}};
  LOG(INFO) << prefix_ << " "
            << absl::StrJoin(values, ",", absl::PairFormatter("="));
}

void MetadataTracker::HandleMetadata(const OpMetadata &metadata) {
  ++instruction_count_;
  if (!metadata.op_type().empty()) {
    ++has_op_type_count_;
  }
  if (!metadata.op_name().empty()) {
    ++has_op_name_count_;
  }
  if (!metadata.source_file().empty()) {
    ++has_source_file_count_;
    if (absl::StrContains(metadata.source_file(), "dummy")) {
      ++has_dummy_source_file_count_;
    }
  }
  if (metadata.source_line() != 0) {
    ++has_source_line_count_;
  }
  if (metadata.size_of_generated_code_in_bytes() != 0) {
    ++has_size_of_generated_code_in_bytes_count_;
  }
  if (metadata.size_of_memory_working_set_in_bytes() != 0) {
    ++has_size_of_memory_working_set_in_bytes_count_;
  }
  if (metadata.has_profile_info()) {
    ++has_profile_info_count_;
  }
}

absl::Status MetadataTracker::DefaultAction(HloInstruction *instruction) {
  HandleMetadata(instruction->metadata());
  return absl::OkStatus();
}

} // namespace zkx
