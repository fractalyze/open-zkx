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

#include "zkx/hlo/ir/hlo_instructions.h"

#include "absl/algorithm/container.h"

#include "zkx/base/logging.h"
#include "zkx/hlo/ir/hlo_casting_utils.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/literal_util.h"
#include "zkx/protobuf_util.h"
#include "zkx/window_util.h"

namespace zkx {
namespace {

bool IsInstructionElementwiseOnOperand(const HloInstruction *instruction,
                                       const HloInstruction *operand) {
  const auto operand_indices = instruction->OperandIndices(operand);
  return absl::c_all_of(operand_indices, [instruction](int64_t operand_index) {
    return instruction->IsElementwiseOnOperand(operand_index);
  });
}

void PrintSparsityDescriptor(HloInstruction::AttributePrinter &printer,
                             absl::Span<const SparsityDescriptor> sparsity) {
  printer.Next([&sparsity](Printer *printer) {
    printer->Append("sparsity=");
    for (int i = 0; i < sparsity.size(); ++i) {
      if (i != 0) {
        printer->Append("_");
      }
      const SparsityDescriptor &cur = sparsity[i];
      printer->Append(cur.index() == 0 ? "L." : "R.");
      printer->Append(cur.dimension());
      printer->Append("@");
      switch (cur.type()) {
        case SPARSITY_STRUCTURED_N_M:
          printer->Append(cur.n());
          printer->Append(":");
          printer->Append(cur.m());
          break;
        default:
          LOG(FATAL) << "Unknown sparsity type: " << cur.type();
      }
    }
  });
}

void SetThreadName(HloComputation *called_computation,
                   std::string_view execution_thread,
                   bool skip_async_execution_thread_overwrite) {
  called_computation->SetExecutionThread(execution_thread);
  for (HloInstruction *instr : called_computation->instructions()) {
    for (HloComputation *nested_called_computation :
         instr->called_computations()) {
      SetThreadName(nested_called_computation, execution_thread,
                    skip_async_execution_thread_overwrite);
    }
  }
}

}  // namespace

bool HloDimensionsInstruction::IdenticalSlowPath(
    const HloInstruction &other,
    absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
        eq_computations) const {
  const auto &casted_other =
      static_cast<const HloDimensionsInstruction &>(other);
  return dimensions() == casted_other.dimensions();
}

HloInstructionProto HloDimensionsInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  for (int64_t dimension : dimensions_) {
    proto.add_dimensions(dimension);
  }
  return proto;
}

void HloDimensionsInstruction::PrintExtraAttributesImpl(
    AttributePrinter &printer, const HloPrintOptions &options) const {
  printer.Next([this](Printer *printer) {
    printer->Append("dimensions={");
    AppendJoin(printer, dimensions(), ",");
    printer->Append("}");
  });
}

HloBroadcastInstruction::HloBroadcastInstruction(
    const Shape &shape, HloInstruction *operand,
    absl::Span<const int64_t> broadcast_dimension)
    : HloDimensionsInstruction(HloOpcode::kBroadcast, shape,
                               broadcast_dimension) {
  AppendOperand(operand);
}

std::unique_ptr<HloInstruction>
HloBroadcastInstruction::CloneWithNewOperandsImpl(
    const Shape &shape, absl::Span<HloInstruction *const> new_operands,
    HloCloneContext *context) const {
  CHECK_EQ(new_operands.size(), 1);
  return std::make_unique<HloBroadcastInstruction>(shape, new_operands[0],
                                                   dimensions());
}

HloFftInstruction::HloFftInstruction(
    const Shape &shape, absl::Span<HloInstruction *const> new_operands,
    FftType fft_type, int64_t fft_length, bool fft_do_bit_reverse)
    : HloInstruction(HloOpcode::kFft, shape),
      fft_type_(fft_type),
      fft_length_(fft_length),
      fft_do_bit_reverse_(fft_do_bit_reverse) {
  CHECK_LE(new_operands.size(), 2);
  AppendOperands(new_operands);
}

HloInstructionProto HloFftInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_fft_type(fft_type_);
  proto.set_fft_length(fft_length_);
  proto.set_fft_do_bit_reverse(fft_do_bit_reverse_);
  return proto;
}

void HloFftInstruction::PrintExtraAttributesImpl(
    AttributePrinter &printer, const HloPrintOptions &options) const {
  printer.Next([this](Printer *printer) {
    AppendCat(printer, "fft_type=", FftType_Name(fft_type()));
  });
  printer.Next([this](Printer *printer) {
    AppendCat(printer, "fft_length=", fft_length());
  });
}

bool HloFftInstruction::IdenticalSlowPath(
    const HloInstruction &other,
    absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
        eq_computations) const {
  const auto &casted_other = static_cast<const HloFftInstruction &>(other);
  return fft_type() == casted_other.fft_type() &&
         fft_length() == casted_other.fft_length() &&
         fft_do_bit_reverse() == casted_other.fft_do_bit_reverse();
}

std::unique_ptr<HloInstruction> HloFftInstruction::CloneWithNewOperandsImpl(
    const Shape &shape, absl::Span<HloInstruction *const> new_operands,
    HloCloneContext *context) const {
  return std::make_unique<HloFftInstruction>(shape, new_operands, fft_type_,
                                             fft_length_, fft_do_bit_reverse_);
}

HloCompareInstruction::HloCompareInstruction(const Shape &shape,
                                             HloInstruction *lhs,
                                             HloInstruction *rhs,
                                             ComparisonDirection direction,
                                             std::optional<PrimitiveType> type)
    : HloInstruction(HloOpcode::kCompare, shape),
      compare_(type.has_value()
                   ? Comparison(direction, *type)
                   : Comparison(direction, lhs->shape().element_type())) {
  AppendOperand(lhs);
  AppendOperand(rhs);
}

HloInstructionProto HloCompareInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_comparison_direction(
      ComparisonDirectionToString(compare_.GetDirection()));
  proto.set_comparison_primitive_type(compare_.GetPrimitiveType());
  return proto;
}

void HloCompareInstruction::PrintExtraAttributesImpl(
    AttributePrinter &printer, const HloPrintOptions &options) const {
  printer.Next([this](Printer *printer) {
    AppendCat(printer, "direction=", ComparisonDirectionToString(direction()));
  });
  if (compare_.GetPrimitiveType() != operand(0)->shape().element_type()) {
    printer.Next([this](Printer *printer) {
      AppendCat(printer, "type=",
                ComparisonPrimitiveTypeToString(compare_.GetPrimitiveType()));
    });
  }
}

bool HloCompareInstruction::IdenticalSlowPath(
    const HloInstruction &other,
    absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
        eq_computations) const {
  const auto &casted_other = static_cast<const HloCompareInstruction &>(other);
  return direction() == casted_other.direction();
}

std::unique_ptr<HloInstruction> HloCompareInstruction::CloneWithNewOperandsImpl(
    const Shape &shape, absl::Span<HloInstruction *const> new_operands,
    HloCloneContext *context) const {
  CHECK_EQ(new_operands.size(), 2);
  return std::make_unique<HloCompareInstruction>(
      shape, new_operands[0], new_operands[1], direction(), primitive_type());
}

HloReverseInstruction::HloReverseInstruction(
    const Shape &shape, HloInstruction *operand,
    absl::Span<const int64_t> dimensions)
    : HloDimensionsInstruction(HloOpcode::kReverse, shape, dimensions) {
  AppendOperand(operand);
}

std::unique_ptr<HloInstruction> HloReverseInstruction::CloneWithNewOperandsImpl(
    const Shape &shape, absl::Span<HloInstruction *const> new_operands,
    HloCloneContext *context) const {
  CHECK_EQ(new_operands.size(), 1);
  return std::make_unique<HloReverseInstruction>(shape, new_operands[0],
                                                 dimensions());
}

HloBitReverseInstruction::HloBitReverseInstruction(
    const Shape &shape, HloInstruction *operand,
    absl::Span<const int64_t> dimensions)
    : HloDimensionsInstruction(HloOpcode::kBitReverse, shape, dimensions) {
  AppendOperand(operand);
}

std::unique_ptr<HloInstruction>
HloBitReverseInstruction::CloneWithNewOperandsImpl(
    const Shape &shape, absl::Span<HloInstruction *const> new_operands,
    HloCloneContext *context) const {
  CHECK_EQ(new_operands.size(), 1);
  return std::make_unique<HloBitReverseInstruction>(shape, new_operands[0],
                                                    dimensions());
}

HloConcatenateInstruction::HloConcatenateInstruction(
    const Shape &shape, absl::Span<HloInstruction *const> operands,
    int64_t dimension)
    : HloDimensionsInstruction(HloOpcode::kConcatenate, shape, {dimension}) {
  for (auto operand : operands) {
    AppendOperand(operand);
  }
}

std::unique_ptr<HloInstruction>
HloConcatenateInstruction::CloneWithNewOperandsImpl(
    const Shape &shape, absl::Span<HloInstruction *const> new_operands,
    HloCloneContext *context) const {
  return std::make_unique<HloConcatenateInstruction>(shape, new_operands,
                                                     concatenate_dimension());
}

HloReduceInstruction::HloReduceInstruction(
    const Shape &shape, absl::Span<HloInstruction *const> args,
    absl::Span<const int64_t> dimensions_to_reduce,
    HloComputation *reduce_computation)
    : HloDimensionsInstruction(HloOpcode::kReduce, shape,
                               dimensions_to_reduce) {
  for (HloInstruction *arg : args) {
    AppendOperand(arg);
  }
  AppendComputation(reduce_computation);
}

bool HloReduceInstruction::IdenticalSlowPath(
    const HloInstruction &other,
    absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
        eq_computations) const {
  const auto &casted_other = static_cast<const HloReduceInstruction &>(other);
  // Reduction results are determined by the reduction dimension and the
  // reduction computation.
  return dimensions() == casted_other.dimensions() &&
         eq_computations(to_apply(), casted_other.to_apply());
}

std::unique_ptr<HloInstruction> HloReduceInstruction::CloneWithNewOperandsImpl(
    const Shape &shape, absl::Span<HloInstruction *const> new_operands,
    HloCloneContext *context) const {
  CHECK_EQ(new_operands.size() % 2, 0);
  return std::make_unique<HloReduceInstruction>(shape, new_operands,
                                                dimensions(), to_apply());
}

HloTransposeInstruction::HloTransposeInstruction(
    const Shape &shape, HloInstruction *operand,
    absl::Span<const int64_t> dimensions)
    : HloDimensionsInstruction(HloOpcode::kTranspose, shape, dimensions) {
  AppendOperand(operand);
}

bool HloTransposeInstruction::IsRank2Transpose() const {
  return dimensions() == std::vector<int64_t>({1, 0}) &&
         shape().dimensions_size() == 2 &&
         std::equal(shape().dimensions().begin(), shape().dimensions().end(),
                    operand(0)->shape().dimensions().rbegin());
}

std::unique_ptr<HloInstruction>
HloTransposeInstruction::CloneWithNewOperandsImpl(
    const Shape &shape, absl::Span<HloInstruction *const> new_operands,
    HloCloneContext *context) const {
  CHECK_EQ(new_operands.size(), 1);
  return std::make_unique<HloTransposeInstruction>(shape, new_operands[0],
                                                   dimensions());
}

HloDynamicReshapeInstruction::HloDynamicReshapeInstruction(
    const Shape &shape, HloInstruction *data_operand,
    absl::Span<HloInstruction *const> dim_sizes)
    : HloInstruction(HloOpcode::kDynamicReshape, shape) {
  AppendOperand(data_operand);
  for (auto operand : dim_sizes) {
    AppendOperand(operand);
  }
}

std::unique_ptr<HloInstruction>
HloDynamicReshapeInstruction::CloneWithNewOperandsImpl(
    const Shape &shape, absl::Span<HloInstruction *const> new_operands,
    HloCloneContext *context) const {
  CHECK_GE(new_operands.size(), 1);
  return std::make_unique<HloDynamicReshapeInstruction>(
      shape, new_operands[0], new_operands.subspan(1));
}

HloReshapeInstruction::HloReshapeInstruction(const Shape &shape,
                                             HloInstruction *operand,
                                             int64_t inferred_dimension)
    : HloInstruction(HloOpcode::kReshape, shape),
      inferred_dimension_(inferred_dimension) {
  AppendOperand(operand);
}

HloInstructionProto HloReshapeInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  if (inferred_dimension_ != -1) {
    proto.add_dimensions(inferred_dimension_);
  }
  return proto;
}

void HloReshapeInstruction::PrintExtraAttributesImpl(
    AttributePrinter &printer, const HloPrintOptions &options) const {
  if (inferred_dimension() != -1) {
    printer.Next([this](Printer *printer) {
      AppendCat(printer, "inferred_dimension=", inferred_dimension());
    });
  }
}

bool HloReshapeInstruction::IdenticalSlowPath(
    const HloInstruction &other,
    absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
        eq_computations) const {
  const auto &casted_other = static_cast<const HloReshapeInstruction &>(other);
  return inferred_dimension() == casted_other.inferred_dimension();
}

std::unique_ptr<HloInstruction> HloReshapeInstruction::CloneWithNewOperandsImpl(
    const Shape &shape, absl::Span<HloInstruction *const> new_operands,
    HloCloneContext *context) const {
  CHECK_EQ(new_operands.size(), 1);
  return std::make_unique<HloReshapeInstruction>(shape, new_operands[0],
                                                 inferred_dimension());
}

HloMapInstruction::HloMapInstruction(const Shape &shape,
                                     absl::Span<HloInstruction *const> operands,
                                     HloComputation *map_computation)
    : HloInstruction(HloOpcode::kMap, shape) {
  for (auto operand : operands) {
    AppendOperand(operand);
  }
  AppendComputation(map_computation);
  // TODO(b/65689298) Remove code below once Map is generalized to accept
  // arbitrary map dimensions.
  dimensions_.resize(shape.rank());
  std::iota(dimensions_.begin(), dimensions_.end(), 0);
}

HloInstructionProto HloMapInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  for (int64_t dimension : dimensions_) {
    proto.add_dimensions(dimension);
  }
  return proto;
}

bool HloMapInstruction::IsElementwiseImpl(
    const std::optional<int64_t> &operand_idx) const {
  if (!dimensions().empty()) {
    // Check that the map is executed in elementwise compatible dimensions.
    if (dimensions().size() != shape().dimensions_size()) {
      return false;
    }
    for (int i = 0; i < dimensions().size(); ++i) {
      if (dimensions()[i] != i) {
        return false;
      }
    }
  }
  return true;
}

void HloMapInstruction::PrintExtraAttributesImpl(
    AttributePrinter &printer, const HloPrintOptions &options) const {
  printer.Next([this](Printer *printer) {
    printer->Append("dimensions={");
    AppendJoin(printer, dimensions(), ",");
    printer->Append("}");
  });
}

bool HloMapInstruction::IdenticalSlowPath(
    const HloInstruction &other,
    absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
        eq_computations) const {
  const auto &casted_other = static_cast<const HloMapInstruction &>(other);
  return eq_computations(to_apply(), casted_other.to_apply()) &&
         dimensions() == casted_other.dimensions();
}

std::unique_ptr<HloInstruction> HloMapInstruction::CloneWithNewOperandsImpl(
    const Shape &shape, absl::Span<HloInstruction *const> new_operands,
    HloCloneContext *context) const {
  return std::make_unique<HloMapInstruction>(shape, new_operands, to_apply());
}

HloSliceInstruction::HloSliceInstruction(
    const Shape &shape, HloInstruction *operand,
    absl::Span<const int64_t> start_indices,
    absl::Span<const int64_t> limit_indices, absl::Span<const int64_t> strides)
    : HloInstruction(HloOpcode::kSlice, shape),
      slice_starts_(start_indices.begin(), start_indices.end()),
      slice_limits_(limit_indices.begin(), limit_indices.end()),
      slice_strides_(strides.begin(), strides.end()) {
  AppendOperand(operand);
  // For backward compatibility with old serialized computations: if there are
  // no strides, assume all strides are 1.
  // TODO(b/63317920): remove this code.
  if (slice_strides_.empty()) {
    slice_strides_ = std::vector<int64_t>(start_indices.size(), 1LL);
  }
}

HloInstructionProto HloSliceInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  for (int i = 0; i < slice_starts_.size(); ++i) {
    auto *slice_dimension = proto.add_slice_dimensions();
    slice_dimension->set_start(slice_starts_[i]);
    slice_dimension->set_limit(slice_limits_[i]);
    slice_dimension->set_stride(slice_strides_[i]);
  }
  return proto;
}

void HloSliceInstruction::PrintExtraAttributesImpl(
    AttributePrinter &printer, const HloPrintOptions &options) const {
  printer.Next([this](Printer *printer) {
    const bool omit_stride = absl::c_all_of(
        slice_strides_, [](int64_t stride) { return stride == 1; });
    printer->Append("slice={");
    AppendJoin(printer, slice_starts_, ", ",
               [&](Printer *printer, auto &slice_start) {
                 const auto i = &slice_start - slice_starts_.data();
                 AppendCat(printer, "[", slice_start, ":", slice_limits_[i]);
                 if (!omit_stride) {
                   AppendCat(printer, ":", slice_strides_[i]);
                 }
                 printer->Append("]");
               });
    printer->Append("}");
  });
}

bool HloSliceInstruction::IdenticalSlowPath(
    const HloInstruction &other,
    absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
        eq_computations) const {
  const auto &other_slice = static_cast<const HloSliceInstruction &>(other);
  return slice_starts_ == other_slice.slice_starts_ &&
         slice_limits_ == other_slice.slice_limits_ &&
         slice_strides_ == other_slice.slice_strides_;
}

std::unique_ptr<HloInstruction> HloSliceInstruction::CloneWithNewOperandsImpl(
    const Shape &shape, absl::Span<HloInstruction *const> new_operands,
    HloCloneContext *context) const {
  CHECK_EQ(new_operands.size(), 1);
  return std::make_unique<HloSliceInstruction>(
      shape, new_operands[0], slice_starts_, slice_limits_, slice_strides_);
}

HloConstantInstruction::HloConstantInstruction(Literal literal)
    : HloInstruction(HloOpcode::kConstant, literal.shape()),
      literal_(new Literal(std::move(literal))) {}

HloConstantInstruction::HloConstantInstruction(Literal literal,
                                               const Shape &shape)
    : HloInstruction(HloOpcode::kConstant, shape),
      literal_(new Literal(std::move(literal))) {}

HloConstantInstruction::HloConstantInstruction(std::shared_ptr<Literal> literal,
                                               const Shape &shape)
    : HloInstruction(HloOpcode::kConstant, shape), literal_(literal) {}

HloConstantInstruction::HloConstantInstruction(const Shape &shape)
    : HloInstruction(HloOpcode::kConstant, shape) {}

HloInstructionProto HloConstantInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  if (literal_) {
    *proto.mutable_literal() = literal_->ToProto();
  }
  return proto;
}

bool HloConstantInstruction::IsElementwiseImpl(
    const std::optional<int64_t> &operand_idx) const {
  return true;
}

void HloConstantInstruction::RelayoutConstant(const Layout &new_layout,
                                              const ShapeIndex &shape_index) {
  Shape *mutable_array_subshape =
      ShapeUtil::GetMutableSubshape(mutable_shape(), shape_index);
  CHECK(mutable_array_subshape->IsArray());

  // Normally array_subshape will always have a layout, but this invariant is
  // temporarily broken in LayoutAssignment::AssignLayouts where all shape
  // layouts are cleared. The inner condition below ensures that we don't
  // unnecessarily relayout literals in that case.

  if (!mutable_array_subshape->has_layout() ||
      !LayoutUtil::Equal(mutable_array_subshape->layout(), new_layout)) {
    if (!LayoutUtil::Equal(
            new_layout,
            ShapeUtil::GetSubshape(literal().shape(), shape_index).layout())) {
      // Only relayout literals if that's really necessary.
      Literal new_literal = literal_->Relayout(new_layout, shape_index);
      *mutable_literal() = std::move(new_literal);
    }
    *mutable_array_subshape->mutable_layout() = new_layout;
  }
}

bool HloConstantInstruction::IdenticalSlowPath(
    const HloInstruction &other,
    absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
        eq_computations) const {
  const auto &casted_other = static_cast<const HloConstantInstruction &>(other);
  return literal() == casted_other.literal();
}

std::unique_ptr<HloInstruction>
HloConstantInstruction::CloneWithNewOperandsImpl(
    const Shape &shape, absl::Span<HloInstruction *const> new_operands,
    HloCloneContext *context) const {
  if (!literal_) {
    return std::make_unique<HloConstantInstruction>(this->shape());
  }
  // Literal's shape may have no/different tiling info. Use this instruction's
  // shape instead.
  CHECK(Shape::Equal().MinorToMajorOnlyInLayout()(literal_->shape(),
                                                  this->shape()));
  return std::make_unique<HloConstantInstruction>(literal_, this->shape());
}

void HloConstantInstruction::PrintOperandsWithCanonicalNameMap(
    Printer *printer, const HloPrintOptions &options,
    CanonicalNameMap *canonical_name_map) const {
  if (options.print_only_essential_constants()) {
    if (!literal_) {
      printer->Append("{...}");
      return;
    }
    if (literal().IsAll(0)) {
      printer->Append("0");
      return;
    }
    if (literal().IsAll(1)) {
      printer->Append("1");
      return;
    }
    if (shape().IsInteger()) {
      // The following prevents high compilation latencies caused by serializing
      // large constant tensors; for example: b/265669625. The limit of 500k was
      // chosen empirically to make sure that serialization of the `literal_` is
      // less than a second.
      if (auto num_constants =
              absl::c_accumulate(shape().dimensions(), 1, std::multiplies<>());
          num_constants <= 500'000) {
        literal_->PrintWithoutShapeOneline(printer);
        return;
      }
    }
    printer->Append("{...}");
    return;
  }

  // For constants, show the actual value in place of an empty operand list.
  if (literal_ &&
      ((shape().IsArray() && ShapeUtil::ElementsIn(shape()) <= 10) ||
       options.print_large_constants())) {
    // Literal::ToString emits multidimensional arrays over multiple
    // lines. Compact this into one line by stripping out white space.
    literal_->PrintWithoutShapeOneline(printer);
  } else {
    // Do not show large constants or tuples.
    printer->Append("{...}");
  }
}

HloCallableInstruction::HloCallableInstruction(HloOpcode opcode,
                                               const Shape &shape)
    : HloInstruction(opcode, shape) {}

HloCallableInstruction::HloCallableInstruction(
    HloOpcode opcode, const Shape &shape,
    absl::Span<HloInstruction *const> operands)
    : HloInstruction(opcode, shape) {
  for (auto operand : operands) {
    AppendOperand(operand);
  }
  SetAndSanitizeName(HloOpcodeString(opcode));
}

HloCallableInstruction::HloCallableInstruction(
    HloOpcode opcode, const Shape &shape,
    absl::Span<HloInstruction *const> operands,
    HloComputation *called_computation, std::string_view prefix)
    : HloInstruction(opcode, shape) {
  for (auto operand : operands) {
    AppendOperand(operand);
  }
  SetAndSanitizeName(absl::StrCat(prefix, HloOpcodeString(opcode)));
  AppendComputation(called_computation);
}

HloCallableInstruction::HloCallableInstruction(
    HloOpcode opcode, const Shape &shape,
    absl::Span<HloInstruction *const> operands,
    absl::Span<HloComputation *const> called_computations)
    : HloInstruction(opcode, shape) {
  for (auto operand : operands) {
    AppendOperand(operand);
  }
  SetAndSanitizeName(HloOpcodeString(opcode));
  for (auto called_computation : called_computations) {
    AppendComputation(called_computation);
  }
}

HloCallableInstruction::HloCallableInstruction(HloOpcode opcode,
                                               const Shape &shape,
                                               const std::string &name,
                                               const std::string &attributes,
                                               int64_t version)
    : HloInstruction(opcode, shape) {
  auto frontend_attributes =
      BuildFrontendAttributesForComposite(name, attributes, version);
  add_frontend_attributes(frontend_attributes);
  set_is_composite(true);
}

HloCallableInstruction::HloCallableInstruction(
    HloOpcode opcode, const Shape &shape,
    absl::Span<HloInstruction *const> operands, HloComputation *decomposition,
    const std::string &name, const std::string &attributes, int64_t version)
    : HloInstruction(opcode, shape) {
  for (auto operand : operands) {
    AppendOperand(operand);
  }
  SetAndSanitizeName(HloOpcodeString(opcode));
  AppendComputation(decomposition);

  auto frontend_attributes =
      BuildFrontendAttributesForComposite(name, attributes, version);
  add_frontend_attributes(frontend_attributes);
  set_is_composite(true);
}

HloCallableInstruction::~HloCallableInstruction() { ClearCalledComputations(); }

HloComputation *HloCallableInstruction::called_computation() const {
  CHECK(!called_computations().empty());
  return called_computations().front();
}

HloInstruction *HloCallableInstruction::called_computation_root() const {
  return called_computation()->root_instruction();
}

HloInstruction *HloCallableInstruction::AddCallOperand(
    HloInstruction *new_operand) {
  CHECK_EQ(operand_count(),
           called_computation()->parameter_instructions().size());
  const int64_t param_no = operand_count();
  std::string param_name = absl::StrCat("param_", param_no);
  HloInstruction *called_computation_parameter =
      called_computation()->AddParameter(HloInstruction::CreateParameter(
          param_no, new_operand->shape(), param_name));
  AppendOperand(new_operand);
  return called_computation_parameter;
}

HloInstruction *HloCallableInstruction::AppendInstructionIntoCalledComputation(
    HloInstruction *instruction_to_append, bool add_output) {
  // When add_output is false, this callable instruction must be a user of
  // instruction_to_append.
  if (!add_output) {
    CHECK(IsUserOf(instruction_to_append));
  }
  return CloneAndAppendInstructionIntoCalledComputation(instruction_to_append,
                                                        add_output);
}

HloInstruction *
HloCallableInstruction::CloneAndAppendInstructionIntoCalledComputation(
    HloInstruction *instruction_to_append, bool add_output) {
  VLOG(3) << "CloneAndAppendInstructionIntoCalledComputation:\n"
          << instruction_to_append->ToString();
  HloInstruction *clone = nullptr;
  bool do_not_clone =
      instruction_to_append->opcode() == HloOpcode::kTuple &&
      absl::c_all_of(instruction_to_append->users(), [](HloInstruction *u) {
        return u->opcode() == HloOpcode::kGetTupleElement;
      });
  if (called_computations().empty()) {
    // New fusion instruction. It should not be a multi-output instruction.
    CHECK(!add_output);
    auto builder = HloComputation::Builder(default_called_computation_name());
    builder.AddInstruction(instruction_to_append->Clone(/*suffix=*/""));
    auto *new_computation =
        CHECK_NOTNULL(GetModule())->AddEmbeddedComputation(builder.Build());
    AppendComputation(new_computation);
    if (opcode() == HloOpcode::kFusion) {
      new_computation->SetFusionInstruction(this);
    }

    clone = called_computation_root();
  } else {
    // When add_output is false, instruction_to_append is necessarily an
    // operand of the callable instruction. After appending this will no
    // longer be the case. Remove the operand from the operand list and remove
    // its corresponding called computation parameter instruction.
    bool in_operand_list =
        absl::c_linear_search(operands(), instruction_to_append);
    CHECK(add_output || in_operand_list);
    if (do_not_clone) {
      // We assume all uses of a kTuple operation are GTE ops. In this case,
      // we don't need to clone 'instruction_to_append'.
      CHECK(!in_operand_list);
      clone = instruction_to_append;
    } else {
      clone = called_computation()->AddInstruction(
          instruction_to_append->Clone(/*suffix=*/""));
    }
    const auto &called_computation_parameters =
        called_computation()->parameter_instructions();
    for (int64_t operand_num = 0; operand_num < operand_count();
         ++operand_num) {
      if (instruction_to_append == operand(operand_num)) {
        // Replace the called computation parameter instruction's uses with
        // the clone.
        HloInstruction *called_computation_parameter =
            called_computation_parameters[operand_num];
        CHECK_OK(called_computation_parameter->ReplaceAllUsesWith(clone));

        // Remove the corresponding called computation parameter and operand
        // from their respective vectors.
        CHECK_OK(called_computation()->RemoveParameter(operand_num));
        RemoveOperandAt(operand_num);
        break;
      }
    }
    // We've cloned instruction_to_append into this callable instruction, so
    // this callable instruction is no longer a use of instruction_to_append.
    if (in_operand_list) {
      DetachFrom(instruction_to_append);
      // When the instruction_to_append does not have other users, we don't
      // need to generate a multi-output instruction.
      if (instruction_to_append->user_count() == 0) {
        add_output = false;
      }
    }
  }

  // Reread the parameters in the computation.
  const auto &called_computation_parameters =
      called_computation()->parameter_instructions();

  // Add each operand of the clone as an operand of the callable instruction.
  // A complication is that some clone operands may already be operands of the
  // callable instruction.
  for (int64_t operand_num = 0; operand_num < clone->operand_count();
       ++operand_num) {
    HloInstruction *operand = clone->mutable_operand(operand_num);

    // See if this operand is already an operand of the callable instruction.
    CHECK_EQ(operands().size(), called_computation_parameters.size());
    HloInstruction *called_computation_parameter = nullptr;
    for (int64_t i = 0; i < operands().size(); ++i) {
      if (this->operand(i) == operand) {
        called_computation_parameter = called_computation_parameters[i];
        break;
      }
    }

    if (called_computation_parameter == nullptr) {
      // Clone's operand was not already an operand of the callable
      // instruction. Add it as an operand and add a corresponding called
      // computation parameter instruction.

      // No need to create an original value for an added parameter as the
      // original value is saved in the corresponding argument.
      called_computation_parameter = AddCallOperand(operand);
    }
    CHECK_OK(
        clone->ReplaceOperandWith(operand_num, called_computation_parameter));
  }

  if (clone != instruction_to_append) {
    // Copy over the original value to the clone of a fused instruction.
    if (auto original_value = instruction_to_append->original_value()) {
      clone->set_original_value(original_value);
    }
    VLOG(2) << "New clone:\n" << clone->ToString();
  }

  if (add_output) {
    int64_t user_count = instruction_to_append->user_count();
    CHECK(user_count > 0 || instruction_to_append->IsRoot())
        << "Unable to append instruction: " << instruction_to_append->ToString()
        << ", which has " << user_count << " users.";
    HloInstruction *root = called_computation_root();
    // Check whether we have replaced an existing fusion root with 'clone'. If
    // yes, no need to add a duplicate root.
    if (root->opcode() == HloOpcode::kTuple) {
      for (int64_t i = 0; i < root->operand_count(); ++i) {
        if (root->operand(i) == clone) {
          HloInstruction *new_gte = AddInstruction(
              HloInstruction::CreateGetTupleElement(clone->shape(), this, i));
          CHECK_OK(instruction_to_append->ReplaceAllUsesWith(new_gte));
          return clone;
        }
      }
    }
    // If this is already a multioutput instruction, expand the root tuple
    // by 1.
    HloInstruction::InstructionVector tuple_elements;
    bool newly_created_tuple_instr = false;
    if (root->opcode() == HloOpcode::kTuple) {
      tuple_elements = root->operands();
    } else {
      tuple_elements.push_back(root);
      newly_created_tuple_instr = true;
    }
    if (clone->opcode() == HloOpcode::kTuple) {
      for (auto inst : clone->operands()) {
        tuple_elements.push_back(inst);
      }
    } else {
      tuple_elements.push_back(clone);
    }
    HloInstruction *new_root = called_computation()->AddInstruction(
        HloInstruction::CreateTuple(tuple_elements));

    // No need to create an original value for a new root with added outputs
    // as the original value is saved in the get-tuple-element instructions
    // that use it.
    called_computation()->set_root_instruction(new_root,
                                               /*accept_different_shape=*/true);
    *mutable_shape() = new_root->shape();
    if (root->opcode() == HloOpcode::kTuple) {
      CHECK_OK(called_computation()->RemoveInstruction(root));
    }

    // If this is a newly created multioutput instruction, we need to update
    // the use of the original callable instruction.
    if (newly_created_tuple_instr) {
      HloInstruction *new_instr = AddInstruction(
          HloInstruction::CreateGetTupleElement(root->shape(), this, 0));
      CHECK_OK(ReplaceAllUsesWithDifferentShape(new_instr));
    }
    int64_t index = tuple_elements.size();
    if (do_not_clone) {
      CHECK_EQ(clone, instruction_to_append);
      index -= instruction_to_append->operand_count();
      std::vector<HloInstruction *> to_be_removed;
      const auto &users = instruction_to_append->users();
      to_be_removed.reserve(users.size());
      for (auto old_gte : users) {
        CHECK_EQ(old_gte->opcode(), HloOpcode::kGetTupleElement);
        int64_t old_tuple_index = old_gte->tuple_index();
        HloInstruction *new_gte =
            AddInstruction(HloInstruction::CreateGetTupleElement(
                old_gte->shape(), this, index + old_tuple_index));
        CHECK_OK(old_gte->ReplaceAllUsesWith(new_gte));
        to_be_removed.push_back(old_gte);
      }
      for (auto old_gte : to_be_removed) {
        CHECK_OK(parent()->RemoveInstruction(old_gte));
      }
    } else {
      HloInstruction *new_gte =
          AddInstruction(HloInstruction::CreateGetTupleElement(
              clone->shape(), this, index - 1));
      CHECK_OK(instruction_to_append->ReplaceAllUsesWith(new_gte));
    }
  }

  return clone;
}

absl::InlinedVector<HloComputation *, 1>
HloCallableInstruction::GetOrCloneCalledComputations(
    HloCloneContext *context) const {
  HloModule *module = context != nullptr ? context->module() : GetModule();
  absl::InlinedVector<HloComputation *, 1> new_called_computations;
  for (auto *comp : called_computations()) {
    HloComputation *new_custom_call_computation = nullptr;
    if (context != nullptr) {
      new_custom_call_computation = context->FindComputation(comp);
    }
    if (new_custom_call_computation == nullptr) {
      new_custom_call_computation =
          module->AddEmbeddedComputation(comp->Clone("clone", context));
    }
    new_called_computations.push_back(new_custom_call_computation);
  }
  return new_called_computations;
}

void HloCallableInstruction::RecursivelySetComputationsThreadName(
    std::string_view execution_thread,
    bool skip_async_execution_thread_overwrite) {
  for (HloComputation *comp : called_computations()) {
    SetThreadName(comp, execution_thread,
                  skip_async_execution_thread_overwrite);
  }
}

HloFusionInstruction::HloFusionInstruction(const Shape &shape,
                                           FusionKind fusion_kind,
                                           HloInstruction *fused_root,
                                           std::string_view prefix)
    : HloCallableInstruction(HloOpcode::kFusion, shape),
      fusion_kind_(fusion_kind) {
  CHECK_NE(fused_root, nullptr);
  SetAndSanitizeName(absl::StrCat(prefix, HloOpcodeString(opcode())));

  set_parent(fused_root->parent());
  set_metadata(fused_root->metadata());
  set_frontend_attributes(fused_root->frontend_attributes());
  // This simplifies some use cases for the original value that involve fusions.
  if (auto original_value = fused_root->original_value()) {
    set_original_value(original_value);
  }
  CHECK(fused_root->IsFusible()) << fused_root->ToString();
  CloneAndAppendInstructionIntoCalledComputation(fused_root);
}

HloFusionInstruction::HloFusionInstruction(
    const Shape &shape, FusionKind fusion_kind,
    absl::Span<HloInstruction *const> operands,
    HloComputation *fusion_computation, std::string_view prefix)
    : HloCallableInstruction(HloOpcode::kFusion, shape, operands,
                             fusion_computation, prefix),
      fusion_kind_(fusion_kind) {
  fusion_computation->SetFusionInstruction(this);
}

HloFusionInstruction::~HloFusionInstruction() {
  ClearFusionComputationInstruction();
}

void HloFusionInstruction::ClearFusionComputationInstruction() {
  // Each fusion calls a single computation, but we use called_computations()
  // instead of fused_instructions_computation(), because the order in which
  // things get destructed can vary; the fusion computation's back-pointer may
  // already be null, which violates a check in
  // fused_instructions_computation.
  for (HloComputation *computation : called_computations()) {
    // Some passes that rewrite fusions may reassign a fusion computation to a
    // different fusion instruction as this instruction gets destructed.
    if (computation->FusionInstruction() == this) {
      computation->SetFusionInstruction(nullptr);
    }
  }
}

void HloFusionInstruction::ClearCalledComputations() {
  ClearFusionComputationInstruction();
  HloInstruction::ClearCalledComputations();
}

HloInstruction *
HloFusionInstruction::CloneAndAppendInstructionIntoCalledComputation(
    HloInstruction *instruction_to_append, bool add_output) {
  CHECK(instruction_to_append->IsFusible())
      << instruction_to_append->ToString();
  return HloCallableInstruction::CloneAndAppendInstructionIntoCalledComputation(
      instruction_to_append, add_output);
}

std::string HloFusionInstruction::ToCategory() const {
  switch (fusion_kind()) {
    case FusionKind::kLoop:
      return "loop fusion";
    case FusionKind::kInput:
      return "input fusion";
    case FusionKind::kOutput:
      return "output fusion";
    case FusionKind::kCustom:
      return "custom fusion";
  }
}

HloInstructionProto HloFusionInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_fusion_kind() = std::string(zkx::ToString(fusion_kind()));
  for (const auto &pair : output_to_operand_aliasing()) {
    auto aliasing = proto.add_output_operand_aliasing();
    aliasing->set_operand_index(pair.second.first);
    for (int64_t index : pair.first) {
      aliasing->add_output_shape_index(index);
    }
    for (int64_t index : pair.second.second) {
      aliasing->add_operand_shape_index(index);
    }
  }
  proto.add_called_computation_ids(
      fused_instructions_computation()->unique_id());
  return proto;
}

bool HloFusionInstruction::IsElementwiseImpl(
    const std::optional<int64_t> &operand_idx) const {
  if (!operand_idx.has_value()) {
    for (auto *fused : fused_instructions()) {
      if (fused->opcode() != HloOpcode::kParameter && !fused->IsElementwise()) {
        return false;
      }
    }
    return true;
  }
  // A loop-fusion is elementwise on an operand if all operations (computed
  // using BFS) between the operand and the fused root are elementwise.
  std::deque<HloInstruction *> worklist;
  absl::flat_hash_set<const HloInstruction *> visited;
  worklist.push_back(fused_parameter(operand_idx.value()));
  visited.insert(fused_parameter(operand_idx.value()));
  while (!worklist.empty()) {
    HloInstruction *operand = worklist.front();
    worklist.pop_front();
    for (HloInstruction *user : operand->users()) {
      CHECK_GE(user->unique_id(), 0);
      if (ContainsKey(visited, user)) {
        continue;
      }
      if (user->IsElementwise() ||
          IsInstructionElementwiseOnOperand(user, operand)) {
        worklist.push_back(user);
        visited.insert(user);
      } else {
        return false;
      }
    }
  }
  return true;
}

HloInstruction *HloFusionInstruction::AddFusionOperand(
    HloInstruction *new_operand) {
  return AddCallOperand(new_operand);
}

void HloFusionInstruction::MergeFusionInstruction(
    HloFusionInstruction *instruction_to_merge) {
  CHECK(absl::c_linear_search(operands(), instruction_to_merge));
  // Clone the instruction from which to merge fused instructions.
  std::unique_ptr<HloInstruction> cloned = instruction_to_merge->Clone();
  HloFusionInstruction *cloned_fusion =
      static_cast<HloFusionInstruction *>(cloned.get());
  // Replace uses of fused parameters with the corresponding operand of the
  // fusion.  Add all non-parameter fused instructions to
  // 'unfused_instructions' to be merged into 'this'.  This is done in reverse
  // post order.
  std::vector<HloInstruction *> unfused_instructions;
  auto fused_instructions = cloned_fusion->fused_instructions_computation()
                                ->MakeInstructionPostOrder();
  for (auto fused_it = fused_instructions.rbegin();
       fused_it != fused_instructions.rend(); ++fused_it) {
    auto fused_instruction = *fused_it;
    if (fused_instruction->opcode() == HloOpcode::kParameter) {
      CHECK_OK(
          fused_instruction->ReplaceAllUsesWith(cloned_fusion->mutable_operand(
              fused_instruction->parameter_number())));
    } else {
      unfused_instructions.push_back(fused_instruction);
    }
  }

  // If there are no unfused instructions, the fused computation must consist
  // only of kParameter instructions. Make the operand of the corresponding
  // parameter number the new root.
  HloInstruction *unfused_root =
      unfused_instructions.empty()
          ? instruction_to_merge->mutable_operand(
                instruction_to_merge->fused_instructions_computation()
                    ->root_instruction()
                    ->parameter_number())
          : unfused_instructions.front();
  CHECK(unfused_root == cloned_fusion->fused_expression_root() ||
        unfused_instructions.empty());
  // Replace instruction_to_merge use of 'this' with unfused_root.
  CHECK_OK(instruction_to_merge->ReplaceUseWith(this, unfused_root));

  // Build a dummy root for the cloned fusion as we may remove the original
  // root in the fusion process.
  if (!unfused_instructions.empty()) {
    HloComputation *computation = unfused_root->parent();
    auto *dummy_root = computation->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::Zero(U32)));
    computation->set_root_instruction(dummy_root,
                                      /*accept_different_shape=*/true);
  }

  // Fuse 'unfused_instructions' into 'this'. Every time we fuse an instruction
  // we remove it from the closed fusion node. This is so that we don't add
  // extra users to the producer of that instruction (we use user count to
  // decide if a side-effectful instruction is fusible).
  for (auto &instruction : unfused_instructions) {
    auto *fused = FuseInstruction(instruction);
    CHECK_OK(instruction->ReplaceAllUsesWith(fused));
    CHECK_OK(instruction->parent()->RemoveInstruction(instruction));
  }
  CHECK_EQ(0, cloned_fusion->user_count());
  CHECK_OK(GetModule()->RemoveEmbeddedComputation(
      cloned_fusion->fused_instructions_computation()));
}

void HloFusionInstruction::MergeFusionInstructionIntoMultiOutput(
    HloFusionInstruction *instruction_to_merge) {
  // Add all non-parameter fused instructions to 'unfused_instructions' to be
  // merged into 'this'. `old_to_new' maps the instructions in the fused node
  // to the disassembled fusion instructions.
  // Note that we add the unfused instructions to this->parent_ computation.
  // This is necessary because the unique_id needs for an instruction and
  // it's only added when inserting to the computation.
  absl::flat_hash_map<HloInstruction *, HloInstruction *> old_to_new;
  std::vector<HloInstruction *> unfused_instructions;
  absl::flat_hash_set<const HloInstruction *> new_roots;
  std::vector<std::pair<HloInstruction *, int64_t>> old_fusion_outputs;
  auto computation_to_merge =
      instruction_to_merge->fused_instructions_computation();
  for (auto fused_instruction :
       computation_to_merge->MakeInstructionPostOrder()) {
    if (fused_instruction->opcode() == HloOpcode::kParameter) {
      InsertOrDie(&old_to_new, fused_instruction,
                  instruction_to_merge->mutable_operand(
                      fused_instruction->parameter_number()));
      continue;
    }
    // If 'instruction_to_merge' is a multi-output fusion, we need to skip the
    // root tuple, but remember which of the fusion outputs need to become
    // fusion outputs of the merged fusion.
    if (fused_instruction->opcode() == HloOpcode::kTuple &&
        fused_instruction == instruction_to_merge->fused_expression_root()) {
      for (const HloInstruction *user : instruction_to_merge->users()) {
        CHECK_EQ(user->opcode(), HloOpcode::kGetTupleElement);
        old_fusion_outputs.emplace_back(
            fused_instruction->mutable_operand(user->tuple_index()),
            user->tuple_index());
        bool has_outside_user = false;
        for (HloInstruction *gte_user : user->users()) {
          if (gte_user != this) {
            has_outside_user = true;
            break;
          }
        }
        if (!has_outside_user && !user->IsRoot()) {
          continue;
        }

        new_roots.insert(
            FindOrDie(old_to_new, old_fusion_outputs.back().first));
      }
      continue;
    }

    // Here we clone the insertion and call FuseInstructionIntoMultiOutput()
    // which clones again. This can be improved.
    std::vector<HloInstruction *> new_operands;
    new_operands.reserve(fused_instruction->operand_count());
    for (HloInstruction *new_operand : fused_instruction->mutable_operands()) {
      new_operands.push_back(FindOrDie(old_to_new, new_operand));
    }
    auto cloned_instruction =
        parent()->AddInstruction(fused_instruction->CloneWithNewOperands(
            fused_instruction->shape(), new_operands, /*suffix=*/"clone"));
    // Copy over the original value to the clone of a fused instruction.
    // This is necessary as the clone will be cloned again when the clone is
    // fused in FuseInstructionIntoMultiOutput(). This can be skipped if we
    // improve the code to only clone once as stated in the preceding comment.
    if (auto original_value = fused_instruction->original_value()) {
      cloned_instruction->set_original_value(original_value);
    }
    unfused_instructions.push_back(cloned_instruction);
    InsertOrDie(&old_to_new, fused_instruction, cloned_instruction);
  }
  if (instruction_to_merge->IsMultiOutputFusion()) {
    for (auto [old_root, tuple_index] : old_fusion_outputs) {
      auto new_root = FindOrDie(old_to_new, old_root);
      // Replace the get-tuple-element op on 'instruction_to_merge' referencing
      // the same tuple index as 'old_root' with 'new_root'.
      for (HloInstruction *gte : instruction_to_merge->users()) {
        if (gte->opcode() == HloOpcode::kGetTupleElement &&
            gte->tuple_index() == tuple_index) {
          CHECK_OK(gte->ReplaceAllUsesWith(new_root));
          CHECK_OK(gte->parent()->RemoveInstruction(gte));
        }
      }
    }
  } else {
    // If there are no unfused instructions, the fused computation must consist
    // only of kParameter instructions. Make the operand of the corresponding
    // parameter number the new root.
    HloInstruction *unfused_root =
        unfused_instructions.empty()
            ? instruction_to_merge->mutable_operand(
                  instruction_to_merge->fused_instructions_computation()
                      ->root_instruction()
                      ->parameter_number())
            : unfused_instructions.back();
    new_roots.insert(unfused_root);
    CHECK_OK(instruction_to_merge->ReplaceAllUsesWith(unfused_root));
  }
  CHECK_OK(
      instruction_to_merge->parent()->RemoveInstruction(instruction_to_merge));
  if (GetModule()) {
    CHECK_OK(GetModule()->RemoveEmbeddedComputation(computation_to_merge));
  }
  for (int64_t i = unfused_instructions.size() - 1; i >= 0; --i) {
    HloInstruction *instruction = unfused_instructions[i];
    if (new_roots.contains(instruction)) {
      FuseInstructionIntoMultiOutput(instruction);
    } else {
      FuseInstruction(instruction);
    }
    CHECK_OK(instruction->parent()->RemoveInstruction(instruction));
  }
}

HloComputation *HloFusionInstruction::fused_instructions_computation() const {
  CHECK_EQ(called_computations().size(), 1);
  auto *fused_instructions_computation = called_computations().front();
  CHECK(fused_instructions_computation->IsFusionComputation())
      << "Computation " << fused_instructions_computation->name()
      << " is not a fusion kind";
  return fused_instructions_computation;
}

HloInstruction *HloFusionInstruction::fused_expression_root() const {
  return fused_instructions_computation()->root_instruction();
}

HloInstruction *HloFusionInstruction::fused_parameter(
    int64_t parameter_number) const {
  return fused_instructions_computation()->parameter_instruction(
      parameter_number);
}

const HloInstruction::InstructionVector &
HloFusionInstruction::fused_parameters() const {
  return fused_instructions_computation()->parameter_instructions();
}

tsl::gtl::iterator_range<HloInstructionUnwrappingConstIterator>
HloFusionInstruction::fused_instructions() const {
  const HloComputation *subcomp = fused_instructions_computation();
  return subcomp->instructions();
}

tsl::gtl::iterator_range<HloInstructionUnwrappingIterator>
HloFusionInstruction::fused_instructions() {
  return fused_instructions_computation()->instructions();
}

int64_t HloFusionInstruction::fused_instruction_count() const {
  return fused_instructions_computation()->instruction_count();
}

void HloFusionInstruction::PrintExtraAttributesImpl(
    AttributePrinter &printer, const HloPrintOptions &options) const {
  printer.Next([this](Printer *printer) {
    AppendCat(printer, "kind=", zkx::ToString(fusion_kind()));
  });
  if (!output_to_operand_aliasing().empty()) {
    printer.Next([this](Printer *printer) {
      printer->Append("output_to_operand_aliasing={");
      AppendJoin(printer, output_to_operand_aliasing(), ", ",
                 [](Printer *printer, auto &pair) {
                   AppendCat(printer, pair.first.ToString(), ": (",
                             pair.second.first, ", ");
                   AppendCat(printer, pair.second.second.ToString(), ")");
                 });
      printer->Append("}");
    });
  }
}

bool HloFusionInstruction::IdenticalSlowPath(
    const HloInstruction &other,
    absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
        eq_computations) const {
  return fusion_kind() == other.fusion_kind() &&
         output_to_operand_aliasing() ==
             static_cast<const HloFusionInstruction &>(other)
                 .output_to_operand_aliasing() &&
         eq_computations(fused_instructions_computation(),
                         other.fused_instructions_computation());
}

std::unique_ptr<HloInstruction> HloFusionInstruction::CloneWithNewOperandsImpl(
    const Shape &shape, absl::Span<HloInstruction *const> new_operands,
    HloCloneContext *context) const {
  auto new_fused_computation = GetOrCloneCalledComputations(context);
  CHECK_EQ(new_fused_computation.size(), 1);
  auto new_fusion_instruction = std::make_unique<HloFusionInstruction>(
      shape, fusion_kind(), new_operands, new_fused_computation.front());
  new_fusion_instruction->set_output_to_operand_aliasing(
      output_to_operand_aliasing());
  return new_fusion_instruction;
}

absl::Status HloFusionInstruction::DeduplicateFusionOperands() {
  if (IsCustomFusion()) {
    return absl::OkStatus();
  }
  absl::flat_hash_map<const HloInstruction *, int> operand_indices;
  std::vector<int> operands_to_remove;
  const int count = operand_count();
  operands_to_remove.reserve(count);
  for (int i = 0; i < count; ++i) {
    auto emplace_result = operand_indices.emplace(operand(i), i);
    if (!emplace_result.second) {
      TF_RETURN_IF_ERROR(fused_parameter(i)->ReplaceAllUsesWith(
          fused_parameter(emplace_result.first->second)));
      operands_to_remove.push_back(i);
    }
  }
  if (operands_to_remove.empty()) {
    return absl::OkStatus();
  }
  TF_RETURN_IF_ERROR(fused_instructions_computation()
                         ->RemoveUnusedParametersFromFusedComputation());
  RemoveOperandsAtAscendingIndices(operands_to_remove);
  return absl::OkStatus();
}

HloCallInstruction::HloCallInstruction(const Shape &shape,
                                       HloInstruction *called_computation_root)
    : HloCallableInstruction(HloOpcode::kCall, shape) {
  CHECK(called_computation_root != nullptr);
  SetAndSanitizeName(HloOpcodeString(opcode()));
  set_parent(called_computation_root->parent());
  set_metadata(called_computation_root->metadata());
  CloneAndAppendInstructionIntoCalledComputation(called_computation_root);
}

HloCallInstruction::HloCallInstruction(
    const Shape &shape, absl::Span<HloInstruction *const> operands,
    HloComputation *called_computation)
    : HloCallableInstruction(HloOpcode::kCall, shape, operands,
                             called_computation) {}

HloCallInstruction::HloCallInstruction(const Shape &shape,
                                       HloInstruction *decomposition_root,
                                       const std::string &name,
                                       const std::string &attributes,
                                       int64_t version)
    : HloCallableInstruction(HloOpcode::kCall, shape, name, attributes,
                             version) {
  CHECK(decomposition_root != nullptr);
  SetAndSanitizeName(HloOpcodeString(opcode()));

  FrontendAttributes frontend_attributes;
  frontend_attributes.mutable_map()->insert({"composite.name", name});
  frontend_attributes.mutable_map()->insert(
      {"composite.attributes", attributes});
  frontend_attributes.mutable_map()->insert(
      {"composite.version", std::to_string(version)});

  add_frontend_attributes(frontend_attributes);
  set_is_composite(true);
  set_parent(decomposition_root->parent());
  set_metadata(decomposition_root->metadata());
  CloneAndAppendInstructionIntoCalledComputation(decomposition_root);
}

HloCallInstruction::HloCallInstruction(
    const Shape &shape, absl::Span<HloInstruction *const> operands,
    HloComputation *decomposition, const std::string &name,
    const std::string &attributes, int64_t version)
    : HloCallableInstruction(HloOpcode::kCall, shape, operands, decomposition,
                             name, attributes, version) {
  FrontendAttributes frontend_attributes;
  frontend_attributes.mutable_map()->insert({"composite.name", name});
  frontend_attributes.mutable_map()->insert(
      {"composite.attributes", attributes});
  frontend_attributes.mutable_map()->insert(
      {"composite.version", std::to_string(version)});

  add_frontend_attributes(frontend_attributes);
  set_is_composite(true);
}

HloParameterInstruction::HloParameterInstruction(int64_t parameter_number,
                                                 const Shape &shape,
                                                 std::string_view name)
    : HloInstruction(HloOpcode::kParameter, shape),
      parameter_number_(parameter_number) {
  SetAndSanitizeName(name);
}

HloInstructionProto HloParameterInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_parameter_number(parameter_number_);
  if (parameter_replicated_at_leaf_buffers_) {
    for (bool replicated : *parameter_replicated_at_leaf_buffers_) {
      proto.mutable_parameter_replication()->add_replicated_at_leaf_buffers(
          replicated);
    }
  }
  return proto;
}

void HloParameterInstruction::PrintExtraAttributesImpl(
    AttributePrinter &printer, const HloPrintOptions &options) const {
  if (!parameter_replicated_at_leaf_buffers_ || !options.print_ids()) {
    return;
  }
  printer.Next([this](Printer *printer) {
    printer->Append("parameter_replication={");
    AppendJoin(printer, *parameter_replicated_at_leaf_buffers_, ",",
               [](Printer *printer, bool replicated) {
                 printer->Append(replicated ? "true" : "false");
               });
    printer->Append("}");
  });
}

void HloParameterInstruction::PrintOperandsWithCanonicalNameMap(
    Printer *printer, const HloPrintOptions &options,
    CanonicalNameMap *canonical_name_map) const {
  if (options.print_parameter_number()) {
    printer->Append(parameter_number_);
  }
}

bool HloParameterInstruction::IdenticalSlowPath(
    const HloInstruction &other,
    absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
        eq_computations) const {
  const auto &casted_other =
      static_cast<const HloParameterInstruction &>(other);
  return parameter_number() == casted_other.parameter_number();
}

std::unique_ptr<HloInstruction>
HloParameterInstruction::CloneWithNewOperandsImpl(
    const Shape &shape, absl::Span<HloInstruction *const> new_operands,
    HloCloneContext *context) const {
  auto clone = std::make_unique<HloParameterInstruction>(parameter_number_,
                                                         shape, name());
  if (parameter_replicated_at_leaf_buffers_ &&
      ShapeUtil::Equal(shape, this->shape())) {
    clone->set_parameter_replicated_at_leaf_buffers(
        *parameter_replicated_at_leaf_buffers_);
  }
  return clone;
}

HloGetTupleElementInstruction::HloGetTupleElementInstruction(
    const Shape &shape, HloInstruction *operand, int64_t index)
    : HloInstruction(HloOpcode::kGetTupleElement, shape), tuple_index_(index) {
  AppendOperand(operand);
}

HloInstructionProto HloGetTupleElementInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_tuple_index(tuple_index_);
  return proto;
}

void HloGetTupleElementInstruction::PrintExtraAttributesImpl(
    AttributePrinter &printer, const HloPrintOptions &options) const {
  printer.Next([this](Printer *printer) {
    AppendCat(printer, "index=", tuple_index());
  });
}

bool HloGetTupleElementInstruction::IdenticalSlowPath(
    const HloInstruction &other,
    absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
        eq_computations) const {
  const auto &casted_other =
      static_cast<const HloGetTupleElementInstruction &>(other);
  return tuple_index() == casted_other.tuple_index();
}

std::unique_ptr<HloInstruction>
HloGetTupleElementInstruction::CloneWithNewOperandsImpl(
    const Shape &shape, absl::Span<HloInstruction *const> new_operands,
    HloCloneContext *context) const {
  CHECK_EQ(new_operands.size(), 1);
  return std::make_unique<HloGetTupleElementInstruction>(shape, new_operands[0],
                                                         tuple_index());
}

HloReduceWindowInstruction::HloReduceWindowInstruction(
    const Shape &shape, HloInstruction *operand, HloInstruction *init_value,
    const Window &window, HloComputation *reduce_computation)
    : HloReduceWindowInstruction(shape, absl::MakeSpan(&operand, 1),
                                 absl::MakeSpan(&init_value, 1), window,
                                 reduce_computation) {}

HloReduceWindowInstruction::HloReduceWindowInstruction(
    const Shape &shape, absl::Span<HloInstruction *const> operands,
    absl::Span<HloInstruction *const> init_values, const Window &window,
    HloComputation *reduce_computation)
    : HloInstruction(HloOpcode::kReduceWindow, shape), window_(window) {
  for (auto *operand : operands) {
    AppendOperand(operand);
  }
  for (auto *init_value : init_values) {
    AppendOperand(init_value);
  }
  AppendComputation(reduce_computation);
}

HloInstructionProto HloReduceWindowInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_window() = window_;
  return proto;
}

void HloReduceWindowInstruction::PrintExtraAttributesImpl(
    AttributePrinter &printer, const HloPrintOptions &options) const {
  if (window_.dimensions_size() != 0) {
    printer.Next([this](Printer *printer) {
      AppendCat(printer, "window={", window_util::ToString(window()), "}");
    });
  }
}

bool HloReduceWindowInstruction::IdenticalSlowPath(
    const HloInstruction &other,
    absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
        eq_computations) const {
  const auto &casted_other =
      static_cast<const HloReduceWindowInstruction &>(other);
  return eq_computations(to_apply(), casted_other.to_apply()) &&
         protobuf_util::ProtobufEquals(window(), casted_other.window());
}

std::unique_ptr<HloInstruction>
HloReduceWindowInstruction::CloneWithNewOperandsImpl(
    const Shape &shape, absl::Span<HloInstruction *const> new_operands,
    HloCloneContext *context) const {
  CHECK_EQ(new_operands.size() % 2, 0);
  int64_t num_operands = new_operands.size() / 2;
  return std::make_unique<HloReduceWindowInstruction>(
      shape, absl::MakeSpan(new_operands).subspan(0, num_operands),
      absl::MakeSpan(new_operands)
          .subspan(num_operands, new_operands.size() / 2),
      window(), to_apply());
}

HloPadInstruction::HloPadInstruction(const Shape &shape,
                                     HloInstruction *operand,
                                     HloInstruction *padding_value,
                                     const PaddingConfig &padding_config)
    : HloInstruction(HloOpcode::kPad, shape), padding_config_(padding_config) {
  AppendOperand(operand);
  AppendOperand(padding_value);
}

HloInstructionProto HloPadInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_padding_config() = padding_config_;
  return proto;
}

void HloPadInstruction::PrintExtraAttributesImpl(
    AttributePrinter &printer, const HloPrintOptions &options) const {
  printer.Next([this](Printer *printer) {
    AppendCat(printer, "padding=", zkx::PaddingConfigToString(padding_config_));
  });
}

bool HloPadInstruction::IdenticalSlowPath(
    const HloInstruction &other,
    absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
        eq_computations) const {
  const auto &casted_other = static_cast<const HloPadInstruction &>(other);
  return protobuf_util::ProtobufEquals(padding_config(),
                                       casted_other.padding_config());
}

std::unique_ptr<HloInstruction> HloPadInstruction::CloneWithNewOperandsImpl(
    const Shape &shape, absl::Span<HloInstruction *const> new_operands,
    HloCloneContext *context) const {
  CHECK_EQ(new_operands.size(), 2);
  return std::make_unique<HloPadInstruction>(shape, new_operands[0],
                                             new_operands[1], padding_config_);
}

HloDynamicSliceInstruction::HloDynamicSliceInstruction(
    const Shape &shape, HloInstruction *operand, HloInstruction *start_indices,
    absl::Span<const int64_t> slice_sizes)
    : HloDynamicIndexInstruction(HloOpcode::kDynamicSlice, shape),
      dynamic_slice_sizes_(slice_sizes.begin(), slice_sizes.end()) {
  AppendOperand(operand);
  AppendOperand(start_indices);
}

HloDynamicSliceInstruction::HloDynamicSliceInstruction(
    const Shape &shape, HloInstruction *operand,
    absl::Span<HloInstruction *const> start_indices,
    absl::Span<const int64_t> slice_sizes)
    : HloDynamicIndexInstruction(HloOpcode::kDynamicSlice, shape),
      dynamic_slice_sizes_(slice_sizes.begin(), slice_sizes.end()) {
  AppendOperand(operand);
  for (HloInstruction *index : start_indices) {
    AppendOperand(index);
  }
}

HloInstructionProto HloDynamicSliceInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  for (int64_t slice_size : dynamic_slice_sizes_) {
    proto.add_dynamic_slice_sizes(slice_size);
  }
  return proto;
}

void HloDynamicSliceInstruction::PrintExtraAttributesImpl(
    AttributePrinter &printer, const HloPrintOptions &options) const {
  printer.Next([this](Printer *printer) {
    printer->Append("dynamic_slice_sizes={");
    AppendJoin(printer, dynamic_slice_sizes(), ",");
    printer->Append("}");
  });
}

bool HloDynamicSliceInstruction::IdenticalSlowPath(
    const HloInstruction &other,
    absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
        eq_computations) const {
  const auto &casted_other =
      static_cast<const HloDynamicSliceInstruction &>(other);
  return dynamic_slice_sizes() == casted_other.dynamic_slice_sizes();
}

std::unique_ptr<HloInstruction>
HloDynamicSliceInstruction::CloneWithNewOperandsImpl(
    const Shape &shape, absl::Span<HloInstruction *const> new_operands,
    HloCloneContext *context) const {
  if (new_operands.size() == 2 && new_operands[1]->shape().rank() == 1) {
    // TODO(b/118437727): Old form, remove this path.
    return std::make_unique<HloDynamicSliceInstruction>(
        shape, new_operands[0], new_operands[1], dynamic_slice_sizes_);
  } else {
    return std::make_unique<HloDynamicSliceInstruction>(
        shape, new_operands[0], new_operands.subspan(1), dynamic_slice_sizes_);
  }
}

HloDynamicUpdateSliceInstruction::HloDynamicUpdateSliceInstruction(
    const Shape &shape, HloInstruction *operand, HloInstruction *update,
    HloInstruction *start_indices)
    : HloDynamicIndexInstruction(HloOpcode::kDynamicUpdateSlice, shape) {
  AppendOperand(operand);
  AppendOperand(update);
  AppendOperand(start_indices);
}

HloDynamicUpdateSliceInstruction::HloDynamicUpdateSliceInstruction(
    const Shape &shape, HloInstruction *operand, HloInstruction *update,
    absl::Span<HloInstruction *const> start_indices)
    : HloDynamicIndexInstruction(HloOpcode::kDynamicUpdateSlice, shape) {
  AppendOperand(operand);
  AppendOperand(update);
  for (HloInstruction *index : start_indices) {
    AppendOperand(index);
  }
}

HloGatherInstruction::HloGatherInstruction(
    const Shape &shape, HloInstruction *operand, HloInstruction *start_indices,
    const GatherDimensionNumbers &gather_dim_numbers,
    absl::Span<const int64_t> slice_sizes, bool indices_are_sorted)
    : HloInstruction(HloOpcode::kGather, shape),
      indices_are_sorted_(indices_are_sorted) {
  AppendOperand(operand);
  AppendOperand(start_indices);
  gather_dimension_numbers_ =
      std::make_unique<GatherDimensionNumbers>(gather_dim_numbers);
  absl::c_copy(slice_sizes, std::back_inserter(gather_slice_sizes_));
}

// static
std::string HloGatherInstruction::GatherDimensionNumbersToString(
    const GatherDimensionNumbers &dim_numbers) {
  StringPrinter printer;
  PrintGatherDimensionNumbers(&printer, dim_numbers);
  return std::move(printer).ToString();
}

// static
void HloGatherInstruction::PrintGatherDimensionNumbers(
    Printer *printer, const GatherDimensionNumbers &dim_numbers) {
  printer->Append("offset_dims={");
  AppendJoin(printer, dim_numbers.offset_dims(), ",");
  printer->Append("}, collapsed_slice_dims={");
  AppendJoin(printer, dim_numbers.collapsed_slice_dims(), ",");
  printer->Append("}, start_index_map={");
  AppendJoin(printer, dim_numbers.start_index_map(), ",");
  if (dim_numbers.operand_batching_dims_size()) {
    printer->Append("}, operand_batching_dims={");
    AppendJoin(printer, dim_numbers.operand_batching_dims(), ",");
  }
  if (dim_numbers.start_indices_batching_dims_size()) {
    printer->Append("}, start_indices_batching_dims={");
    AppendJoin(printer, dim_numbers.start_indices_batching_dims(), ",");
  }
  AppendCat(printer, "}, index_vector_dim=", dim_numbers.index_vector_dim());
}

// static
GatherDimensionNumbers HloGatherInstruction::MakeGatherDimNumbers(
    absl::Span<const int64_t> offset_dims,
    absl::Span<const int64_t> collapsed_slice_dims,
    absl::Span<const int64_t> start_index_map, int64_t index_vector_dim,
    absl::Span<const int64_t> operand_batching_dims,
    absl::Span<const int64_t> start_indices_batching_dims) {
  GatherDimensionNumbers gather_dim_numbers;
  for (int64_t output_window_dim : offset_dims) {
    gather_dim_numbers.add_offset_dims(output_window_dim);
  }
  for (int64_t elided_window_dim : collapsed_slice_dims) {
    gather_dim_numbers.add_collapsed_slice_dims(elided_window_dim);
  }
  for (int64_t gather_dim_to_input_dim : start_index_map) {
    gather_dim_numbers.add_start_index_map(gather_dim_to_input_dim);
  }
  for (int64_t operand_batching_dim : operand_batching_dims) {
    gather_dim_numbers.add_operand_batching_dims(operand_batching_dim);
  }
  for (int64_t start_indices_batching_dim : start_indices_batching_dims) {
    gather_dim_numbers.add_start_indices_batching_dims(
        start_indices_batching_dim);
  }

  gather_dim_numbers.set_index_vector_dim(index_vector_dim);
  return gather_dim_numbers;
}

HloInstructionProto HloGatherInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_gather_dimension_numbers() = gather_dimension_numbers();
  for (int64_t bound : gather_slice_sizes()) {
    proto.add_gather_slice_sizes(bound);
  }
  proto.set_indices_are_sorted(indices_are_sorted());
  return proto;
}

void HloGatherInstruction::PrintExtraAttributesImpl(
    AttributePrinter &printer, const HloPrintOptions &options) const {
  printer.Next([this](Printer *printer) {
    PrintGatherDimensionNumbers(printer, gather_dimension_numbers());
  });
  printer.Next([this](Printer *printer) {
    printer->Append("slice_sizes={");
    AppendJoin(printer, gather_slice_sizes(), ",");
    printer->Append("}");
  });
  if (indices_are_sorted()) {
    printer.Next(
        [](Printer *printer) { printer->Append("indices_are_sorted=true"); });
  }
}

bool HloGatherInstruction::IdenticalSlowPath(
    const HloInstruction &other,
    absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
        eq_computations) const {
  const auto &casted_other = static_cast<const HloGatherInstruction &>(other);
  return protobuf_util::ProtobufEquals(
             gather_dimension_numbers(),
             casted_other.gather_dimension_numbers()) &&
         gather_slice_sizes() == casted_other.gather_slice_sizes() &&
         indices_are_sorted() == casted_other.indices_are_sorted();
}

std::unique_ptr<HloInstruction> HloGatherInstruction::CloneWithNewOperandsImpl(
    const Shape &shape, absl::Span<HloInstruction *const> new_operands,
    HloCloneContext *context) const {
  CHECK_EQ(new_operands.size(), 2);
  return std::make_unique<HloGatherInstruction>(
      shape, new_operands[0], new_operands[1], gather_dimension_numbers(),
      gather_slice_sizes(), indices_are_sorted());
}

HloScatterInstruction::HloScatterInstruction(
    const Shape &shape, absl::Span<HloInstruction *const> args,
    HloComputation *update_computation,
    const ScatterDimensionNumbers &scatter_dim_numbers, bool indices_are_sorted,
    bool unique_indices)
    : HloInstruction(HloOpcode::kScatter, shape),
      indices_are_sorted_(indices_are_sorted),
      unique_indices_(unique_indices) {
  mutable_operands().reserve(args.size());
  for (HloInstruction *arg : args) {
    AppendOperand(arg);
  }
  AppendComputation(update_computation);
  scatter_dimension_numbers_ =
      std::make_unique<ScatterDimensionNumbers>(scatter_dim_numbers);
}

// static
std::string HloScatterInstruction::ScatterDimensionNumbersToString(
    const ScatterDimensionNumbers &dim_numbers) {
  StringPrinter printer;
  PrintScatterDimensionNumbers(&printer, dim_numbers);
  return std::move(printer).ToString();
}

// static
void HloScatterInstruction::PrintScatterDimensionNumbers(
    Printer *printer, const ScatterDimensionNumbers &dim_numbers) {
  printer->Append("update_window_dims={");
  AppendJoin(printer, dim_numbers.update_window_dims(), ",");
  printer->Append("}, inserted_window_dims={");
  AppendJoin(printer, dim_numbers.inserted_window_dims(), ",");
  printer->Append("}, scatter_dims_to_operand_dims={");
  AppendJoin(printer, dim_numbers.scatter_dims_to_operand_dims(), ",");
  if (dim_numbers.input_batching_dims_size()) {
    printer->Append("}, input_batching_dims={");
    AppendJoin(printer, dim_numbers.input_batching_dims(), ",");
  }
  if (dim_numbers.scatter_indices_batching_dims_size()) {
    printer->Append("}, scatter_indices_batching_dims={");
    AppendJoin(printer, dim_numbers.scatter_indices_batching_dims(), ",");
  }
  AppendCat(printer, "}, index_vector_dim=", dim_numbers.index_vector_dim());
}

// static
ScatterDimensionNumbers HloScatterInstruction::MakeScatterDimNumbers(
    absl::Span<const int64_t> update_window_dims,
    absl::Span<const int64_t> inserted_window_dims,
    absl::Span<const int64_t> scatter_dims_to_operand_dims,
    int64_t index_vector_dim, absl::Span<const int64_t> input_batching_dims,
    absl::Span<const int64_t> scatter_indices_batching_dims) {
  ScatterDimensionNumbers scatter_dim_numbers;
  for (int64_t update_window_dim : update_window_dims) {
    scatter_dim_numbers.add_update_window_dims(update_window_dim);
  }
  for (int64_t inserted_window_dim : inserted_window_dims) {
    scatter_dim_numbers.add_inserted_window_dims(inserted_window_dim);
  }
  for (int64_t scatter_dim_to_operand_dim : scatter_dims_to_operand_dims) {
    scatter_dim_numbers.add_scatter_dims_to_operand_dims(
        scatter_dim_to_operand_dim);
  }
  for (int64_t input_batching_dim : input_batching_dims) {
    scatter_dim_numbers.add_input_batching_dims(input_batching_dim);
  }
  for (int64_t scatter_indices_batching_dim : scatter_indices_batching_dims) {
    scatter_dim_numbers.add_scatter_indices_batching_dims(
        scatter_indices_batching_dim);
  }
  scatter_dim_numbers.set_index_vector_dim(index_vector_dim);
  return scatter_dim_numbers;
}

HloInstructionProto HloScatterInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_scatter_dimension_numbers() = scatter_dimension_numbers();
  proto.set_indices_are_sorted(indices_are_sorted());
  proto.set_unique_indices(unique_indices());
  return proto;
}

void HloScatterInstruction::PrintExtraAttributesImpl(
    AttributePrinter &printer, const HloPrintOptions &options) const {
  printer.Next([this](Printer *printer) {
    printer->Append(
        ScatterDimensionNumbersToString(scatter_dimension_numbers()));
  });
  if (indices_are_sorted()) {
    printer.Next(
        [](Printer *printer) { printer->Append("indices_are_sorted=true"); });
  }
  if (unique_indices()) {
    printer.Next(
        [](Printer *printer) { printer->Append("unique_indices=true"); });
  }
}

bool HloScatterInstruction::IdenticalSlowPath(
    const HloInstruction &other,
    absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
        eq_computations) const {
  const auto &casted_other = static_cast<const HloScatterInstruction &>(other);
  return protobuf_util::ProtobufEquals(
             scatter_dimension_numbers(),
             casted_other.scatter_dimension_numbers()) &&
         eq_computations(to_apply(), casted_other.to_apply()) &&
         indices_are_sorted() == casted_other.indices_are_sorted() &&
         unique_indices() == casted_other.unique_indices();
}

std::unique_ptr<HloInstruction> HloScatterInstruction::CloneWithNewOperandsImpl(
    const Shape &shape, absl::Span<HloInstruction *const> new_operands,
    HloCloneContext *context) const {
  return std::make_unique<HloScatterInstruction>(
      shape, new_operands, to_apply(), scatter_dimension_numbers(),
      indices_are_sorted(), unique_indices());
}

HloIotaInstruction::HloIotaInstruction(const Shape &shape,
                                       int64_t iota_dimension)
    : HloInstruction(HloOpcode::kIota, shape),
      iota_dimension_(iota_dimension) {}

HloInstructionProto HloIotaInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.add_dimensions(iota_dimension());
  return proto;
}

void HloIotaInstruction::PrintExtraAttributesImpl(
    AttributePrinter &printer, const HloPrintOptions &options) const {
  printer.Next([this](Printer *printer) {
    AppendCat(printer, "iota_dimension=", iota_dimension());
  });
}

bool HloIotaInstruction::IdenticalSlowPath(
    const HloInstruction &other,
    absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
        eq_computations) const {
  const auto &casted_other = static_cast<const HloIotaInstruction &>(other);
  return iota_dimension() == casted_other.iota_dimension();
}

std::unique_ptr<HloInstruction> HloIotaInstruction::CloneWithNewOperandsImpl(
    const Shape &shape, absl::Span<HloInstruction *const> new_operands,
    HloCloneContext *context) const {
  return std::make_unique<HloIotaInstruction>(shape, iota_dimension());
}

HloDotInstruction::HloDotInstruction(
    const Shape &shape, HloInstruction *lhs, HloInstruction *rhs,
    const DotDimensionNumbers &dimension_numbers,
    std::vector<SparsityDescriptor> sparsity,
    absl::Span<HloInstruction *const> sparse_meta)
    : HloInstruction(HloOpcode::kDot, shape),
      dot_dimension_numbers_(dimension_numbers),
      sparsity_(std::move(sparsity)) {
  AppendOperand(lhs);
  AppendOperand(rhs);
  CHECK_LE(sparsity_.size(), kOperands);
  CHECK_EQ(sparsity_.size(), sparse_meta.size());
  for (HloInstruction *meta : sparse_meta) {
    AppendOperand(meta);
  }
  if (sparsity_.size() == kOperands &&
      sparsity_[0].index() > sparsity_[1].index()) {
    std::swap(sparsity_[0], sparsity_[1]);  // Keep descriptors ordered.
    std::swap(mutable_operands()[2], mutable_operands()[3]);
  }
}

HloInstructionProto HloDotInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_dot_dimension_numbers() = dot_dimension_numbers_;
  for (const SparsityDescriptor &descriptor : sparsity_) {
    *proto.add_dot_sparsity() = descriptor;
  }
  return proto;
}

void HloDotInstruction::PrintExtraAttributesImpl(
    AttributePrinter &printer, const HloPrintOptions &options) const {
  printer.Next([this](Printer *printer) {
    printer->Append(DotDimensionNumbersToString(dot_dimension_numbers_));
  });
  if (!sparsity_.empty()) {
    PrintSparsityDescriptor(printer, absl::MakeSpan(sparsity_));
  }
}

bool HloDotInstruction::IdenticalSlowPath(
    const HloInstruction &other,
    absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
        eq_computations) const {
  const auto &casted_other = static_cast<const HloDotInstruction &>(other);
  return protobuf_util::ProtobufEquals(dot_dimension_numbers(),
                                       casted_other.dot_dimension_numbers()) &&
         absl::c_equal(sparsity_, casted_other.sparsity_,
                       protobuf_util::ProtobufEquals);
}

std::unique_ptr<HloInstruction> HloDotInstruction::CloneWithNewOperandsImpl(
    const Shape &shape, absl::Span<HloInstruction *const> new_operands,
    HloCloneContext *context) const {
  CHECK_EQ(new_operands.size(), kOperands + sparse_operands());
  return std::make_unique<HloDotInstruction>(
      shape, new_operands[0], new_operands[1], dot_dimension_numbers_,
      sparsity_, new_operands.subspan(kOperands));
}

HloGetDimensionSizeInstruction::HloGetDimensionSizeInstruction(
    const Shape &shape, HloInstruction *operand, int64_t dimension)
    : HloInstruction(HloOpcode::kGetDimensionSize, shape),
      dimension_(dimension) {
  AppendOperand(operand);
}

HloInstructionProto HloGetDimensionSizeInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.add_dimensions(dimension());
  return proto;
}

void HloGetDimensionSizeInstruction::PrintExtraAttributesImpl(
    AttributePrinter &printer, const HloPrintOptions & /*options*/) const {
  printer.Next([this](Printer *printer) {
    AppendCat(printer, "dimensions={", dimension(), "}");
  });
}

bool HloGetDimensionSizeInstruction::IdenticalSlowPath(
    const HloInstruction &other,
    absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
    /*eq_computations*/) const {
  const auto &casted_other =
      static_cast<const HloGetDimensionSizeInstruction &>(other);
  return dimension() == casted_other.dimension();
}

std::unique_ptr<HloInstruction>
HloGetDimensionSizeInstruction::CloneWithNewOperandsImpl(
    const Shape &shape, absl::Span<HloInstruction *const> new_operands,
    HloCloneContext * /*context*/) const {
  if (new_operands.size() != 1) {
    LOG(FATAL) << "expects 1 operand";
  }
  return std::make_unique<HloGetDimensionSizeInstruction>(
      shape, new_operands[0], dimension());
}

HloSetDimensionSizeInstruction::HloSetDimensionSizeInstruction(
    const Shape &shape, HloInstruction *operand, HloInstruction *val,
    int64_t dimension)
    : HloInstruction(HloOpcode::kSetDimensionSize, shape),
      dimension_(dimension) {
  AppendOperand(operand);
  AppendOperand(val);
}

HloInstructionProto HloSetDimensionSizeInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.add_dimensions(dimension());
  return proto;
}

void HloSetDimensionSizeInstruction::PrintExtraAttributesImpl(
    AttributePrinter &printer, const HloPrintOptions & /*options*/) const {
  printer.Next([this](Printer *printer) {
    AppendCat(printer, "dimensions={", dimension(), "}");
  });
}

bool HloSetDimensionSizeInstruction::IdenticalSlowPath(
    const HloInstruction &other,
    absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
    /*eq_computations*/) const {
  const auto &casted_other =
      static_cast<const HloSetDimensionSizeInstruction &>(other);
  return dimension() == casted_other.dimension();
}

std::unique_ptr<HloInstruction>
HloSetDimensionSizeInstruction::CloneWithNewOperandsImpl(
    const Shape &shape, absl::Span<HloInstruction *const> new_operands,
    HloCloneContext * /*context*/) const {
  if (new_operands.size() != 2) {
    LOG(FATAL) << "expects 2 operand";
  }
  return std::make_unique<HloSetDimensionSizeInstruction>(
      shape, new_operands[0], new_operands[1], dimension());
}

}  // namespace zkx
