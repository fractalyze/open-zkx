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

// All HloInstruction subclasses are put in this file.

#ifndef ZKX_HLO_IR_HLO_INSTRUCTIONS_H_
#define ZKX_HLO_IR_HLO_INSTRUCTIONS_H_

#include <stdint.h>

#include <optional>
#include <string>
#include <vector>

#include "absl/log/check.h"

#include "zkx/comparison_util.h"
#include "zkx/hlo/ir/hlo_clone_context.h"
#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/literal_pool.h"
#include "zkx/shape.h"

namespace zkx {

// Base class for instructions with a dimensions vector.
class HloDimensionsInstruction : public HloInstruction {
 public:
  absl::Span<const int64_t> dimensions() const override { return dimensions_; }

  std::vector<int64_t> *mutable_dimensions() override { return &dimensions_; }

  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction *hlo) {
    switch (hlo->opcode()) {
      case HloOpcode::kBitReverse:
      case HloOpcode::kBroadcast:
      case HloOpcode::kConcatenate:
      case HloOpcode::kReduce:
      case HloOpcode::kReverse:
      case HloOpcode::kTranspose:
        return true;
      default:
        return false;
    }
  }

 protected:
  HloDimensionsInstruction(HloOpcode opcode, const Shape &shape,
                           absl::Span<const int64_t> dimensions)
      : HloInstruction(opcode, shape),
        dimensions_(dimensions.begin(), dimensions.end()) {}
  void PrintExtraAttributesImpl(AttributePrinter &printer,
                                const HloPrintOptions &options) const override;

  bool IdenticalSlowPath(
      const HloInstruction &other,
      absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
          eq_computations) const override;

  std::vector<int64_t> dimensions_;
};

class HloBroadcastInstruction : public HloDimensionsInstruction {
 public:
  explicit HloBroadcastInstruction(
      const Shape &shape, HloInstruction *operand,
      absl::Span<const int64_t> broadcast_dimension);

  static bool ClassOf(const HloInstruction *hlo) {
    return hlo->opcode() == HloOpcode::kBroadcast;
  }

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape &shape, absl::Span<HloInstruction *const> new_operands,
      HloCloneContext *context) const override;
};

class HloFftInstruction : public HloInstruction {
 public:
  explicit HloFftInstruction(const Shape &shape,
                             absl::Span<HloInstruction *const> new_operands,
                             FftType fft_type, int64_t fft_length,
                             bool fft_do_bit_reverse);
  FftType fft_type() const { return fft_type_; }

  int64_t fft_length() const { return fft_length_; }

  bool fft_do_bit_reverse() const { return fft_do_bit_reverse_; }

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction *hlo) {
    return hlo->opcode() == HloOpcode::kFft;
  }

 private:
  void PrintExtraAttributesImpl(AttributePrinter &printer,
                                const HloPrintOptions &options) const override;
  bool IdenticalSlowPath(
      const HloInstruction &other,
      absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
          eq_computations) const override;

  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape &shape, absl::Span<HloInstruction *const> new_operands,
      HloCloneContext *context) const override;

  // Describes FFT type for an FFT instruction.
  FftType fft_type_ = FftType::FFT;

  // Indicates the FFT length for an FFT instruction.
  int64_t fft_length_;

  // Indicates whether to apply bit-reverse to the FFT.
  bool fft_do_bit_reverse_ = true;
};

class HloCompareInstruction : public HloInstruction {
 public:
  explicit HloCompareInstruction(const Shape &shape, HloInstruction *lhs,
                                 HloInstruction *rhs,
                                 ComparisonDirection direction,
                                 std::optional<PrimitiveType> type);
  ComparisonDirection direction() const { return compare_.GetDirection(); }
  ComparisonOrder order() const { return compare_.GetOrder(); }
  PrimitiveType primitive_type() const { return compare_.GetPrimitiveType(); }
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction *hlo) {
    return hlo->opcode() == HloOpcode::kCompare;
  }

 private:
  void PrintExtraAttributesImpl(AttributePrinter &printer,
                                const HloPrintOptions &options) const override;
  bool IdenticalSlowPath(
      const HloInstruction &other,
      absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
          eq_computations) const override;
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape &shape, absl::Span<HloInstruction *const> new_operands,
      HloCloneContext *context) const override;

  Comparison compare_;
};

class HloReverseInstruction : public HloDimensionsInstruction {
 public:
  explicit HloReverseInstruction(const Shape &shape, HloInstruction *operand,
                                 absl::Span<const int64_t> dimensions);

  static bool ClassOf(const HloInstruction *hlo) {
    return hlo->opcode() == HloOpcode::kReverse;
  }

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape &shape, absl::Span<HloInstruction *const> new_operands,
      HloCloneContext *context) const override;
};

class HloBitReverseInstruction : public HloDimensionsInstruction {
 public:
  explicit HloBitReverseInstruction(const Shape &shape, HloInstruction *operand,
                                    absl::Span<const int64_t> dimensions);

  static bool ClassOf(const HloInstruction *hlo) {
    return hlo->opcode() == HloOpcode::kBitReverse;
  }

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape &shape, absl::Span<HloInstruction *const> new_operands,
      HloCloneContext *context) const override;
};

class HloConcatenateInstruction : public HloDimensionsInstruction {
 public:
  explicit HloConcatenateInstruction(const Shape &shape,
                                     absl::Span<HloInstruction *const> operands,
                                     int64_t dimension);
  // Accessor for the dimension in which a concatenate HLO should occur.
  int64_t concatenate_dimension() const {
    return HloInstruction::dimensions(0);
  }

  static bool ClassOf(const HloInstruction *hlo) {
    return hlo->opcode() == HloOpcode::kConcatenate;
  }

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape &shape, absl::Span<HloInstruction *const> new_operands,
      HloCloneContext *context) const override;
};

class HloReduceInstruction : public HloDimensionsInstruction {
 public:
  explicit HloReduceInstruction(const Shape &shape,
                                absl::Span<HloInstruction *const> args,
                                absl::Span<const int64_t> dimensions_to_reduce,
                                HloComputation *reduce_computation);

  // Returns the number of input arrays (and, consequently, the number of
  // init values) this reduce has.
  int64_t input_count() const { return operand_count() / 2; }

  // Returns the input tensors to be reduced.
  absl::Span<HloInstruction *const> inputs() const {
    return absl::MakeSpan(operands()).subspan(0, input_count());
  }

  // Returns the init values of the reduction.
  absl::Span<HloInstruction *const> init_values() const {
    return absl::MakeSpan(operands()).subspan(input_count(), operand_count());
  }

  static bool ClassOf(const HloInstruction *hlo) {
    return hlo->opcode() == HloOpcode::kReduce;
  }

 private:
  bool IdenticalSlowPath(
      const HloInstruction &other,
      absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape &shape, absl::Span<HloInstruction *const> new_operands,
      HloCloneContext *context) const override;
};

class HloTransposeInstruction : public HloDimensionsInstruction {
 public:
  explicit HloTransposeInstruction(const Shape &shape, HloInstruction *operand,
                                   absl::Span<const int64_t> dimensions);
  // Returns whether this instruction does a rank-2 transposition.
  bool IsRank2Transpose() const;

  static bool ClassOf(const HloInstruction *hlo) {
    return hlo->opcode() == HloOpcode::kTranspose;
  }

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape &shape, absl::Span<HloInstruction *const> new_operands,
      HloCloneContext *context) const override;
};

class HloDynamicReshapeInstruction : public HloInstruction {
 public:
  explicit HloDynamicReshapeInstruction(
      const Shape &shape, HloInstruction *data_operand,
      absl::Span<HloInstruction *const> dim_sizes);

  // Returns the input dim sizes dimensions, which is operands[1:]
  absl::Span<HloInstruction *const> dim_sizes() const {
    return absl::MakeSpan(operands()).subspan(1, operand_count());
  }

  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape &shape, absl::Span<HloInstruction *const> new_operands,
      HloCloneContext *context) const override;

  // Returns the input dim size dimension, which is operands[1+i]
  HloInstruction *dim_sizes(int64_t i) const { return operands()[i + 1]; }

  static bool ClassOf(const HloInstruction *hlo) {
    return hlo->opcode() == HloOpcode::kDynamicReshape;
  }
};

class HloReshapeInstruction : public HloInstruction {
 public:
  explicit HloReshapeInstruction(const Shape &shape, HloInstruction *operand,
                                 int64_t inferred_dimension);
  int64_t inferred_dimension() const { return inferred_dimension_; }
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction *hlo) {
    return hlo->opcode() == HloOpcode::kReshape;
  }

 private:
  void PrintExtraAttributesImpl(AttributePrinter &printer,
                                const HloPrintOptions &options) const override;
  bool IdenticalSlowPath(
      const HloInstruction &other,
      absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape &shape, absl::Span<HloInstruction *const> new_operands,
      HloCloneContext *context) const override;
  int64_t inferred_dimension_;
};

class HloMapInstruction : public HloInstruction {
 public:
  explicit HloMapInstruction(const Shape &shape,
                             absl::Span<HloInstruction *const> operands,
                             HloComputation *map_computation);
  // Returns the dimension sizes or numbers associated with this instruction.
  absl::Span<const int64_t> dimensions() const override { return dimensions_; }

  std::vector<int64_t> *mutable_dimensions() override { return &dimensions_; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction *hlo) {
    return hlo->opcode() == HloOpcode::kMap;
  }

 private:
  bool IsElementwiseImpl(
      const std::optional<int64_t> &operand_idx) const override;
  void PrintExtraAttributesImpl(AttributePrinter &printer,
                                const HloPrintOptions &options) const override;
  bool IdenticalSlowPath(
      const HloInstruction &other,
      absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape &shape, absl::Span<HloInstruction *const> new_operands,
      HloCloneContext *context) const override;

  std::vector<int64_t> dimensions_;
};

class HloSliceInstruction : public HloInstruction {
 public:
  explicit HloSliceInstruction(const Shape &shape, HloInstruction *operand,
                               absl::Span<const int64_t> start_indices,
                               absl::Span<const int64_t> limit_indices,
                               absl::Span<const int64_t> strides);

  HloInstructionProto ToProto() const override;

  // Returns the start index in the given dimension for a slice node.
  int64_t slice_starts(int64_t dimension) const {
    return slice_starts_[dimension];
  }
  const std::vector<int64_t> &slice_starts() const { return slice_starts_; }
  std::vector<int64_t> *mutable_slice_starts() { return &slice_starts_; }

  // Returns the (exclusive) limit index in the given dimension for a slice
  // node.
  int64_t slice_limits(int64_t dimension) const {
    return slice_limits_[dimension];
  }
  const std::vector<int64_t> &slice_limits() const { return slice_limits_; }
  std::vector<int64_t> *mutable_slice_limits() { return &slice_limits_; }

  // Returns the stride in the given dimension for a slice node.
  int64_t slice_strides(int64_t dimension) const {
    return slice_strides_[dimension];
  }
  const std::vector<int64_t> &slice_strides() const { return slice_strides_; }
  std::vector<int64_t> *mutable_slice_strides() { return &slice_strides_; }

  static bool ClassOf(const HloInstruction *hlo) {
    return hlo->opcode() == HloOpcode::kSlice;
  }

 private:
  void PrintExtraAttributesImpl(AttributePrinter &printer,
                                const HloPrintOptions &options) const override;
  bool IdenticalSlowPath(
      const HloInstruction &other,
      absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape &shape, absl::Span<HloInstruction *const> new_operands,
      HloCloneContext *context) const override;

  // Describes the [begin, end) index range for a slice.
  std::vector<int64_t> slice_starts_;
  std::vector<int64_t> slice_limits_;
  std::vector<int64_t> slice_strides_;
};

class HloConstantInstruction : public HloInstruction {
 public:
  explicit HloConstantInstruction(Literal literal);
  HloConstantInstruction(Literal literal, const Shape &shape);
  HloConstantInstruction(std::shared_ptr<Literal> literal, const Shape &shape);
  // Used when the literal is too large and dropped.
  explicit HloConstantInstruction(const Shape &shape);
  // Returns the literal associated with this instruction.
  const Literal &literal() const { return *literal_; }
  // Returns the (mutable) literal associated with this instruction.
  // Clone the literal if necessary (do not modify the shared instance).
  Literal *mutable_literal() {
    if (literal_.use_count() > 1) {
      literal_.reset(new Literal(literal_->Clone()));
    }
    return literal_.get();
  }
  // Returns whether there is literal associated with this instruction.
  bool HasLiteral() const { return static_cast<bool>(literal_); }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  // Changes the layout of the Constant Hlo instruction to match new_layout. For
  // tuple shaped constants shape_index is the path to the internal array
  // subshape whose layout needs to be changed.
  void RelayoutConstant(const Layout &new_layout,
                        const ShapeIndex &shape_index = {});

  static bool ClassOf(const HloInstruction *hlo) {
    return hlo->opcode() == HloOpcode::kConstant;
  }

  // Canonicalize constant literal using the given literal pool.
  bool Canonicalize(LiteralPool *literal_pool) {
    if (literal_pool && literal_) {
      auto canonical = literal_pool->GetCanonicalLiteral(literal_);
      if (canonical != literal_) {
        literal_ = std::move(canonical);
        return true;
      }
    }
    return false;
  }

  // Add literal to the hash state.
  void HashAdditionalAttributes(absl::HashState h) const override {
    if (HasLiteral()) {
      absl::HashState::combine(std::move(h),
                               Literal::AbslHashable<true>(literal()));
    }
  }

 private:
  bool IsElementwiseImpl(
      const std::optional<int64_t> &operand_idx) const override;
  bool IdenticalSlowPath(
      const HloInstruction &other,
      absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
          eq_computations) const override;
  void PrintOperandsWithCanonicalNameMap(
      Printer *printer, const HloPrintOptions &options,
      CanonicalNameMap *canonical_name_map) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape &shape, absl::Span<HloInstruction *const> new_operands,
      HloCloneContext *context) const override;
  std::shared_ptr<Literal> literal_;
};

// Abstract class that represents an HLO instruction that "calls" a computation.
// Fusion and Call HLOs inherit from this class.
class HloCallableInstruction : public HloInstruction {
 public:
  HloCallableInstruction(HloOpcode opcode, const Shape &shape);

  HloCallableInstruction(HloOpcode opcode, const Shape &shape,
                         absl::Span<HloInstruction *const> operands);

  HloCallableInstruction(HloOpcode opcode, const Shape &shape,
                         absl::Span<HloInstruction *const> operands,
                         HloComputation *called_computation,
                         std::string_view prefix = "");

  HloCallableInstruction(HloOpcode opcode, const Shape &shape,
                         absl::Span<HloInstruction *const> operands,
                         absl::Span<HloComputation *const> called_computations);

  HloCallableInstruction(HloOpcode opcode, const Shape &shape,
                         const std::string &name, const std::string &attributes,
                         int64_t version);

  HloCallableInstruction(HloOpcode opcode, const Shape &shape,
                         absl::Span<HloInstruction *const> operands,
                         HloComputation *decomposition, const std::string &name,
                         const std::string &attributes, int64_t version);

  ~HloCallableInstruction() override;

  // Adds a new operand to the callable instruction.
  HloInstruction *AddCallOperand(HloInstruction *new_operand);

  // Appends (fuses) the given instruction into this callable instruction.
  // instruction_to_append is cloned and the clone is placed in the callable
  // instruction.  The users of instruction_to_append will be redirected to this
  // callable instruction. instruction_to_append is unchanged otherwise. When
  // add_output is true, a clone of the instruction_to_append will be added as
  // additional output resulting in a multi-output callable instruction.
  HloInstruction *AppendInstructionIntoCalledComputation(
      HloInstruction *instruction_to_append, bool add_output = false);
  // Clones the given instruction_to_append and inserts the clone into this
  // callable instruction. If add_output is true, a clone of
  // instruction_to_append will be in the output of the this callable
  // instruction (part of the tuple of the callable root).
  HloInstruction *CloneAndAppendInstructionIntoCalledComputation(
      HloInstruction *instruction_to_append, bool add_output = false);

  // Retrieves the called computations of an HloCallableInstruction that is
  // being cloned. If the called computations have not yet been cloned, then
  // they are first cloned and added to the context.
  absl::InlinedVector<HloComputation *, 1> GetOrCloneCalledComputations(
      HloCloneContext *context) const;

  HloComputation *called_computation() const;

  HloInstruction *called_computation_root() const;

  // Recursively sets all nested called computation to have thread name as
  // `execution_thread`. if `skip_async_execution_thread_overwrite` is true,
  // skip overwrite async instruction and its computations thread name
  // overwriting.
  void RecursivelySetComputationsThreadName(
      std::string_view execution_thread,
      bool skip_async_execution_thread_overwrite);

  static bool ClassOf(const HloInstruction *hlo) {
    return hlo->opcode() == HloOpcode::kFusion ||
           hlo->opcode() == HloOpcode::kCall;
  }

  // Gets a list of output/operand buffer pairs that alias each other, where the
  // output buffer is represented as a ShapeIndex, and the operand buffer is
  // represented as the operand index and the ShapeIndex. By default this list
  // is empty.
  const std::vector<std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>> &
  output_to_operand_aliasing() const {
    return output_to_operand_aliasing_;
  }
  // Sets the list of output/operand buffer pairs that alias each other.
  void set_output_to_operand_aliasing(
      std::vector<std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
          aliasing) {
    output_to_operand_aliasing_ = std::move(aliasing);
  }

  FrontendAttributes BuildFrontendAttributesForComposite(
      const std::string &name,
      std::optional<std::string_view> attributes = std::nullopt,
      std::optional<int64_t> version = std::nullopt) {
    FrontendAttributes frontend_attributes;
    frontend_attributes.mutable_map()->insert({"composite.name", name});
    frontend_attributes.mutable_map()->insert(
        {"composite.attributes",
         attributes.has_value() ? std::string(*attributes) : "{}"});
    frontend_attributes.mutable_map()->insert(
        {"composite.version",
         version.has_value() ? std::to_string(*version) : "0"});
    return frontend_attributes;
  }

 protected:
  // Returns the default called computation name.
  virtual std::string default_called_computation_name() const = 0;

 private:
  // A list of output/operand buffer pairs that alias each other. See comment of
  // output_to_operand_aliasing().
  std::vector<std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
      output_to_operand_aliasing_;
};

class HloFusionInstruction : public HloCallableInstruction {
 public:
  explicit HloFusionInstruction(const Shape &shape, FusionKind fusion_kind,
                                HloInstruction *fused_root,
                                std::string_view prefix = "");

  explicit HloFusionInstruction(const Shape &shape, FusionKind fusion_kind,
                                absl::Span<HloInstruction *const> operands,
                                HloComputation *fusion_computation,
                                std::string_view prefix = "");

  ~HloFusionInstruction() override;

  void ClearCalledComputations() override;

  // When a fusion instruction is being destructed, clear the back pointer of
  // its fusion computation, to avoid referencing freed memory.
  void ClearFusionComputationInstruction();

  // Clones the given instruction_to_append and inserts the clone into this
  // callable instruction.
  HloInstruction *CloneAndAppendInstructionIntoCalledComputation(
      HloInstruction *instruction_to_append, bool add_output = false);

  std::string ToCategory() const override;
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  // Adds a new operand the fusion instruction.
  HloInstruction *AddFusionOperand(HloInstruction *new_operand);

  // Merges the fused instructions from 'instruction_to_merge' into the
  // fused instruction set of 'this', updating operands as necessary.
  //
  // Precondition: 'instruction_to_merge' must be an operand of 'this'.
  void MergeFusionInstruction(HloFusionInstruction *instruction_to_merge);

  // Merges the fused instructions from instruction_to_merge into the fused
  // instruction set of 'this' and generates multi-output fusion instructions.
  // All the users of instruction_to_merge will be redirected to 'this'
  // instruction. instruction_to_merge will be removed from its parent
  // computation.
  void MergeFusionInstructionIntoMultiOutput(
      HloFusionInstruction *instruction_to_merge);

  // Fuses the given instruction in this fusion instruction. instruction_to_fuse
  // is cloned and the clone is placed in the fusion
  // instruction. instruction_to_fuse is unchanged. Instruction is cloned rather
  // than moved to cleanly handle the case where the instruction has a use
  // outside the fusion instruction. Moving such an instruction into a fusion
  // instruction would violate the single-result invariant of HLO instructions
  // and significantly complicate code generation.
  HloInstruction *FuseInstruction(HloInstruction *instruction_to_fuse) {
    CHECK(instruction_to_fuse->IsFusible()) << instruction_to_fuse->ToString();
    return AppendInstructionIntoCalledComputation(instruction_to_fuse);
  }

  // Fuses the given instruction in this fusion instruction and generates a
  // multioutput fusion instruction. A clone of the instruction_to_fuse will
  // be part of the output of fusion instructions. The users of
  // instruction_to_fuse will be redirected to this fusion instructions.
  // instruction_to_fuse is unchanged otherwise.
  HloInstruction *FuseInstructionIntoMultiOutput(
      HloInstruction *instruction_to_fuse) {
    return AppendInstructionIntoCalledComputation(instruction_to_fuse,
                                                  /*add_output=*/true);
  }

  // Returns the computation for this fused instruction.
  HloComputation *fused_instructions_computation() const;

  // Returns the root instruction of the fused expression contained within this
  // fusion instruction.
  HloInstruction *fused_expression_root() const;

  // Returns the list of fused instructions inside this fusion instruction.  The
  // returned type is a range of HloInstruction*s.
  tsl::gtl::iterator_range<HloInstructionUnwrappingConstIterator>
  fused_instructions() const;

  tsl::gtl::iterator_range<HloInstructionUnwrappingIterator>
  fused_instructions();

  // Gets the number of instructions inside this fusion instruction.
  int64_t fused_instruction_count() const;

  // Returns the fused parameter instruction in this fusion instruction
  // corresponding to the given parameter number.
  HloInstruction *fused_parameter(int64_t parameter_number) const;

  // Returns the vector of fused parameters inside this fusion instruction.
  const HloInstruction::InstructionVector &fused_parameters() const;

  // Returns true if this instruction is a fusion instruction that generates
  // multiple outputs.
  bool IsMultiOutputFusion() const {
    return fused_expression_root()->opcode() == HloOpcode::kTuple;
  }

  FusionKind fusion_kind() const { return fusion_kind_; }

  void set_fusion_kind(FusionKind kind) { fusion_kind_ = kind; }

  // If multiple operands are the same instruction, keeps only one of them.
  absl::Status DeduplicateFusionOperands();

  static bool ClassOf(const HloInstruction *hlo) {
    return hlo->opcode() == HloOpcode::kFusion;
  }

  // Add various fusion parameters to the hash.
  void HashAdditionalAttributes(absl::HashState h) const override {
    absl::HashState::combine(std::move(h), *fused_expression_root(),
                             fusion_kind(), fused_instruction_count(),
                             fused_parameters().size());
  }

 protected:
  std::string default_called_computation_name() const override {
    return "fused_computation";
  }

 private:
  bool IsElementwiseImpl(
      const std::optional<int64_t> &operand_idx) const override;
  void PrintExtraAttributesImpl(AttributePrinter &printer,
                                const HloPrintOptions &options) const override;
  bool IdenticalSlowPath(
      const HloInstruction &other,
      absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
          eq_computations) const override;

  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape &shape, absl::Span<HloInstruction *const> new_operands,
      HloCloneContext *context) const override;

  // The type of the fusion.
  FusionKind fusion_kind_;
};

class HloCallInstruction : public HloCallableInstruction {
 public:
  HloCallInstruction(const Shape &shape,
                     HloInstruction *called_computation_root);

  HloCallInstruction(const Shape &shape,
                     absl::Span<HloInstruction *const> operands,
                     HloComputation *called_computation);

  HloCallInstruction(const Shape &shape, HloInstruction *decomposition_root,
                     const std::string &name, const std::string &attributes,
                     int64_t version);

  HloCallInstruction(const Shape &shape,
                     absl::Span<HloInstruction *const> operands,
                     HloComputation *decomposition, const std::string &name,
                     const std::string &attributes, int64_t version);

  static bool ClassOf(const HloInstruction *hlo) {
    return hlo->opcode() == HloOpcode::kCall;
  }

 protected:
  std::string default_called_computation_name() const override {
    return "called_computation";
  }
};

class HloParameterInstruction : public HloInstruction {
 public:
  explicit HloParameterInstruction(int64_t parameter_number, const Shape &shape,
                                   std::string_view name);
  int64_t parameter_number() const { return parameter_number_; }

  // Sets and gets the whether all replicas will receive the same parameter data
  // for each leaf buffer in data parallelism.
  void set_parameter_replicated_at_leaf_buffers(
      absl::Span<const bool> parameter_replicated_at_leaf_buffers) {
    CHECK_EQ(ShapeUtil::GetLeafCount(shape()),
             parameter_replicated_at_leaf_buffers.size());
    parameter_replicated_at_leaf_buffers_.emplace(
        parameter_replicated_at_leaf_buffers.begin(),
        parameter_replicated_at_leaf_buffers.end());
  }
  void set_parameter_replicated_at_leaf_buffers(
      const std::vector<bool> &parameter_replicated_at_leaf_buffers) {
    CHECK_EQ(ShapeUtil::GetLeafCount(shape()),
             parameter_replicated_at_leaf_buffers.size());
    parameter_replicated_at_leaf_buffers_ =
        parameter_replicated_at_leaf_buffers;
  }
  const std::optional<std::vector<bool>> &parameter_replicated_at_leaf_buffers()
      const {
    return parameter_replicated_at_leaf_buffers_;
  }

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction *hlo) {
    return hlo->opcode() == HloOpcode::kParameter;
  }

  // Add parameter number to the hash.
  void HashAdditionalAttributes(absl::HashState h) const override {
    absl::HashState::combine(std::move(h), parameter_number());
  }

 private:
  void PrintExtraAttributesImpl(AttributePrinter &printer,
                                const HloPrintOptions &options) const override;
  bool IdenticalSlowPath(
      const HloInstruction &other,
      absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
          eq_computations) const override;
  void PrintOperandsWithCanonicalNameMap(
      Printer *printer, const HloPrintOptions &options,
      CanonicalNameMap *canonical_name_map) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape &shape, absl::Span<HloInstruction *const> new_operands,
      HloCloneContext *context) const override;

  int64_t parameter_number_ = 0;

  // Specifies whether each buffer has the same parameter value on all replicas
  // in data parallelism.
  std::optional<std::vector<bool>> parameter_replicated_at_leaf_buffers_;
};

class HloGetTupleElementInstruction : public HloInstruction {
 public:
  explicit HloGetTupleElementInstruction(const Shape &shape,
                                         HloInstruction *operand,
                                         int64_t index);
  // Returns the tuple index associated with this instruction.
  int64_t tuple_index() const { return tuple_index_; }
  // Sets the tuple index associated with this instruction.
  void set_tuple_index(int64_t new_tuple_index) {
    tuple_index_ = new_tuple_index;
  }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction *hlo) {
    return hlo->opcode() == HloOpcode::kGetTupleElement;
  }

 private:
  void PrintExtraAttributesImpl(AttributePrinter &printer,
                                const HloPrintOptions &options) const override;
  bool IdenticalSlowPath(
      const HloInstruction &other,
      absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape &shape, absl::Span<HloInstruction *const> new_operands,
      HloCloneContext *context) const override;

  int64_t tuple_index_ = -1;
};

class HloReduceWindowInstruction : public HloInstruction {
 public:
  explicit HloReduceWindowInstruction(const Shape &shape,
                                      HloInstruction *operand,
                                      HloInstruction *init_value,
                                      const Window &window,
                                      HloComputation *reduce_computation);
  explicit HloReduceWindowInstruction(
      const Shape &shape, absl::Span<HloInstruction *const> operands,
      absl::Span<HloInstruction *const> init_values, const Window &window,
      HloComputation *reduce_computation);
  const Window &window() const override { return window_; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;
  // Returns the number of input arrays (and, consequently, the number of
  // init values) this reduce has.
  int64_t input_count() const { return operand_count() / 2; }
  // Returns the input tensors to be reduced.
  absl::Span<HloInstruction *const> inputs() const {
    return absl::MakeSpan(operands()).subspan(0, input_count());
  }
  // Returns the init values of the reduction.
  absl::Span<HloInstruction *const> init_values() const {
    return absl::MakeSpan(operands()).subspan(input_count(), operand_count());
  }
  // Returns the shapes of input tensors to be reduced.
  absl::InlinedVector<const Shape *, 2> input_shapes() const {
    absl::InlinedVector<const Shape *, 2> shapes;
    for (const auto *op : inputs()) {
      shapes.push_back(&op->shape());
    }
    return shapes;
  }
  // Returns the init values of the reduction.
  absl::InlinedVector<const Shape *, 2> init_value_shapes() const {
    absl::InlinedVector<const Shape *, 2> shapes;
    for (const auto *op : init_values()) {
      shapes.push_back(&op->shape());
    }
    return shapes;
  }

  static bool ClassOf(const HloInstruction *hlo) {
    return hlo->opcode() == HloOpcode::kReduceWindow;
  }

 private:
  void PrintExtraAttributesImpl(AttributePrinter &printer,
                                const HloPrintOptions &options) const override;
  bool IdenticalSlowPath(
      const HloInstruction &other,
      absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape &shape, absl::Span<HloInstruction *const> new_operands,
      HloCloneContext *context) const override;

  Window window_;
};

class HloPadInstruction : public HloInstruction {
 public:
  explicit HloPadInstruction(const Shape &shape, HloInstruction *operand,
                             HloInstruction *padding_value,
                             const PaddingConfig &padding_config);
  // Returns the padding configuration for a pad node.
  const PaddingConfig &padding_config() const { return padding_config_; }
  PaddingConfig *mutable_padding_config() { return &padding_config_; }
  // Returns the operand being padded.
  const HloInstruction *padded_operand() const { return operand(0); }
  HloInstruction *mutable_padded_operand() { return mutable_operand(0); }
  // Returns the padding value.
  const HloInstruction *padding_value() const { return operand(1); }
  HloInstruction *mutable_padding_value() { return mutable_operand(1); }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction *hlo) {
    return hlo->opcode() == HloOpcode::kPad;
  }

 private:
  void PrintExtraAttributesImpl(AttributePrinter &printer,
                                const HloPrintOptions &options) const override;
  bool IdenticalSlowPath(
      const HloInstruction &other,
      absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape &shape, absl::Span<HloInstruction *const> new_operands,
      HloCloneContext *context) const override;

  // The padding configuration that describes the edge padding of this pad
  // instruction.
  PaddingConfig padding_config_;
};

class HloDynamicIndexInstruction : public HloInstruction {
 public:
  explicit HloDynamicIndexInstruction(HloOpcode opcode, const Shape &shape)
      : HloInstruction(opcode, shape) {}
  virtual int64_t first_index_operand_number() const = 0;

  // Returns a subspan of operands which represent the start indices.
  absl::Span<HloInstruction *const> index_operands() const {
    return absl::MakeSpan(operands()).subspan(first_index_operand_number());
  }

  // Returns the shapes of the index operands.
  std::vector<Shape> index_shapes() const {
    std::vector<Shape> shapes;
    auto indices = index_operands();
    for (const HloInstruction *index : indices) {
      shapes.push_back(index->shape());
    }
    return shapes;
  }

  static bool ClassOf(const HloInstruction *hlo) {
    return hlo->opcode() == HloOpcode::kDynamicSlice ||
           hlo->opcode() == HloOpcode::kDynamicUpdateSlice;
  }
};

class HloDynamicSliceInstruction : public HloDynamicIndexInstruction {
 public:
  explicit HloDynamicSliceInstruction(const Shape &shape,
                                      HloInstruction *operand,
                                      HloInstruction *start_indices,
                                      absl::Span<const int64_t> slice_sizes);
  explicit HloDynamicSliceInstruction(
      const Shape &shape, HloInstruction *operand,
      absl::Span<HloInstruction *const> start_indices,
      absl::Span<const int64_t> slice_sizes);
  // Old methods kept for smooth subclassing transition END.
  // Returns the size of the slice in the given dimension for a dynamic
  // slice node.
  int64_t slice_sizes(int64_t dimension) const {
    return dynamic_slice_sizes_[dimension];
  }
  const std::vector<int64_t> &dynamic_slice_sizes() const {
    return dynamic_slice_sizes_;
  }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  int64_t first_index_operand_number() const override { return 1; }
  static bool ClassOf(const HloInstruction *hlo) {
    return hlo->opcode() == HloOpcode::kDynamicSlice;
  }

 private:
  void PrintExtraAttributesImpl(AttributePrinter &printer,
                                const HloPrintOptions &options) const override;
  bool IdenticalSlowPath(
      const HloInstruction &other,
      absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape &shape, absl::Span<HloInstruction *const> new_operands,
      HloCloneContext *context) const override;

  // Describes the [start, start + size) range size for a dynamic slice
  // ('start' is specified dynamically in the second operand of the operation).
  std::vector<int64_t> dynamic_slice_sizes_;
};

class HloDynamicUpdateSliceInstruction : public HloDynamicIndexInstruction {
 public:
  explicit HloDynamicUpdateSliceInstruction(const Shape &shape,
                                            HloInstruction *operand,
                                            HloInstruction *update,
                                            HloInstruction *start_indices);
  explicit HloDynamicUpdateSliceInstruction(
      const Shape &shape, HloInstruction *operand, HloInstruction *update,
      absl::Span<HloInstruction *const> start_indices);

  int64_t first_index_operand_number() const override { return 2; }

  const HloInstruction *update() const { return operand(1); }

  static bool ClassOf(const HloInstruction *hlo) {
    return hlo->opcode() == HloOpcode::kDynamicUpdateSlice;
  }
};

class HloGatherInstruction : public HloInstruction {
 public:
  explicit HloGatherInstruction(
      const Shape &shape, HloInstruction *operand,
      HloInstruction *start_indices,
      const GatherDimensionNumbers &gather_dim_numbers,
      absl::Span<const int64_t> slice_sizes, bool indices_are_sorted);
  const GatherDimensionNumbers &gather_dimension_numbers() const {
    CHECK_NE(gather_dimension_numbers_, nullptr);
    return *gather_dimension_numbers_;
  }
  absl::Span<const int64_t> gather_slice_sizes() const {
    return gather_slice_sizes_;
  }
  bool indices_are_sorted() const { return indices_are_sorted_; }
  void set_indices_are_sorted(bool indices_are_sorted) {
    indices_are_sorted_ = indices_are_sorted;
  }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  // Creates an instance of GatherDimensionNumbers.
  static GatherDimensionNumbers MakeGatherDimNumbers(
      absl::Span<const int64_t> offset_dims,
      absl::Span<const int64_t> collapsed_slice_dims,
      absl::Span<const int64_t> start_index_map, int64_t index_vector_dim,
      absl::Span<const int64_t> operand_batching_dims = {},
      absl::Span<const int64_t> start_indices_batching_dims = {});
  // Returns the dump string of the given gather dimension numbers.
  static std::string GatherDimensionNumbersToString(
      const GatherDimensionNumbers &dim_numbers);
  // Prints the dump string of the given gather dimension numbers.
  static void PrintGatherDimensionNumbers(
      Printer *printer, const GatherDimensionNumbers &dim_numbers);

  static bool ClassOf(const HloInstruction *hlo) {
    return hlo->opcode() == HloOpcode::kGather;
  }

 private:
  void PrintExtraAttributesImpl(AttributePrinter &printer,
                                const HloPrintOptions &options) const override;
  bool IdenticalSlowPath(
      const HloInstruction &other,
      absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
          eq_computations) const override;
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape &shape, absl::Span<HloInstruction *const> new_operands,
      HloCloneContext *context) const override;

  std::unique_ptr<GatherDimensionNumbers> gather_dimension_numbers_;
  std::vector<int64_t> gather_slice_sizes_;
  bool indices_are_sorted_;
};

class HloScatterInstruction : public HloInstruction {
 public:
  explicit HloScatterInstruction(
      const Shape &shape, absl::Span<HloInstruction *const> args,
      HloComputation *update_computation,
      const ScatterDimensionNumbers &scatter_dim_numbers,
      bool indices_are_sorted, bool unique_indices);
  const ScatterDimensionNumbers &scatter_dimension_numbers() const {
    CHECK_NE(scatter_dimension_numbers_, nullptr);
    return *scatter_dimension_numbers_;
  }
  bool indices_are_sorted() const { return indices_are_sorted_; }
  void set_indices_are_sorted(bool indices_are_sorted) {
    indices_are_sorted_ = indices_are_sorted;
  }
  bool unique_indices() const override { return unique_indices_; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;
  int64_t scatter_operand_count() const { return operand_count() / 2; }
  absl::Span<HloInstruction *const> scatter_operands() const {
    return absl::MakeConstSpan(operands()).first(scatter_operand_count());
  }
  absl::Span<HloInstruction *const> scatter_updates() const {
    return absl::MakeConstSpan(operands()).last(scatter_operand_count());
  }
  const HloInstruction *scatter_indices() const {
    return operand(scatter_operand_count());
  }
  HloInstruction *scatter_indices() {
    return mutable_operand(scatter_operand_count());
  }

  // Creates an instance of ScatterDimensionNumbers.
  static ScatterDimensionNumbers MakeScatterDimNumbers(
      absl::Span<const int64_t> update_window_dims,
      absl::Span<const int64_t> inserted_window_dims,
      absl::Span<const int64_t> scatter_dims_to_operand_dims,
      int64_t index_vector_dim,
      absl::Span<const int64_t> input_batching_dims = {},
      absl::Span<const int64_t> scatter_indices_batching_dims = {});
  // Returns the dump string of the given scatter dimension numbers.
  static std::string ScatterDimensionNumbersToString(
      const ScatterDimensionNumbers &dim_numbers);
  // Prints the dump string of the given scatter dimension numbers.
  static void PrintScatterDimensionNumbers(
      Printer *printer, const ScatterDimensionNumbers &dim_numbers);

  static bool ClassOf(const HloInstruction *hlo) {
    return hlo->opcode() == HloOpcode::kScatter;
  }

 private:
  void PrintExtraAttributesImpl(AttributePrinter &printer,
                                const HloPrintOptions &options) const override;
  bool IdenticalSlowPath(
      const HloInstruction &other,
      absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape &shape, absl::Span<HloInstruction *const> new_operands,
      HloCloneContext *context) const override;

  std::unique_ptr<ScatterDimensionNumbers> scatter_dimension_numbers_;
  bool indices_are_sorted_;
  bool unique_indices_;
};

class HloIotaInstruction : public HloInstruction {
 public:
  explicit HloIotaInstruction(const Shape &shape, int64_t iota_dimension);

  // Returns the dimension sizes or numbers associated with this instruction.
  int64_t iota_dimension() const { return iota_dimension_; }
  absl::Span<const int64_t> dimensions() const override {
    return absl::MakeConstSpan(&iota_dimension_, 1);
  }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction *hlo) {
    return hlo->opcode() == HloOpcode::kIota;
  }

 private:
  void PrintExtraAttributesImpl(AttributePrinter &printer,
                                const HloPrintOptions &options) const override;
  bool IdenticalSlowPath(
      const HloInstruction &other,
      absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape &shape, absl::Span<HloInstruction *const> new_operands,
      HloCloneContext *context) const override;

  int64_t iota_dimension_;
};

class HloDotInstruction : public HloInstruction {
 public:
  static const int kOperands = 2;

  // Creates a dot op with operands `lhs` and `rhs` with contracting and batch
  // dimensions specified in `dimension_numbers`. If `sparsity` is set, then
  // `sparse_meta` must also be present (and have the same size).
  explicit HloDotInstruction(
      const Shape &shape, HloInstruction *lhs, HloInstruction *rhs,
      const DotDimensionNumbers &dimension_numbers,
      std::vector<SparsityDescriptor> sparsity = {},
      absl::Span<HloInstruction *const> sparse_meta = {});

  // Returns data on the dimension numbers used for a dot operation.
  const DotDimensionNumbers &dot_dimension_numbers() const {
    return dot_dimension_numbers_;
  }

  // Sets dimension numbers used for a dot operation.
  DotDimensionNumbers *mutable_dot_dimension_numbers() {
    return &dot_dimension_numbers_;
  }

  // Sparsity descriptors are optional. If present, additional operands define
  // how the data is read for the dot inputs.
  int sparse_operands() const { return sparsity_.size(); }
  absl::Span<const SparsityDescriptor> sparsity() const {
    return absl::MakeSpan(sparsity_);
  }

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction *hlo) {
    return hlo->opcode() == HloOpcode::kDot;
  }

 private:
  void PrintExtraAttributesImpl(AttributePrinter &printer,
                                const HloPrintOptions &options) const override;
  bool IdenticalSlowPath(
      const HloInstruction &other,
      absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape &shape, absl::Span<HloInstruction *const> new_operands,
      HloCloneContext *context) const override;

  // Describes the dimension numbers used for a dot.
  DotDimensionNumbers dot_dimension_numbers_;

  // Sparsity descriptors are set if some operands are sparse. In this case, the
  // additional metadata operands contain the information that defines how
  // the data is read.
  std::vector<SparsityDescriptor> sparsity_;
};

class HloGetDimensionSizeInstruction : public HloInstruction {
 public:
  explicit HloGetDimensionSizeInstruction(const Shape &shape,
                                          HloInstruction *operand,
                                          int64_t dimension);

  // Returns the dimension sizes or numbers associated with this instruction.
  int64_t dimension() const { return dimension_; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction *hlo) {
    return hlo->opcode() == HloOpcode::kGetDimensionSize;
  }

 private:
  void PrintExtraAttributesImpl(AttributePrinter &printer,
                                const HloPrintOptions &options) const override;
  bool IdenticalSlowPath(
      const HloInstruction &other,
      absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape &shape, absl::Span<HloInstruction *const> new_operands,
      HloCloneContext *context) const override;

  int64_t dimension_;
};

class HloSetDimensionSizeInstruction : public HloInstruction {
 public:
  explicit HloSetDimensionSizeInstruction(const Shape &shape,
                                          HloInstruction *operand,
                                          HloInstruction *val,
                                          int64_t dimension);

  // Returns the dimension sizes or numbers associated with this instruction.
  int64_t dimension() const { return dimension_; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction *hlo) {
    return hlo->opcode() == HloOpcode::kSetDimensionSize;
  }

 private:
  void PrintExtraAttributesImpl(AttributePrinter &printer,
                                const HloPrintOptions &options) const override;
  bool IdenticalSlowPath(
      const HloInstruction &other,
      absl::FunctionRef<bool(const HloComputation *, const HloComputation *)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape &shape, absl::Span<HloInstruction *const> new_operands,
      HloCloneContext *context) const override;

  int64_t dimension_;
};

}  // namespace zkx

#endif  // ZKX_HLO_IR_HLO_INSTRUCTIONS_H_
