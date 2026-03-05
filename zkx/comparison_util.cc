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

#include "zkx/comparison_util.h"

#include "absl/container/flat_hash_map.h"
#include "absl/debugging/leak_check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"

#include "zkx/base/random.h"

namespace zkx {
namespace {

// Verifies that this is a valid Comparison: (1) not a partial ordering on
// integers, and (2) a valid PrimitiveType.
bool IsValidComparison(PrimitiveType type, Comparison::Order order) {
  if (primitive_util::IsIntegralType(type) ||
      primitive_util::IsFieldType(type) ||
      primitive_util::IsEcPointType(type) || type == PRED) {
    return order == Comparison::Order::kTotal;
  }
  LOG(FATAL) << "Unsupported type: " << PrimitiveType_Name(type);
}

// Returns the expected ordering for each primitive type.
Comparison::Order DefaultOrdering(PrimitiveType type) {
  if (primitive_util::IsIntegralType(type) ||
      primitive_util::IsFieldType(type) ||
      primitive_util::IsEcPointType(type) || type == PRED) {
    return Comparison::Order::kTotal;
  }
  LOG(FATAL) << "Unsupported type: " << PrimitiveType_Name(type);
}

// Returns the converse of `direction`.
Comparison::Direction Converse(Comparison::Direction direction) {
  switch (direction) {
    case Comparison::Direction::kEq:
      return Comparison::Direction::kEq;
    case Comparison::Direction::kNe:
      return Comparison::Direction::kNe;
    case Comparison::Direction::kGe:
      return Comparison::Direction::kLe;
    case Comparison::Direction::kGt:
      return Comparison::Direction::kLt;
    case Comparison::Direction::kLe:
      return Comparison::Direction::kGe;
    case Comparison::Direction::kLt:
      return Comparison::Direction::kGt;
  }
}

// Returns the inverse of `direction`.
Comparison::Direction Inverse(Comparison::Direction direction) {
  switch (direction) {
    case Comparison::Direction::kEq:
      return Comparison::Direction::kNe;
    case Comparison::Direction::kNe:
      return Comparison::Direction::kEq;
    case Comparison::Direction::kGe:
      return Comparison::Direction::kLt;
    case Comparison::Direction::kGt:
      return Comparison::Direction::kLe;
    case Comparison::Direction::kLe:
      return Comparison::Direction::kGt;
    case Comparison::Direction::kLt:
      return Comparison::Direction::kGe;
  }
}

}  // namespace

std::string ComparisonDirectionToString(Comparison::Direction direction) {
  switch (direction) {
    case Comparison::Direction::kEq:
      return "EQ";
    case Comparison::Direction::kNe:
      return "NE";
    case Comparison::Direction::kGe:
      return "GE";
    case Comparison::Direction::kGt:
      return "GT";
    case Comparison::Direction::kLe:
      return "LE";
    case Comparison::Direction::kLt:
      return "LT";
    default:
      LOG(FATAL) << "Attempted to print uninitialized comparison direction";
  }
}

std::string_view ComparisonPrimitiveTypeToString(PrimitiveType type) {
  return PrimitiveType_Name(type);
}

std::string_view ComparisonOrderToString(Comparison::Order order) {
  switch (order) {
    case Comparison::Order::kPartial:
      return "PARTIALORDER";
    case Comparison::Order::kTotal:
      return "TOTALORDER";
  }
}

absl::StatusOr<Comparison::Direction> StringToComparisonDirection(
    std::string_view direction) {
  static auto* map = absl::IgnoreLeak(
      new absl::flat_hash_map<std::string, Comparison::Direction>({
          {"EQ", Comparison::Direction::kEq},
          {"NE", Comparison::Direction::kNe},
          {"GE", Comparison::Direction::kGe},
          {"GT", Comparison::Direction::kGt},
          {"LE", Comparison::Direction::kLe},
          {"LT", Comparison::Direction::kLt},
      }));
  auto it = map->find(direction);
  if (it == map->end()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Unknown comparison direction: %s", direction));
  }
  return it->second;
}

absl::StatusOr<Comparison::Order> StringToComparisonOrder(
    std::string_view order) {
  static auto* map =
      absl::IgnoreLeak(new absl::flat_hash_map<std::string, Comparison::Order>({
          {"PARTIALORDER", Comparison::Order::kPartial},
          {"TOTALORDER", Comparison::Order::kTotal},
      }));
  auto it = map->find(order);
  if (it == map->end()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Unknown comparison order: %s", order));
  }
  return it->second;
}

Comparison::Comparison(Direction dir, PrimitiveType type, Order order)
    : dir_(dir), primitive_type_(type), order_(order) {
  CHECK(IsValidComparison(primitive_type_, order_));
}

Comparison::Comparison(Direction dir, PrimitiveType type)
    : dir_(dir), primitive_type_(type), order_(DefaultOrdering(type)) {
  CHECK(IsValidComparison(primitive_type_, order_));
}

Comparison Comparison::Converse() const {
  return Comparison(zkx::Converse(dir_), primitive_type_, order_);
}

std::optional<Comparison> Comparison::Inverse() const {
  if (IsPartialOrder()) {
    // We assume comparisons don't have inverses unless they are total order,
    // e.g., a partial order floating point comparison can return true if one
    // operand is NaN.
    return std::nullopt;
  }
  if (primitive_util::IsArrayType(primitive_type_)) {
    return Comparison(zkx::Inverse(dir_), primitive_type_, order_);
  }
  return std::nullopt;
}

bool Comparison::IsReflexive() const {
  switch (dir_) {
    case Direction::kEq:
    case Direction::kGe:
    case Direction::kLe:
      return IsTotalOrder();
    case Direction::kNe:
    case Direction::kGt:
    case Direction::kLt:
      return false;
  }
}

bool Comparison::IsAntireflexive() const {
  switch (dir_) {
    case Direction::kNe:
      return IsTotalOrder();
    case Direction::kGt:
    case Direction::kLt:
      return true;
    case Direction::kEq:
    case Direction::kGe:
    case Direction::kLe:
      return false;
  }
}

std::string Comparison::ToString(std::string prefix1, std::string prefix2,
                                 std::string prefix3) const {
  return absl::StrCat(prefix1, ComparisonDirectionToString(dir_), prefix2,
                      ComparisonPrimitiveTypeToString(primitive_type_), prefix3,
                      ComparisonOrderToString(order_));
}

ComparisonDirection RandomComparisonDirection() {
  return static_cast<ComparisonDirection>(base::Uniform<uint8_t>() % 6);
}

ComparisonDirection RandomEqualityComparisonDirection() {
  return static_cast<ComparisonDirection>(base::Uniform<uint8_t>() % 2);
}

}  // namespace zkx
