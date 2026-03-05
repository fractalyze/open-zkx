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

#ifndef ZKX_COMPARISON_UTIL_H_
#define ZKX_COMPARISON_UTIL_H_

#include <stdint.h>

#include <functional>
#include <optional>
#include <ostream>
#include <string>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"

#include "zkx/primitive_util.h"
#include "zkx/zkx_data.pb.h"

namespace zkx {

// A utility class for primitive comparisons. A comparison includes three
// components: the type of the elements being compared (F32, S16, etc), whether
// it is a partial or total order comparison, and the actual comparison operator
// (==, <=, >, etc).
//
// Note that integer comparisons are always total order. Float comparisons can
// be either total or partial order.
//
// Some examples:
//
//   Comparison a(
//     Comparison::Direction::kLt,
//     zkx::PrimitiveType::BF16,
//     Comparison::Order::kTotal
//   );
//   a.ToString(); /* ".LT.BF16.TOTALORDER" */
//
//   Comparison b(Comparison::Direction::kEq, zkx::PrimitiveType::U32);
//   b.IsTotalOrder(); /* true */
class Comparison {
 public:
  // Represents the ordering of the comparison.
  enum class Order : uint8_t {
    // https://en.wikipedia.org/wiki/Total_order
    kTotal,
    // https://en.wikipedia.org/wiki/Partially_ordered_set
    kPartial,
  };

  friend std::string_view ComparisonOrderToString(Comparison::Order order);

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Order& p) {
    absl::Format(&sink, "%s", ComparisonOrderToString(p));
  }

  // Represents different comparison operations.
  enum class Direction : uint8_t {
    kEq,
    kNe,
    kGe,
    kGt,
    kLe,
    kLt,
  };

  Comparison() = delete;

  // This will default to the expected behavior for Comparison::Order: integers
  // will use total ordering, and floats will use partial ordering.
  explicit Comparison(Direction dir, PrimitiveType type);

  // Pass in a Comparison::Order to specify a non-default ordering, e.g., some
  // targets may support total order floating point type comparisons.
  explicit Comparison(Direction dir, PrimitiveType type, Order order);

  Direction GetDirection() const { return dir_; }
  PrimitiveType GetPrimitiveType() const { return primitive_type_; }
  Order GetOrder() const { return order_; }

  bool IsEq() const { return dir_ == Direction::kEq; }
  bool IsNe() const { return dir_ == Direction::kNe; }
  bool IsGe() const { return dir_ == Direction::kGe; }
  bool IsGt() const { return dir_ == Direction::kGt; }
  bool IsLt() const { return dir_ == Direction::kLt; }
  bool IsTotalOrder() const { return order_ == Order::kTotal; }
  bool IsPartialOrder() const { return order_ == Order::kPartial; }

  bool IsStandardS32() const {
    return primitive_type_ == PrimitiveType::S32 && IsTotalOrder();
  }
  bool IsStandardU32() const {
    return primitive_type_ == PrimitiveType::U32 && IsTotalOrder();
  }

  bool IsIntegralPrimitiveType() const {
    return primitive_util::IsIntegralType(primitive_type_);
  }

  // Returns whether (a dir a) is always true for this comparison.
  bool IsReflexive() const;

  // Returns whether (a dir a) is always false for this comparison.
  bool IsAntireflexive() const;

  // Gets the converse of the given comparison direction (e.g. >= turns to <=).
  // Useful when commuting operands to get constants into immediate-accepting
  // positions in the ISA.
  Comparison Converse() const;

  // Gets the inverse of the given comparison if it exists (e.g. >= turns to <).
  // Returns optional value because not all inversions may be supported.
  std::optional<Comparison> Inverse() const;

  // Returns a string version of this comparison, e.g., ".GT.F32.TOTALORDER"
  std::string ToString(std::string prefix1 = ".", std::string prefix2 = ".",
                       std::string prefix3 = ".") const;

  // Returns a comparison operator: (T, T) -> bool for this Comparison's
  // Direction.
  template <typename T>
  std::function<bool(T, T)> GetComparator() const {
    switch (GetDirection()) {
      case Direction::kEq:
        return std::equal_to<T>();
      case Direction::kNe:
        return std::not_equal_to<T>();
      case Direction::kGe:
        return std::greater_equal<T>();
      case Direction::kGt:
        return std::greater<T>();
      case Direction::kLe:
        return std::less_equal<T>();
      case Direction::kLt:
        return std::less<T>();
    }
  }

  template <typename T>
  bool Compare(const T a, const T b) const {
    DCHECK(primitive_util::IsCanonicalRepresentation<T>(primitive_type_));
    // Applies the comparison from this Comparison's direction and ordering.
    return GetComparator<T>()(a, b);
  }

 private:
  // The direction of the Comparison, e.g., GT.
  const Direction dir_;
  // The primitive type of the Comparison operands, e.g., U32.
  const PrimitiveType primitive_type_;
  // The ordering of the Comparison, e.g., kPartial.
  const Order order_;
};

using ComparisonDirection = Comparison::Direction;
using ComparisonOrder = Comparison::Order;

inline std::ostream& operator<<(std::ostream& os, const Comparison& cmp) {
  return os << cmp.ToString();
}

std::string ComparisonDirectionToString(Comparison::Direction direction);
std::string_view ComparisonPrimitiveTypeToString(PrimitiveType type);

absl::StatusOr<Comparison::Direction> StringToComparisonDirection(
    std::string_view direction);
absl::StatusOr<Comparison::Order> StringToComparisonOrder(
    std::string_view order);

// Returns a comparison function using the provided key function on each value,
// i.e. `key_fn(a) < key_fn(b)`.
template <typename KeyFn>
auto LessThanByKey(KeyFn&& key_fn) {
  return [=](const auto& a, const auto& b) { return key_fn(a) < key_fn(b); };
}

// Two comparisons are equivalent iff they have the same direction, precision,
// and ordering.
inline bool operator==(const Comparison& a, const Comparison& b) {
  return a.GetDirection() == b.GetDirection() &&
         a.GetPrimitiveType() == b.GetPrimitiveType() &&
         a.GetOrder() == b.GetOrder();
}

inline bool operator!=(const Comparison& a, const Comparison& b) {
  return !(a == b);
}

ComparisonDirection RandomComparisonDirection();
ComparisonDirection RandomEqualityComparisonDirection();

}  // namespace zkx

#endif  // ZKX_COMPARISON_UTIL_H_
