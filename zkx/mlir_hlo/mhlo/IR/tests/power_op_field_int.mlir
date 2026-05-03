// RUN: emitters_opt %s --split-input-file --verify-diagnostics | FileCheck %s

// Verifies that mhlo.power accepts a field-typed base with an integer-typed
// exponent and a field-typed result. Mirrors stablehlo.power's shape so the
// stablehlo -> mhlo legalization can pass through unchanged. Regression guard
// for the case where MHLO_PowOp incorrectly required matching element types
// across operands and result, blocking maci_quin_generate_path_indices in
// llzk-to-shlo.

!bn254 = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256, true>

// CHECK-LABEL: @power_field_base_int_exp
// CHECK: mhlo.power %{{.*}}, %{{.*}} : (tensor<![[FIELD:[a-zA-Z0-9_]+]]>, tensor<i256>) -> tensor<![[FIELD]]>
func.func @power_field_base_int_exp(%base: tensor<!bn254>, %exp: tensor<i256>)
    -> tensor<!bn254> {
  %0 = mhlo.power %base, %exp : (tensor<!bn254>, tensor<i256>) -> tensor<!bn254>
  return %0 : tensor<!bn254>
}

// CHECK-LABEL: @power_int_base_int_exp
// CHECK: mhlo.power %{{.*}}, %{{.*}} : (tensor<6xi64>, tensor<6xi64>) -> tensor<6xi64>
func.func @power_int_base_int_exp(%lhs: tensor<6xi64>, %rhs: tensor<6xi64>)
    -> tensor<6xi64> {
  %0 = mhlo.power %lhs, %rhs : (tensor<6xi64>, tensor<6xi64>) -> tensor<6xi64>
  return %0 : tensor<6xi64>
}

// -----

// Negative case: a non-integer rhs is still rejected by the verifier.
!bn254 = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256, true>

func.func @power_field_rhs_rejected(%base: tensor<!bn254>, %exp: tensor<!bn254>)
    -> tensor<!bn254> {
  // expected-error @+1 {{op operand #1 must be ranked tensor of}}
  %0 = mhlo.power %base, %exp : (tensor<!bn254>, tensor<!bn254>) -> tensor<!bn254>
  return %0 : tensor<!bn254>
}
