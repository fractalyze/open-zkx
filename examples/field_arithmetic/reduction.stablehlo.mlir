// Field reduction example: sum-reduce a vector of BabyBear field elements.
//
// This program computes: result = sum(a[0..1024])
// The reduction folds all 1024 elements into a single field element using
// field addition.
//
// BabyBear prime: p = 2013265921 = 15 * 2²⁷ + 1

!bf = !field.pf<2013265921 : i32, true>

module @field_reduction {
  func.func public @main(%a: tensor<1024x!bf>) -> tensor<!bf> {
    %init = stablehlo.constant dense<0> : tensor<!bf>
    %result = stablehlo.reduce(%a init: %init) across dimensions = [0] : (tensor<1024x!bf>, tensor<!bf>) -> tensor<!bf>
      reducer(%x: tensor<!bf>, %y: tensor<!bf>) {
        %sum = stablehlo.add %x, %y : tensor<!bf>
        stablehlo.return %sum : tensor<!bf>
      }
    return %result : tensor<!bf>
  }
}
