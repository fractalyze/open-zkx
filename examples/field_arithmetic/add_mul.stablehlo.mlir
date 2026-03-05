// Field arithmetic example: add and multiply chain on BabyBear field elements.
//
// This program computes: result = (a + b) * a
// where a and b are tensors of BabyBear prime field elements.
//
// BabyBear prime: p = 2013265921 = 15 * 2²⁷ + 1

!bf = !field.pf<2013265921 : i32, true>

module @field_add_mul {
  func.func public @main(%a: tensor<1024x!bf>, %b: tensor<1024x!bf>) -> tensor<1024x!bf> {
    %sum = stablehlo.add %a, %b : tensor<1024x!bf>
    %result = stablehlo.multiply %sum, %a : tensor<1024x!bf>
    return %result : tensor<1024x!bf>
  }
}
