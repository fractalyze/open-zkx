// Input Fusion: result = sum((a + b) * a)
//
// Before fusion:
//   %add = stablehlo.add %a, %b           → separate elementwise kernel
//   %mul = stablehlo.multiply %add, %a    → separate elementwise kernel
//   %sum = stablehlo.reduce(%mul, ...)     → separate reduction kernel
//
// After priority fusion (1 kReduction fusion):
//   The elementwise add + multiply are absorbed as "input" computations
//   into the reduction fusion. No intermediate tensors hit global memory;
//   they live in registers and are reduced via shared memory.
//
// BabyBear prime: p = 2013265921 = 15 * 2²⁷ + 1

!bf = !field.pf<2013265921 : i32, true>

module @input_fusion {
  func.func public @main(%a: tensor<1024x!bf>, %b: tensor<1024x!bf>) -> tensor<!bf> {
    %add = stablehlo.add %a, %b : tensor<1024x!bf>
    %mul = stablehlo.multiply %add, %a : tensor<1024x!bf>
    %zero = stablehlo.constant dense<0> : tensor<!bf>
    %sum = stablehlo.reduce(%mul init: %zero) across dimensions = [0]
      : (tensor<1024x!bf>, tensor<!bf>) -> tensor<!bf>
      reducer(%x: tensor<!bf>, %y: tensor<!bf>) {
        %s = stablehlo.add %x, %y : tensor<!bf>
        stablehlo.return %s : tensor<!bf>
      }
    return %sum : tensor<!bf>
  }
}
