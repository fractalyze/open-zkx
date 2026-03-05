// Multi-Output / Sibling Fusion:
//   r1 = sum((a + b) * a)
//   r2 = sum((a + b) * b)
//
// Fusion process (two stages):
//
// Stage 1 — Priority Fusion:
//   Branch 1: add + mul_a + reduce → kReduction fusion (r1)
//   Branch 2: add + mul_b + reduce → kReduction fusion (r2)
//   The (a + b) computation is duplicated in each fusion.
//
// Stage 2 — Multi-Output Fusion:
//   Both branches share the same inputs (a, b) and the same (a + b)
//   subexpression. The compiler merges the two sibling reductions
//   into a single multi-output fusion, computing both r1 and r2
//   in one kernel launch with shared memory reuse.
//
// BabyBear prime: p = 2013265921 = 15 * 2²⁷ + 1

!bf = !field.pf<2013265921 : i32, true>

module @multi_output_fusion {
  func.func public @main(%a: tensor<1024x!bf>, %b: tensor<1024x!bf>) -> (tensor<!bf>, tensor<!bf>) {
    %add = stablehlo.add %a, %b : tensor<1024x!bf>
    %mul_a = stablehlo.multiply %add, %a : tensor<1024x!bf>
    %mul_b = stablehlo.multiply %add, %b : tensor<1024x!bf>
    %zero = stablehlo.constant dense<0> : tensor<!bf>
    %r1 = stablehlo.reduce(%mul_a init: %zero) across dimensions = [0]
      : (tensor<1024x!bf>, tensor<!bf>) -> tensor<!bf>
      reducer(%x1: tensor<!bf>, %y1: tensor<!bf>) {
        %s1 = stablehlo.add %x1, %y1 : tensor<!bf>
        stablehlo.return %s1 : tensor<!bf>
      }
    %r2 = stablehlo.reduce(%mul_b init: %zero) across dimensions = [0]
      : (tensor<1024x!bf>, tensor<!bf>) -> tensor<!bf>
      reducer(%x2: tensor<!bf>, %y2: tensor<!bf>) {
        %s2 = stablehlo.add %x2, %y2 : tensor<!bf>
        stablehlo.return %s2 : tensor<!bf>
      }
    return %r1, %r2 : tensor<!bf>, tensor<!bf>
  }
}
