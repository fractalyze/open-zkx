// Concatenate + Elementwise Fusion:
//   result = concat(a, b, c) * broadcast(scale)
//
// Before fusion (3 separate ops):
//   %cat    = stablehlo.concatenate %a, %b, %c   → concatenate kernel
//   %bcast  = stablehlo.broadcast_in_dim %scale   → materializes full tensor
//   %result = stablehlo.multiply %cat, %bcast     → elementwise kernel
//
// After fusion (1 kConcatenate fusion):
//   The concatenate, broadcast, and multiply are fused into one kernel.
//   Each thread:
//     1. Determines which input (a, b, or c) to read based on output index
//     2. Loads the element from the correct input
//     3. Multiplies by the broadcast scale in-register
//     4. Writes the result
//
// BabyBear prime: p = 2013265921 = 15 * 2²⁷ + 1

!bf = !field.pf<2013265921 : i32, true>

module @concatenate_fusion {
  func.func public @main(%a: tensor<256x!bf>, %b: tensor<512x!bf>,
                          %c: tensor<256x!bf>,
                          %scale: tensor<!bf>) -> tensor<1024x!bf> {
    %cat = stablehlo.concatenate %a, %b, %c, dim = 0
      : (tensor<256x!bf>, tensor<512x!bf>, tensor<256x!bf>) -> tensor<1024x!bf>
    %bcast = stablehlo.broadcast_in_dim %scale, dims = []
      : (tensor<!bf>) -> tensor<1024x!bf>
    %result = stablehlo.multiply %cat, %bcast : tensor<1024x!bf>
    return %result : tensor<1024x!bf>
  }
}
