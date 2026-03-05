// Broadcast + Elementwise Chain Fusion:
//   result = ((a + b) * broadcast(scale))²
//
// Before fusion (4 separate ops):
//   %add       = stablehlo.add %a, %b                    → elementwise kernel
//   %bcast     = stablehlo.broadcast_in_dim %scale, ...  → materializes full tensor
//   %mul       = stablehlo.multiply %add, %bcast         → elementwise kernel
//   %result    = stablehlo.multiply %mul, %mul            → elementwise kernel
//
// After loop fusion (1 kLoop fusion):
//   All four ops are fused into a single elementwise loop kernel.
//   The broadcast is absorbed — no full-size tensor is materialized;
//   the scalar scale is loaded once and reused per element.
//
// BabyBear prime: p = 2013265921 = 15 * 2²⁷ + 1

!bf = !field.pf<2013265921 : i32, true>

module @broadcast_fusion {
  func.func public @main(%a: tensor<1024x!bf>, %b: tensor<1024x!bf>,
                          %scale: tensor<!bf>) -> tensor<1024x!bf> {
    %add = stablehlo.add %a, %b : tensor<1024x!bf>
    %bcast = stablehlo.broadcast_in_dim %scale, dims = []
      : (tensor<!bf>) -> tensor<1024x!bf>
    %mul = stablehlo.multiply %add, %bcast : tensor<1024x!bf>
    %result = stablehlo.multiply %mul, %mul : tensor<1024x!bf>
    return %result : tensor<1024x!bf>
  }
}
