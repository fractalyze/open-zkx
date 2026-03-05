// Scatter + Elementwise Fusion:
//   result[i] = operand[i],            for i >= 256
//   result[i] = (a[i] + b[i]) * a[i],  for i in {0, ..., 255}
//
// Before fusion (4 separate ops):
//   %add     = stablehlo.add %a, %b               → elementwise kernel
//   %mul     = stablehlo.multiply %add, %a         → elementwise kernel
//   %indices = stablehlo.iota ...                  → constant materialization
//   %scatter = stablehlo.scatter(operand, ...)     → scatter kernel
//
// After fusion (1 kScatter fusion):
//   The elementwise add/multiply and iota are absorbed into the scatter
//   kernel. Each thread computes (a[i] + b[i]) * a[i] in-register and
//   writes directly to the output at position i.
//
// BabyBear prime: p = 2013265921 = 15 * 2²⁷ + 1

!bf = !field.pf<2013265921 : i32, true>

module @scatter_fusion {
  func.func public @main(%operand: tensor<1024x!bf>,
                          %a: tensor<256x!bf>,
                          %b: tensor<256x!bf>) -> tensor<1024x!bf> {
    %add = stablehlo.add %a, %b : tensor<256x!bf>
    %updates = stablehlo.multiply %add, %a : tensor<256x!bf>
    %indices = stablehlo.iota dim = 0 : tensor<256x1xi32>
    %result = "stablehlo.scatter"(%operand, %indices, %updates) <{
      indices_are_sorted = true,
      scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [],
        inserted_window_dims = [0],
        scatter_dims_to_operand_dims = [0],
        index_vector_dim = 1
      >,
      unique_indices = true
    }> ({
      ^bb0(%x: tensor<!bf>, %y: tensor<!bf>):
        stablehlo.return %y : tensor<!bf>
    }) : (tensor<1024x!bf>, tensor<256x1xi32>, tensor<256x!bf>) -> tensor<1024x!bf>
    return %result : tensor<1024x!bf>
  }
}
