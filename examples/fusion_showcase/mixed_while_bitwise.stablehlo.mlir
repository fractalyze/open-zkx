// Mixed-type while-body fusion regression.
// The loop body converts BabyBear -> i32, performs bitwise ops, converts back,
// and also advances an i32 loop counter via add.
// Historically this shape could leave standalone elementwise ops in the while
// body, which later hit IrEmitterUnnested with "Unsupported instruction opcode".

!bf = !field.pf<2013265921 : i32, true>

module @mixed_while_bitwise {
  func.func public @main(%arg0: tensor<!bf>) -> tensor<!bf> {
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %0:2 = stablehlo.while(%iterArg = %c0, %iterArg_0 = %arg0) : tensor<i32>, tensor<!bf>
     cond {
      %1 = stablehlo.compare LT, %iterArg, %c1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %one = stablehlo.constant dense<1> : tensor<i32>
      %as_i32 = stablehlo.convert %iterArg_0 : (tensor<!bf>) -> tensor<i32>
      %shifted = stablehlo.shift_right_logical %as_i32, %one : tensor<i32>
      %masked = stablehlo.and %shifted, %one : tensor<i32>
      %next_state = stablehlo.convert %masked : (tensor<i32>) -> tensor<!bf>
      %next_counter = stablehlo.add %iterArg, %one : tensor<i32>
      stablehlo.return %next_counter, %next_state : tensor<i32>, tensor<!bf>
    }
    return %0#1 : tensor<!bf>
  }
}
