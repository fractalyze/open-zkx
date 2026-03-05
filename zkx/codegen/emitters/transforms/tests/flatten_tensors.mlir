// RUN: emitters_opt %s -split-input-file -zkx-flatten-tensors \
// RUN: --verify-diagnostics | FileCheck %s

func.func @tensor_extract(
    %arg0: tensor<2x3xf32, dense<[0, 1]> : tensor<2xi64>>,
    %arg1: index, %arg2: index) -> f32 {
  %v = tensor.extract %arg0[%arg1, %arg2]
      : tensor<2x3xf32, dense<[0, 1]> : tensor<2xi64>>
  func.return %v : f32
}
// CHECK: #[[$MAP:.+]] = #zkx.indexing_map<"(d0, d1) -> (d1 * 2 + d0), domain: d0 in [0, 1], d1 in [0, 2]">

// CHECK-LABEL: func.func @tensor_extract(
// CHECK-SAME:      %[[SRC:.*]]: tensor<6xf32>,
// CHECK-SAME:      %[[I:.*]]: index, %[[J:.*]]: index) -> f32 {
// CHECK:        %[[INDEX:.*]] = zkx.apply_indexing #[[$MAP]](%[[I]], %[[J]])
// CHECK:        tensor.extract %[[SRC]][%[[INDEX]]] : tensor<6xf32>

// -----

func.func @tensor_insert(%arg0: tensor<10x24xf32>, %i: index) -> tensor<10x24xf32> {
  %scalar = arith.constant 3.0 : f32
  %out = tensor.insert %scalar into %arg0[%i, %i] : tensor<10x24xf32>
  func.return %out : tensor<10x24xf32>
}
// CHECK-LABEL: func.func @tensor_insert(
// CHECK-SAME:      %[[TENSOR:.*]]: tensor<240xf32>, %[[I:.*]]: index) -> tensor<240xf32> {
// CHECK:         %[[SCALAR:.*]] = arith.constant
// CHECK:        %[[INDEX:.*]] = zkx.apply_indexing #[[$MAP]](%[[I]], %[[I]])
// CHECK:         tensor.insert %[[SCALAR]] into %[[TENSOR]][%[[INDEX]]]

// -----

func.func @for_loop(%t0: tensor<32x1024xf32>, %t1: tensor<64x8x4xf32>)
    -> (tensor<32x1024xf32>, tensor<64x8x4xf32>, f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c0_f32 = arith.constant 0.0 : f32
  %for:2 = scf.for %i = %c0 to %c64 step %c32 iter_args(%t0_ = %t0, %t1_ = %t1)
    -> (tensor<32x1024xf32>, tensor<64x8x4xf32>) {
    %update0 = tensor.insert %c0_f32 into %t0_[%c1, %i] : tensor<32x1024xf32>
    %update1 = tensor.insert %c0_f32 into %t1_[%i, %c1, %c1]
      : tensor<64x8x4xf32>
    scf.yield %update0, %update1 : tensor<32x1024xf32>, tensor<64x8x4xf32>
  } {some_attr}
    return %for#0, %for#1, %c0_f32 : tensor<32x1024xf32>, tensor<64x8x4xf32>, f32
}
// CHECK: #[[$MAP0:.+]] = #zkx.indexing_map<"(d0) -> (d0 + 1024)
// CHECK: #[[$MAP1:.+]] = #zkx.indexing_map<"(d0) -> (d0 * 32 + 5)
// CHECK-LABEL: func.func @for_loop(
// CHECK-SAME:      %[[T0:.*]]: tensor<32768xf32>,
// CHECK-SAME:      %[[T1:.*]]: tensor<2048xf32>) -> (tensor<32768xf32>, tensor<2048xf32>, f32) {

// CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:  %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG:  %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG:  %[[F32:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:      %[[FOR:.*]]:2 = scf.for %[[I:.*]] = %[[C0]] to %[[C64]]
// CHECK-SAME:     step %[[C32]]
// CHECK-SAME:     iter_args(%[[T0_:.*]] = %[[T0]], %[[T1_:.*]] = %[[T1]])
// CHECK:        %[[IND0:.*]] = zkx.apply_indexing #[[$MAP0]](%[[I]])
// CHECK:        %[[UPD0:.*]] = tensor.insert %[[F32]] into %[[T0_]][%[[IND0]]]
// CHECK:        %[[IND1:.*]] = zkx.apply_indexing #[[$MAP1]](%[[I]])
// CHECK:        %[[UPD1:.*]] = tensor.insert %[[F32]] into %[[T1_]][%[[IND1]]]
// CHECK:        scf.yield %[[UPD0]], %[[UPD1]] : tensor<32768xf32>, tensor<2048xf32>

// -----

// Verify that non-cast multi-dimensional field vector init args in scf.for
// are properly flattened by RewriteFor.
!pf = !field.pf<2013265921 : i32>

func.func @for_field_vector_init(%v: vector<2x3x!pf>, %val: !pf)
    -> vector<2x3x!pf> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %result = scf.for %i = %c0 to %c2 step %c1
      iter_args(%acc = %v) -> (vector<2x3x!pf>) {
    %inserted = vector.insert %val, %acc [%i, %c0]
      : !pf into vector<2x3x!pf>
    scf.yield %inserted : vector<2x3x!pf>
  }
  return %result : vector<2x3x!pf>
}
// CHECK-LABEL: func.func @for_field_vector_init
// CHECK-SAME: vector<6x!pf_babybear>
// CHECK-SAME: !pf_babybear
// CHECK: scf.for {{.*}} -> (vector<6x!pf_babybear>)
// CHECK-NOT:  builtin.unrealized_conversion_cast

// -----

// Verify that cross-type casts (field ↔ integer from shuffle lowering)
// do not trigger FlattenTensors convergence failure.
!pf = !field.pf<2013265921 : i32>

func.func @cross_type_cast(%a: !pf) -> !pf {
  %int = builtin.unrealized_conversion_cast %a
    : !pf to i32
  %back = builtin.unrealized_conversion_cast %int
    : i32 to !pf
  return %back : !pf
}
// The cross-type casts (field ↔ integer) may fold away, but the important
// thing is the pass succeeds without a convergence failure.
// CHECK-LABEL: func.func @cross_type_cast

// -----

func.func @allocate_shared() -> tensor<10x15xf32> {
  %shmem = zkx_gpu.allocate_shared : tensor<10x15xf32>
  func.return %shmem : tensor<10x15xf32>
}
// CHECK-LABEL: func.func @allocate_shared() -> tensor<150xf32>
// CHECK:         zkx_gpu.allocate_shared : tensor<150xf32>
// CHECK-NOT:     builtin.unrealized_conversion_cast

// -----

func.func @sync() -> (tensor<8x4xf32>, tensor<8x4xf32>) {
  %shared1 = zkx_gpu.allocate_shared : tensor<8x4xf32>
  %shared2 = zkx_gpu.allocate_shared : tensor<8x4xf32>
  %sync:2 = zkx_gpu.sync_threads %shared1, %shared2
    : tensor<8x4xf32>, tensor<8x4xf32>
  return %sync#0, %sync#1 : tensor<8x4xf32>, tensor<8x4xf32>
}
// CHECK-LABEL: func.func @sync() -> (tensor<32xf32>, tensor<32xf32>) {
// CHECK:         %[[SHARED1:.*]] = zkx_gpu.allocate_shared : tensor<32xf32>
// CHECK:         %[[SHARED2:.*]] = zkx_gpu.allocate_shared : tensor<32xf32>
// CHECK:         %[[SYNC:.*]] = zkx_gpu.sync_threads %[[SHARED1]], %[[SHARED2]]
// CHECK-SAME:      : tensor<32xf32>, tensor<32xf32>
// CHECK-NEXT:    return

// -----

func.func @constant() -> tensor<2x3xf32> {
   %cst = arith.constant dense<[
    [-3.000000e+00, 2.000000e+00, 1.000000e+00],
    [0.000000e+00, -3.000000e+00, 1.000000e+00]
   ]> : tensor<2x3xf32>
   return %cst : tensor<2x3xf32>
}
// CHECK-LABEL: func.func @constant
// CHECK-SAME: -> tensor<6xf32>
// CHECK-NOT:  builtin.unrealized_conversion_cast

// -----

func.func @vector_extract(%arg0: vector<2x3xf32>, %arg1: index) -> f32 {
  %v = vector.extract %arg0[%arg1, 2] : f32 from vector<2x3xf32>
  func.return %v : f32
}
// CHECK: #[[$MAP:.+]] = #zkx.indexing_map<"(d0) -> (d0 * 3 + 2),
// CHECK-SAME: domain: d0 in [0, 1]

// CHECK-LABEL: func.func @vector_extract(
// CHECK-SAME:      %[[SRC:.*]]: vector<6xf32>, %[[I:.*]]: index) -> f32 {
// CHECK:        %[[INDEX:.*]] = zkx.apply_indexing #[[$MAP]](%[[I]])
// CHECK:        vector.extract %[[SRC]][%[[INDEX]]] : f32 from vector<6xf32>

// -----

func.func @vector_insert(%arg0: vector<10x24xf32>, %i: index)
  -> vector<10x24xf32> {
  %scalar = arith.constant 3.0 : f32
  %out = vector.insert %scalar, %arg0 [1, %i] : f32 into vector<10x24xf32>
  func.return %out : vector<10x24xf32>
}
// CHECK: #[[$MAP:.+]] = #zkx.indexing_map<"(d0) -> (d0 + 24),
// CHECK-SAME: domain: d0 in [0, 23]
// CHECK-LABEL: func.func @vector_insert(
// CHECK-SAME:      %[[VECTOR:.*]]: vector<240xf32>, %[[I:.*]]: index) ->
// CHECK-SAME:      vector<240xf32> {
// CHECK:         %[[INDEX:.*]] = zkx.apply_indexing #[[$MAP]](%[[I]])
// CHECK:         vector.insert {{.*}}, %[[VECTOR]] [%[[INDEX]]]
// CHECK-SAME:      : f32 into vector<240xf32>

// -----
