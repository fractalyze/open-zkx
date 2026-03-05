// Transpose + Elementwise Fusion:
//   result = transpose(a + b)
//
// Before fusion (2 separate ops):
//   %add = stablehlo.add %a, %b              → elementwise kernel
//   %t   = stablehlo.transpose %add, ...     → separate transpose kernel
//
// After fusion (1 kTranspose fusion):
//   The elementwise add and transpose are fused into a single kernel.
//   The transpose uses shared memory with +1 padding to avoid bank conflicts:
//     1. Threads cooperatively compute (a + b) and write tiles to shmem
//     2. __syncthreads() barrier
//     3. Threads read from shmem in transposed order and write to output
//
// To verify shared memory usage, dump the LLVM IR:
//   ZKX_FLAGS="--zkx_dump_to=/tmp/dump --zkx_dump_hlo_as_text" \
//     examples/run.sh examples/fusion_showcase/transpose_fusion.stablehlo.mlir
//   grep -c "addrspace(3)" /tmp/dump/*.ir-no-opt.ll
//
// Expected LLVM IR evidence:
//   @shared_0 = private addrspace(3) global [...]       ← allocate_shared
//   call void @llvm.nvvm.barrier.cta.sync.aligned.all() ← sync_threads

!bf = !field.pf<2013265921 : i32, true>

module @transpose_fusion {
  func.func public @main(%a: tensor<256x256x!bf>,
                          %b: tensor<256x256x!bf>) -> tensor<256x256x!bf> {
    %add = stablehlo.add %a, %b : tensor<256x256x!bf>
    %t = stablehlo.transpose %add, dims = [1, 0]
      : (tensor<256x256x!bf>) -> tensor<256x256x!bf>
    return %t : tensor<256x256x!bf>
  }
}
