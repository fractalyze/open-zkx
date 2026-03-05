# Tiling and Shared Memory Optimization

This document describes the loop tiling, shared memory management, and memory
optimization strategies used in ZKX GPU kernel generation.

## Overview

GPU performance depends critically on:

1. **Tiling** — Partitioning work across thread blocks and threads
1. **Shared memory** — Fast on-chip memory shared within a thread block
1. **Memory coalescing** — Ensuring adjacent threads access adjacent memory
1. **GPU occupancy** — Maximizing active warps per SM

ZKX GPU addresses all four through specialized emitters, MLIR passes, and
analysis infrastructure.

## Reduction Tiling

**File:** `zkx/service/gpu/reduction_utils.cc`

Reduction operations are partitioned into a 3D tile `(Z, Y, X)` representing
`(batch, kept, reduced)` dimensions.

### Tiling Strategy

```cpp
// Row reduction: reduce along the minor (contiguous) dimension
tile = {min(batch_size, RaceFreeBound), 1, 16}

// Column reduction: reduce along a non-minor dimension
tile = {1, 128, 1}
```

For row reductions, the minor dimension tile size of 16 enables vectorized
memory access while keeping shared memory usage bounded.

### Reduction Emitter Variants

**File:** `zkx/backends/gpu/codegen/emitters/reduction.cc` (~1,118 lines)

Four reduction strategies are available, selected based on the reduction shape:

| Strategy         | When Used                          | Threads          |
| ---------------- | ---------------------------------- | ---------------- |
| **RowReduction** | Standard row reduction             | 256 (tunable)    |
| **MultiRow**     | Small power-of-2 reduced dimension | Warp-aligned     |
| **Column**       | Column (non-minor) reduction       | 128 × vectorized |
| **SmallColumn**  | Small column reduction             | Compact blocks   |

### Two-Level Reduction

Row reductions use a two-level strategy:

```
Step 1: Per-thread reduction across tile elements
        Each thread accumulates partial results

Step 2: Warp-level shuffle reduction
        zkx_gpu.shuffle_reduce combines results within a warp

Step 3: (if multi-warp) Shared memory reduction
        Warp results written to shared memory
        zkx_gpu.sync_threads barrier
        Final warp reads and reduces shared memory values
```

## Shared Memory Usage

### ZKX GPU Dialect Operations

**File:** `zkx/backends/gpu/codegen/emitters/ir/zkx_gpu_ops.td`

The ZKX GPU dialect provides dedicated operations for shared memory management:

```mlir
// Allocate a block of shared memory
%shmem = zkx_gpu.allocate_shared : tensor<32x33xf32>

// Synchronize all threads in the block (preserves tensor state)
%synced = zkx_gpu.sync_threads %values : tensor<32x33xf32>

// Warp-level shuffle reduction with custom combiner
%result = zkx_gpu.shuffle_reduce (%partial) to 16 combiner=@add_fn
```

### Transpose: Shared Memory Tiling

**File:** `zkx/backends/gpu/codegen/emitters/transpose.cc` (~530 lines)

Transposing data on the GPU is inherently non-coalesced for one of the two
access patterns (read or write). The transpose emitter solves this with shared
memory tiling:

```
Step 1: Coalesced read from global memory → shared memory
        (threads read contiguous input rows)

Step 2: Synchronize threads (zkx_gpu.sync_threads)

Step 3: Coalesced write from shared memory → global memory
        (threads read transposed columns from shmem, write contiguous rows)
```

**Bank conflict avoidance:** The shared memory tile is padded by +1 column to
avoid bank conflicts. For example, a 32×32 tile becomes 32×33:

```
// From transpose.cc — avoid bank conflicts
if (MostMinorDimensionUnchanged()) {
    ++shmem_tensor_size[shmem_tensor_size.size() - 2];
} else {
    ++shmem_tensor_size.back();
}
```

**Thread configuration:**

- 128 threads per block (`kNumThreadsPerBlock`)
- 4 rows per thread (`kNumRows`)
- Up to 4 bytes vectorized per load (`kMaxVectorizedBytes`)

### Reduction: Shared Memory Accumulation

**File:** `zkx/backends/gpu/codegen/emitters/reduction.cc`

Row reductions with multiple warps use shared memory for cross-warp
communication:

```
Warp 0: shuffle_reduce → write lane 0 result to shmem[0]
Warp 1: shuffle_reduce → write lane 0 result to shmem[1]
...
sync_threads
Warp 0: read shmem[0..N], shuffle_reduce → final result
```

Column reductions use shared memory with minimal padding to avoid bank
conflicts:

```cpp
// Minimal padding that avoids all bank conflicts
int padding = (num_threads == 32 && vector_size == 1)
    ? 0
    : CeilOfRatio(input_shape[2], num_threads * vector_size);
```

## Memory Coalescing Analysis

**File:** `zkx/service/gpu/model/coalescing_analysis.cc` (~628 lines)

The coalescing analysis determines whether memory accesses within a fusion are
coalesced (adjacent threads access adjacent memory addresses).

### Heuristic-Based Analysis

```cpp
bool IsReadCoalescedHeuristic(
    EmitterFusionKind fusion_kind,
    const DeviceDescription& device_info,
    const HloInstruction* producer,
    const HloInstruction* consumer)
```

**Coalescing breakers** (patterns that destroy coalescing):

- Transposes of the minor (contiguous) dimension
- Dual row reductions
- Non-contiguous memory access patterns

### Indexing-Based Analysis

For more precise analysis, the system traces the indexing maps from thread IDs
to memory addresses. If adjacent thread IDs map to adjacent memory offsets, the
access is coalesced.

The analysis result feeds into the performance model, which adjusts memory read
time estimates based on coalescing status.

## Loop Optimization Passes

### Vectorize Loads and Stores

**File:**
`zkx/backends/gpu/codegen/emitters/transforms/vectorize_loads_stores.cc`

Converts scalar memory operations into vector operations when the access pattern
permits:

- Analyzes stride and alignment of affine expressions
- Detects coalesced access patterns suitable for vectorization
- Applies vector loads up to the maximum vectorized width

### Optimize Loops

**File:** `zkx/backends/gpu/codegen/emitters/transforms/optimize_loops.cc`

Performs loop-level optimizations:

- **Induction variable replacement** — Replaces complex affine expressions with
  simpler stride-based computations
- **Loop pipelining** — Overlaps computation with memory access
- **LICM** — Hoists loop-invariant operations (via standard MLIR pass)

### Peel Loops

**File:** `zkx/backends/gpu/codegen/emitters/transforms/peel_loops.cc`

Peels loop iterations to handle boundary conditions, enabling the main loop body
to use simpler (no-bounds-check) code paths.

## Launch Dimension Calculation

**File:** `zkx/service/gpu/launch_dimensions.cc`

The launch dimensions (grid size × block size) are computed to maximize GPU
occupancy:

```cpp
LaunchDimensions CalculateLaunchDimensions(
    const Shape& shape,
    const DeviceDescription& gpu_device_info,
    LaunchDimensionsConfig dim_config)
```

### NVIDIA Strategy

```
threads_per_block = min(warp_size × 4, num_elements / unroll_factor)
num_blocks = ceil(num_elements / threads_per_block)
grid = (num_blocks_x, num_blocks_y, 1)
```

- Default: 4 warp schedulers per SM → `warp_size × 4 = 128` threads per block
- Blocks distributed across X and Y dimensions to stay within hardware limits

### Emitter-Specific Launch Dimensions

Specialized emitters compute their own launch dimensions:

- **Reduction:** Based on reduction tile sizes and warp count
- **Transpose:** `Product(block_counts) × 128` (fixed block size)
- **Loop:** Standard occupancy calculation

## Memory Hierarchy Summary

| Level             | Mechanism                           | Managed By             |
| ----------------- | ----------------------------------- | ---------------------- |
| **Registers**     | Per-thread scalars and vectors      | MLIR → LLVM allocation |
| **Shared Memory** | `zkx_gpu.allocate_shared` + padding | Emitters               |
| **L1/L2 Cache**   | Implicit (hardware managed)         | Coalescing analysis    |
| **Global DRAM**   | Bandwidth model in perf model       | Performance model      |

## MLIR Lit Tests

The loop optimization and memory passes are tested via MLIR lit tests:

| Test File                                       | What It Tests                    |
| ----------------------------------------------- | -------------------------------- |
| `transforms/tests/optimize_loops.mlir`          | Loop induction variable opt      |
| `transforms/tests/peel_loops.mlir`              | Loop peeling                     |
| `transforms/tests/vectorize_loads_stores.mlir`  | Load/store vectorization         |
| `transforms/tests/lower_zkx_to_scf.mlir`        | ZKX dialect → SCF lowering       |
| `transforms/tests/flatten_tensors.mlir`         | Tensor flattening                |
| `transforms/tests/simplify_arith.mlir`          | Arithmetic simplification        |
| `transforms/tests/simplify_affine.mlir`         | Affine expression simplification |
| `transforms/tests/lower_tensors.mlir`           | Tensor → buffer lowering         |
| `transforms/tests/unswitch_loops.mlir`          | Loop unswitching                 |
| `transforms/tests/convert_index_type.mlir`      | Index type conversion            |
| `transforms/tests/propagate_slice_indices.mlir` | Slice index propagation          |

## Key Source Files

| File                                                                     | Purpose               |
| ------------------------------------------------------------------------ | --------------------- |
| `zkx/service/gpu/reduction_utils.cc`                                     | Reduction tiling      |
| `zkx/backends/gpu/codegen/emitters/reduction.cc`                         | Reduction emitter     |
| `zkx/backends/gpu/codegen/emitters/transpose.cc`                         | Transpose emitter     |
| `zkx/backends/gpu/codegen/emitters/ir/zkx_gpu_ops.td`                    | Shared memory ops     |
| `zkx/backends/gpu/codegen/emitters/transforms/optimize_loops.cc`         | Loop optimization     |
| `zkx/backends/gpu/codegen/emitters/transforms/peel_loops.cc`             | Loop peeling          |
| `zkx/backends/gpu/codegen/emitters/transforms/vectorize_loads_stores.cc` | Vectorization         |
| `zkx/service/gpu/model/coalescing_analysis.cc`                           | Coalescing analysis   |
| `zkx/service/gpu/launch_dimensions.cc`                                   | Launch dimension calc |
