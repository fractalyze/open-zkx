# Fusion Passes

This document describes the fusion passes that consolidate HLO operations into
unified computational units for efficient GPU execution.

## Overview

Fusion is the process of combining multiple HLO operations into a single GPU
kernel to reduce memory traffic and kernel launch overhead. ZKX GPU implements
two complementary fusion strategies orchestrated by the fusion pipeline.

The fusion pipeline (`zkx/service/gpu/fusion_pipeline.cc`) runs inside
`HloPassFix` (repeated until no more changes):

```
┌─────────────────────────────────────────────┐
│              Fusion Pipeline                │
│                                             │
│  ┌───────────────────┐                      │
│  │  Priority Fusion  │  cost-model driven   │
│  └────────┬──────────┘                      │
│           ▼                                 │
│  ┌───────────────────┐                      │
│  │    CSE + DCE      │  cleanup             │
│  └────────┬──────────┘                      │
│           ▼                                 │
│  ┌───────────────────┐                      │
│  │ Multi-Output      │  sibling + producer  │
│  │ Fusion            │  consumer fusion     │
│  └────────┬──────────┘                      │
│           ▼                                 │
│  ┌───────────────────┐                      │
│  │    CSE + DCE      │  cleanup             │
│  └───────────────────┘                      │
│         (repeat until convergence)          │
└─────────────────────────────────────────────┘
```

## Priority Fusion

**File:** `zkx/service/gpu/transforms/priority_fusion.cc` (826 lines)

Priority fusion is the primary fusion pass. It uses a cost model to make greedy
fusion decisions that maximize performance.

### Algorithm

1. Build a priority queue of all candidate (producer, consumer) pairs.
1. For each pair, estimate runtime with and without fusion using
   `GpuPerformanceModel`.
1. Priority = `time_unfused - time_fused`. Higher priority = more beneficial.
1. Greedily fuse the highest-priority pair.
1. Recompute priorities for affected instructions.
1. Repeat until no profitable fusions remain.

### Cost Model

The cost model (`zkx/service/gpu/model/gpu_performance_model.cc`) estimates
execution time by combining:

- **Compute time** — Based on FLOP count and GPU compute throughput
- **Memory read time** — Per-operand analysis with coalescing heuristics
- **Memory write time** — Output bytes × memory bandwidth
- **Combined time** — `max(compute_time, memory_time)` or weighted combination

```
exec_time = CombineComputeAndMemoryAccessTime(
    compute_time,
    read_time + write_time
)
```

For each operand, the model considers:

- Total bytes accessed vs. net bytes (accounting for reuse)
- Whether reads are coalesced (via `CoalescingAnalysis`)
- DRAM bandwidth utilization heuristics

### Fusion Legality

Not all instruction pairs can be fused. The pass checks:

- Instruction is fusible (not a parameter, not certain collective ops)
- Fusion would not exceed shared memory limits
- Fusion would not create excessively large kernels
- The fused computation can be handled by an available emitter

### What Gets Fused

Typical fusion patterns:

- **Elementwise chains**: `add → mul → add` fused into a single kernel
- **Broadcast + elementwise**: Broadcast folded into the consumer
- **Reduction + epilogue**: Reduction followed by elementwise post-processing
- **Transpose + elementwise**: Using shared memory for the transpose

## Multi-Output Fusion

**File:** `zkx/service/gpu/transforms/multi_output_fusion.cc` (497 lines)

Multi-output fusion combines operations that share common operands into a single
kernel with multiple outputs, reducing redundant memory reads.

### Two Strategies

**Sibling fusion:** Two consumers of the same producer are fused together.

```
Before:                 After:
    P                      P
   / \                     |
  A   B                [A + B]  (multi-output)
```

**Producer-consumer fusion:** A producer is fused into its consumer when the
consumer already has multiple outputs.

### Decision Process

1. Traverse HLO in reverse post-order (use-before-def).
1. For each instruction, attempt sibling fusion first.
1. Then attempt producer-consumer fusion.
1. Each candidate is validated with:
   - `GpuPerformanceModel::EstimateRunTimesForMultiOutputFusion()`
   - `time_fused <= time_unfused` check (conservative: only fuse if not slower)

### Profitability Heuristic

The pass skips operands that are "effective scalars" (very small tensors), since
fusing them provides negligible memory savings.

## Emitter Fusion Kinds

After fusion, each fused computation is classified into an emitter kind that
determines code generation strategy:

| Kind           | Description                                        |
| -------------- | -------------------------------------------------- |
| `kLoop`        | General elementwise and loop-based operations      |
| `kReduction`   | Row, column, multi-row, or small column reductions |
| `kTranspose`   | Tiled transpose using shared memory                |
| `kScatter`     | Scatter update operations                          |
| `kConcatenate` | Concatenation operations                           |

## Inspecting Fusion Decisions

### Dump Fusion Visualization

```sh
bazel-bin/zkx/tools/stablehlo_runner/stablehlo_runner_main \
    --zkx_dump_to=/tmp/dump \
    --zkx_dump_hlo_pass_re='fusion' \
    examples/fusion_showcase/broadcast_fusion.stablehlo.mlir
```

This dumps the HLO module before and after each fusion-related pass, allowing
you to compare the graph structure and see which operations were fused.

### Reading Fusion Dumps

In the dumped HLO, fused computations appear as:

```
%fused_computation {
  %p0 = babybear_mont[] parameter(0)
  %p1 = babybear_mont[] parameter(1)
  %add = babybear_mont[] add(%p0, %p1)
  %mul = babybear_mont[] multiply(%add, %p0)
  ROOT %result = babybear_mont[] add(%mul, %p1)
}

%fusion = babybear_mont[] fusion(%a, %b), kind=kLoop, calls=%fused_computation
```

The `kind=` annotation shows the emitter that will be used. Fused computations
are named `%fused_computation`, `%fused_computation.1`, etc.

## Key Source Files

| File                                                | Purpose                    |
| --------------------------------------------------- | -------------------------- |
| `zkx/service/gpu/fusion_pipeline.cc`                | Pipeline orchestration     |
| `zkx/service/gpu/transforms/priority_fusion.cc`     | Cost-model priority fusion |
| `zkx/service/gpu/transforms/multi_output_fusion.cc` | Multi-output fusion        |
| `zkx/service/gpu/model/gpu_performance_model.cc`    | Fusion cost estimation     |
| `zkx/service/gpu/model/coalescing_analysis.cc`      | Memory coalescing analysis |
| `zkx/backends/gpu/codegen/fusions.cc`               | Emitter dispatch           |
