# GPU Compilation Pipeline

This document describes the full compilation pipeline that transforms StableHLO
programs with ZK-specific types into optimized GPU kernels.

## Pipeline Overview

```
StableHLO (with ZK types)
    │
    ▼
┌─────────┐
│  HLO IR │  ← StableHLO → HLO conversion
└────┬────┘
     │
     ▼
┌──────────────────┐
│  Fusion Pipeline │  ← Priority / Multi-Output fusion
└────────┬─────────┘
     │
     ▼
┌──────────────────┐
│  MLIR (PrimeIR)  │  ← HLO → MLIR emission with ZkxGpu dialect
└────────┬─────────┘
     │
     ▼
┌───────────────────┐
│  ZK Type Lowering │  ← EllipticCurve → Field → ModArith → Arith
└────────┬──────────┘
     │
     ▼
┌──────────────────┐
│  LLVM IR         │  ← MLIR → LLVM lowering
└────────┬─────────┘
     │
     ▼
┌──────────────────┐
│  PTX             │  ← Target code generation
└────────┬─────────┘
     │
     ▼
┌──────────────────┐
│  GPU Execution   │  ← Kernel launch via Stream Executor
└──────────────────┘
```

## Stage 1: StableHLO → HLO

The entry point is a StableHLO program that uses PrimeIR type aliases for
ZK-specific types. For example, a BabyBear field element is represented as:

```mlir
!pf_babybear_mont = !field.pf<2013265921 : i32, true>
```

The StableHLO-to-HLO conversion translates standard StableHLO operations (add,
multiply, etc.) into HLO instructions while preserving the ZK primitive types
through the `PrimitiveType` enum.

**Key files:**

- `zkx/hlo/translate/stablehlo_to_hlo/` — StableHLO → HLO translation

## Stage 2: HLO Optimization and Fusion

The HLO graph undergoes optimization passes, culminating in the **fusion
pipeline** that groups operations into fused computations for efficient GPU
execution. See [fusion-passes.md](fusion-passes.md) for details.

The fusion pipeline runs inside `HloPassFix` (repeated until convergence):

1. **Priority Fusion** — Cost-model-driven producer-consumer fusion
1. **CSE + DCE** — Clean up after fusion
1. **Multi-Output Fusion** — Sibling and producer-consumer multi-output fusion
1. **CSE + DCE** — Clean up again

**Key file:** `zkx/service/gpu/fusion_pipeline.cc`

## Stage 3: Emitter Selection

Each fused computation is classified by `HloFusionAnalysis` into an
`EmitterFusionKind`, which determines the code generation strategy:

| Kind           | Emitter Class       | Strategy                         |
| -------------- | ------------------- | -------------------------------- |
| `kLoop`        | `LoopFusion`        | Elementwise / general ops        |
| `kReduction`   | `ReductionFusion`   | Row / column / multi-row / small |
| `kTranspose`   | `TransposeFusion`   | Shared-memory tiled transpose    |
| `kScatter`     | `ScatterFusion`     | Scatter update                   |
| `kConcatenate` | `ConcatenateFusion` | Concatenation fusion             |

Special case within `kLoop`:

- **InPlaceDynamicUpdateSliceFusion** — In-place DUS when buffer reuse is safe

**Key file:** `zkx/backends/gpu/codegen/fusions.cc`

## Stage 4: HLO → MLIR Emission

Each emitter translates its fused HLO computation into an MLIR module. The MLIR
module uses multiple dialects:

**Standard MLIR dialects:**

- `arith`, `scf`, `cf`, `func`, `tensor`, `vector`, `affine`, `gpu`
- `NVVM` (NVIDIA), `ROCDL` (AMD)

**PrimeIR dialects (ZK-specific):**

- `prime_ir::elliptic_curve` — Elliptic curve point operations (add, double,
  scalar multiply)
- `prime_ir::field` — Field arithmetic (add, sub, mul, constants)
- `prime_ir::mod_arith` — Modular arithmetic (add, sub, mul, reduce)
- `prime_ir::poly` — Polynomial operations

**ZKX GPU dialect:**

- `zkx_gpu.allocate_shared` — Shared memory allocation
- `zkx_gpu.sync_threads` — Thread barrier synchronization
- `zkx_gpu.shuffle_reduce` — Warp-level shuffle reduction
- `zkx_gpu.materialize` — Tensor → register materialization
- `zkx_gpu.insert` — Register → tensor insertion
- `zkx_gpu.reduce` — Block-level reduction

**Key files:**

- `zkx/backends/gpu/codegen/emitters/emitter_base.cc` — MLIR module creation and
  dialect loading
- `zkx/codegen/emitters/elemental_hlo_to_mlir.cc` — HLO → MLIR element-wise
  emission
- `zkx/backends/gpu/codegen/emitters/ir/zkx_gpu_ops.td` — ZKX GPU op definitions

## Stage 5: MLIR Pass Pipeline

The emitted MLIR module goes through three phases of passes:

### Phase 1: ZKX GPU Optimization

```
SimplifyArith → Canonicalize → CSE → EraseDeadFunctions → CSE
```

### Phase 2: Loop Transformations

```
LowerZkxToScf           (device-dependent lowering)
  → Inliner + CSE       (inline and deduplicate)
  → Canonicalize + CSE
  → PeelLoops            (loop peeling for boundary handling)
  → LowerZkxLoopsToScf   (remaining loop lowering)
  → ConvertToSignless
  → PropagateSliceIndices
  → FlattenTensors
  → LICM                 (loop invariant code motion)
  → UnswitchLoops        (hoist conditions out of loops)
  → LICM                 (second pass after unswitching)
  → VectorizeLoadsAndStores
  → OptimizeLoops         (induction variable optimization)
  → Canonicalize + CSE
```

### Phase 3: ZK Type Lowering + Final Lowering

This is where PrimeIR dialects get progressively lowered:

```
ConvertPureCallOps

── PrimeIR lowering ──────────────────────────────
EllipticCurveToField     EC point ops  → field arithmetic
FieldToModArith          field ops     → modular arithmetic
ModArithToArith          mod_arith ops → standard integer ops
EllipticCurveToLLVM      remaining EC  → LLVM (struct types)
ExtFieldToLLVM           ext field     → LLVM (struct types)
──────────────────────────────────────────────────

LowerTensors → MergePointersToSameSlice
Canonicalize → CSE → SimplifyArith → SimplifyAffine
ConvertIndexType → LowerAffine
LICM → SymbolDCE → CSE
LowerAffine → SCFToControlFlow → LowerToLLVM
ReconcileUnrealizedCasts
```

**Key file:** `zkx/backends/gpu/codegen/emitters/emitter_base.cc:590-659`

## Stage 6: LLVM IR → PTX

The final MLIR module is translated to LLVM IR using
`mlir::translateModuleToLLVMIR`, then compiled to PTX (NVIDIA) or AMDGPU (AMD)
assembly using LLVM's target backend.

**Key file:** `zkx/backends/gpu/codegen/emitters/emitter_base.cc:305-314`

## Stage 7: GPU Execution

The compiled kernel is loaded and executed through the Stream Executor
abstraction, which manages:

- Kernel loading and caching
- Buffer allocation and management
- Launch dimension configuration
- Kernel dispatch

**Key file:** `zkx/stream_executor/`

## ZK Type Flow Example

Consider a BabyBear field multiplication. The type flows through the pipeline
as:

```
StableHLO:  stablehlo.multiply %a, %b : tensor<1024x!pf_babybear_mont_std>
    │
    ▼
HLO:        multiply(p0, p1) : babybear_mont[1024]
    │
    ▼
MLIR:       field.mul %a, %b : !field.pf<2013265921 : i32>
    │
    ▼  (FieldToModArith)
MLIR:       mod_arith.mul %a, %b : !mod_arith.int<2013265921 : i32>
    │
    ▼  (ModArithToArith)
MLIR:       arith.muli %a, %b : i64   (+ modular reduction)
    │
    ▼  (LowerToLLVM)
LLVM IR:    mul i64 %a, %b  +  urem i64 ...
    │
    ▼
PTX:        mul.lo.u64 / mad.lo.u64 + reduction logic
```

## Inspecting the Pipeline

Use the following flags to dump intermediate representations at each stage:

```sh
# Dump all HLO passes (before/after each pass)
--zkx_dump_to=/tmp/dump --zkx_dump_hlo_pass_re='.*'

# Dump fusion passes only
--zkx_dump_to=/tmp/dump --zkx_dump_hlo_pass_re='fusion'
```

Example using the StableHLO runner:

```sh
bazel-bin/zkx/tools/stablehlo_runner/stablehlo_runner_main \
    --zkx_dump_to=/tmp/dump \
    --zkx_dump_hlo_pass_re='.*' \
    examples/poseidon2_babybear/poseidon2_permutation.stablehlo.mlir
```

## Key Source Files

| File                                                  | Purpose                              |
| ----------------------------------------------------- | ------------------------------------ |
| `zkx/service/gpu/fusion_pipeline.cc`                  | Fusion pass orchestration            |
| `zkx/service/gpu/transforms/priority_fusion.cc`       | Cost-model priority fusion           |
| `zkx/service/gpu/transforms/multi_output_fusion.cc`   | Multi-output fusion                  |
| `zkx/backends/gpu/codegen/fusions.cc`                 | Emitter selection dispatch           |
| `zkx/backends/gpu/codegen/emitters/emitter_base.cc`   | MLIR creation + pass pipeline        |
| `zkx/codegen/emitters/elemental_hlo_to_mlir.cc`       | HLO → MLIR element-wise emission     |
| `zkx/backends/gpu/codegen/emitters/ir/zkx_gpu_ops.td` | ZKX GPU dialect op definitions       |
| `zkx/backends/gpu/codegen/emitters/loop.cc`           | Loop fusion emitter                  |
| `zkx/backends/gpu/codegen/emitters/reduction.cc`      | Reduction emitter (4 strategies)     |
| `zkx/backends/gpu/codegen/emitters/transpose.cc`      | Shared-memory transpose emitter      |
| `zkx/mlir/mlir_utils.cc`                              | `PopulateTypeConverterWithPrimeIR()` |
| `zkx/service/gpu/model/gpu_performance_model.cc`      | Fusion cost estimation               |
| `zkx/service/gpu/model/coalescing_analysis.cc`        | Memory coalescing analysis           |
| `zkx/service/gpu/launch_dimensions.cc`                | GPU occupancy / launch dim calc      |
