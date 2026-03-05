# ZKX GPU: Zero Knowledge GPU Compiler

ZKX GPU is the GPU compilation backend framework for
zero-knowledge proofs. It compiles
[StableHLO](https://github.com/openxla/stablehlo) programs with ZK-specific
types into efficient GPU code using
[PrimeIR](https://github.com/fractalyze/prime-ir) as its intermediate
representation.

## Overview

This repository contains the shared compilation infrastructure and GPU backend:

- **HLO IR** — Intermediate representation for ZK computations
- **GPU Backend** — CUDA code generation, fusion pipeline, kernel emitters
- **Stream Executor** — GPU device abstraction and execution runtime
- **StableHLO Runner** — CLI tool for compiling and running StableHLO programs

## Prerequisite

1. Follow the [Bazel installation guide](https://bazel.build/install).
1. CUDA 12.x with a compatible GPU.
1. Clang 18+ (used as the host compiler).

## Build instructions

1. Clone the repo

   ```sh
   git clone https://github.com/fractalyze/open-zkx
   ```

1. Build the StableHLO runner

   ```sh
   bazel build //zkx/tools/stablehlo_runner:stablehlo_runner_main
   ```

1. Run a StableHLO program

   ```sh
   bazel-bin/zkx/tools/stablehlo_runner/stablehlo_runner_main \
     examples/field_arithmetic/add_mul.stablehlo.mlir
   ```

## Supported ZK Types

- Koalabear
- Babybear
- Mersenne31
- Goldilocks
- BN254 (scalar field, extension field, EC points)

## GPU Compilation Pipeline

ZKX GPU compiles StableHLO programs through a multi-stage pipeline:

```
StableHLO → HLO → Fusion → MLIR (PrimeIR) → LLVM IR → PTX → GPU
```

See [docs/gpu-compilation-pipeline.md](docs/gpu-compilation-pipeline.md) for the
full pipeline documentation.

### Fusion Pass

The compiler consolidates elementwise operations, reductions, and tiling-based
instructions into unified computational units using a cost-model-driven fusion
pipeline:

- **Priority Fusion** — Greedy producer-consumer fusion guided by a GPU
  performance model that estimates compute time, memory bandwidth, and
  coalescing
- **Multi-Output Fusion** — Sibling and producer-consumer fusion for shared
  operands

See [docs/fusion-passes.md](docs/fusion-passes.md) for details.

**Key files:** `zkx/service/gpu/fusion_pipeline.cc`,
`zkx/service/gpu/transforms/priority_fusion.cc`,
`zkx/service/gpu/transforms/multi_output_fusion.cc`

### GPU Kernel Lowering with PrimeIR

Fused HLO instructions are lowered into GPU kernels through PrimeIR's
domain-specific MLIR dialects:

```
EllipticCurve → Field → ModArith → Arith → LLVM → PTX
```

The PrimeIR lowering pipeline converts high-level ZK operations (field
arithmetic, elliptic curve point operations) into optimized integer arithmetic
for GPU execution.

See [docs/gpu-compilation-pipeline.md](docs/gpu-compilation-pipeline.md) for the
lowering pass details.

**Key files:** `zkx/backends/gpu/codegen/emitters/emitter_base.cc`,
`zkx/codegen/emitters/elemental_hlo_to_mlir.cc`

### Tiling and Shared Memory

Generated kernels use loop tiling and shared memory management to maximize GPU
occupancy and achieve coalesced memory access:

- **Reduction tiling** — Row/column/multi-row strategies with warp shuffle and
  shared memory accumulation
- **Transpose tiling** — Shared memory tiles with bank-conflict-avoiding padding
- **Loop optimization** — Vectorized loads/stores, loop peeling, induction
  variable optimization
- **Coalescing analysis** — Memory access pattern analysis for the cost model

See [docs/tiling-and-memory.md](docs/tiling-and-memory.md) for details.

**Key files:** `zkx/service/gpu/reduction_utils.cc`,
`zkx/backends/gpu/codegen/emitters/reduction.cc`,
`zkx/backends/gpu/codegen/emitters/transpose.cc`,
`zkx/backends/gpu/codegen/emitters/transforms/`

## Examples

- [**Field Arithmetic**](examples/field_arithmetic/) — BabyBear field add/mul
  and reduction
- [**Poseidon2 BabyBear**](examples/poseidon2_babybear/) — Poseidon2 hash
  permutation (end-to-end ZK workload)

## Community

Building a substantial ZK compiler requires collaboration across the broader ZK
ecosystem — and we'd love your help in shaping ZKX. See
[CONTRIBUTING.md](https://github.com/fractalyze/.github/blob/main/CONTRIBUTING.md)
for more details.

## License

[Apache License 2.0](LICENSE)
