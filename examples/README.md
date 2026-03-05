# Examples

Runnable StableHLO programs that demonstrate the ZKX GPU compilation pipeline.

## Quick Start

```sh
# Run any example
examples/run.sh examples/field_arithmetic/add_mul.stablehlo.mlir

# Run with full IR dumps (all passes + fusion visualization)
examples/run.sh examples/field_arithmetic/add_mul.stablehlo.mlir --dump
```

## `run.sh` Usage

```
examples/run.sh <path.stablehlo.mlir> [--dump] [--platform=gpu|cpu]
```

- Builds `stablehlo_runner_main` automatically
- Detects a sibling `.input.json` file (same basename) and passes it via
  `--input_json`; falls back to `--use_random_inputs` if none exists
- `--dump` writes HLO passes, fusion visualization, and MLIR to
  `/tmp/zkx_dump_<name>/`

## Examples

| Directory             | Program                                | Demonstrates                                                                     |
| --------------------- | -------------------------------------- | -------------------------------------------------------------------------------- |
| `field_arithmetic/`   | `add_mul.stablehlo.mlir`               | BabyBear add + multiply chain → `kLoop` fusion                                   |
| `field_arithmetic/`   | `reduction.stablehlo.mlir`             | Sum-reduction of 1024 BabyBear elements → `kReduction` emitter with warp shuffle |
| `fusion_showcase/`    | `broadcast_fusion.stablehlo.mlir`      | Broadcast absorbed into elementwise chain → single `kLoop` fusion                |
| `fusion_showcase/`    | `input_fusion.stablehlo.mlir`          | Elementwise ops absorbed into reduction → single `kReduction` fusion             |
| `fusion_showcase/`    | `multi_output_fusion.stablehlo.mlir`   | Shared-input reductions merged into one multi-output kernel                      |
| `fusion_showcase/`    | `transpose_fusion.stablehlo.mlir`      | Transpose + elementwise → shared-memory tiled `kTranspose` fusion                |
| `fusion_showcase/`    | `scatter_fusion.stablehlo.mlir`        | Elementwise ops absorbed into scatter → single `kScatter` fusion                 |
| `fusion_showcase/`    | `concatenate_fusion.stablehlo.mlir`    | Concatenate + broadcast epilogue → single `kConcatenate` fusion                  |
| `poseidon2_babybear/` | `poseidon2_permutation.stablehlo.mlir` | End-to-end Poseidon2 hash permutation (field S-box, MDS, round constants)        |

## Running Every Example

```sh
examples/run.sh examples/field_arithmetic/add_mul.stablehlo.mlir
examples/run.sh examples/field_arithmetic/reduction.stablehlo.mlir
examples/run.sh examples/fusion_showcase/broadcast_fusion.stablehlo.mlir
examples/run.sh examples/fusion_showcase/concatenate_fusion.stablehlo.mlir
examples/run.sh examples/fusion_showcase/input_fusion.stablehlo.mlir
examples/run.sh examples/fusion_showcase/multi_output_fusion.stablehlo.mlir
examples/run.sh examples/fusion_showcase/scatter_fusion.stablehlo.mlir
examples/run.sh examples/fusion_showcase/transpose_fusion.stablehlo.mlir
examples/run.sh examples/poseidon2_babybear/poseidon2_permutation.stablehlo.mlir
```
