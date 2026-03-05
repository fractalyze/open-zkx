#!/usr/bin/env bash
# Copyright 2026 The ZKX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Unified StableHLO example runner.
#
# Usage:
#   examples/run.sh <path.stablehlo.mlir> [--dump]
#
# If a sibling .input.json file exists (same basename), it is passed via
# --input_json.  Otherwise --use_random_inputs is used.
#
# --dump writes HLO and IR to /tmp/zkx_dump_<name>/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# --- Parse arguments ---
MLIR_FILE=""
DUMP=false

for arg in "$@"; do
  case "$arg" in
    --dump) DUMP=true ;;
    --platform=*) PLATFORM="${arg#--platform=}" ;;
    *) MLIR_FILE="$arg" ;;
  esac
done

if [[ -z "$MLIR_FILE" ]]; then
  echo "Usage: examples/run.sh <path.stablehlo.mlir> [--dump]"
  exit 1
fi

# Resolve to absolute path
if [[ "$MLIR_FILE" != /* ]]; then
  MLIR_FILE="$(cd "$(dirname "$MLIR_FILE")" && pwd)/$(basename "$MLIR_FILE")"
fi

NAME="$(basename "$MLIR_FILE" .stablehlo.mlir)"
RUNNER_TARGET="//zkx/tools/stablehlo_runner:stablehlo_runner_main"

# --- Build ---
echo "=== Building stablehlo_runner_main ==="
cd "$REPO_ROOT"
bazel build --config cuda_clang "$RUNNER_TARGET"
RUNNER_BIN="$(bazel info bazel-bin)/zkx/tools/stablehlo_runner/stablehlo_runner_main"

# --- Input detection ---
INPUT_JSON="${MLIR_FILE%.stablehlo.mlir}.input.json"
INPUT_FLAGS=()
if [[ -f "$INPUT_JSON" ]]; then
  INPUT_FLAGS+=(--input_json="$INPUT_JSON")
  echo "  Using input: $(basename "$INPUT_JSON")"
else
  INPUT_FLAGS+=(--use_random_inputs)
  echo "  Using random inputs"
fi

# --- Dump flags ---
DUMP_FLAGS=()
if $DUMP; then
  DUMP_DIR="/tmp/zkx_dump_${NAME}"
  mkdir -p "$DUMP_DIR"
  DUMP_FLAGS+=(
    --dump_hlo_to="${DUMP_DIR}/${NAME}.hlo.txt"
    --zkx_dump_to="$DUMP_DIR"
    --zkx_dump_hlo_pass_re='.*'
  )
fi

# --- Run ---
echo ""
echo "=== Running: ${NAME}==="
"$RUNNER_BIN" \
  "${INPUT_FLAGS[@]}" \
  "${DUMP_FLAGS[@]}" \
  "$MLIR_FILE"

if $DUMP; then
  echo ""
  echo "=== IR dumps written to: ${DUMP_DIR}/ ==="
fi
