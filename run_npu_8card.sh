#!/usr/bin/env bash
set -euo pipefail

# 8 cards = Ulysses 8 + TP 1. Vocoder still builds on CPU, then moves to NPU.
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARGS=(
  --input-file "${INPUT_FILE:?set INPUT_FILE to JavisBench CSV}"
  --output-dir "${OUTPUT_DIR:-samples/JavisBench}"
  --model "${MODEL:?set MODEL to a local/HuggingFace vllm-omni model}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}"
  --ulysses-degree "${ULYSSES_DEGREE:-8}"
)
if [[ "${ENABLE_CPU_OFFLOAD:-1}" == "1" ]]; then ARGS+=(--enable-cpu-offload); fi
if [[ "${VAE_USE_TILING:-1}" == "1" ]]; then ARGS+=(--vae-use-tiling); fi
if [[ -n "${LIMIT:-}" ]]; then ARGS+=(--limit "${LIMIT}"); fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then read -r -a EXTRA <<< "${EXTRA_ARGS}"; ARGS+=("${EXTRA[@]}"); fi
if [[ "${JAVISBENCH_OFFICIAL:-0}" == "1" ]]; then ARGS+=(--javisbench-official); fi
python3 "${SCRIPT_DIR}/generate_javisbench.py" "${ARGS[@]}"
