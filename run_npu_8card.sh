#!/usr/bin/env bash
set -euo pipefail

# Match the known-good LTX-2.3 NPU recipe:
# 8 cards = Ulysses 4 + TP 2, vocoder constructed on CPU, then moved to NPU.
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARGS=(
  --input-file "${INPUT_FILE:?set INPUT_FILE to JavisBench CSV}"
  --output-dir "${OUTPUT_DIR:-samples/JavisBench}"
  --model "${MODEL:?set MODEL to a local/HuggingFace vllm-omni model}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-2}"
  --ulysses-degree "${ULYSSES_DEGREE:-4}"
)
if [[ "${ENABLE_CPU_OFFLOAD:-1}" == "1" ]]; then ARGS+=(--enable-cpu-offload); fi
if [[ "${VAE_USE_TILING:-1}" == "1" ]]; then ARGS+=(--vae-use-tiling); fi
if [[ -n "${LIMIT:-}" ]]; then ARGS+=(--limit "${LIMIT}"); fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then read -r -a EXTRA <<< "${EXTRA_ARGS}"; ARGS+=("${EXTRA[@]}"); fi
if [[ "${JAVISBENCH_OFFICIAL:-0}" == "1" ]]; then ARGS+=(--javisbench-official); fi
python3 "${SCRIPT_DIR}/generate_javisbench.py" "${ARGS[@]}"
