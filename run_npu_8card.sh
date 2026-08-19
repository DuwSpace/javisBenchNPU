#!/usr/bin/env bash
set -euo pipefail

# vLLM-Omni launches its diffusion workers internally; one Python process is
# required. TP=8 shards the DiT across all eight NPUs.
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARGS=(
  --input-file "${INPUT_FILE:?set INPUT_FILE to JavisBench CSV}"
  --output-dir "${OUTPUT_DIR:-samples/JavisBench}"
  --model "${MODEL:?set MODEL to a local/HuggingFace vllm-omni model}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-8}"
)
if [[ -n "${LIMIT:-}" ]]; then ARGS+=(--limit "${LIMIT}"); fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then read -r -a EXTRA <<< "${EXTRA_ARGS}"; ARGS+=("${EXTRA[@]}"); fi
if [[ "${JAVISBENCH_OFFICIAL:-0}" == "1" ]]; then ARGS+=(--javisbench-official); fi
python3 "${SCRIPT_DIR}/generate_javisbench.py" "${ARGS[@]}"
