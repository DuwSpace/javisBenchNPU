# JavisBench NPU generation

This repository provides offline generation for JavisBench with vLLM-Omni on
Ascend NPU. The model is loaded once; generation is sequential and completed
MP4 files are atomically renamed into place. Re-running the command after an
interruption skips existing non-empty files and continues from the remaining
rows.

## Run on eight NPUs

Install a vLLM-Omni NPU environment first, then set the model and CSV paths:

```bash
export MODEL=/models/your-video-model
export INPUT_FILE=/data/eval/JavisBench/JavisBench.csv
export OUTPUT_DIR=/data/samples/JavisBench
export LIMIT=1000                 # optional; omit to process all rows
export EXTRA_ARGS="--height 512 --width 768 --num-frames 121 --num-inference-steps 30"
bash run_npu_8card.sh
```

For the LTX-2/LTX-2.3 checkpoint in the NPU environment, the registered class
name is `LTX2Pipeline`; LTX-2.3 is selected from checkpoint metadata.

`ASCEND_RT_VISIBLE_DEVICES` defaults to `0,1,2,3,4,5,6,7`. Override it when
using a different eight-device allocation. `TENSOR_PARALLEL_SIZE` defaults to
8. To resume a selected range, set `EXTRA_ARGS="--start 400"`; existing
`sample_XXXX.mp4` files are always skipped regardless of the range.

Outputs follow the evaluator convention:

```text
OUTPUT_DIR/sample_0000.mp4
OUTPUT_DIR/sample_0001.mp4
```

The script uses the CSV `text` column (falling back to `prompt` or `caption`).
When the checkpoint returns audio through vLLM-Omni's multimodal output, it is
automatically muxed into the MP4 file.
