# JavisBench NPU 生成

本仓库使用 vLLM-Omni 在 Ascend NPU 上离线生成 JavisBench 的音视频结果。
模型只加载一次，按 CSV 顺序处理。每条样本同时原子写入 MP4 和 WAV；
任务中断后重新运行会跳过已经同时存在的非空 `sample_XXXX.mp4` 和 `sample_XXXX.wav`。

## 八卡运行

请先准备好 vLLM-Omni NPU 环境，然后设置模型、CSV 和输出目录：

```bash
export MODEL=/models/your-video-model
export INPUT_FILE=/workspace/data/eval/JavisBench/JavisBench.csv
export OUTPUT_DIR=/workspace/runtime/javisbench_ltx23
export LIMIT=1000                 # 可选；不设置时处理全部数据
export JAVISBENCH_OFFICIAL=1      # 使用 JavisBench 官方 240p 规格
export EXTRA_ARGS="--num-inference-steps 30 --guidance-scale 4.0"
bash run_npu_8card.sh
```

默认 8 卡并行是 `Ulysses=8` + `TP=1`，并打开 `--enable-cpu-offload`、`--vae-use-tiling`。
LTX-2.3 视频/音频注意力头数都是 32，可以被 8 整除。
Ascend 上 LTX vocoder 的 `kaiser_window` 没有 NPU 核，必须先在 CPU 上构建。
若要改回官方示例的 `Ulysses=4` + `TP=2`：

```bash
export ULYSSES_DEGREE=4
export TENSOR_PARALLEL_SIZE=2
```

当前环境中的 LTX-2/LTX-2.3 使用注册类名 `LTX2Pipeline`，LTX-2.3 的具体组件
版本会根据模型目录元数据自动选择。

`ASCEND_RT_VISIBLE_DEVICES` 默认使用 `0,1,2,3,4,5,6,7`。例如从第 400 行继续：

```bash
export EXTRA_ARGS="--start 400 --num-inference-steps 30"
bash run_npu_8card.sh
```

## JavisBench 输出规格

`JAVISBENCH_OFFICIAL=1` 按官方 `240p / 9:16 / 约 4 秒 / 24 FPS` 落盘。
JavisDiT 的 240p 9:16 桶是 `240x426`（高 x 宽）。LTX VAE 要求空间边长为 32 的倍数，
因此先以 `256x448` 生成，再中心裁剪到官方 240p，输出文件分辨率保证是 `240x426`。

```text
落盘分辨率：240x426
生成分辨率：256x448（仅推理，不作为评测文件）
帧数：97
帧率：24 FPS
音频采样率：16 kHz
```

评测器要求成对文件：

```text
OUTPUT_DIR/sample_0000.mp4
OUTPUT_DIR/sample_0000.wav
OUTPUT_DIR/sample_0001.mp4
OUTPUT_DIR/sample_0001.wav
```

只有 MP4 不够。官方 `gather_audio_video_pred()` 会同时检查 `.mp4` 和 `.wav`。
音频指标从独立 WAV 读取，不会从 MP4 音轨拆音频。

## 样例

仓库中的冒烟样例在：

```text
samples/batch4_smoke/sample_0000.mp4
samples/batch4_smoke/sample_0000.wav
samples/batch4_smoke/sample_0001.mp4
samples/batch4_smoke/sample_0001.wav
samples/batch4_smoke/sample_0002.mp4
samples/batch4_smoke/sample_0002.wav
samples/batch4_smoke/sample_0003.mp4
samples/batch4_smoke/sample_0003.wav
```

完整 1,000 / 10,140 条结果体积很大，不放进 Git。

## 多 Prompt 批量生成

使用 `--batch-size` 可以一次提交多个 Prompt，并分别生成对应的视频和音频。例如：

```bash
export EXTRA_ARGS="--batch-size 4 --num-inference-steps 30"
bash run_npu_8card.sh
```

这会一次提交 4 个 Prompt，并分别写入 4 对 MP4/WAV。批大小越大，吞吐量通常越高，
但 NPU 内存占用也越大；建议从 `2` 或 `4` 开始测试。

Prompt 固定使用官方 CSV 的 `text` 列，文件按 CSV 行号命名为 `sample_0000.mp4` / `sample_0000.wav`。
