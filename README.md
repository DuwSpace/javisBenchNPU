# JavisBench NPU 生成

本仓库使用 vLLM-Omni 在 Ascend NPU 上离线生成 JavisBench 的音视频结果。
模型只加载一次，生成结果按 CSV 顺序处理。每个 MP4 采用原子重命名写入，
任务中断后重新运行会自动跳过已有的非空文件，从未完成的位置继续。

## 八卡运行

请先准备好 vLLM-Omni NPU 环境，然后设置模型、CSV 和输出目录：

```bash
export MODEL=/models/your-video-model
export INPUT_FILE=/workspace/data/eval/JavisBench/JavisBench.csv
export OUTPUT_DIR=/workspace/runtime/javisbench_ltx23
export LIMIT=1000                 # 可选；不设置时处理全部数据
export JAVISBENCH_OFFICIAL=1      # 使用 JavisBench 官方规格预设
export EXTRA_ARGS="--num-inference-steps 30 --guidance-scale 4.0"
bash run_npu_8card.sh
```

当前环境中的 LTX-2/LTX-2.3 使用注册类名 `LTX2Pipeline`，LTX-2.3 的具体组件
版本会根据模型目录元数据自动选择。

`ASCEND_RT_VISIBLE_DEVICES` 默认使用 `0,1,2,3,4,5,6,7`，可以按实际设备分配覆盖。
`TENSOR_PARALLEL_SIZE` 默认是 `8`。例如从第 400 行继续：

```bash
export EXTRA_ARGS="--start 400 --num-inference-steps 30"
bash run_npu_8card.sh
```

## JavisBench 输出规格

`JAVISBENCH_OFFICIAL=1` 使用 JavisBench 官方 `240p/9:16/约 4 秒/24 FPS` 约定。
由于 LTX-2 的 VAE 要求空间和时间维度对齐，实际采用最近的合法规格：

```text
分辨率：256x448
帧数：97
帧率：24 FPS
音频采样率：16 kHz
```

输出文件符合 JavisBench 评估器的命名约定：

```text
OUTPUT_DIR/sample_0000.mp4
OUTPUT_DIR/sample_0001.mp4
```

## 多 Prompt 批量生成

使用 `--batch-size` 可以一次提交多个 Prompt，并分别生成对应的视频。例如：

```bash
export EXTRA_ARGS="--batch-size 4 --num-inference-steps 30"
bash run_npu_8card.sh
```

这会一次提交 4 个 Prompt，并分别写入 4 个 MP4。批大小越大，吞吐量通常越高，
但 NPU 内存占用也越大；建议从 `2` 或 `4` 开始测试。

默认使用 CSV 的 `text` 列作为 Prompt。如果需要把多个 CSV 字段合并成每个样本的
一个 Prompt，可以使用：

```bash
export EXTRA_ARGS="--prompt-columns video_text,audio_text"
bash run_npu_8card.sh
```

当模型返回音频时，脚本会自动将音频封装到对应的 MP4 中。
