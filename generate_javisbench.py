#!/usr/bin/env python3
"""使用 vLLM-Omni 离线生成 JavisBench 音视频结果。

模型只加载一次，按 CSV 顺序生成。每条样本同时原子写入
`sample_XXXX.mp4` 和 `sample_XXXX.wav`；中断后重新运行会跳过两文件都已存在的样本。

`--javisbench-official` 按官方 240p / 9:16 输出 240x426。LTX 先以 32 对齐的
256x448 生成，再中心裁剪到 240p，保证评测文件分辨率就是 240p。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import wave
from pathlib import Path

import numpy as np
import torch

# JavisDiT ASPECT_RATIO_240P["0.56"] for 9:16, height x width.
JAVISBENCH_240P_HEIGHT = 240
JAVISBENCH_240P_WIDTH = 426
# LTX VAE: spatial multiple of 32, (num_frames - 1) multiple of 8.
LTX_GENERATE_HEIGHT = 256
LTX_GENERATE_WIDTH = 448
JAVISBENCH_FRAMES = 97
JAVISBENCH_FPS = 24
JAVISBENCH_AUDIO_SR = 16000


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--javisbench-official",
        action="store_true",
        help="按 JavisBench 官方 240p/9:16/约4秒/24fps 输出，并同时落盘 mp4 与 wav",
    )
    p.add_argument("--input-file", required=True, help="JavisBench 输入 CSV，通常包含 text 列")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--model-class-name", default="LTX2Pipeline")
    p.add_argument("--limit", type=int, default=None, help="最多生成多少条")
    p.add_argument("--batch-size", type=int, default=1, help="一次提交的 Prompt 数量")
    p.add_argument("--start", type=int, default=0, help="CSV 起始行，从 0 开始")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--output-height", type=int, default=None, help="落盘分辨率高；默认与生成高度相同")
    p.add_argument("--output-width", type=int, default=None, help="落盘分辨率宽；默认与生成宽度相同")
    p.add_argument("--num-frames", type=int, default=121)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--frame-rate", type=float, default=None)
    p.add_argument("--audio-sample-rate", type=int, default=48000)
    p.add_argument("--num-inference-steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--extra-body", default="{}", help="模型额外采样参数 JSON")
    p.add_argument("--enable-cpu-offload", action="store_true")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--ulysses-degree", type=int, default=8)
    p.add_argument("--ring-degree", type=int, default=1)
    p.add_argument("--vae-patch-parallel-size", type=int, default=1)
    p.add_argument("--pipeline-parallel-size", type=int, default=1)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--vae-use-tiling", action="store_true")
    return p.parse_args()


def nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def sample_done(mp4: Path, wav: Path) -> bool:
    return nonempty(mp4) and nonempty(wav)


def atomic_replace(path: Path, write_tmp) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.stem + ".", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    try:
        write_tmp(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def to_uint8_hwc(frames) -> np.ndarray:
    if isinstance(frames, torch.Tensor):
        x = frames.detach().cpu()
        if x.ndim == 5:
            x = x[0]
        if x.ndim == 4 and x.shape[0] in (3, 4):
            x = x.permute(1, 2, 3, 0)
        x = (x.clamp(-1, 1) * 0.5 + 0.5).float().numpy()
    elif isinstance(frames, np.ndarray):
        x = frames[0] if frames.ndim == 5 else frames
        if np.issubdtype(x.dtype, np.integer):
            return np.clip(x, 0, 255).astype(np.uint8)
        x = x.astype(np.float32)
    else:
        x = np.asarray(frames)
        if np.issubdtype(x.dtype, np.integer):
            return np.clip(x, 0, 255).astype(np.uint8)
    return (np.clip(x, 0, 1) * 255).round().astype(np.uint8)


def fit_frames(frames_u8: np.ndarray, height: int, width: int) -> np.ndarray:
    if frames_u8.ndim != 4:
        raise ValueError(f"expected THWC frames, got shape {frames_u8.shape}")
    _, src_h, src_w, _ = frames_u8.shape
    if src_h == height and src_w == width:
        return frames_u8
    if src_h >= height and src_w >= width:
        top = (src_h - height) // 2
        left = (src_w - width) // 2
        return frames_u8[:, top : top + height, left : left + width]
    t = torch.from_numpy(frames_u8).permute(0, 3, 1, 2).float()
    t = torch.nn.functional.interpolate(t, size=(height, width), mode="bilinear", align_corners=False)
    return t.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8).numpy()


def to_audio_cn(audio) -> np.ndarray:
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().float().numpy()
    audio = np.squeeze(np.asarray(audio)).astype(np.float32)
    if audio.ndim == 0:
        raise ValueError("empty audio")
    if audio.ndim == 1:
        return audio[None, :]
    if audio.ndim != 2:
        raise ValueError(f"unsupported audio shape {audio.shape}")
    if audio.shape[0] in (1, 2) and audio.shape[0] < audio.shape[1]:
        return audio
    return audio.T


def resample_audio(audio_cn: np.ndarray, sample_rate: int, target_rate: int) -> tuple[np.ndarray, int]:
    if sample_rate == target_rate:
        return audio_cn, sample_rate
    n_out = max(1, int(round(audio_cn.shape[1] * target_rate / float(sample_rate))))
    t = torch.from_numpy(audio_cn)[None]
    out = torch.nn.functional.interpolate(t, size=n_out, mode="linear", align_corners=False)
    return out[0].contiguous().numpy(), target_rate


def save_wav(audio_cn: np.ndarray, path: Path, sample_rate: int) -> None:
    pcm = np.clip(audio_cn, -1.0, 1.0)
    pcm = (pcm.T * 32767.0).round().astype(np.int16)

    def write_tmp(tmp: str) -> None:
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(int(audio_cn.shape[0]))
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(pcm.tobytes())

    atomic_replace(path, write_tmp)


def save_video(frames_u8: np.ndarray, path: Path, fps: int, audio_cn=None, audio_sample_rate: int = 24000) -> None:
    from diffusers.utils import export_to_video

    def write_tmp(tmp: str) -> None:
        if audio_cn is None:
            export_to_video(list(frames_u8), tmp, fps=fps)
            return
        from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes

        mux_audio = audio_cn if audio_cn.shape[0] > 1 else audio_cn[0]
        video_bytes = mux_video_audio_bytes(
            frames_u8, mux_audio, fps=float(fps), audio_sample_rate=audio_sample_rate
        )
        with open(tmp, "wb") as f:
            f.write(video_bytes)

    atomic_replace(path, write_tmp)


def unwrap(result):
    audio = None
    audio_sample_rate = 24000
    multimodal = getattr(result, "multimodal_output", None) or {}
    audio = multimodal.get("audio")
    audio_sample_rate = multimodal.get("audio_sample_rate", audio_sample_rate)
    if hasattr(result, "request_output") and result.request_output is not None:
        result = result.request_output
        multimodal = getattr(result, "multimodal_output", None) or {}
        audio = multimodal.get("audio", audio)
        audio_sample_rate = multimodal.get("audio_sample_rate", audio_sample_rate)
    if hasattr(result, "images"):
        images = result.images
        if not images:
            return None
        result = images[0]
    if isinstance(result, dict):
        audio = result.get("audio", audio)
        audio_sample_rate = result.get("audio_sample_rate", audio_sample_rate)
        result = result.get("frames") or result.get("video")
    if isinstance(result, tuple):
        result, audio = result
    return result, audio, audio_sample_rate


def main() -> None:
    a = args()
    if a.javisbench_official:
        a.height, a.width = LTX_GENERATE_HEIGHT, LTX_GENERATE_WIDTH
        a.output_height = JAVISBENCH_240P_HEIGHT
        a.output_width = JAVISBENCH_240P_WIDTH
        a.num_frames, a.fps = JAVISBENCH_FRAMES, JAVISBENCH_FPS
        a.frame_rate = float(JAVISBENCH_FPS)
        a.audio_sample_rate = JAVISBENCH_AUDIO_SR
    out_h = a.output_height if a.output_height is not None else a.height
    out_w = a.output_width if a.output_width is not None else a.width
    out = Path(a.output_dir)
    with open(a.input_file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = rows[a.start : a.start + a.limit if a.limit is not None else None]
    if not rows:
        raise SystemExit("No rows selected")

    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    omni = Omni(
        model=a.model,
        model_class_name=a.model_class_name,
        enforce_eager=a.enforce_eager,
        vae_use_tiling=a.vae_use_tiling,
        enable_cpu_offload=a.enable_cpu_offload,
        parallel_config=DiffusionParallelConfig(
            tensor_parallel_size=a.tensor_parallel_size,
            ulysses_degree=a.ulysses_degree,
            ring_degree=a.ring_degree,
            vae_patch_parallel_size=a.vae_patch_parallel_size,
            pipeline_parallel_size=a.pipeline_parallel_size,
        ),
    )
    if a.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    pending = []
    for offset, row in enumerate(rows, start=a.start):
        mp4 = out / f"sample_{offset:04d}.mp4"
        wav = out / f"sample_{offset:04d}.wav"
        if sample_done(mp4, wav):
            print(f"[{offset}] exists, skip: {mp4} {wav}", flush=True)
            continue
        prompt = (row.get("text") or "").strip()
        if prompt:
            pending.append((offset, mp4, wav, prompt))
        else:
            print(f"[{offset}] missing text, skip", flush=True)

    for batch_start in range(0, len(pending), a.batch_size):
        batch = pending[batch_start : batch_start + a.batch_size]
        print(f"generating batch rows {batch[0][0]}-{batch[-1][0]} ({len(batch)} prompts)", flush=True)
        params = OmniDiffusionSamplingParams(
            height=a.height, width=a.width, num_frames=a.num_frames,
            num_inference_steps=a.num_inference_steps,
            guidance_scale=a.guidance_scale,
            frame_rate=float(a.frame_rate or a.fps),
            generator=torch.Generator(device="npu").manual_seed(a.seed + batch[0][0]),
            extra_args=json.loads(a.extra_body),
        )
        requests = [{"prompt": item[3], "negative_prompt": a.negative_prompt} for item in batch]
        result = omni.generate(requests, params)
        outputs = result if isinstance(result, list) else [result]
        if len(outputs) != len(batch):
            raise RuntimeError(f"Batch returned {len(outputs)} outputs for {len(batch)} prompts")
        for item, output in zip(batch, outputs):
            frames, audio, audio_sample_rate = unwrap(output)
            if frames is None:
                raise RuntimeError(f"No frames returned for row {item[0]}")
            if audio is None:
                raise RuntimeError(f"No audio returned for row {item[0]}; JavisBench needs sample_XXXX.wav")
            frames_u8 = fit_frames(to_uint8_hwc(frames), out_h, out_w)
            audio_cn = to_audio_cn(audio)
            audio_cn, audio_sample_rate = resample_audio(audio_cn, int(audio_sample_rate), int(a.audio_sample_rate))
            save_wav(audio_cn, item[2], audio_sample_rate)
            save_video(frames_u8, item[1], a.fps, audio_cn, audio_sample_rate)
            print(f"[{item[0]}] saved {item[1]} {item[2]} ({out_h}x{out_w})", flush=True)


if __name__ == "__main__":
    main()
