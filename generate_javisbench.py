#!/usr/bin/env python3
"""Offline JavisBench generation with vLLM-Omni.

The model is initialized once and requests are processed in CSV order. A
completed sample is committed with an atomic rename, so rerunning this
program after an interruption simply skips finished samples.
"""
from __future__ import annotations

import argparse
import csv
import os
import tempfile
import json
from pathlib import Path

import numpy as np
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--javisbench-official",
        action="store_true",
        help="Use JavisBench official 240p/9:16/4s/24fps settings, rounded to LTX-compatible dimensions/frames.",
    )
    p.add_argument("--input-file", required=True, help="JavisBench CSV containing a text column")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--model-class-name", default="LTX2Pipeline")
    p.add_argument("--limit", type=int, default=None, help="Generate at most this many rows")
    p.add_argument("--start", type=int, default=0, help="CSV row offset (zero based)")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--num-frames", type=int, default=121)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--frame-rate", type=float, default=None)
    p.add_argument("--audio-sample-rate", type=int, default=48000)
    p.add_argument("--num-inference-steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--extra-body", default="{}", help="JSON object of model-specific sampling parameters")
    p.add_argument("--enable-cpu-offload", action="store_true")
    p.add_argument("--tensor-parallel-size", type=int, default=8)
    p.add_argument("--ulysses-degree", type=int, default=1)
    p.add_argument("--ring-degree", type=int, default=1)
    p.add_argument("--vae-patch-parallel-size", type=int, default=1)
    p.add_argument("--pipeline-parallel-size", type=int, default=1)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--vae-use-tiling", action="store_true")
    return p.parse_args()


def save_video(frames, path: Path, fps: int, audio=None, audio_sample_rate: int = 24000) -> None:
    from diffusers.utils import export_to_video

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
            x = x.astype(np.float32) / 255.0
    else:
        x = frames
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.stem + ".", suffix=".mp4", dir=path.parent)
    os.close(fd)
    try:
        video_frames = list(x) if isinstance(x, np.ndarray) else x
        if audio is None:
            export_to_video(video_frames, tmp, fps=fps)
        else:
            from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes

            video_array = np.asarray(video_frames)
            if np.issubdtype(video_array.dtype, np.integer):
                frames_u8 = video_array.astype(np.uint8)
            else:
                frames_u8 = (np.clip(video_array, 0, 1) * 255).round().astype(np.uint8)
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().float().numpy()
            audio_array = np.squeeze(np.asarray(audio)).astype(np.float32)
            video_bytes = mux_video_audio_bytes(
                frames_u8, audio_array, fps=float(fps), audio_sample_rate=audio_sample_rate
            )
            with open(tmp, "wb") as f:
                f.write(video_bytes)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def unwrap(result):
    audio = None
    audio_sample_rate = 24000
    if isinstance(result, list):
        result = result[0] if result else None
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
        # JavisBench's native bucket is 240x426 and 102 frames. LTX-2 requires
        # spatial multiples of 32 and (frames - 1) divisible by 8, so use the
        # nearest valid bucket while preserving the official aspect/duration.
        a.height, a.width = 256, 448
        a.num_frames, a.fps = 97, 24
        a.frame_rate = 24.0
        a.audio_sample_rate = 16000
    out = Path(a.output_dir)
    with open(a.input_file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = rows[a.start : a.start + a.limit if a.limit is not None else None]
    if not rows:
        raise SystemExit("No rows selected")

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
    for offset, row in enumerate(rows, start=a.start):
        sample_id = offset
        if row.get("id", "").isdigit():
            sample_id = int(row["id"])
        target = out / f"sample_{sample_id:04d}.mp4"
        if target.exists() and target.stat().st_size > 0:
            print(f"[{offset}] exists, skip: {target}", flush=True)
            continue
        prompt = row.get("text") or row.get("prompt") or row.get("caption")
        if not prompt:
            print(f"[{offset}] missing text, skip", flush=True)
            continue
        print(f"[{offset}/{a.start + len(rows) - 1}] generating {target.name}", flush=True)
        params = OmniDiffusionSamplingParams(
            height=a.height, width=a.width, num_frames=a.num_frames,
            num_inference_steps=a.num_inference_steps,
            guidance_scale=a.guidance_scale,
            frame_rate=float(a.frame_rate or a.fps),
            generator=torch.Generator(device="npu").manual_seed(a.seed + sample_id),
            extra_args=json.loads(a.extra_body),
        )
        result = omni.generate({"prompt": prompt, "negative_prompt": a.negative_prompt}, params)
        frames, audio, audio_sample_rate = unwrap(result)
        if frames is None:
            raise RuntimeError(f"No frames returned for row {offset}")
        save_video(frames, target, a.fps, audio, audio_sample_rate)
        print(f"[{offset}] saved {target}", flush=True)


if __name__ == "__main__":
    main()
