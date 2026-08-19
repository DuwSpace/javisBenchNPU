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
from pathlib import Path

import numpy as np
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-file", required=True, help="JavisBench CSV containing a text column")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--model-class-name", default=None)
    p.add_argument("--limit", type=int, default=None, help="Generate at most this many rows")
    p.add_argument("--start", type=int, default=0, help="CSV row offset (zero based)")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--num-frames", type=int, default=121)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--num-inference-steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--tensor-parallel-size", type=int, default=8)
    p.add_argument("--ulysses-degree", type=int, default=1)
    p.add_argument("--ring-degree", type=int, default=1)
    p.add_argument("--vae-patch-parallel-size", type=int, default=1)
    p.add_argument("--pipeline-parallel-size", type=int, default=1)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--vae-use-tiling", action="store_true")
    return p.parse_args()


def save_video(frames, path: Path, fps: int) -> None:
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
        export_to_video(list(x) if isinstance(x, np.ndarray) else x, tmp, fps=fps)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def unwrap(result):
    if isinstance(result, list):
        result = result[0] if result else None
    if hasattr(result, "request_output") and result.request_output is not None:
        result = result.request_output
    if hasattr(result, "images"):
        images = result.images
        if not images:
            return None
        result = images[0]
    if isinstance(result, dict):
        return result.get("frames") or result.get("video")
    if isinstance(result, tuple):
        return result[0]
    return result


def main() -> None:
    a = args()
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
            generator=torch.Generator(device="npu").manual_seed(a.seed + sample_id),
        )
        result = omni.generate({"prompt": prompt, "negative_prompt": a.negative_prompt}, params)
        frames = unwrap(result)
        if frames is None:
            raise RuntimeError(f"No frames returned for row {offset}")
        save_video(frames, target, a.fps)
        print(f"[{offset}] saved {target}", flush=True)


if __name__ == "__main__":
    main()
