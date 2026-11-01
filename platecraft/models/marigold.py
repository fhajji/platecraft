"""Marigold Normals v1-1 adapter: per-frame normals with a commercially usable license.

Runs diffusers' MarigoldNormalsPipeline (in the normalcrafter env) on each plate frame
at the inference size. Every frame starts from the same noise, which keeps the
per-frame results as similar as possible; the remaining flicker is what
--temporal-filter (RAFT) removes.

Raw output (job.out_dir): 16-bit PNG, inference resolution, camera-space normals
+X right, +Y up, +Z toward the camera (same convention as NormalCrafter), n * 0.5 + 0.5.
Frames are independent, so resume = skip the frames that exist.
"""
from __future__ import annotations

from pathlib import Path

from .. import paths
from .base import Job, spawn_worker, worker_main, worker_progress, worker_result
from .registry import MODELS

INFO = MODELS["marigold-n"]
# Marigold's output axes relative to ours (+X right, +Y up, +Z to camera); measured on kitty.
AXES = (1.0, 1.0, 1.0)


def job_settings(preset: dict) -> dict:
    return {"steps": 4, "batch": 4 if preset["max_res"] <= 768 else 2, "seed": 0, "axes": list(AXES)}


def run(job: Job) -> dict:
    job.settings = {**job.settings, "weights": str(paths.MODELS / "marigold-n")}
    return spawn_worker(INFO, __name__, job)


# ------------------------------------------------------------ worker side --

def _work(job: Job) -> None:
    import time
    from concurrent.futures import ThreadPoolExecutor

    from .. import frames
    import cv2
    import numpy as np
    import torch
    from diffusers import MarigoldNormalsPipeline

    s = job.settings
    out_dir, in_dir = Path(job.out_dir), Path(job.input_dir)
    start, end = job.frames
    w, h = job.infer_size
    total = end - start
    todo = [i for i in range(start, end) if not frames.frame_path(out_dir, i).is_file()]
    if not todo:
        worker_progress(total, total, 0)
        worker_result(frames=0, skipped=total)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the normalcrafter env")

    t_load = time.time()
    pipe = MarigoldNormalsPipeline.from_pretrained(s["weights"], variant="fp16", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    try:
        pipe.enable_xformers_memory_efficient_attention()
        xformers = True
    except Exception as e:  # noqa: BLE001 - optional speed-up
        print(f"xformers not used: {e}")
        xformers = False
    pipe.set_progress_bar_config(disable=True)
    t_load = time.time() - t_load
    torch.cuda.reset_peak_memory_stats()

    g = torch.Generator("cuda").manual_seed(s["seed"])
    noise = torch.randn((1, 4, -(-h // 8), -(-w // 8)), generator=g, device="cuda", dtype=torch.float16)
    axes = np.array(s["axes"], np.float32)

    def load(i: int) -> np.ndarray:
        img = frames.read(frames.frame_path(in_dir, i))[..., :3]
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    io = ThreadPoolExecutor(max_workers=8)
    done_before = total - len(todo)
    written = 0
    t_inf = time.time()
    b = s["batch"]
    for k in range(0, len(todo), b):
        idx = todo[k:k + b]
        imgs = torch.from_numpy(np.stack(list(io.map(load, idx))).transpose(0, 3, 1, 2)).cuda()  # (N, 3, h, w) 0..1
        with torch.inference_mode():
            out = pipe(imgs, num_inference_steps=s["steps"], ensemble_size=1, processing_resolution=max(w, h),
                       match_input_resolution=True, batch_size=len(idx), latents=noise.repeat(len(idx), 1, 1, 1),
                       output_type="np").prediction            # (N, h, w, 3) in -1..1
        n = out.astype(np.float32) * axes
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
        list(io.map(lambda j: frames.write(frames.frame_path(out_dir, idx[j]), n[j] * 0.5 + 0.5, 16),
                    range(len(idx))))
        written += len(idx)
        worker_progress(done_before + written, total, written)
    io.shutdown()
    t_inf = time.time() - t_inf
    worker_result(frames=written, skipped=done_before, load_s=round(t_load, 1), infer_s=round(t_inf, 1),
                  fps=round(written / max(t_inf, 1e-6), 3), infer_size=[w, h],
                  peak_alloc_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                  peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2),
                  device=torch.cuda.get_device_name(0), xformers=xformers)


if __name__ == "__main__":
    worker_main(_work)
