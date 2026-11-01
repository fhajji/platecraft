"""Normals from depth: the commercial-safe fallback when no normal model is installed.

Runs in the CLI process (no model, no worker): reads the depth model's raw disparity,
normalises it clip-wide and takes its gradients as a height field. Smooth and
temporally as stable as the depth, but without fine detail; the pass is marked heuristic.

Raw output (job.out_dir): 16-bit PNG at the depth model's resolution, camera-space
normals (+X right, +Y up, +Z toward the camera), n * 0.5 + 0.5.
"""
from __future__ import annotations

from pathlib import Path

from .base import Job

RELIEF = 0.3   # the subject's full depth range, as a fraction of the frame width


def job_settings(preset: dict) -> dict:
    return {"relief": RELIEF}


def run(job: Job) -> dict:
    import time

    import numpy as np

    from .. import frames, postprocess

    s = job.settings
    src, out = Path(s["depth_raw"]), Path(job.out_dir)
    alpha = Path(s["alpha_dir"]) if s.get("alpha_dir") else None
    start, end = job.frames
    idx = [i for i in range(start, end) if frames.frame_path(src, i, s["depth_ext"]).is_file()]
    if not idx:
        raise RuntimeError(f"no depth frames in {src}: run a depth model first (e.g. --model vda-s,depth-normals)")
    lo, hi = postprocess.depth_range(src, alpha, idx, s["depth_ext"])
    t0 = time.time()

    def one(j: int) -> None:
        i = idx[j]
        d = np.clip((frames.read(frames.frame_path(src, i, s["depth_ext"])) - lo) / (hi - lo), 0.0, 1.0)
        frames.write(frames.frame_path(out, i), postprocess.normals_from_depth(d, s["relief"]) * 0.5 + 0.5, 16)

    frames.parallel(one, len(idx), "depth-normals")
    t = time.time() - t0
    return {"frames": len(idx), "infer_s": round(t, 1), "fps": round(len(idx) / max(t, 1e-6), 2),
            "peak_alloc_gb": 0, "peak_reserved_gb": 0}
