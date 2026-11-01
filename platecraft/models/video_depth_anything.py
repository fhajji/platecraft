"""Video Depth Anything adapter (vda-s, vda-l): temporally consistent relative depth.

Raw output (job.out_dir): float32 EXR, model resolution (short side 518, multiples of 14),
relative inverse depth (disparity: larger = nearer) on one scale for the whole run.
postprocess.py normalises it clip-wide and upscales it.

This re-implements upstream's `infer_video_depth` as a stream: 32-frame windows that
advance 22 frames, each window re-feeding 10 key frames of the previous one (always
including the run's first frame), scale/shift-aligned to the previous window and
crossfaded over 8 frames. Frames are written as soon as no later window can change
them; the state needed by the next window (8 pending frames + 1 reference depth) is
kept in vda_state.npz, so an interrupted run continues with identical results.
"""
from __future__ import annotations

import math
from pathlib import Path

from .. import paths
from .base import Job, spawn_worker, worker_main, worker_progress, worker_result
from .registry import MODELS

RAW_EXT = ".exr"
CONFIGS = {
    "vda-s": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384],
              "weights": "vda-s/video_depth_anything_vits.pth"},
    "vda-l": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024],
              "weights": "vda-l/video_depth_anything_vitl.pth"},
}
INPUT_SIZE = 518   # the model's training resolution (short side)


def infer_size(w: int, h: int, preset: dict) -> tuple[int, int]:
    """Upstream's resize: short side 518 (less for very wide frames), both sides multiples of 14."""
    size = INPUT_SIZE
    ratio = max(w, h) / min(w, h)
    if ratio > 1.78:
        size = round(int(size * 1.777 / ratio) / 14) * 14
    s = size / min(w, h)
    return math.ceil(w * s / 14) * 14, math.ceil(h * s / 14) * 14


def job_settings(preset: dict) -> dict:
    return {"input_size": INPUT_SIZE, "fp32": False}


def run(job: Job) -> dict:
    cfg = CONFIGS[job.model]
    job.settings = {**job.settings, **{k: v for k, v in cfg.items() if k != "weights"},
                    "weights": str(paths.MODELS / cfg["weights"])}
    return spawn_worker(MODELS[job.model], __name__, job)


# ------------------------------------------------------------ worker side --

def _work(job: Job) -> None:
    import json
    import time
    from concurrent.futures import ThreadPoolExecutor

    from .. import frames   # before cv2 elsewhere: enables OpenEXR
    import cv2
    import numpy as np
    import torch
    from utils.util import compute_scale_and_shift, get_interpolate_frames
    from video_depth_anything.video_depth import (INFER_LEN, INTERP_LEN, KEYFRAMES, OVERLAP,
                                                  VideoDepthAnything)

    s = job.settings
    out_dir, in_dir = Path(job.out_dir), Path(job.input_dir)
    start, end = job.frames
    w, h = job.infer_size
    step = INFER_LEN - OVERLAP                       # 22
    align_kf = KEYFRAMES[:OVERLAP - INTERP_LEN]      # [0, 12]: positions used for scale/shift
    total = end - start

    def raw(i: int) -> Path:
        return frames.frame_path(out_dir, i, RAW_EXT)

    done = {i for i in range(start, end) if raw(i).is_file()}
    if len(done) == total:
        worker_progress(total, total, 0)
        worker_result(frames=0, skipped=total)
        return

    # Continue a chain only if it was cut short with this exact range; otherwise start over at `start`.
    state_file = out_dir / "vda_state.npz"
    state = None
    if state_file.is_file():
        with np.load(state_file) as st:     # closes the file (Windows can't replace an open one)
            meta = json.loads(str(st["meta"]))
            if meta["anchor"] <= start and meta["end"] == end and meta["next"] < end:
                state = (meta, st["tail"].copy(), st["ref"].copy())
    if state is None:
        anchor, s0, tail, refs = start, start, None, None
    else:
        meta, tail, ref1 = state
        anchor, s0 = meta["anchor"], meta["next"]
        tail = list(tail)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the vda env")

    t_load = time.time()
    model = VideoDepthAnything(encoder=s["encoder"], features=s["features"], out_channels=s["out_channels"])
    model.load_state_dict(torch.load(s["weights"], map_location="cpu", weights_only=True), strict=True)
    model = model.to("cuda").eval()
    t_load = time.time() - t_load
    torch.cuda.reset_peak_memory_stats()

    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)

    def load(i: int) -> np.ndarray:
        """Model input for clip frame i (past the end: the last frame, as upstream pads)."""
        img = frames.read(frames.frame_path(in_dir, min(i, end - 1)))[..., :3]
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        return ((img - mean) / std).transpose(2, 0, 1)

    if state is not None:
        refs = [frames.read(raw(anchor)), ref1]
    io = ThreadPoolExecutor(max_workers=8)
    anchor_input = load(anchor)
    written = 0
    t_inf = time.time()

    def save(indices, depths) -> int:
        todo = [(i, d) for i, d in zip(indices, depths) if start <= i < end]
        list(io.map(lambda a: frames.write_exr(raw(a[0]), a[1]), todo))
        return sum(1 for i, _ in todo if i not in done)

    s_win = s0
    while s_win < end:
        first = state is None and s_win == anchor
        # Window positions: 0 = run anchor, 1 = previous window's position 12, 2..9 = previous
        # window's 24..31 (= s+2..s+9), 10..31 = new frames. The first window is plain s..s+31.
        if first:
            idx = list(range(s_win, s_win + INFER_LEN))
            inputs = [anchor_input] + list(io.map(load, idx[1:]))
        else:
            idx = [anchor, s_win - OVERLAP] + list(range(s_win + 2, s_win + INFER_LEN))
            inputs = [anchor_input] + list(io.map(load, idx[1:]))
        x = torch.from_numpy(np.stack(inputs))[None].to("cuda")
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=not s["fp32"]):
            d = model.forward(x)
        d = d[0].float().cpu().numpy()                 # (32, h, w)

        if first:
            aligned = list(d)
            refs = [d[0], d[12]]
            written += save(range(s_win, s_win + INFER_LEN - INTERP_LEN), aligned[:INFER_LEN - INTERP_LEN])
            tail = aligned[INFER_LEN - INTERP_LEN:]
        else:
            if s.get("metric"):
                scale, shift = 1.0, 0.0
            else:
                scale, shift = compute_scale_and_shift(np.concatenate([d[k] for k in range(len(align_kf))]),
                                                       np.concatenate(refs), np.ones_like(np.concatenate(refs)))
            fix = lambda a: np.maximum(a * scale + shift, 0.0)   # noqa: E731
            post = [fix(d[k]) for k in range(len(align_kf), OVERLAP)]
            blended = get_interpolate_frames(tail, post)       # previous window's last 8 frames
            new = [fix(d[k]) for k in range(OVERLAP, INFER_LEN)]
            written += save(range(s_win + 2, s_win + OVERLAP), blended)
            written += save(range(s_win + OVERLAP, s_win + INFER_LEN - INTERP_LEN), new[:-INTERP_LEN])
            tail = new[-INTERP_LEN:]
            refs = [refs[0], fix(d[12])]

        nxt = s_win + step
        if nxt >= end:   # last window: its pending frames are final
            written += save(range(s_win + INFER_LEN - INTERP_LEN, s_win + INFER_LEN), tail)
        meta = {"anchor": anchor, "end": end, "next": nxt}
        tmp = state_file.with_name("vda_state.tmp.npz")
        np.savez(tmp, meta=json.dumps(meta), tail=np.stack(tail), ref=refs[1])
        tmp.replace(state_file)
        state = None
        s_win = nxt
        worker_progress(min(total, len(done) + written), total, written)
    io.shutdown()
    t_inf = time.time() - t_inf
    worker_result(frames=written, skipped=len(done), load_s=round(t_load, 1), infer_s=round(t_inf, 1),
                  fps=round(written / max(t_inf, 1e-6), 3), infer_size=[w, h],
                  peak_alloc_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                  peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2),
                  device=torch.cuda.get_device_name(0))


if __name__ == "__main__":
    worker_main(_work)
