"""--temporal-filter: optical-flow guided smoothing of per-frame normals (RAFT, torchvision).

Recursive filter, frame by frame:
    out[t] = normalize(w * warp(out[t-1]) + (1 - w) * raw[t])
where warp follows the RAFT flow from frame t back to t-1 (computed on the plate), and
w = `strength` where the flow is reliable (forward/backward consistent, inside the
subject in both frames) and 0 elsewhere, so disocclusions take the fresh frame.
Detail that moves with the footage is carried along by the warp; detail that changes
from frame to frame (Marigold's fur) is averaged away, so higher strength = steadier
but softer. Measured on kitty (Marigold, flow-compensated change per frame):
raw 0.077, strength 0.7 0.036, 0.8 0.029, 0.9 0.022 (NormalCrafter 0.025).

Input: raw normals (16-bit PNG, n * 0.5 + 0.5) of a normal model. Output (job.out_dir):
the same format. Resume: out[first missing - 1] is read back from disk.
"""
from __future__ import annotations

from pathlib import Path

from .base import Job, spawn_worker, worker_main, worker_progress, worker_result
from .registry import MODELS

INFO = MODELS["raft"]


def job_settings(strength: float) -> dict:
    return {"strength": strength, "flow_iters": 12, "consistency_px": 1.0}


def run(job: Job) -> dict:
    return spawn_worker(INFO, __name__, job)


# ------------------------------------------------------------ worker side --

def _work(job: Job) -> None:
    import time
    from concurrent.futures import ThreadPoolExecutor

    from .. import frames
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    s = job.settings
    out_dir, in_dir, src = Path(job.out_dir), Path(job.input_dir), Path(s["source_dir"])
    alpha_dir = Path(s["alpha_dir"]) if s.get("alpha_dir") else None
    start, end = job.frames
    w, h = job.infer_size
    total = end - start
    first = next((i for i in range(start, end) if not frames.frame_path(out_dir, i).is_file()), end)
    if first >= end:
        worker_progress(total, total, 0)
        worker_result(frames=0, skipped=total)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the vda env")

    t_load = time.time()
    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to("cuda").eval()
    t_load = time.time() - t_load
    torch.cuda.reset_peak_memory_stats()
    pw, ph = -(-w // 8) * 8, -(-h // 8) * 8        # RAFT needs multiples of 8

    def plate(i: int) -> torch.Tensor:
        img = cv2.resize(frames.read(frames.frame_path(in_dir, i))[..., :3], (pw, ph), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(img.transpose(2, 0, 1) * 2 - 1)

    def alpha(i: int) -> torch.Tensor:
        if alpha_dir is None:
            return torch.ones((h, w))
        return torch.from_numpy(cv2.resize(frames.read(frames.frame_path(alpha_dir, i)), (w, h),
                                           interpolation=cv2.INTER_AREA))

    def normal(folder: Path, i: int) -> torch.Tensor:
        return torch.from_numpy((frames.read(frames.frame_path(folder, i))[..., :3] * 2 - 1).transpose(2, 0, 1))

    ys, xs = torch.meshgrid(torch.arange(h, device="cuda"), torch.arange(w, device="cuda"), indexing="ij")

    def warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Sample img (C, h, w) at p + flow(p)."""
        gx = (xs + flow[0]) / (w - 1) * 2 - 1
        gy = (ys + flow[1]) / (h - 1) * 2 - 1
        grid = torch.stack([gx, gy], -1)[None]
        return F.grid_sample(img[None], grid, mode="bilinear", padding_mode="border", align_corners=True)[0]

    def flows(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Flow a->b and b->a at (h, w) in pixels."""
        x1 = torch.stack([a, b]).cuda()
        x2 = torch.stack([b, a]).cuda()
        with torch.no_grad():
            f = model(x1, x2, num_flow_updates=s["flow_iters"])[-1]
        if (pw, ph) != (w, h):
            f = F.interpolate(f, (h, w), mode="bilinear", align_corners=False)
            f = f * torch.tensor([w / pw, h / ph], device="cuda")[None, :, None, None]
        return f[0], f[1]

    io = ThreadPoolExecutor(max_workers=4)
    written = 0
    valid = []      # share of the subject that took the warped history
    t_inf = time.time()
    if first > start:
        prev_out = normal(out_dir, first - 1).cuda()
        prev_plate, prev_alpha = plate(first - 1), alpha(first - 1).cuda()
    else:
        prev_out = None
    nxt = io.submit(lambda i: (plate(i), alpha(i), normal(src, i)), first)
    for i in range(first, end):
        cur_plate, cur_alpha, cur = nxt.result()
        if i + 1 < end:
            nxt = io.submit(lambda k: (plate(k), alpha(k), normal(src, k)), i + 1)
        cur, cur_alpha = cur.cuda(), cur_alpha.cuda()
        if prev_out is None:
            out = cur
        else:
            back, fwd = flows(cur_plate, prev_plate)            # t -> t-1, t-1 -> t
            warped = warp(prev_out, back)
            # forward/backward check: following back then fwd should return to the start
            err = (back + warp(fwd, back)).norm(dim=0)
            ok = (err < s["consistency_px"] + 0.05 * back.norm(dim=0)).float()
            ok *= (cur_alpha > 0.5).float() * (warp(prev_alpha[None], back)[0] > 0.5).float()
            ok = F.avg_pool2d(ok[None, None], 5, 1, 2)[0, 0]     # soften the mask edges
            wgt = s["strength"] * ok
            valid.append(float((ok * (cur_alpha > 0.5)).sum() / (cur_alpha > 0.5).sum().clamp_min(1)))
            out = wgt * warped + (1 - wgt) * cur
            out = out / out.norm(dim=0, keepdim=True).clamp_min(1e-6)
        res = (out.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).astype(np.float32)
        io.submit(frames.write, frames.frame_path(out_dir, i), res, 16)
        prev_out, prev_plate, prev_alpha = out, cur_plate, cur_alpha
        written += 1
        if written % 4 == 0 or i == end - 1:
            worker_progress(first - start + written, total, written)
    io.shutdown(wait=True)
    t_inf = time.time() - t_inf
    worker_result(frames=written, skipped=first - start, load_s=round(t_load, 1), infer_s=round(t_inf, 1),
                  fps=round(written / max(t_inf, 1e-6), 3), infer_size=[w, h],
                  flow_valid=round(float(np.mean(valid)), 3) if valid else None,
                  peak_alloc_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                  peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2),
                  device=torch.cuda.get_device_name(0))


if __name__ == "__main__":
    worker_main(_work)
