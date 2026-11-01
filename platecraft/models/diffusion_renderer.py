"""Cosmos DiffusionRenderer adapter: video inverse rendering (albedo, normal, depth, roughness, metallic).

NVIDIA's 7B video diffusion model (nv-tlabs/cosmos-transfer1-diffusion-renderer). It needs Linux
(transformer-engine) and about 27 GB of VRAM for a 57-frame chunk (upstream's figure; 16 GB is
their minimum for single images), so it can't run on the 3060. On Windows the worker runs
inside WSL2 when engine/setup_engine.sh was run there (see paths.worker_launcher).

The model was trained on 57-frame clips at 1280x704 and generates one G-buffer pass per
run, so every chunk is sampled five times (once per pass). The plate is scaled to fit
1280x704 and padded with the plate grey. Chunks overlap by `overlap` frames and are
crossfaded there; depth comes out normalised per chunk (near..far -> 0..1), so each chunk's
depth is first fitted (scale + shift) to the previous chunk over the overlap. Overlap
frames are written provisionally and rewritten by the next chunk. Chunks sit on a fixed grid
from the run's start, and each chunk's pre-blend tail is kept in dr_state.npz, so a resumed run
joins exactly like an uninterrupted one.

Raw output (job.out_dir/<pass>/), inference resolution without the padding:
    basecolor/  16-bit PNG RGB, sRGB-encoded as the model outputs it (assumed: not documented)
    normal/     16-bit PNG, n * 0.5 + 0.5, axes multiplied by AXES (not verified on this model yet)
    depth/      float EXR, 1 - normalised depth: larger = nearer (affine-invariant, not inverse depth)
    roughness/ metallic/   16-bit PNG gray
"""
from __future__ import annotations

import os
from pathlib import Path

from .. import paths
from .base import Job, chunks, spawn_worker, worker_main, worker_progress, worker_result
from .registry import MODELS

INFO = MODELS["diffusion-renderer"]
MODEL_SIZE = (1280, 704)      # (w, h) the model was trained at; its latent shape is fixed to this
FRAMES = 57                   # training clip length (must be 8n + 1)
OVERLAP = 8
# Model normal axes relative to ours (+X right, +Y up, +Z toward the camera). Not measured yet:
# run() prints a silhouette check after each run; flip a sign here if it reports a negative axis.
AXES = (1.0, 1.0, 1.0)
# Our pass name -> (upstream G-buffer name, raw subfolder, raw extension)
RAW = {
    "albedo": ("basecolor", "basecolor", ".png"),
    "normal": ("normal", "normal", ".png"),
    "depth": ("depth", "depth", ".exr"),
    "roughness": ("roughness", "roughness", ".png"),
    "metallic": ("metallic", "metallic", ".png"),
}
DEPTH_ENCODING = ("relative depth (affine-invariant, linear in depth), 1 = nearest, 0 = farthest; "
                  "normalised over the clip (not metric)")
OFFLOAD_BELOW_GB = 40         # move the DiT to CPU RAM while the tokenizer decodes


def infer_size(w: int, h: int, preset: dict) -> tuple[int, int]:
    """Fit inside 1280x704 keeping the aspect ratio, multiples of 8; the rest is padding."""
    s = min(MODEL_SIZE[0] / w, MODEL_SIZE[1] / h)
    return min(MODEL_SIZE[0], round(w * s / 8) * 8), min(MODEL_SIZE[1], round(h * s / 8) * 8)


def raw_pass(raw: Path, pass_name: str) -> tuple[Path, str]:
    _, sub, ext = RAW[pass_name]
    return raw / sub, ext


def job_settings(preset: dict) -> dict:
    return {"model_size": list(MODEL_SIZE), "frames": FRAMES, "overlap": OVERLAP, "steps": 15,
            "guidance": 0.0, "seed": 1000, "passes": list(INFO.passes), "axes": list(AXES)}


def preflight() -> float | None:
    """Fail before any work if this PC can't run the model. Returns the detected VRAM (GB)."""
    from ..doctor import check_gpu
    vram = check_gpu()[1]
    if vram is not None and vram < INFO.min_vram_gb and not os.environ.get("PLATECRAFT_IGNORE_VRAM"):
        raise RuntimeError(f"{INFO.name} needs a GPU with at least {INFO.min_vram_gb:g} GB VRAM (about 27 GB "
                           f"for a 57-frame chunk without offloading); this one has {vram:.0f} GB. "
                           "Set PLATECRAFT_IGNORE_VRAM=1 to try anyway.")
    if not paths.worker_launcher(INFO.env).exists():
        where = "inside WSL2 (from this folder)" if paths.IS_WINDOWS else "on this PC"
        raise FileNotFoundError(f"{INFO.name} needs its Linux environment: run engine/setup_engine.sh {where}")
    return vram


def run(job: Job) -> dict:
    vram = preflight()
    job.settings = {**job.settings, "checkpoint_dir": str(paths.MODELS / "diffusion-renderer"),
                    "repo": str(paths.REPOS / INFO.repo),
                    "offload": vram is None or vram < OFFLOAD_BELOW_GB}
    result = spawn_worker(INFO, __name__, job)
    _check_axes(job)
    return result


def _check_axes(job: Job) -> None:
    """Silhouette test: at the matte's edge a normal should point away from the subject.
    Prints the correlation per axis (positive = matches +X right / +Y up)."""
    import numpy as np

    from .. import frames
    import cv2

    alpha_dir = Path(job.clip_dir) / "alpha"
    ndir, ext = raw_pass(Path(job.out_dir), "normal")
    idx = [i for i in range(*job.frames) if frames.frame_path(ndir, i, ext).is_file()]
    if not alpha_dir.is_dir() or not idx:
        return
    acc = np.zeros((2, 2))   # per axis: sum of n . outward, count
    for i in idx[:: max(1, len(idx) // 8)][:8]:
        n = frames.read(frames.frame_path(ndir, i, ext))[..., :3] * 2 - 1
        a = frames.read(frames.frame_path(alpha_dir, i))
        a = cv2.GaussianBlur(cv2.resize(a, (n.shape[1], n.shape[0]), interpolation=cv2.INTER_AREA), (0, 0), 1.5)
        gx, gy = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)
        sel = mag > 0.25 * max(float(mag.max()), 1e-6)
        b = max(4, round(0.03 * max(a.shape)))      # matte cut by the frame or a garbage matte: no real edge
        sel[:b], sel[-b:], sel[:, :b], sel[:, -b:] = False, False, False, False
        # Outward = -gradient in image coordinates; image y points down, ours points up.
        out = (-gx / np.maximum(mag, 1e-6), gy / np.maximum(mag, 1e-6))
        for k in (0, 1):                             # score each axis on edges running across it
            m = sel & (np.abs(out[k]) > 0.7)
            acc[k] += (float((n[..., k][m] * out[k][m]).sum()), float(m.sum()))
    if acc[:, 1].min() > 0:
        cx, cy = acc[:, 0] / acc[:, 1]
        print(f"  normal axes check (silhouette): X {cx:+.2f}, Y {cy:+.2f} (positive = +X right, +Y up)")
        if min(cx, cy) < -0.1:
            print("  WARNING: a normal axis points the wrong way; flip it in AXES in "
                  "platecraft/models/diffusion_renderer.py and rerun infer with --force")
        elif min(cx, cy) < 0.1:
            print("  (inconclusive: the matte edge says little about these normals)")


# ------------------------------------------------------------ worker side --

def _generate(pipe, batch: dict, offload: bool):
    """DiffusionRendererPipeline.generate_video without its uint8 quantisation.
    Returns (T, H, W, 3) float32 in -1..1."""
    import torch
    from cosmos_predict1.utils import misc

    batch = dict(batch, video=batch["rgb"])
    batch = misc.to(batch, device="cuda", dtype=torch.bfloat16)
    m = pipe.model
    f = m.tokenizer.spatial_compression_factor
    v = batch["video"]
    state_shape = [m.tokenizer.channel, (v.shape[2] - 1) // 8 + 1, v.shape[3] // f, v.shape[4] // f]
    sample = m.generate_samples_from_batch(batch, guidance=pipe.guidance, state_shape=state_shape,
                                           num_steps=pipe.num_steps, is_negative_prompt=False, seed=pipe.seed)
    if offload:   # the DiT (~14 GB) waits in CPU RAM while the tokenizer decodes 57 frames
        m.model.to("cpu")
        torch.cuda.empty_cache()
    video = m.decode(sample)
    if offload:
        m.model.to("cuda")
    return video[0].permute(1, 2, 3, 0).float().cpu().numpy()


def _work(job: Job) -> None:
    import json
    import time
    from concurrent.futures import ThreadPoolExecutor

    from .. import frames
    import cv2
    import numpy as np
    import torch

    s = job.settings
    out_dir, in_dir = Path(job.out_dir).resolve(), Path(job.input_dir).resolve()
    start, end = job.frames
    w, h = job.infer_size
    mw, mh = s["model_size"]
    top, left = (mh - h) // 2, (mw - w) // 2
    T, ov = s["frames"], s["overlap"]
    passes = s["passes"]
    axes = np.array(s["axes"], np.float32)
    total = end - start

    def raw(name: str, i: int) -> Path:
        d, ext = raw_pass(out_dir, name)
        return frames.frame_path(d, i, ext)

    done = {i for i in range(start, end) if all(raw(p, i).is_file() for p in passes)}
    first = next((i for i in range(start, end) if i not in done), end)
    if first >= end:
        worker_progress(total, total, 0)
        worker_result(frames=0, skipped=total)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the cosmos env")

    os.chdir(s["repo"])   # upstream loads its config files by relative path
    from cosmos_predict1.diffusion.inference.diffusion_renderer_pipeline import DiffusionRendererPipeline
    from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.dataloader_utils import dict_collation_fn
    from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.rendering_utils import GBUFFER_INDEX_MAPPING
    from cosmos_predict1.utils import misc

    misc.set_random_seed(s["seed"])
    t_load = time.time()
    pipe = DiffusionRendererPipeline(checkpoint_dir=s["checkpoint_dir"],
                                     checkpoint_name="Diffusion_Renderer_Inverse_Cosmos_7B",
                                     guidance=s["guidance"], num_steps=s["steps"], height=mh, width=mw, fps=24,
                                     num_video_frames=T, seed=s["seed"])
    t_load = time.time() - t_load
    torch.cuda.reset_peak_memory_stats()

    def load(i: int) -> np.ndarray:
        img = cv2.resize(frames.read(frames.frame_path(in_dir, i))[..., :3], (w, h), interpolation=cv2.INTER_AREA)
        canvas = np.full((mh, mw, 3), job.pad_value, np.float32)
        canvas[top:top + h, left:left + w] = np.clip(img, 0, 1)
        return canvas

    def decode(name: str, v: np.ndarray) -> np.ndarray:
        """Model output (t, h, w, 3) in -1..1 -> our raw encoding (normals as unit vectors)."""
        if name == "normal":
            n = v * axes
            return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
        x = np.clip((v + 1) / 2, 0.0, 1.0)
        if name == "albedo":
            return x
        g = x.mean(axis=-1)
        return 1.0 - g if name == "depth" else g    # depth: model 0 = near; ours 1 = near

    def read_back(name: str, i: int) -> np.ndarray:
        x = frames.read(raw(name, i))
        if name == "normal":
            return x[..., :3] * 2 - 1
        return x[..., :3] if name == "albedo" else (x if x.ndim == 2 else x[..., 0])

    def write(name: str, i: int, x: np.ndarray) -> None:
        if name == "depth":
            frames.write_exr(raw(name, i), x.astype(np.float32))
        else:
            frames.write(raw(name, i), x * 0.5 + 0.5 if name == "normal" else x, 16)

    for name in passes:
        raw_pass(out_dir, name)[0].mkdir(parents=True, exist_ok=True)

    # Chunks sit on a grid anchored at the run's start, and after each chunk its tail (the frames the
    # next chunk crossfades with, before that blend) is saved to dr_state.npz. A resumed run restarts at
    # the first chunk with a missing frame and joins exactly as an uninterrupted run would.
    grid = chunks(start, end, T, ov, T)
    j = next(k for k, (cs, ce) in enumerate(grid) if any(i not in done for i in range(cs, ce)))
    state_file = out_dir / "dr_state.npz"
    meta = {"start": start, "end": end, "frames": T, "overlap": ov}
    prev: dict[int, dict[str, np.ndarray]] = {}
    note = ""
    if j > 0:
        saved = None
        if state_file.is_file():
            with np.load(state_file) as st:     # closes the file (Windows can't replace an open one)
                if json.loads(str(st["meta"])) == {**meta, "next": int(st["next"])} and int(st["next"]) <= j:
                    saved = int(st["next"]), {int(i): {q: st[q][n] for q in passes}
                                              for n, i in enumerate(st["idx"])}
        if saved:
            j, prev = saved
        else:
            note = ("no saved chunk tail for this resume point: joined onto the frames on disk "
                    "(seam close to, but not exactly, an uninterrupted run)")
            prev = {i: {q: read_back(q, i) for q in passes}
                    for i in range(grid[j][0], grid[j - 1][1]) if i in done}

    io = ThreadPoolExecutor(max_workers=8)
    written: set[int] = set()
    t_inf = time.time()
    for k, (cs, ce) in enumerate(grid[j:], start=j):
        idx = list(range(cs, ce))
        imgs = list(io.map(load, idx))
        imgs += [imgs[-1]] * (T - len(imgs))            # short clip: repeat the last frame, as upstream
        sample = {"clip_name": "platecraft", "chunk_index": "0000", "is_preprocessed": True,
                  "num_frames": torch.tensor(T, dtype=torch.float),
                  "image_size": torch.tensor([mh, mw]), "fps": torch.tensor(24, dtype=torch.float),
                  "padding_mask": torch.zeros(1, mh, mw),
                  "t5_text_embeddings": torch.zeros(512, 1024),
                  "t5_text_mask": torch.nn.functional.one_hot(torch.tensor(0), 512).float(),
                  "context_index": torch.LongTensor([0]),
                  "rgb": torch.from_numpy(np.stack(imgs)).permute(3, 0, 1, 2) * 2 - 1}   # (3, T, H, W)
        batch = dict_collation_fn([sample])
        res = {}
        for name in passes:
            batch["context_index"] = torch.full_like(batch["context_index"], GBUFFER_INDEX_MAPPING[RAW[name][0]])
            with torch.inference_mode():
                v = _generate(pipe, batch, s["offload"])
            res[name] = decode(name, v[:len(idx), top:top + h, left:left + w])

        over = [i for i in idx if i in prev]
        if over:
            if "depth" in res:   # each chunk has its own near/far: fit it to the previous chunk first
                # Closed-form least squares in float64 (np.polyfit on ~10^6 float32 values is badly off).
                a = np.concatenate([res["depth"][i - cs][::4, ::4].ravel() for i in over]).astype(np.float64)
                b = np.concatenate([prev[i]["depth"][::4, ::4].ravel() for i in over]).astype(np.float64)
                va = a.var()
                scale = float(((a - a.mean()) * (b - b.mean())).mean() / va) if va > 1e-12 else 1.0
                if scale <= 0:
                    scale = 1.0
                shift = float(b.mean() - scale * a.mean())
                res["depth"] = res["depth"] * scale + shift
                print(f"chunk {k} ({cs}..{ce - 1}): depth fitted to the previous chunk, "
                      f"scale {scale:.4f} shift {shift:+.4f} over {len(over)} frames", flush=True)
            for m, i in enumerate(over):                 # linear crossfade, new chunk weight rising
                wgt = (m + 1) / (len(over) + 1)
                for name in passes:
                    x = prev[i][name] * (1 - wgt) + res[name][i - cs] * wgt
                    if name == "normal":
                        x = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)
                    res[name][i - cs] = x
        # A last chunk pulled back to full length also covers frames before the crossfade that are
        # already final: keep those rather than switch chunks abruptly.
        lo = min(prev) if prev else cs
        todo = [i for i in idx if i >= lo]
        list(io.map(lambda i: [write(q, i, res[q][i - cs]) for q in passes], todo))
        written.update(i for i in todo if i not in done)
        tail = idx[-ov:]
        prev = {i: {q: res[q][i - cs] for q in passes} for i in tail}
        tmp = state_file.with_name("dr_state.tmp.npz")
        np.savez(tmp, meta=json.dumps({**meta, "next": k + 1}), next=k + 1, idx=np.array(tail),
                 **{q: np.stack([prev[i][q] for i in tail]).astype(np.float32) for q in passes})
        tmp.replace(state_file)
        worker_progress(len(done | written), total, len(written))
    io.shutdown()
    t_inf = time.time() - t_inf
    worker_result(frames=len(written), skipped=len(done), load_s=round(t_load, 1), infer_s=round(t_inf, 1),
                  fps=round(len(written) / max(t_inf, 1e-6), 3), infer_size=[w, h], padded_size=[mw, mh],
                  peak_alloc_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                  peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2),
                  device=torch.cuda.get_device_name(0), offload=s["offload"], note=note)


if __name__ == "__main__":
    worker_main(_work)
