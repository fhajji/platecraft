"""NormalCrafter adapter: temporally consistent video normals (single-step SVD-based).

Raw output (job.out_dir): 16-bit PNG, inference resolution, camera-space normals
as the model returns them, encoded n * 0.5 + 0.5. postprocess.py converts them.

Long clips are cut into chunks of `chunk` frames; inside a chunk the pipeline runs
its own `window`-frame windows overlapping by `overlap`. Each chunk is conditioned on
the previous chunk's latents (raw/.../latents/, one small .npy per frame), so chunk
seams and resumed runs behave like the model's own window seams. Existing raw
frames are kept, so an interrupted run continues from the first missing frame.
"""
from __future__ import annotations

from pathlib import Path

from .. import paths
from .base import Job, chunks, spawn_worker, worker_main, worker_progress, worker_result
from .registry import MODELS

INFO = MODELS["normalcrafter"]
SIZE_MULTIPLE = 64      # the pipeline pads to this; we pad ourselves with the plate grey instead of white


def job_settings(preset: dict) -> dict:
    return {
        "unet_vae": str(paths.MODELS / "normalcrafter" / "normalcrafter"),
        "svd": str(paths.MODELS / "normalcrafter" / "svd-xt"),
        "offload": preset["offload"],
        "window": preset["window"],
        "overlap": preset["overlap"],
        "decode_chunk": preset["decode_chunk"],
        "chunk": preset["chunk"],
    }


def run(job: Job) -> dict:
    return spawn_worker(INFO, __name__, job)


# ------------------------------------------------------------ worker side --

def _infer_chunk(pipe, images, prev, window: int, overlap: int, decode_chunk: int):
    """NormalCrafterPipeline.__call__ for one chunk, conditioned on the previous chunk's latents.

    `prev` holds the latents of the chunk's first frames (1, m, C, h, w) or None. The first
    window is seeded with them exactly like the pipeline seeds each window from the one
    before, so chunk seams behave like the model's own window seams.
    Returns (normals (T, H, W, 3) in -1..1, latents (T, C, h, w)).
    """
    import torch

    n = len(images)
    window = min(window, n)
    step = max(1, window - overlap)
    width, height = images[0].size
    device = pipe._execution_device
    pipe._guidance_scale = 1.0
    emb = pipe._encode_image(images, device, 1, do_classifier_free_guidance=False, scale=1, image_size=None)
    img_lat = pipe.ecnode_video_vae(images, chunk_size=decode_chunk).to(emb.dtype)[None]
    time_ids = pipe._get_add_time_ids(7, 127, 0.0, emb.dtype, 1, 1, False).to(device)  # upstream constants

    starts = list(range(0, n - window + 1, step))
    if starts[-1] + window < n:
        starts.append(n - window)
    pred = None
    for k, s0 in enumerate(starts):
        if k == 0:
            rep = prev.to(device, emb.dtype) if prev is not None else None
        else:
            rep = pred[:, -(starts[k - 1] + window - s0):]
        lat = pipe.generate(1, device, 1, 1, window, height, width, emb[s0:s0 + window], None, True, rep,
                            False, img_lat[:, s0:s0 + window], time_ids)
        if rep is not None:   # linear merge over the overlap, as upstream
            m = rep.shape[1]
            wgt = torch.linspace(1.0, 0.0, m + 2)[1:-1].to(device)[None, :, None, None, None]
            lat[:, :m] = rep * wgt + lat[:, :m] * (1 - wgt)
        pred = lat if pred is None else torch.cat([pred[:, :s0], lat], dim=1)

    out = pipe.decode_latents(pred, n, decode_chunk)
    out = pipe.video_processor.postprocess_video(video=out, output_type="np") * 2 - 1
    pipe.maybe_free_model_hooks()
    return out[0].astype("float32"), pred[0]


def _work(job: Job) -> None:
    import time
    from concurrent.futures import ThreadPoolExecutor

    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import AutoencoderKLTemporalDecoder
    from normalcrafter.normal_crafter_ppl import NormalCrafterPipeline
    from normalcrafter.unet import DiffusersUNetSpatioTemporalConditionModelNormalCrafter

    from .. import frames

    s = job.settings
    out_dir, in_dir = Path(job.out_dir), Path(job.input_dir)
    start, end = job.frames
    w, h = job.infer_size
    pw, ph = (-(-w // SIZE_MULTIPLE) * SIZE_MULTIPLE, -(-h // SIZE_MULTIPLE) * SIZE_MULTIPLE)
    top, left = (ph - h) // 2, (pw - w) // 2

    done = {i for i in range(start, end) if frames.frame_path(out_dir, i).is_file()}
    first = next((i for i in range(start, end) if i not in done), end)
    total = end - start
    if first >= end:
        worker_progress(total, total, 0)
        worker_result(frames=0, skipped=total)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the normalcrafter env")
    t_load = time.time()
    unet = DiffusersUNetSpatioTemporalConditionModelNormalCrafter.from_pretrained(
        s["unet_vae"], subfolder="unet", low_cpu_mem_usage=True, torch_dtype=torch.float16)
    vae = AutoencoderKLTemporalDecoder.from_pretrained(s["unet_vae"], subfolder="vae", torch_dtype=torch.float16)
    pipe = NormalCrafterPipeline.from_pretrained(s["svd"], unet=unet, vae=vae,
                                                 torch_dtype=torch.float16, variant="fp16")
    if s["offload"] == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif s["offload"] == "model":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    try:
        pipe.enable_xformers_memory_efficient_attention()
        xformers = True
    except Exception as e:  # noqa: BLE001 - optional speed-up
        print(f"xformers not used: {e}")
        xformers = False
    pipe.set_progress_bar_config(disable=True)
    t_load = time.time() - t_load
    torch.cuda.reset_peak_memory_stats()

    def load(i: int) -> Image.Image:
        img = cv2.resize(frames.read(frames.frame_path(in_dir, i))[..., :3], (w, h), interpolation=cv2.INTER_AREA)
        canvas = np.full((ph, pw, 3), job.pad_value, np.float32)
        canvas[top:top + h, left:left + w] = img
        return Image.fromarray((np.clip(canvas, 0, 1) * 255 + 0.5).astype(np.uint8))

    def unit(n: np.ndarray) -> np.ndarray:
        return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)

    lat_dir = out_dir / "latents"
    lat_dir.mkdir(parents=True, exist_ok=True)

    def lat_path(i: int) -> Path:
        return lat_dir / f"frame_{i + 1:06d}.npy"

    # Resume: condition the first chunk on the cached latents just before the first missing frame.
    ov = s["overlap"]
    known = {}
    if first - ov >= 0 and all(lat_path(i).is_file() for i in range(first - ov, first)):
        known = {i: torch.from_numpy(np.load(lat_path(i))) for i in range(first - ov, first)}
    written = 0
    t_inf = time.time()
    io = ThreadPoolExecutor(max_workers=8)
    for cs, ce in chunks(first - len(known), end, s["chunk"], ov, s["window"]):
        images = list(io.map(load, range(cs, ce)))
        prev = [known[i] for i in range(cs, ce) if i in known]
        prev = torch.stack(prev)[None] if prev else None
        with torch.inference_mode():
            res, lat = _infer_chunk(pipe, images, prev, s["window"], s["overlap"], s["decode_chunk"])
        res = unit(res[:, top:top + h, left:left + w])
        lat = lat.to("cpu", torch.float16)
        # Overlap frames are rewritten with the merged result, as the pipeline does between its windows.
        todo = [i for i in range(cs, ce) if i >= first or i in known]
        list(io.map(lambda i: (frames.write(frames.frame_path(out_dir, i), res[i - cs] * 0.5 + 0.5, 16),
                               np.save(lat_path(i), lat[i - cs].numpy())), todo))
        written += sum(1 for i in todo if i not in known and i not in done)
        known = {i: lat[i - cs] for i in range(cs, ce)}
        worker_progress(len(done) + written, total, written)
    io.shutdown()
    t_inf = time.time() - t_inf
    worker_result(frames=written, skipped=len(done), load_s=round(t_load, 1), infer_s=round(t_inf, 1),
                  fps=round(written / max(t_inf, 1e-6), 3), infer_size=[w, h], padded_size=[pw, ph],
                  peak_alloc_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                  peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2),
                  device=torch.cuda.get_device_name(0), xformers=xformers)


if __name__ == "__main__":
    worker_main(_work)
