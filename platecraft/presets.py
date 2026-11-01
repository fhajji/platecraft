"""Inference presets by GPU memory.

Measured with NormalCrafter on an RTX 3060 12 GB (model CPU offload, xformers, fp16,
window 14, decode chunk 4), peak torch allocated / reserved:
    640x360  4.1 / 6.6 GB  1.27 fps
    768x432  5.0 / 7.8 GB  0.96 fps   <- 3060 default: clearly sharper than 640
    1024x576 7.5 / 11.9 GB 0.42 fps   (at the limit with the desktop's own VRAM use)
"""
from __future__ import annotations

PRESETS: dict[str, dict] = {
    "low":  {"min_vram_gb": 0,  "max_res": 512,  "offload": "sequential", "window": 14, "overlap": 4, "decode_chunk": 2, "chunk": 48},
    "3060": {"min_vram_gb": 10, "max_res": 768,  "offload": "model",      "window": 14, "overlap": 4, "decode_chunk": 4, "chunk": 70},
    "mid":  {"min_vram_gb": 16, "max_res": 1024, "offload": "model",      "window": 14, "overlap": 4, "decode_chunk": 7, "chunk": 70},
    "high": {"min_vram_gb": 22, "max_res": 1024, "offload": "none",       "window": 14, "overlap": 4, "decode_chunk": 7, "chunk": 70},
}


def recommend(vram_gb: float | None) -> str:
    if vram_gb is None:
        return "low"
    best = "low"
    for name, p in PRESETS.items():
        if vram_gb >= p["min_vram_gb"]:
            best = name
    return best
