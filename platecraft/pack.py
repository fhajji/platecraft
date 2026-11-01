"""`platecraft pack`: pack a clip's passes into one 2x2 atlas image per frame for Unreal.

One image sequence keeps every pass in sync behind a single Img Media Source and player:

    +--------------------------+--------------------------+
    | albedo RGB (linear)      | normal RGB               |
    |                          | n * 0.5 + 0.5, tangent   |
    +--------------------------+--------------------------+
    | R = AO, G = roughness,   | depth (R = G = B)        |
    | B = metallic (0 = none)  | 1 = nearest              |
    +--------------------------+--------------------------+
    A = alpha in every quadrant (straight, not premultiplied)

Every value is linear (albedo is converted from sRGB here), so the material needs no
per-region color correction. Region rects, channels and encodings go to
manifest.json["atlas"] for the Unreal importer. Missing passes become constants.

Format: half-float EXR (default) or 16-bit PNG. UE 5.8's Img Media keeps both at 16 bits
per channel (EXR -> PF_FloatRGBA, RGBA16 PNG -> PF_A16B16G16R16); 8-bit PNG would be
truncated to 8 bits, which data maps must never be. EXR decodes faster (multi-threaded).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import ffmpeg_io, frames, manifest, postprocess
import cv2  # noqa: E402  (after frames: OpenEXR enabled)
import numpy as np  # noqa: E402

FORMATS = ("exr", "png")
COMPRESSIONS = tuple(frames.EXR_COMPRESSION)
DIR = "atlas"

# Quadrants as (column, row); UV origin top-left, as in Unreal.
TILE = {"albedo": (0, 0), "normal": (1, 0), "orm": (0, 1), "depth": (1, 1)}
LAYOUT_VERSION = 2  # 2: depth under transparent pixels filled from the subject
CONSTANT = {"normal": 0.0, "ao": 1.0, "roughness": 0.7, "metallic": 0.0, "depth": 0.0}


@dataclass
class PackOptions:
    clip_dir: Path
    format: str = "exr"
    compression: str = "zip"
    scale: float = 1.0               # quadrant size relative to the clip
    start: str | None = None
    end: str | None = None           # exclusive
    max_frames: int | None = None
    force: bool = False


def _rect(tile: str) -> list[float]:
    c, r = TILE[tile]
    return [c * 0.5, r * 0.5, c * 0.5 + 0.5, r * 0.5 + 0.5]


def _source(p: dict, name: str) -> dict:
    """What a region was made from: the pass entry without its progress counter."""
    return {k: v for k, v in p[name].items() if k not in ("frames_done", "pattern", "start_number")}


def regions(m: dict) -> dict:
    """Layout record for manifest.json: where each pass lives in the atlas and how it is encoded."""
    p = m["passes"]
    out = {}
    albedo = {"rect": _rect("albedo"), "channels": "rgb", "colorspace": "linear", "alpha": "straight",
              "edge_bleed": True, "note": "converted from sRGB; transparent pixels hold nearby colour"}
    if "albedo" in p:
        out["albedo"] = {**albedo, "source": "albedo", "heuristic": p["albedo"].get("heuristic", False)}
    else:
        out["albedo"] = {**albedo, "source": next(k for k in ("fg", "plate", "src") if k in p),
                         "heuristic": False}
    out["alpha"] = {"rect": [0.0, 0.0, 1.0, 1.0], "channels": "a", "colorspace": "linear",
                    "source": "alpha" if "alpha" in p else None, "constant": None if "alpha" in p else 1.0,
                    "note": "the A channel of every quadrant"}
    if "normal" in p:
        out["normal"] = {"rect": _rect("normal"), "channels": "rgb", "colorspace": "linear",
                         "encoding": "n * 0.5 + 0.5", "normal_space": p["normal"].get("normal_space", "directx"),
                         "source": "normal", "model": p["normal"].get("model"),
                         "heuristic": p["normal"].get("heuristic", False)}
    else:
        out["normal"] = {"rect": _rect("normal"), "channels": "rgb", "colorspace": "linear",
                         "encoding": "n * 0.5 + 0.5", "normal_space": "directx", "source": None,
                         "constant": [0.5, 0.5, 1.0], "heuristic": True}
    for name, ch in (("ao", "r"), ("roughness", "g"), ("metallic", "b")):
        e = {"rect": _rect("orm"), "channels": ch, "colorspace": "linear"}
        if name in p:
            out[name] = {**e, "source": name, "heuristic": p[name].get("heuristic", False)}
        else:
            out[name] = {**e, "source": None, "constant": CONSTANT[name], "heuristic": True}
    e = {"rect": _rect("depth"), "channels": "r", "colorspace": "linear", "edge_bleed": True,
         "note": "replicated to G and B; pixels with alpha < 0.5 hold nearby subject depth"}
    if "depth" in p:
        out["depth"] = {**e, "source": "depth", "model": p["depth"].get("model"),
                        "encoding": p["depth"].get("encoding"), "heuristic": False}
    else:
        out["depth"] = {**e, "source": None, "constant": CONSTANT["depth"], "heuristic": True}
    return out


def _read_gray(path: Path) -> np.ndarray:
    g = frames.read(path)
    return g if g.ndim == 2 else g[..., 0]


def _fit(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if (img.shape[1], img.shape[0]) == size:
        return img
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def pack_frame(clip: Path, m: dict, i: int, tile: tuple[int, int]) -> np.ndarray:
    """The atlas for frame i: float32 RGBA of size (2 * tile_h, 2 * tile_w), all linear."""
    p = m["passes"]
    tw, th = tile
    if "albedo" in p:
        rgb = frames.read(frames.frame_path(clip / p["albedo"]["dir"], i))[..., :3]
        a = _read_gray(frames.frame_path(clip / p["alpha"]["dir"], i)) if "alpha" in p else None
    else:
        rgb, a = postprocess.straight_color(clip, m, i)
    if a is None:
        a = np.ones(rgb.shape[:2], np.float32)
    rgb = frames.bleed_color(rgb, a)
    rgb, a = _fit(rgb, tile), _fit(a, tile)

    def gray(name: str) -> np.ndarray:
        if name in p:
            return _fit(_read_gray(frames.frame_path(clip / p[name]["dir"], i)), tile)
        return np.full((th, tw), CONSTANT[name], np.float32)

    if "normal" in p:
        n = frames.read(frames.frame_path(clip / p["normal"]["dir"], i))[..., :3]
        if (n.shape[1], n.shape[0]) != tile:
            n = cv2.resize(n * 2 - 1, tile, interpolation=cv2.INTER_AREA)
            n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6) * 0.5 + 0.5
    else:
        n = np.broadcast_to(np.array([0.5, 0.5, 1.0], np.float32), (th, tw, 3))
    d = gray("depth")
    if "depth" in p:
        # Below alpha 0.5 the model's depth is background or a mix; fill it from the subject so
        # a card bent by depth (Unreal material) does not fold back along the silhouette.
        d = frames.bleed_color(np.dstack([d, d, d]), (a >= 0.5).astype(np.float32))[..., 0]
    orm = np.dstack([gray("ao"), gray("roughness"), gray("metallic")])

    out = np.empty((th * 2, tw * 2, 4), np.float32)
    for name, img in (("albedo", postprocess.srgb_to_linear(rgb)), ("normal", n), ("orm", orm),
                      ("depth", np.dstack([d, d, d]))):
        c, r = TILE[name]
        out[r * th:(r + 1) * th, c * tw:(c + 1) * tw, :3] = img
        out[r * th:(r + 1) * th, c * tw:(c + 1) * tw, 3] = a
    return out


def _clear(folder: Path) -> None:
    for f in list(folder.glob("frame_*.exr")) + list(folder.glob("frame_*.png")):
        f.unlink()


def run(o: PackOptions) -> None:
    clip = Path(o.clip_dir)
    m = manifest.load(clip)
    if not m:
        raise FileNotFoundError(f"no {manifest.FILENAME} in {clip}: run 'platecraft extract' first")
    if o.format not in FORMATS:
        raise ValueError(f"--format must be one of {', '.join(FORMATS)}")
    if o.compression not in COMPRESSIONS:
        raise ValueError(f"--exr-compression must be one of {', '.join(COMPRESSIONS)}")
    if not 0.0 < o.scale <= 1.0:
        raise ValueError("--scale must be in (0, 1]")
    p = m["passes"]
    fr = m["frames"]
    total = fr["count"]
    start = ffmpeg_io.parse_time(o.start, fr["fps_float"]) or 0
    end = min(total, ffmpeg_io.parse_time(o.end, fr["fps_float"]) or total)
    if o.max_frames:
        end = min(end, start + o.max_frames)
    if not 0 <= start < end:
        raise ValueError(f"empty frame range {start}..{end} (clip has {total} frames)")

    tile = (max(2, round(fr["width"] * o.scale / 2) * 2), max(2, round(fr["height"] * o.scale / 2) * 2))
    layout = regions(m)
    missing = [k for k in ("albedo", "normal", "depth", "roughness", "ao") if k not in p]
    if missing:
        print(f"Note: no {', '.join(missing)} pass: packing constants instead (run 'platecraft infer' first?)")

    # Inputs per frame: every pass folder the atlas reads from.
    src_names = [k for k in ("albedo", "alpha", "normal", "depth", "roughness", "ao", "metallic") if k in p]
    if "albedo" not in p:
        src_names.append(next(k for k in ("fg", "plate", "src") if k in p))
    src_dirs = [clip / p[k]["dir"] for k in dict.fromkeys(src_names)]
    ready = [i for i in range(start, end) if all(frames.frame_path(f, i).is_file() for f in src_dirs)]
    if len(ready) < end - start:
        print(f"Warning: only {len(ready)} of {end - start} frames have every pass; the others are skipped "
              f"and the atlas sequence will have gaps until they are inferred")
    if not ready:
        raise FileNotFoundError("no frame has all passes")

    out_dir = clip / DIR
    ext = "." + o.format
    settings = {"layout": LAYOUT_VERSION, "format": o.format, "compression": o.compression if o.format == "exr" else None,
                "tile": list(tile), "sources": {k: _source(p, k) for k in dict.fromkeys(src_names)}}
    sfile = out_dir / "settings.json"
    old = json.loads(sfile.read_text(encoding="utf-8")) if sfile.is_file() else None
    if out_dir.is_dir() and old != settings:
        _clear(out_dir)          # frames from other settings or sources must not mix into the sequence
    out_dir.mkdir(parents=True, exist_ok=True)
    sfile.write_text(json.dumps(settings, indent=2), encoding="utf-8")

    def stale(i: int) -> bool:
        f = frames.frame_path(out_dir, i, ext)
        if o.force or not f.is_file():
            return True
        t = f.stat().st_mtime
        return any(frames.frame_path(d, i).stat().st_mtime > t for d in src_dirs)

    todo = [i for i in ready if stale(i)]
    size = (tile[0] * 2, tile[1] * 2)
    print(f"Atlas {size[0]}x{size[1]} ({o.format}{', ' + o.compression if o.format == 'exr' else ''}): "
          f"{len(todo)} to pack, {len(ready) - len(todo)} up to date -> {out_dir}")

    def one(j: int) -> None:
        i = todo[j]
        img = pack_frame(clip, m, i, tile)
        if o.format == "exr":
            frames.write_exr_half(frames.frame_path(out_dir, i, ext), img, o.compression)
        else:
            frames.write(frames.frame_path(out_dir, i, ext), img, 16)

    if todo:
        frames.parallel(one, len(todo), "atlas")

    done = sum(1 for _ in out_dir.glob(f"frame_*{ext}"))
    m["atlas"] = {"dir": DIR, "pattern": f"frame_%06d{ext}", "start_number": manifest.START_NUMBER,
                  "format": o.format, "pixel": "half float" if o.format == "exr" else "16-bit unorm",
                  "compression": settings["compression"], "width": size[0], "height": size[1],
                  "tile": list(tile), "colorspace": "linear", "fps": fr["fps"], "fps_float": fr["fps_float"],
                  "frame_count": total, "frames_done": done, "regions": layout,
                  "note": "sample each region with UV * 0.5 + offset; clamp half a texel inside "
                          "the quadrant so bilinear filtering does not reach the neighbour"}
    manifest.log(m, "pack", range=[start, end], frames=len(todo), format=o.format, size=list(size))
    manifest.save(clip, m)
    if done < total:
        print(f"Note: {done} of {total} frames packed so far")
    print(f"Updated {manifest.path_for(clip)}")
