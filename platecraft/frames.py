"""Frame-level image work: reading/writing PNG/EXR, despill, plate preparation.

Images are handled as float32 RGB in 0..1 (sRGB-encoded unless stated) and
written back at the clip's bit depth.
"""
from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")  # must be set before cv2 is imported
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from .manifest import FRAME_PATTERN, START_NUMBER  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr", ".bmp"}


# ------------------------------------------------------------------ I/O --

def read(path: Path) -> np.ndarray:
    """Read as float32, RGB(A) order, 0..1 (EXR values are returned as stored)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"cannot read image {path}")
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    else:
        img = img.astype(np.float32)
    if img.ndim == 3:
        img = img[..., [2, 1, 0, 3]] if img.shape[2] == 4 else img[..., ::-1]
    return np.ascontiguousarray(img)


def write(path: Path, img: np.ndarray, bit_depth: int) -> None:
    """Write float 0..1 RGB(A) or gray as 8- or 16-bit PNG."""
    out = np.clip(img, 0.0, 1.0)
    if out.ndim == 3:
        out = out[..., [2, 1, 0, 3]] if out.shape[2] == 4 else out[..., ::-1]
    scale, dtype = (65535.0, np.uint16) if bit_depth == 16 else (255.0, np.uint8)
    out = (out * scale + 0.5).astype(dtype)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(out), [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise IOError(f"cannot encode {path}")
    # Write then rename: an interrupted run never leaves a truncated frame that looks up to date.
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(buf.tobytes())
    os.replace(tmp, path)


def frame_path(folder: Path, index: int, ext: str = ".png") -> Path:
    """index is 0-based; files are numbered from START_NUMBER."""
    return Path(folder) / (FRAME_PATTERN % (index + START_NUMBER)).replace(".png", ext)


def write_exr(path: Path, img: np.ndarray) -> None:
    """Write float32 data as it is (raw model output that has no fixed range)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out = img
    if img.ndim == 3:
        out = img[..., [2, 1, 0, 3]] if img.shape[2] == 4 else img[..., ::-1]
    if not cv2.imwrite(str(path), np.ascontiguousarray(out, np.float32),
                       [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]):
        raise IOError(f"cannot write {path}")


EXR_COMPRESSION = {"zip": "ZIP_COMPRESSION", "piz": "PIZ_COMPRESSION", "none": "NO_COMPRESSION"}


def write_exr_half(path: Path, img: np.ndarray, compression: str = "zip") -> None:
    """Half-float RGB(A) EXR through the OpenEXR library (core env only): it compresses on its own
    thread pool, ~5x faster than cv2's writer for a 4K frame."""
    import OpenEXR
    if OpenEXR.global_thread_count() == 0:
        OpenEXR.set_global_thread_count(min(8, os.cpu_count() or 4))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    name = "RGBA" if img.ndim == 3 and img.shape[2] == 4 else "RGB" if img.ndim == 3 else "Y"
    header = {"compression": getattr(OpenEXR, EXR_COMPRESSION[compression]), "type": OpenEXR.scanlineimage}
    with OpenEXR.File(header, {name: np.ascontiguousarray(img, np.float16)}) as f:
        f.write(str(path))


def frame_files(folder: Path) -> list[Path]:
    return sorted(Path(folder).glob("frame_*.png"))


def clear_frames(folder: Path) -> None:
    for f in frame_files(folder):
        f.unlink()


def _natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def list_sequence(folder: Path) -> list[Path]:
    """Image files in a folder, in natural order (frame2 before frame10)."""
    files = [p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    if not files:
        raise FileNotFoundError(f"no images ({', '.join(sorted(IMAGE_EXTS))}) in {folder}")
    exts = {p.suffix.lower() for p in files}
    if len(exts) > 1:
        raise ValueError(f"mixed image types in {folder}: {', '.join(sorted(exts))}")
    return sorted(files, key=_natural_key)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def parallel(fn: Callable[[int], None], count: int, label: str, workers: int | None = None) -> None:
    """Run fn(i) for every frame on a thread pool (OpenCV releases the GIL)."""
    workers = workers or min(8, os.cpu_count() or 4)
    done = 0
    step = max(1, count // 20)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(fn, range(count)):
            done += 1
            if done % step == 0 or done == count:
                print(f"\r  {label}: {done}/{count}", end="", flush=True)
    print()


# ------------------------------------------------------- color / matting --

def despill(rgb: np.ndarray, color: str, amount: float = 1.0) -> np.ndarray:
    """Limit the screen channel to the average of the other two ('average' despill)."""
    if color not in ("green", "blue") or amount <= 0:
        return rgb
    out = rgb.copy()
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if color == "green":
        limit = (r + b) * 0.5
        out[..., 1] = g - amount * np.maximum(g - limit, 0.0)
    else:
        limit = (r + g) * 0.5
        out[..., 2] = b - amount * np.maximum(b - limit, 0.0)
    return out


def normalize_alpha(folder: Path, source_bits: int, count: int) -> None:
    """ffmpeg widens N-bit alpha to 16-bit by bit-shifting, so opaque lands at e.g. 65280
    (8-bit) or 65520 (12-bit) instead of 65535. Rescale so opaque is exactly 1.0."""
    if source_bits >= 16:
        return
    full = ((1 << source_bits) - 1) << (16 - source_bits)
    lut = np.clip(np.round(np.arange(65536, dtype=np.float64) * 65535.0 / full), 0, 65535).astype(np.uint16)

    def one(i: int) -> None:
        p = frame_path(folder, i)
        a = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if a is None or a.dtype != np.uint16:
            return
        cv2.imwrite(str(p), lut[a], [cv2.IMWRITE_PNG_COMPRESSION, 3])

    parallel(one, count, "alpha")


def make_plate(rgb: np.ndarray, alpha: np.ndarray | None, grey: float) -> np.ndarray:
    """Composite the (straight, not premultiplied) foreground over neutral grey."""
    if alpha is None:
        return rgb
    a = alpha[..., None] if alpha.ndim == 2 else alpha
    return rgb * a + grey * (1.0 - a)


def bleed_color(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Fill fully transparent pixels with the nearby foreground color (push-pull pyramid).

    Straight-alpha color is black outside the subject; bilinear filtering or mips at a soft
    edge would then mix that black in and leave a dark fringe. Visible pixels are unchanged.
    """
    known = (alpha > 0.0).astype(np.float32)
    if known.min() > 0.0 or known.max() == 0.0:
        return rgb
    pyr = [(rgb * known[..., None], known)]
    while max(pyr[-1][1].shape) > 1:          # down to 1x1, so every pixel gets some color
        c, w = pyr[-1]
        pyr.append((cv2.pyrDown(c), cv2.pyrDown(w)))
    c, w = pyr[-1]
    fill = c / np.maximum(w, 1e-6)[..., None]
    for c, w in reversed(pyr[:-1]):
        fill = cv2.resize(fill, (w.shape[1], w.shape[0]), interpolation=cv2.INTER_LINEAR)
        t = np.clip(w, 0.0, 1.0)[..., None]
        fill = c / np.maximum(w, 1e-6)[..., None] * t + fill * (1.0 - t)
    return np.where(known[..., None] > 0.0, rgb, fill).astype(np.float32)


def _ellipse(size: int) -> np.ndarray:
    size = max(1, int(size) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _norm_blur(img: np.ndarray, weight: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Weighted (normalized) Gaussian blur: average of img over pixels where weight=1."""
    w = cv2.GaussianBlur(weight, (0, 0), sigma)
    s = cv2.GaussianBlur(img * weight[..., None], (0, 0), sigma)
    return s / np.maximum(w[..., None], 1e-4), w


def luma_key_black(rgb: np.ndarray, *, edge: int, threshold: float = 0.10, min_area: float = 0.002,
                   garbage: Iterable[tuple[int, int, int, int]] = ()) -> tuple[np.ndarray, np.ndarray]:
    """Key a subject that was shot or composited over pure black.

    Returns (alpha, foreground) with the foreground as straight (un-premultiplied) color.
      1. Silhouette = pixels brighter than `threshold`, small specks dropped, enclosed holes
         filled (dark fur and shadows inside the body stay solid).
      2. Inside the silhouette, shrunk by `edge` px: alpha 1, original color.
      3. In the edge band: over black, pixel = fur color x alpha, so alpha = brightness /
         brightness of the nearby solid fur, and the color is that nearby fur color. This
         turns a dark compositing fringe into soft translucency instead of a dark outline.
    `garbage` rectangles (x0, y0, x1, y1) are forced transparent (e.g. a floor reflection).
    """
    h, w = rgb.shape[:2]
    v = rgb.max(axis=2)
    m = (v > threshold).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _ellipse(7))
    rects = [tuple(int(c) for c in r) for r in garbage]
    for x0, y0, x1, y1 in rects:
        m[y0:y1, x0:x1] = 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area * h * w]
    keep = np.isin(lab, big).astype(np.uint8)
    flood = keep.copy()
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    filled = keep | (flood == 0).astype(np.uint8)        # enclosed holes -> solid

    core = cv2.erode(filled, _ellipse(edge)).astype(np.float32)
    region = cv2.dilate(filled, _ellipse(edge * 2 // 3)).astype(np.float32)

    # Local fur color from the solid core; far-away edge pixels use a wider, cheaper estimate.
    fg_est, wt = _norm_blur(rgb, core, max(2.0, edge * 0.4))
    small = cv2.resize(rgb, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    core_s = cv2.resize(core, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    far, _ = _norm_blur(small, core_s, max(2.0, edge * 0.6))
    far = cv2.resize(far, (w, h), interpolation=cv2.INTER_LINEAR)
    fg_est = np.where((wt < 0.05)[..., None], far, fg_est)

    a_edge = np.clip(v / np.maximum(fg_est.max(axis=2), 0.05), 0.0, 1.0) ** 1.3
    alpha = np.maximum(core, a_edge) * region
    for x0, y0, x1, y1 in rects:
        alpha[y0:y1, x0:x1] = 0.0

    unpremult = np.clip(rgb / np.maximum(alpha[..., None], 1e-3), 0.0, 1.0)
    fg = np.where(core[..., None] > 0.5, rgb,
                  np.where(alpha[..., None] >= 0.6, unpremult, fg_est))
    fg = np.where(alpha[..., None] > 0.0, fg, 0.0)
    return alpha.astype(np.float32), fg.astype(np.float32)


def premultiplied_stats(rgb: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evidence for premultiplied color in a frame's soft edge (values as encoded, 0..1).

    Returns per edge pixel (alpha 0.05..0.9): `over` = max(rgb) exceeds alpha beyond codec noise
    (impossible when premultiplied), and for alpha 0.15..0.85 the brightness relative to the
    nearby solid subject `ratio` with its `alpha`: premultiplied color tracks alpha, straight
    color stays near 1 (darker fur tips: somewhat below).
    """
    band = (alpha > 0.05) & (alpha < 0.9)
    over = rgb.max(axis=2)[band] > alpha[band] * 1.15 + 0.03
    lum = rgb.mean(axis=2)
    solid = (alpha > 0.98).astype(np.float32)
    ref, wt = _norm_blur(rgb, solid, max(2.0, rgb.shape[0] / 70))
    ref = ref.mean(axis=2)
    mid = (alpha > 0.15) & (alpha < 0.85) & (wt > 0.05) & (ref > 0.05)
    return over, (lum / np.maximum(ref, 1e-4))[mid], alpha[mid]


def is_premultiplied(over: np.ndarray, ratio: np.ndarray, alpha: np.ndarray) -> tuple[bool, str]:
    """Decide from premultiplied_stats gathered over a few frames; returns (verdict, reason)."""
    if over.size < 500 or ratio.size < 500:
        return False, f"too few soft-edge pixels to tell ({ratio.size}); assuming straight"
    share_over = float(over.mean())
    d_pre, d_str = float(np.median(np.abs(ratio - alpha))), float(np.median(np.abs(ratio - 1.0)))
    verdict = share_over < 0.05 and d_pre < d_str
    return verdict, (f"{share_over:.1%} of edge pixels brighter than their alpha; edge brightness vs alpha "
                     f"{d_pre:.2f} / vs 1 {d_str:.2f}")


def unpremultiply(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Straight color = premultiplied / alpha (0 where alpha is 0)."""
    a = alpha[..., None]
    return np.where(a > 1.0 / 1024, np.clip(rgb / np.maximum(a, 1.0 / 1024), 0.0, 1.0), 0.0).astype(np.float32)


def parse_rects(values: Iterable[str]) -> list[tuple[int, int, int, int]]:
    """'x0:y0:x1:y1' pixel rectangles (end exclusive)."""
    out = []
    for v in values or []:
        parts = v.replace(",", ":").split(":")
        if len(parts) != 4:
            raise ValueError(f"--garbage expects x0:y0:x1:y1, got '{v}'")
        x0, y0, x1, y1 = (int(p) for p in parts)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"--garbage rectangle '{v}' is empty")
        out.append((x0, y0, x1, y1))
    return out


def sample_key_color(img: np.ndarray, screen: str) -> str:
    """Median color of the clearly screen-colored pixels, as 0xRRGGBB for ffmpeg."""
    rgb = img[..., :3]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if screen == "green":
        mask = (g > r * 1.25) & (g > b * 1.25) & (g > 0.15)
    else:
        mask = (b > r * 1.25) & (b > g * 1.15) & (b > 0.15)
    if mask.mean() < 0.02:
        raise ValueError(f"could not find a {screen} screen in the first frame; "
                         f"pass the key color explicitly, e.g. --key #3FB24A")
    med = np.median(rgb[mask], axis=0)
    return "0x" + "".join(f"{int(round(c * 255)):02X}" for c in med)


def parse_key(value: str) -> tuple[str, str | None]:
    """'green' | 'blue' | '#RRGGBB' | '0xRRGGBB' -> (screen kind, explicit color or None)."""
    v = value.strip().lower()
    if v in ("green", "blue", "black"):
        return v, None
    hexpart = v[1:] if v.startswith("#") else v[2:] if v.startswith("0x") else v
    if not re.fullmatch(r"[0-9a-f]{6}", hexpart):
        raise ValueError(f"--key must be green, blue, black or a hex color like #3FB24A, not '{value}'")
    r, g, b = (int(hexpart[i:i + 2], 16) for i in (0, 2, 4))
    return ("green" if g >= b else "blue"), "0x" + hexpart.upper()


# ------------------------------------------------------ image sequences --

def import_sequence(files: Iterable[Path], out_rgb: Path, out_alpha: Path | None, bit_depth: int) -> dict:
    """Copy an arbitrary image sequence into frame_%06d.png folders. EXR is treated as linear."""
    files = list(files)
    first = read(files[0])
    h, w = first.shape[:2]
    has_alpha = first.ndim == 3 and first.shape[2] == 4
    is_exr = files[0].suffix.lower() == ".exr"

    def one(i: int) -> None:
        img = read(files[i])
        if img.shape[:2] != (h, w):
            raise ValueError(f"{files[i].name} is {img.shape[1]}x{img.shape[0]}, expected {w}x{h}")
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        rgb = img[..., :3]
        if is_exr:
            rgb = linear_to_srgb(rgb)
        write(frame_path(out_rgb, i), rgb, bit_depth)
        if out_alpha is not None and img.shape[2] == 4:
            write(frame_path(out_alpha, i), img[..., 3], bit_depth)

    parallel(one, len(files), "import")
    return {"width": w, "height": h, "has_alpha": has_alpha, "linear_exr": is_exr}


def sequence_info(first_file: Path) -> dict:
    """Size, channels and bit depth of an image sequence, from its first file."""
    raw = cv2.imread(str(first_file), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise IOError(f"cannot read image {first_file}")
    channels = 1 if raw.ndim == 2 else raw.shape[2]
    depth = 8 if raw.dtype == np.uint8 else 16      # uint16 and float (EXR) both keep 16-bit
    return {"width": raw.shape[1], "height": raw.shape[0], "channels": channels,
            "has_alpha": channels == 4, "bit_depth": depth, "dtype": str(raw.dtype)}
