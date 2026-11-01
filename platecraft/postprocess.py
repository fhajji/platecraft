"""Raw model output -> final full-resolution passes, plus the derived (heuristic) maps.

Model conventions (measured, see CLAUDE.md):
    raw normals (NormalCrafter, Marigold after its adapter): camera space, +X right, +Y up,
    +Z toward the camera.
    raw depth (Video Depth Anything): relative inverse depth, larger = nearer, float EXR.
For footage on a camera-facing card, camera space equals the card's tangent space,
so the tangent-space conversion is only an axis convention:
    directx (UE): green = -Y (down);  opengl: green = +Y (up).

Upscaling to source resolution is guided by the full-resolution plate (fast guided
filter, He & Sun 2015): the linear fit between plate and model output is computed at
inference resolution and applied to the sharp plate, so silhouettes and edges land
where the footage has them. --upscale bicubic turns that off.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import frames   # first: enables OpenEXR before cv2 is imported
import cv2  # noqa: E402
import numpy as np  # noqa: E402

NORMAL_SPACES = ("directx", "opengl")
UPSCALE_MODES = ("guided", "bicubic")
FLAT = np.array([0.0, 0.0, 1.0], np.float32)


def _unit(n: np.ndarray) -> np.ndarray:
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)


# --------------------------------------------------------- upscaling --

def _box(x: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1)^2 window; keeps the array's shape whatever its channel count."""
    flat = x.reshape(x.shape[0], x.shape[1], -1)
    out = cv2.boxFilter(flat, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)
    return out.reshape(x.shape)


def _resize(x: np.ndarray, size: tuple[int, int], interp: int) -> np.ndarray:
    """cv2.resize for any channel count (cv2 handles at most 4 at once for some modes)."""
    flat = x.reshape(x.shape[0], x.shape[1], -1)
    parts = [cv2.resize(np.ascontiguousarray(flat[..., c:c + 4]), size, interpolation=interp)
             for c in range(0, flat.shape[2], 4)]
    parts = [p[..., None] if p.ndim == 2 else p for p in parts]
    return np.concatenate(parts, axis=2).reshape(size[1], size[0], *x.shape[2:])


def _inv3(m: np.ndarray) -> np.ndarray:
    """Inverse of a stack of symmetric 3x3 matrices (..., 3, 3) by cofactors
    (np.linalg.inv/solve on ~300k tiny matrices is ~100x slower)."""
    a, b, c = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    e, f, i = m[..., 1, 1], m[..., 1, 2], m[..., 2, 2]
    c00, c01, c02 = e * i - f * f, c * f - b * i, b * f - c * e
    c11, c12, c22 = a * i - c * c, b * c - a * f, a * e - b * b
    det = a * c00 + b * c01 + c * c02
    inv = np.stack([c00, c01, c02, c01, c11, c12, c02, c12, c22], -1).reshape(m.shape)
    return inv / det[..., None, None]


def guided_upsample(src: np.ndarray, guide_lo: np.ndarray, guide_hi: np.ndarray,
                    radius: int = 2, eps: float = 1e-3) -> np.ndarray:
    """Fast guided upsampling with a color guide.

    src (h, w[, C]) at low resolution; guide_lo (h, w, 3) = guide_hi (H, W, 3) downscaled.
    Per window, src ~ a . guide + b is fitted at low resolution (eps regularises a: guide
    contrast below sqrt(eps) is not transferred); a and b are averaged, upsampled and
    applied to the full-resolution guide.
    """
    p = src[..., None] if src.ndim == 2 else src
    i_lo = guide_lo
    m_i = _box(i_lo, radius)
    m_p = _box(p, radius)
    cov_ii = _box(i_lo[..., :, None] * i_lo[..., None, :], radius) - m_i[..., :, None] * m_i[..., None, :]
    cov_ii += eps * np.eye(3, dtype=np.float32)
    cov_ip = _box(i_lo[..., :, None] * p[..., None, :], radius) - m_i[..., :, None] * m_p[..., None, :]
    a = np.einsum("hwij,hwjc->hwic", _inv3(cov_ii), cov_ip)          # (h, w, 3, C)
    b = m_p - np.einsum("hwic,hwi->hwc", a, m_i)
    H, W = guide_hi.shape[:2]
    a = _resize(_box(a, radius), (W, H), cv2.INTER_LINEAR)
    b = _resize(_box(b, radius), (W, H), cv2.INTER_LINEAR)
    q = np.einsum("hwic,hwi->hwc", a, guide_hi) + b
    return q[..., 0] if src.ndim == 2 else q


@dataclass
class Upscaler:
    """Brings inference-resolution maps to the clip's resolution, optionally guided by the plate."""
    size: tuple[int, int]                 # (w, h) of the clip
    guide_dir: Path | None = None         # full-resolution plate
    mode: str = "guided"

    def __post_init__(self):
        if self.mode not in UPSCALE_MODES:
            raise ValueError(f"upscale must be one of {', '.join(UPSCALE_MODES)}")

    def __call__(self, img: np.ndarray, i: int, eps: float) -> np.ndarray:
        h, w = img.shape[:2]
        if (w, h) == tuple(self.size):
            return img
        if self.mode == "bicubic" or self.guide_dir is None:
            return cv2.resize(img, self.size, interpolation=cv2.INTER_CUBIC)
        hi = frames.read(frames.frame_path(self.guide_dir, i))[..., :3]
        if (hi.shape[1], hi.shape[0]) != tuple(self.size):
            hi = cv2.resize(hi, self.size, interpolation=cv2.INTER_AREA)
        lo = cv2.resize(hi, (w, h), interpolation=cv2.INTER_AREA)
        radius = max(1, round(2 * max(w, h) / 768))
        return guided_upsample(img, lo, hi, radius, eps)


def _alpha(alpha_dir: Path | None, i: int, size: tuple[int, int] | None = None) -> np.ndarray | None:
    if alpha_dir is None:
        return None
    a = frames.read(frames.frame_path(alpha_dir, i))
    if size and (a.shape[1], a.shape[0]) != tuple(size):
        a = cv2.resize(a, size, interpolation=cv2.INTER_AREA)
    return a


# ----------------------------------------------------------- normals --

def normals(raw_dir: Path, out_dir: Path, alpha_dir: Path | None, indices: list[int],
            up: Upscaler, space: str = "directx") -> None:
    """Upscale raw camera-space normals, flatten the background, encode for UE (16-bit)."""
    if space not in NORMAL_SPACES:
        raise ValueError(f"normal space must be one of {', '.join(NORMAL_SPACES)}")
    flip = np.array([1.0, -1.0 if space == "directx" else 1.0, 1.0], np.float32)

    def one(j: int) -> None:
        i = indices[j]
        n = frames.read(frames.frame_path(raw_dir, i))[..., :3] * 2 - 1
        n = _unit(up(n, i, eps=2e-3))
        a = _alpha(alpha_dir, i)
        if a is not None:
            n[a <= 0.0] = FLAT
        frames.write(frames.frame_path(out_dir, i), n * flip * 0.5 + 0.5, 16)

    out_dir.mkdir(parents=True, exist_ok=True)
    frames.parallel(one, len(indices), "normal")


# ------------------------------------------------------------- depth --

def depth_range(raw_dir: Path, alpha_dir: Path | None, indices: list[int], ext: str = ".exr",
                samples: int = 48) -> tuple[float, float]:
    """Clip-wide disparity range of the subject (0.5 / 99.5 percentiles over sampled frames)."""
    pick = [indices[k] for k in np.linspace(0, len(indices) - 1, min(samples, len(indices))).round().astype(int)]
    vals = []
    for i in dict.fromkeys(pick):
        d = frames.read(frames.frame_path(raw_dir, i, ext))
        a = _alpha(alpha_dir, i, (d.shape[1], d.shape[0]))
        v = d[a > 0.5] if a is not None else d.ravel()
        vals.append(v[:: max(1, v.size // 20000)] if v.size else v)
    v = np.concatenate(vals) if vals else np.zeros(1, np.float32)
    if v.size < 16:
        v = np.concatenate([frames.read(frames.frame_path(raw_dir, i, ext)).ravel() for i in pick[:4]])
    lo, hi = np.percentile(v, [0.5, 99.5])
    return float(lo), float(max(hi, lo + 1e-6))


def depth(raw_dir: Path, out_dir: Path, alpha_dir: Path | None, indices: list[int], up: Upscaler,
          rng: tuple[float, float], ext: str = ".exr") -> None:
    """Relative inverse depth -> 16-bit: 1 = nearest subject point, 0 = farthest (and background)."""
    lo, hi = rng

    def one(j: int) -> None:
        i = indices[j]
        d = (frames.read(frames.frame_path(raw_dir, i, ext)) - lo) / (hi - lo)
        d = np.clip(up(d.astype(np.float32), i, eps=1e-3), 0.0, 1.0)
        a = _alpha(alpha_dir, i)
        if a is not None:
            d[a <= 0.0] = 0.0
        frames.write(frames.frame_path(out_dir, i), d, 16)

    out_dir.mkdir(parents=True, exist_ok=True)
    frames.parallel(one, len(indices), "depth")


# ------------------------------------------------ model material maps --

def model_albedo(raw_dir: Path, out_dir: Path, alpha_dir: Path | None, indices: list[int], up: Upscaler,
                 ext: str = ".png") -> None:
    """A model's base colour (sRGB-encoded) -> RGBA, straight alpha, 16-bit, source resolution."""
    def one(j: int) -> None:
        i = indices[j]
        rgb = np.clip(up(frames.read(frames.frame_path(raw_dir, i, ext))[..., :3], i, eps=1e-3), 0.0, 1.0)
        a = _alpha(alpha_dir, i)
        frames.write(frames.frame_path(out_dir, i),
                     np.dstack([rgb, a if a is not None else np.ones(rgb.shape[:2], np.float32)]), 16)

    out_dir.mkdir(parents=True, exist_ok=True)
    frames.parallel(one, len(indices), "albedo")


def model_gray(raw_dir: Path, out_dir: Path, indices: list[int], up: Upscaler, label: str,
               ext: str = ".png") -> None:
    """A model's single-channel material map (roughness, metallic) -> 16-bit, source resolution."""
    def one(j: int) -> None:
        i = indices[j]
        g = frames.read(frames.frame_path(raw_dir, i, ext))
        g = g if g.ndim == 2 else g[..., 0]
        frames.write(frames.frame_path(out_dir, i), np.clip(up(g.astype(np.float32), i, eps=1e-3), 0.0, 1.0), 16)

    out_dir.mkdir(parents=True, exist_ok=True)
    frames.parallel(one, len(indices), label)


def normals_from_depth(d: np.ndarray, relief: float, sigma: float = 1.5) -> np.ndarray:
    """Camera-space normals (+X right, +Y up, +Z to camera) of a normalised disparity map,
    read as a height field whose full 0..1 range spans `relief` x the image width."""
    z = cv2.GaussianBlur(d.astype(np.float32), (0, 0), sigma) * relief * d.shape[1]
    gx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3) / 8.0     # image y points down = -Y
    return _unit(np.dstack([-gx, gy, np.ones_like(z)]))


# ------------------------------------------------------ derived maps --

@dataclass
class DeriveOptions:
    roughness: float = 0.7          # base roughness inside the subject
    roughness_cavity: float = 0.15  # + this much in occluded creases (from AO)
    ao_strength: float = 1.0
    delight: float = 0.0            # 0..1: divide out low-frequency shading from the albedo
    edge_color: float = 1.0         # 0..1: lift darkened soft-edge color to the nearby solid color


def _masked_blur(x: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    w = cv2.GaussianBlur(weight, (0, 0), sigma)
    return cv2.GaussianBlur(x * weight, (0, 0), sigma) / np.maximum(w, 1e-4)


def ambient_occlusion(depth01: np.ndarray | None, normal: np.ndarray | None, alpha: np.ndarray | None,
                      strength: float = 1.0) -> np.ndarray:
    """Heuristic AO at the given resolution (1 = unoccluded).

    depth term: how far a pixel lies behind its masked neighbourhood at three radii
    (recesses between nearer surfaces); normal term: concavity = negative divergence
    of the normals' XY tilt (creases, folds).
    """
    ref = depth01 if depth01 is not None else normal[..., 2]
    h, w = ref.shape[:2]
    mask = alpha.astype(np.float32) if alpha is not None else np.ones((h, w), np.float32)
    occ = np.zeros((h, w), np.float32)
    if depth01 is not None:
        for frac, gain in ((1 / 200, 6.0), (1 / 80, 4.0), (1 / 30, 2.5)):
            around = _masked_blur(depth01, mask, max(1.0, h * frac))
            occ += gain * np.maximum(around - depth01, 0.0)
    if normal is not None:
        nx = cv2.GaussianBlur(normal[..., 0], (0, 0), max(1.0, h / 540))
        ny = cv2.GaussianBlur(normal[..., 1], (0, 0), max(1.0, h / 540))
        div = cv2.Sobel(nx, cv2.CV_32F, 1, 0, ksize=3) / 8.0 - cv2.Sobel(ny, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        occ += np.maximum(-div, 0.0) * (h / 108.0)          # resolution independent
    # The silhouette is where depth falls off and normals turn to the flat background: that
    # reads as a crease to both terms, but an outline is not occluded. Fade AO in from the edge.
    edge = max(3, int(h / 25))
    core = cv2.erode((mask > 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge, edge)))
    occ *= cv2.GaussianBlur(core.astype(np.float32), (0, 0), edge / 2)
    ao = np.clip(1.0 - strength * occ, 0.0, 1.0)
    ao = cv2.GaussianBlur(ao, (0, 0), max(0.75, h / 720))
    return np.where(mask > 0.0, ao, 1.0).astype(np.float32)


def straight_color(clip: Path, m: dict, i: int) -> tuple[np.ndarray, np.ndarray | None]:
    """The subject's own (un-premultiplied, despilled) color and its alpha."""
    p = m["passes"]
    a = frames.read(frames.frame_path(clip / p["alpha"]["dir"], i)) if "alpha" in p else None
    if "fg" in p:
        return frames.read(frames.frame_path(clip / p["fg"]["dir"], i))[..., :3], a
    plate = p.get("plate") or p["src"]
    rgb = frames.read(frames.frame_path(clip / plate["dir"], i))[..., :3]
    if a is not None and plate.get("model_input") and plate["dir"] != p["src"]["dir"]:
        grey = plate.get("grey", 0.5)       # plate = fg * a + grey * (1 - a)
        aa = a[..., None]
        rgb = np.where(aa > 1e-3, (rgb - grey * (1 - aa)) / np.maximum(aa, 1e-3), 0.0)
    return np.clip(rgb, 0.0, 1.0), a


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def delight(rgb: np.ndarray, alpha: np.ndarray | None, amount: float) -> np.ndarray:
    """Rough de-lighting: divide the linear color by its low-frequency luminance, keep the mean."""
    if amount <= 0:
        return rgb
    lin = srgb_to_linear(rgb)
    lum = lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    h = rgb.shape[0]
    mask = (alpha > 0.5).astype(np.float32) if alpha is not None else np.ones_like(lum)
    if mask.sum() < 16:
        return rgb
    shade = _masked_blur(lum, mask, h / 25)
    target = float((lum * mask).sum() / mask.sum())
    gain = (target / np.maximum(shade, 1e-3)) ** amount
    return frames.linear_to_srgb(lin * np.clip(gain, 0.25, 4.0)[..., None])


def edge_color(rgb: np.ndarray, alpha: np.ndarray | None, amount: float,
               lo: float = 0.5, hi: float = 0.98) -> np.ndarray:
    """Fix the color of soft-edge pixels from the nearby solid subject.

    Brightness: thin fur/hair keyed off a dark background keeps some of it in its color (kitty3:
    edges 10-25% darker than the fur next to them), a dark rim over a bright backdrop. In linear
    light that is a factor, so each edge pixel gets gain = solid / own luminance (only >= 1: never
    darkens); pixels too dark to lift (gain capped at 4) move toward the solid luminance instead.
    Hue: 4:2:0 sources smear the subject's chroma 1-2 px into the dark background, where it decodes
    as saturated red/blue (kitty3: R/G 8 at alpha < 0.1, fur 1.5), so the chromaticity comes from
    the nearby solid color. Both apply fully below alpha `lo` and fade out to none at `hi`.
    """
    if amount <= 0 or alpha is None:
        return rgb
    solid = (alpha >= hi).astype(np.float32)
    if solid.sum() < 16:
        return rgb
    lin = srgb_to_linear(rgb)
    lum_w = np.array([0.2126, 0.7152, 0.0722], np.float32)
    h = rgb.shape[0]
    est, wt = frames._norm_blur(lin, solid, max(2.0, h / 100))          # nearby solid color
    wide, _ = frames._norm_blur(lin, solid, max(4.0, h / 25))           # for pixels far from it
    est = np.where((wt < 0.05)[..., None], wide, est)
    ref = est @ lum_w
    lum = lin @ lum_w
    t = np.clip((hi - alpha) / (hi - lo), 0.0, 1.0) * min(amount, 1.0)
    lum_out = lum * np.clip(ref / np.maximum(lum, 1e-4), 1.0, 4.0) ** t
    lum_out += (ref - lum_out) * np.clip(1.0 - lum_out / np.maximum(ref, 1e-4), 0.0, 1.0) * t
    chroma = lin / np.maximum(lum, 1e-4)[..., None]                     # color at luminance 1
    chroma += (est / np.maximum(ref, 1e-4)[..., None] - chroma) * t[..., None]
    out = chroma * lum_out[..., None]
    out = out / np.maximum(out.max(axis=2, keepdims=True), 1.0)         # keep the hue when > 1
    return np.where((alpha > 0.0)[..., None], frames.linear_to_srgb(out), rgb).astype(np.float32)


def derive(clip: Path, m: dict, indices: list[int], o: DeriveOptions,
           which: set[str] = frozenset({"albedo", "ao", "roughness"})) -> list[str]:
    """Write albedo/ (RGBA), ao/ and roughness/ (those in `which`; a model may have made the others)
    for the given frames. Returns the passes written."""
    p = m["passes"]
    has_depth, has_normal = "depth" in p, "normal" in p
    bits = max(8, m["passes"]["src"]["bit_depth"])
    written = [k for k in ("albedo", "roughness") if k in which] + (["ao"] if has_depth or has_normal else [])

    def one(j: int) -> None:
        i = indices[j]
        rgb, a = straight_color(clip, m, i)
        alpha = a if a is not None else np.ones(rgb.shape[:2], np.float32)
        if "albedo" in which:
            if m.get("alpha", {}).get("source") != "lumakey":   # the black key's fg/ has its edge color fixed
                rgb = edge_color(rgb, a, o.edge_color)
            rgb = delight(rgb, a, o.delight)
            frames.write(frames.frame_path(clip / "albedo", i), np.dstack([rgb, alpha]), bits)

        h, w = rgb.shape[:2]
        small = (max(1, w // 2), max(1, h // 2))       # AO is low-frequency: half resolution is plenty
        d = n = None
        if has_depth:
            d = cv2.resize(frames.read(frames.frame_path(clip / p["depth"]["dir"], i)), small,
                           interpolation=cv2.INTER_AREA)
        if has_normal:
            n = cv2.resize(frames.read(frames.frame_path(clip / p["normal"]["dir"], i))[..., :3], small,
                           interpolation=cv2.INTER_AREA) * 2 - 1
            if p["normal"].get("normal_space", "directx") == "directx":
                n[..., 1] *= -1                       # back to +Y up
        a_small = cv2.resize(alpha, small, interpolation=cv2.INTER_AREA)
        ao = np.ones(small[::-1], np.float32)
        if d is not None or n is not None:
            ao = ambient_occlusion(d, n, a_small, o.ao_strength)
            frames.write(frames.frame_path(clip / "ao", i), cv2.resize(ao, (w, h), interpolation=cv2.INTER_LINEAR), 16)
        if "roughness" not in which:
            return
        rough = o.roughness + o.roughness_cavity * (1.0 - ao)
        rough = np.where(a_small > 0.0, np.clip(rough, 0.0, 1.0), o.roughness)
        frames.write(frames.frame_path(clip / "roughness", i),
                     cv2.resize(rough.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR), 16)

    frames.parallel(one, len(indices), "/".join(written))
    return written
