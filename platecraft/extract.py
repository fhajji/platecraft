"""`platecraft extract`: video or image sequence -> frame folders + manifest.json.

Output in <frames-dir>:
    src/    RGB frames as decoded (sRGB, 8- or 16-bit)
    alpha/  matte frames, white = foreground (only when there is an alpha)
    plate/  model input: despilled foreground over neutral grey (only when needed)
    manifest.json
"""
from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import ffmpeg_io, frames, manifest, paths


@dataclass
class ExtractOptions:
    input: Path
    frames_dir: Path | None = None
    start: str | None = None          # frame number, f120, 4.5s, 00:00:04.5
    end: str | None = None            # exclusive
    max_frames: int | None = None
    fps: str | None = None            # required for image sequences without a manifest; overrides video fps
    bit_depth: str = "auto"           # auto | 8 | 16
    alpha: str = "auto"               # auto | embedded | none
    matte: Path | None = None
    key: str | None = None            # green | blue | #RRGGBB
    key_similarity: float = 0.10
    key_blend: float = 0.08
    key_edge: int | None = None       # black key: edge band in px (default: height / 35)
    garbage: list[str] = field(default_factory=list)   # x0:y0:x1:y1 rectangles forced transparent
    despill: str = "auto"             # auto | green | blue | none
    despill_amount: float = 1.0
    plate_grey: float = 0.5
    premultiplied: str = "auto"       # auto | yes | no: color of embedded alpha / matte footage
    force: bool = False


def default_frames_dir(input_path: Path) -> Path:
    return paths.ROOT / "work" / Path(input_path).stem


def _settings(o: ExtractOptions) -> dict:
    """What decides the extracted content; a change here means re-extract."""
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(o).items()
            if k not in ("frames_dir", "force")}


def _resolve_alpha_mode(o: ExtractOptions, has_embedded: bool) -> str:
    if o.matte and o.key:
        raise ValueError("use either --matte or --key, not both")
    if o.garbage and not o.key:
        raise ValueError("--garbage is only used with --key")
    if o.matte:
        return "matte"
    if o.key:
        return "lumakey" if frames.parse_key(o.key)[0] == "black" else "chromakey"
    if o.alpha == "embedded":
        if not has_embedded:
            raise ValueError("--alpha embedded: the input has no alpha channel")
        return "embedded"
    if o.alpha == "auto" and has_embedded:
        return "embedded"
    return "none"


def run(o: ExtractOptions) -> Path:
    src_path = Path(o.input)
    if not src_path.exists():
        raise FileNotFoundError(f"input not found: {src_path}")
    clip = Path(o.frames_dir) if o.frames_dir else default_frames_dir(src_path)
    clip.mkdir(parents=True, exist_ok=True)
    d_src, d_alpha, d_plate = clip / "src", clip / "alpha", clip / "plate"

    m = manifest.load(clip)
    settings = _settings(o)
    prev = m.get("extract", {}).get("settings") if m else None
    if prev is not None:      # options added since that extraction count as their defaults
        prev = {**_settings(ExtractOptions(input=o.input)), **prev}
    if m and not o.force and prev == settings \
            and manifest.count_frames(d_src) == m.get("frames", {}).get("count", -1) > 0:
        print(f"Already extracted ({m['frames']['count']} frames) in {clip}. Use --force to redo.")
        return clip
    if m and not o.force and m.get("models"):
        raise RuntimeError(f"{clip} already has model outputs. Re-extracting would invalidate them; "
                           "use --force or a different --frames-dir.")
    for d in (d_src, d_alpha, d_plate):
        frames.clear_frames(d)
    m = manifest.new(clip)
    warnings: list[str] = []
    rects = frames.parse_rects(o.garbage)

    # ---------------------------------------------------------- decode --
    if src_path.is_dir():
        files = frames.list_sequence(src_path)
        si = frames.sequence_info(files[0])
        fps = o.fps or "24"
        if not o.fps:
            warnings.append("image sequence without --fps: assuming 24 fps")
        fps_float = float(eval_fps(fps))
        start = ffmpeg_io.parse_time(o.start, fps_float) or 0
        end = ffmpeg_io.parse_time(o.end, fps_float)
        files = files[start:end]
        if o.max_frames:
            files = files[:o.max_frames]
        depth = si["bit_depth"] if o.bit_depth == "auto" else int(o.bit_depth)
        mode = _resolve_alpha_mode(o, si["has_alpha"])
        if mode == "chromakey":
            raise ValueError("--key works on video input only; key the sequence first or pass --matte")
        info = frames.import_sequence(files, d_src, d_alpha if mode == "embedded" else None, depth)
        if info["linear_exr"]:
            warnings.append("EXR input treated as linear Rec.709 and converted to sRGB")
        width, height, count = info["width"], info["height"], len(files)
        m["source"] = {"path": str(src_path), "type": "sequence", "first_file": files[0].name,
                       "width": width, "height": height, "dtype": si["dtype"], "has_alpha": si["has_alpha"]}
        key_info = None
        if mode == "matte":
            _sequence_matte(Path(o.matte), d_alpha, depth, (width, height), start, end, o.max_frames)
    else:
        vi = ffmpeg_io.probe(src_path)
        warnings += vi.warnings
        fps = o.fps or vi.fps
        fps_float = float(eval_fps(fps))
        start = ffmpeg_io.parse_time(o.start, fps_float) or 0
        end = ffmpeg_io.parse_time(o.end, fps_float)
        if o.max_frames:
            end = min(end if end is not None else start + o.max_frames, start + o.max_frames)
        if start >= vi.frame_count > 0:
            raise ValueError(f"--start {o.start} is past the end of the clip ({vi.frame_count} frames)")
        depth = (16 if vi.bit_depth > 8 else 8) if o.bit_depth == "auto" else int(o.bit_depth)
        mode = _resolve_alpha_mode(o, vi.has_alpha)
        width, height = vi.width, vi.height

        key_info = None
        if mode == "chromakey":
            screen, color = frames.parse_key(o.key)
            if color is None:
                with tempfile.TemporaryDirectory() as tmp:
                    first = frames.read(ffmpeg_io.grab_frame(vi, start, Path(tmp) / "key.png"))
                color = frames.sample_key_color(first, screen)
                print(f"  sampled {screen} screen color {color}")
            key_info = {"screen": screen, "color": color, "similarity": o.key_similarity, "blend": o.key_blend,
                        "garbage": [list(r) for r in rects]}
            warnings.append("basic ffmpeg chroma key: a proper key from Resolve/Nuke/AE gives cleaner edges")

        print(f"Extracting {src_path.name}: {width}x{height} @ {fps} fps, {depth}-bit, alpha: {mode}")
        ffmpeg_io.extract_frames(
            vi, d_src, bit_depth=depth, start_frame=start, end_frame=end,
            out_alpha=d_alpha if mode in ("embedded", "chromakey") else None,
            key=None if mode != "chromakey" else {"color": key_info["color"],
                                                  "similarity": key_info["similarity"],
                                                  "blend": key_info["blend"]})
        alpha_bits = 16      # lumakey alpha is computed in float and written as true 16-bit
        if mode == "chromakey":
            alpha_bits = 8
        elif mode == "embedded":
            alpha_bits = vi.bit_depth
        if mode == "matte":
            mi = ffmpeg_io.probe(Path(o.matte)) if Path(o.matte).is_file() else None
            if mi:
                ffmpeg_io.extract_matte(mi, d_alpha, bit_depth=depth, size=(width, height),
                                        start_frame=start, end_frame=end)
                alpha_bits = mi.bit_depth
            else:
                _sequence_matte(Path(o.matte), d_alpha, depth, (width, height), start, end, o.max_frames)
        count = manifest.count_frames(d_src)
        if mode in ("embedded", "chromakey", "matte") and depth == 16:
            frames.normalize_alpha(d_alpha, alpha_bits, manifest.count_frames(d_alpha))
        m["source"] = {"type": "video", **vi.to_dict()}
        m["source"].pop("warnings", None)

    if count == 0:
        raise RuntimeError("no frames were extracted (check --start/--end)")

    # ------------------------------------------------------ black key --
    d_fg = clip / "fg"
    frames.clear_frames(d_fg)
    if mode == "lumakey":
        edge = o.key_edge or (round(height / 35) | 1)
        key_info = {"screen": "black", "edge_px": edge, "garbage": [list(r) for r in rects]}
        print(f"Keying black background (edge band {edge}px"
              + (f", {len(rects)} garbage rectangle(s)" if rects else "") + ")")

        def key_one(i: int) -> None:
            rgb = frames.read(frames.frame_path(d_src, i))[..., :3]
            a, fg = frames.luma_key_black(rgb, edge=edge, garbage=rects)
            frames.write(frames.frame_path(d_alpha, i), a, 16)
            frames.write(frames.frame_path(d_fg, i), fg, depth)

        frames.parallel(key_one, count, "key")
        warnings.append("black key assumes the subject was shot or composited over pure black; "
                        "check fg/ and alpha/ before relying on it")
    elif mode == "chromakey" and rects:
        print(f"Applying {len(rects)} garbage rectangle(s) to the chroma key")

        def garbage_one(i: int) -> None:
            p = frames.frame_path(d_alpha, i)
            a = frames.read(p)
            for x0, y0, x1, y1 in rects:
                a[y0:y1, x0:x1] = 0.0
            frames.write(p, a, depth)

        frames.parallel(garbage_one, count, "garbage")

    # ---------------------------------------------------------- alpha --
    if mode != "none":
        n_alpha = manifest.count_frames(d_alpha)
        if n_alpha < count:
            raise RuntimeError(f"matte has {n_alpha} frames but the plate has {count}")
        for extra in frames.frame_files(d_alpha)[count:]:
            extra.unlink()       # a longer matte is trimmed to the plate

    # ---------------------------------------------------- premultiplied --
    premult = None
    if mode in ("embedded", "matte"):
        premult, reason = _premultiplied(o.premultiplied, d_src, d_alpha, count)
        print(f"Color: {'premultiplied' if premult else 'straight'} ({reason})")
        if premult:
            def unpremult_one(i: int) -> None:
                rgb = frames.read(frames.frame_path(d_src, i))[..., :3]
                a = frames.read(frames.frame_path(d_alpha, i))
                frames.write(frames.frame_path(d_fg, i), frames.unpremultiply(rgb, a), depth)

            frames.parallel(unpremult_one, count, "unpremultiply")

    # ---------------------------------------------------------- plate --
    spill = o.despill
    if spill == "auto":
        spill = key_info["screen"] if key_info and key_info["screen"] in ("green", "blue") else "none"
    needs_plate = mode != "none" or spill != "none"
    if needs_plate:
        print(f"Preparing model plate (despill: {spill}, background grey {o.plate_grey})")

        color_dir = d_fg if mode == "lumakey" or premult else d_src

        def one(i: int) -> None:
            rgb = frames.read(frames.frame_path(color_dir, i))[..., :3]
            rgb = frames.despill(rgb, spill, o.despill_amount)
            a = frames.read(frames.frame_path(d_alpha, i)) if mode != "none" else None
            if a is not None and a.ndim == 3:
                a = a[..., 0]
            frames.write(frames.frame_path(d_plate, i), frames.make_plate(rgb, a, o.plate_grey), depth)

        frames.parallel(one, count, "plate")
    else:
        print("No alpha and no despill: the models will use src/ directly (no plate/ folder).")
        if d_plate.exists():
            shutil.rmtree(d_plate, ignore_errors=True)

    # ------------------------------------------------------- manifest --
    m["frames"] = {"count": count, "width": width, "height": height, "fps": str(fps),
                   "fps_float": round(fps_float, 6), "pattern": manifest.FRAME_PATTERN,
                   "start_number": manifest.START_NUMBER, "source_start_frame": start}
    m["alpha"] = {"source": mode, **({"key": key_info} if key_info else {}),
                  **({"matte": str(o.matte)} if o.matte else {}),
                  **({"premultiplied": premult} if premult is not None else {})}
    manifest.add_pass(m, "src", dir="src", colorspace="sRGB", bit_depth=depth, channels=3)
    if mode == "lumakey":
        manifest.add_pass(m, "fg", dir="fg", colorspace="sRGB", bit_depth=depth, channels=3,
                          note="straight (un-premultiplied) foreground color with the black fringe removed")
    elif premult:
        manifest.add_pass(m, "fg", dir="fg", colorspace="sRGB", bit_depth=depth, channels=3,
                          note="straight foreground color: src/ divided by alpha (src/ is premultiplied)")
    if mode != "none":
        manifest.add_pass(m, "alpha", dir="alpha", colorspace="linear",
                          bit_depth=16 if mode == "lumakey" else depth, channels=1)
    manifest.add_pass(m, "plate", dir="plate" if needs_plate else "src", colorspace="sRGB",
                      bit_depth=depth, channels=3, despill=spill, grey=o.plate_grey, model_input=True)
    m["extract"] = {"settings": settings, "warnings": warnings}
    manifest.log(m, "extract", frames=count)
    manifest.save(clip, m)

    for w in warnings:
        print(f"  WARNING: {w}")
    print(f"Done: {count} frames -> {clip}")
    return clip


def _premultiplied(choice: str, d_src: Path, d_alpha: Path, count: int) -> tuple[bool, str]:
    """--premultiplied yes/no, or auto: tested on 6 frames spread over the clip."""
    if choice in ("yes", "no"):
        return choice == "yes", f"--premultiplied {choice}"
    import numpy as np
    over, ratio, alpha = [], [], []
    for i in sorted({int(round(x)) for x in np.linspace(0, count - 1, 6)}):
        a = frames.read(frames.frame_path(d_alpha, i))
        o, r, aa = frames.premultiplied_stats(frames.read(frames.frame_path(d_src, i))[..., :3],
                                              a[..., 0] if a.ndim == 3 else a)
        over.append(o), ratio.append(r), alpha.append(aa)
    verdict, reason = frames.is_premultiplied(np.concatenate(over), np.concatenate(ratio), np.concatenate(alpha))
    return verdict, f"detected: {reason}; override with --premultiplied yes|no"


def _sequence_matte(folder: Path, d_alpha: Path, depth: int, size: tuple[int, int],
                    start: int, end: int | None, max_frames: int | None) -> None:
    """A matte given as an image folder: first channel (or alpha if present) = matte."""
    import cv2
    files = frames.list_sequence(folder)[start:end]
    if max_frames:
        files = files[:max_frames]

    def one(i: int) -> None:
        img = frames.read(files[i])
        a = img[..., 3] if img.ndim == 3 and img.shape[2] == 4 else (img[..., 0] if img.ndim == 3 else img)
        if (a.shape[1], a.shape[0]) != size:
            a = cv2.resize(a, size, interpolation=cv2.INTER_CUBIC)
        frames.write(frames.frame_path(d_alpha, i), a, depth)

    frames.parallel(one, len(files), "matte")


def eval_fps(fps: str):
    from fractions import Fraction
    f = Fraction(str(fps))
    if f <= 0:
        raise ValueError(f"invalid fps '{fps}'")
    return f
