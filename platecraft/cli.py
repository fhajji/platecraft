"""PlateCraft command line."""
from __future__ import annotations

import argparse
import sys

from . import __version__
from .models.registry import MODELS, MODEL_SETS, COMMERCIAL_NO


def cmd_doctor(args) -> int:
    from . import doctor
    models = [m.strip() for m in args.model.split(",")] if args.model else None
    return doctor.run(models=models, as_json=args.json)


def cmd_list_models(args) -> int:
    shown = [m for m in MODELS.values() if not (args.commercial and m.commercial == COMMERCIAL_NO)]
    kw = max(len(m.key) for m in shown) + 2
    pw = max(len(",".join(m.passes) or "(filter)") for m in shown) + 2
    print(f"{'key':<{kw}}{'passes':<{pw}}{'commercial':<16}license")
    for m in shown:
        print(f"{m.key:<{kw}}{','.join(m.passes) or '(filter)':<{pw}}{m.commercial:<16}{m.license}")
    print("\nsets:")
    for name, keys in MODEL_SETS.items():
        print(f"  {name:<10}{', '.join(keys)}")
    return 0


def _extract(args):
    from pathlib import Path
    from .extract import ExtractOptions, run
    return run(ExtractOptions(
        input=Path(args.input), frames_dir=Path(args.frames_dir) if args.frames_dir else None,
        start=args.start, end=args.end, max_frames=args.max_frames, fps=args.fps,
        bit_depth=args.bit_depth, alpha=args.alpha, matte=Path(args.matte) if args.matte else None,
        key=args.key, key_similarity=args.key_similarity, key_blend=args.key_blend,
        key_edge=args.key_edge, garbage=args.garbage,
        despill=args.despill, despill_amount=args.despill_amount, plate_grey=args.plate_grey,
        premultiplied=args.premultiplied, force=args.force))


def cmd_extract(args) -> int:
    _extract(args)
    return 0


def cmd_encode(args) -> int:
    from pathlib import Path
    from . import ffmpeg_io, manifest
    folder = Path(args.folder)
    frames_dir, m, note = manifest.resolve_pass_dir(folder)
    if note:
        print(f"Note: {note}")
    fps = args.fps or (m["frames"]["fps"] if m else None)
    if not fps:
        print("No manifest found next to the frames: pass --fps.", file=sys.stderr)
        return 2
    alpha_dir = None
    if args.alpha:
        if not m or "alpha" not in m.get("passes", {}):
            print("--alpha: this clip has no alpha pass (extract it with --key, --matte or embedded alpha).",
                  file=sys.stderr)
            return 2
        alpha_dir = frames_dir.parent / m["passes"]["alpha"]["dir"]
        if frames_dir.name == "src" and "fg" in m["passes"]:
            print("Note: src/ is premultiplied over black for this clip; fg/ has the clean edge colors.")
    out = Path(args.output) if args.output else folder.with_name(folder.name + "_preview")
    result = ffmpeg_io.encode(frames_dir, out, fps=fps, codec=args.codec, alpha_dir=alpha_dir)
    print(f"Wrote {result}")
    return 0


def cmd_infer(args) -> int:
    from pathlib import Path
    from .infer import InferOptions, run
    from .postprocess import DeriveOptions
    run(InferOptions(clip_dir=Path(args.clip), models=args.model, preset=args.preset, max_res=args.max_res,
                     start=args.start, end=args.end, max_frames=args.max_frames,
                     normal_space=args.normal_space, temporal_filter=args.temporal_filter,
                     temporal_strength=args.temporal_strength, upscale=args.upscale, derive=not args.no_derive,
                     derive_options=DeriveOptions(roughness=args.roughness, roughness_cavity=args.roughness_cavity,
                                                  ao_strength=args.ao_strength, delight=args.delight,
                                                  edge_color=args.edge_color),
                     commercial=args.commercial, force=args.force, redo_passes=args.redo_passes))
    return 0


def cmd_pack(args) -> int:
    from pathlib import Path
    from .pack import PackOptions, run
    run(PackOptions(clip_dir=Path(args.clip), format=args.format, compression=args.exr_compression,
                    scale=args.scale, start=args.start, end=args.end, max_frames=args.max_frames,
                    force=args.force))
    return 0


def cmd_process(args) -> int:
    from .models import registry
    models = registry.resolve(args.model)      # fail on bad models before spending time on extraction
    registry.check_commercial(models, args.commercial)
    for m in models:
        adapter = registry.adapter(m)
        if hasattr(adapter, "preflight"):   # e.g. VRAM / Linux env checks
            adapter.preflight()
    clip = _extract(args)
    # --start/--end/--max-frames trim the extraction; infer and pack then take the whole clip.
    whole = argparse.Namespace(**{**vars(args), "clip": str(clip), "start": None, "end": None, "max_frames": None})
    print("\n== infer ==")
    cmd_infer(whole)
    print("\n== pack ==")
    cmd_pack(whole)
    print(f"\nDone: {clip}")
    return 0


def _add_range(p: argparse.ArgumentParser) -> None:
    p.add_argument("--start", help="first frame: 120, f120, 4.5s or 00:00:04.5")
    p.add_argument("--end", help="end (exclusive), same formats as --start")
    p.add_argument("--max-frames", type=int, help="limit the number of frames (handy for tests)")


def _add_extract_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--frames-dir", help="where to put the frames (default: work/<input name>)")
    p.add_argument("--fps", help="frame rate, e.g. 25 or 24000/1001 (needed for image sequences)")
    p.add_argument("--bit-depth", choices=["auto", "8", "16"], default="auto",
                   help="PNG bit depth; auto = 16 for 10-bit+ sources")
    g = p.add_argument_group("alpha / keying")
    g.add_argument("--alpha", choices=["auto", "embedded", "none"], default="auto",
                   help="use the input's own alpha channel (auto: if present)")
    g.add_argument("--matte", help="separate matte video or image folder (white = foreground)")
    g.add_argument("--premultiplied", choices=["auto", "yes", "no"], default="auto",
                   help="color of embedded-alpha / matte footage is premultiplied (auto: detected from "
                        "the soft edges); yes -> fg/ = src/ divided by alpha")
    g.add_argument("--key", help="green / blue (chroma key, color sampled), #RRGGBB, "
                                 "or black (subject over a pure black background)")
    g.add_argument("--key-similarity", type=float, default=0.10, help="chroma key tolerance (default 0.10)")
    g.add_argument("--key-blend", type=float, default=0.08, help="chroma key edge softness (default 0.08)")
    g.add_argument("--key-edge", type=int, help="black key: soft edge band in px (default height/35, 31 at 1080p)")
    g.add_argument("--garbage", action="append", default=[], metavar="X0:Y0:X1:Y1",
                   help="rectangle forced transparent, e.g. a floor strip (repeatable)")
    g.add_argument("--despill", choices=["auto", "green", "blue", "none"], default="auto",
                   help="spill removal for the model plate (auto: the --key screen color)")
    g.add_argument("--despill-amount", type=float, default=1.0)
    g.add_argument("--plate-grey", type=float, default=0.5, help="background value behind the subject (0..1)")


def _add_infer_options(p: argparse.ArgumentParser) -> None:
    from .infer import TEMPORAL_FILTER
    from .postprocess import NORMAL_SPACES, UPSCALE_MODES
    from .presets import PRESETS
    p.add_argument("--model", default="quality",
                   help="model keys or a set (comma-separated; see list-models). Default: quality "
                        "(NormalCrafter + VDA Small); commercial-safe: safe")
    p.add_argument("--preset", default="auto", choices=["auto", *PRESETS],
                   help="GPU memory preset (auto: from the detected VRAM)")
    p.add_argument("--max-res", type=int, help="inference long side in px (overrides the preset)")
    p.add_argument("--normal-space", choices=NORMAL_SPACES, default="directx",
                   help="tangent-space green convention: directx (Unreal, default) or opengl")
    p.add_argument("--temporal-filter", choices=TEMPORAL_FILTER, default="auto",
                   help="RAFT optical-flow smoothing of the normals (auto: for per-frame models like marigold-n)")
    p.add_argument("--temporal-strength", type=float, default=0.8,
                   help="share of the motion-compensated previous frame, 0..1 (default 0.8)")
    p.add_argument("--upscale", choices=UPSCALE_MODES, default="guided",
                   help="to source resolution: guided by the plate (default) or plain bicubic")
    dg = p.add_argument_group("derived maps (albedo, AO, roughness; heuristic)")
    dg.add_argument("--no-derive", action="store_true", help="skip albedo/ao/roughness")
    dg.add_argument("--roughness", type=float, default=0.7, help="base roughness (default 0.7)")
    dg.add_argument("--roughness-cavity", type=float, default=0.15,
                    help="extra roughness in occluded creases (default 0.15)")
    dg.add_argument("--ao-strength", type=float, default=1.0, help="ambient occlusion strength (default 1)")
    dg.add_argument("--delight", type=float, default=0.0,
                    help="0..1: remove low-frequency shading from the albedo (default 0 = keyed plate as is)")
    dg.add_argument("--edge-color", type=float, default=1.0,
                    help="0..1: fix soft-edge color (darkened or hue-shifted by a dark background) from the "
                         "nearby solid color (default 1; 0 = off; not used with the black key, whose fg/ is fixed "
                         "already)")
    p.add_argument("--commercial", action="store_true", help="refuse models not cleared for commercial use")
    p.add_argument("--redo-passes", action="store_true",
                   help="rewrite normal/depth/albedo/ao/roughness even where they look up to date "
                        "(normally skipped when newer than their inputs and made with the same settings); "
                        "model output stays cached")


def _add_pack_options(p: argparse.ArgumentParser) -> None:
    from .pack import COMPRESSIONS, FORMATS
    p.add_argument("--format", choices=FORMATS, default="exr",
                   help="exr (half float, default: faster to decode in Img Media) or png (16-bit)")
    p.add_argument("--exr-compression", choices=COMPRESSIONS, default="zip",
                   help="lossless EXR compression (default zip; none = largest files, cheapest to decode)")
    p.add_argument("--scale", type=float, default=1.0,
                   help="quadrant size relative to the clip (default 1: a 1080p clip gives a 4K atlas)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="platecraft",
                                description="Turn video into temporally consistent PBR frame sequences.")
    p.add_argument("--version", action="version", version=f"platecraft {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="check GPU, ffmpeg, engine environments and weights")
    d.add_argument("--model", help="models that must be installed (comma-separated); others are skipped if absent")
    d.add_argument("--json", action="store_true", help="machine-readable output")
    d.set_defaults(func=cmd_doctor)

    lm = sub.add_parser("list-models", help="show models, passes and licenses")
    lm.add_argument("--commercial", action="store_true", help="hide non-commercial models")
    lm.set_defaults(func=cmd_list_models)

    ex = sub.add_parser("extract", help="split a video (or image sequence) into frames + manifest",
                        description="Decode every frame losslessly, pull alpha (embedded, matte or chroma key) "
                                    "and prepare the model plate.")
    ex.add_argument("input", help="video file, or a folder with an image sequence")
    _add_range(ex)
    _add_extract_options(ex)
    ex.add_argument("--force", action="store_true", help="re-extract even if frames exist")
    ex.set_defaults(func=cmd_extract)

    en = sub.add_parser("encode", help="encode a frame folder to a preview video")
    en.add_argument("folder", help="a frame folder, e.g. work/clip/plate")
    en.add_argument("-o", "--output", help="output file (default: <folder>_preview.mov)")
    en.add_argument("--fps", help="frame rate (default: from manifest.json)")
    en.add_argument("--alpha", action="store_true",
                    help="merge the clip's alpha in: transparent ProRes 4444 (.mov)")
    en.add_argument("--codec", choices=["prores", "prores4444", "nvenc", "mpeg4"], default="prores",
                    help="prores (default), prores4444 (keeps alpha), nvenc (H.264 on the GPU), mpeg4")
    en.set_defaults(func=cmd_encode)

    inf = sub.add_parser("infer", help="run models on an extracted clip",
                         description="Run models on the clip's plate frames and write full-resolution passes. "
                                     "Interrupted runs continue where they stopped.")
    inf.add_argument("clip", help="clip folder from 'extract' (with manifest.json), e.g. work/clip")
    _add_range(inf)
    _add_infer_options(inf)
    inf.add_argument("--force", action="store_true", help="discard cached model output and run again")
    inf.set_defaults(func=cmd_infer)

    pk = sub.add_parser("pack", help="pack passes into 2x2 atlas frames for Unreal",
                        description="One image per frame: albedo+alpha | normal / AO-roughness-metallic | depth, "
                                    "all linear, 16 bits per channel. Writes <clip>/atlas/ and manifest['atlas'].")
    pk.add_argument("clip", help="clip folder with manifest.json, e.g. work/clip")
    _add_pack_options(pk)
    _add_range(pk)
    pk.add_argument("--force", action="store_true", help="repack frames that are up to date")
    pk.set_defaults(func=cmd_pack)

    pr = sub.add_parser("process", help="extract + infer + pack in one go",
                        description="Run extract, infer and pack on a video. Each step resumes or skips what is "
                                    "already done, so rerunning after an interruption continues.")
    pr.add_argument("input", help="video file, or a folder with an image sequence")
    _add_range(pr)
    _add_extract_options(pr)
    _add_infer_options(pr)
    _add_pack_options(pr)
    pr.add_argument("--force", action="store_true", help="redo every step (re-extract, rerun the models, repack)")
    pr.set_defaults(func=cmd_process)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except (ValueError, FileNotFoundError, RuntimeError, PermissionError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
