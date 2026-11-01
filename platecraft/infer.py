"""`platecraft infer`: run models on an extracted clip and write the final passes.

In <clip>:
    raw/<model>/      model output at inference resolution + settings.json (resume cache)
    raw/<model>+tf/   the same after --temporal-filter (per-frame normal models)
    <pass>/           final pass at source resolution: normal/, depth/
    albedo/ ao/ roughness/   derived from the plate and the passes (heuristic maps)
The models, their licenses and the settings are recorded in manifest.json.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from . import ffmpeg_io, frames, manifest, postprocess
from .models import registry
from .models.base import Job
from .presets import PRESETS, recommend

TEMPORAL_FILTER = ("auto", "on", "off")
DEPTH_ENCODING = "relative inverse depth, 1 = nearest, 0 = farthest; normalised over the clip (not metric)"


@dataclass
class InferOptions:
    clip_dir: Path
    models: str = "normalcrafter"
    preset: str = "auto"
    max_res: int | None = None
    start: str | None = None
    end: str | None = None            # exclusive
    max_frames: int | None = None
    normal_space: str = "directx"
    temporal_filter: str = "auto"     # auto = on for per-frame normal models
    temporal_strength: float = 0.8
    upscale: str = "guided"
    derive: bool = True
    derive_options: postprocess.DeriveOptions = field(default_factory=postprocess.DeriveOptions)
    commercial: bool = False
    force: bool = False
    redo_passes: bool = False         # rewrite final passes even where they look up to date


def _preset(o: InferOptions) -> tuple[str, dict]:
    name = o.preset
    if name == "auto":
        from .doctor import check_gpu
        name = recommend(check_gpu()[1])
    if name not in PRESETS:
        raise ValueError(f"unknown preset '{name}'. Known: auto, {', '.join(PRESETS)}")
    p = dict(PRESETS[name])
    if o.max_res:
        p["max_res"] = o.max_res
    return name, p


def infer_size(w: int, h: int, max_res: int) -> tuple[int, int]:
    """Keep the aspect ratio, long side <= max_res, both sides multiples of 8 (VAE latents)."""
    s = min(1.0, max_res / max(w, h))
    return max(8, round(w * s / 8) * 8), max(8, round(h * s / 8) * 8)


def _prepare_cache(raw: Path, settings: dict, force: bool, derived: bool = False) -> None:
    """Keep raw frames only if they were made with the same settings (that's what resume relies on).

    derived=True: the cache is computed from another cache, so a change just discards it.
    """
    sfile = raw / "settings.json"
    old = json.loads(sfile.read_text(encoding="utf-8")) if sfile.is_file() else None
    if force or (old is not None and old != settings):
        if old is not None and not force and not derived and manifest.count_frames(raw):
            changed = sorted(k for k in set(old) | set(settings) if old.get(k) != settings.get(k))
            raise ValueError(f"{raw} has frames made with different settings ({', '.join(changed)}); "
                             "pass --force to discard them")
        if raw.is_dir():
            shutil.rmtree(raw)
    raw.mkdir(parents=True, exist_ok=True)
    sfile.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def _public(settings: dict) -> dict:
    """Settings without local paths (those differ per PC and don't change the result)."""
    return {k: v for k, v in settings.items() if k not in ("unet_vae", "svd", "weights")}


def _pass_entry(name: str, **kw) -> dict:
    """The manifest entry add_pass would write (without frames_done)."""
    tmp = {"passes": {}}
    manifest.add_pass(tmp, name, **kw)
    return tmp["passes"][name]


def _unchanged(m: dict, entries: dict[str, dict]) -> bool:
    """True if the manifest already has these pass entries (same model, settings, encoding)."""
    for name, entry in entries.items():
        old = dict(m["passes"].get(name) or {})
        old.pop("frames_done", None)
        if old != entry:
            return False
    return True


def _todo(outs: list[Path], inputs: list[tuple[Path, str]], indices: list[int], redo: bool) -> list[int]:
    """Frames to (re)write: missing, or older than one of their input frames (folder, extension).
    Same rule as pack."""
    if redo:
        return list(indices)

    def stale(i: int) -> bool:
        files = [frames.frame_path(d, i) for d in outs]
        if not all(f.is_file() for f in files):
            return True
        t = min(f.stat().st_mtime for f in files)
        return any(frames.frame_path(d, i, ext).stat().st_mtime > t for d, ext in inputs)
    return [i for i in indices if stale(i)]


def _write_pass(label: str, todo: list[int], total: int, fn) -> None:
    print(f"  {label}: {len(todo)} to write, {total - len(todo)} up to date")
    if todo:
        fn(todo)


def _print_stats(stats: dict) -> None:
    if stats.get("frames"):
        print(f"  {stats['frames']} frames in {stats['infer_s']} s ({stats['fps']} fps), "
              f"peak VRAM {stats['peak_alloc_gb']} GB allocated / {stats['peak_reserved_gb']} GB reserved")


def run(o: InferOptions) -> None:
    clip = Path(o.clip_dir)
    m = manifest.load(clip)
    if not m:
        raise FileNotFoundError(f"no {manifest.FILENAME} in {clip}: run 'platecraft extract' first")
    if o.temporal_filter not in TEMPORAL_FILTER:
        raise ValueError(f"--temporal-filter must be one of {', '.join(TEMPORAL_FILTER)}")
    models = registry.resolve(o.models)
    for w in registry.check_commercial(models, o.commercial):
        print(f"License warning: {w}")
    preset_name, preset = _preset(o)
    for info in models:
        adapter = registry.adapter(info)
        if hasattr(adapter, "preflight"):   # e.g. VRAM / Linux env checks, before any model runs
            adapter.preflight()

    fr = m["frames"]
    total, fps = fr["count"], fr["fps_float"]
    full = (fr["width"], fr["height"])
    start = ffmpeg_io.parse_time(o.start, fps) or 0
    end = min(total, ffmpeg_io.parse_time(o.end, fps) or total)
    if o.max_frames:
        end = min(end, start + o.max_frames)
    if not 0 <= start < end:
        raise ValueError(f"empty frame range {start}..{end} (clip has {total} frames)")
    plate = m["passes"].get("plate") or m["passes"]["src"]
    input_dir = clip / plate["dir"]
    alpha_dir = clip / m["passes"]["alpha"]["dir"] if "alpha" in m["passes"] else None
    up = postprocess.Upscaler(full, input_dir, o.upscale)
    guides = [(input_dir, ".png")] + ([(alpha_dir, ".png")] if alpha_dir else [])
    print(f"Clip {clip.name}: frames {start}..{end - 1} of {total}, preset {preset_name}, upscale {o.upscale}")

    for info in models:
        adapter = registry.adapter(info)
        size = (adapter.infer_size(*full, preset) if hasattr(adapter, "infer_size")
                else infer_size(*full, preset["max_res"]))
        raw_ext = getattr(adapter, "RAW_EXT", ".png")
        settings = adapter.job_settings(preset)
        raw = clip / "raw" / info.key
        cache = {"input": plate["dir"], "infer_size": list(size), **_public(settings)}
        derived = bool(info.requires)
        if derived:   # computed from another model's raw output: follow that model's cache
            src_key = m["passes"].get("depth", {}).get("model")
            if not src_key or src_key not in registry.MODELS:
                raise ValueError(f"{info.name} needs a depth pass: run a depth model first (e.g. --model vda-s)")
            src_raw = clip / "raw" / src_key
            src_settings = json.loads((src_raw / "settings.json").read_text(encoding="utf-8"))
            size = tuple(src_settings["infer_size"])
            cache = {"source": src_key, "source_settings": src_settings, **_public(settings)}
            settings = {**settings, "depth_raw": str(src_raw), "alpha_dir": str(alpha_dir) if alpha_dir else "",
                        "depth_ext": getattr(registry.adapter(registry.MODELS[src_key]), "RAW_EXT", ".png")}
        _prepare_cache(raw, cache, o.force or derived, derived)
        print(f"{info.name} at {size[0]}x{size[1]} -> {raw}")
        stats = adapter.run(Job(model=info.key, clip_dir=str(clip), input_dir=str(input_dir), out_dir=str(raw),
                                frames=[start, end], infer_size=list(size), pad_value=plate.get("grey", 0.5),
                                settings=settings))
        _print_stats(stats)
        entry = {"key": info.key, "name": info.name, "passes": list(info.passes), "preset": preset_name,
                 "settings": _public(settings), "infer_size": list(size)}

        for pass_name in info.passes:
            out = clip / pass_name
            # Multi-pass models keep each pass's raw frames in its own subfolder.
            rp, rext = adapter.raw_pass(raw, pass_name) if hasattr(adapter, "raw_pass") else (raw, raw_ext)
            done = [i for i in range(start, end) if frames.frame_path(rp, i, rext).is_file()]
            if out.is_dir() and m["passes"].get(pass_name, {}).get("model") != info.key:
                frames.clear_frames(out)       # another model's (or the heuristic) pass: don't mix
            if pass_name == "normal":
                src = rp
                tf = o.temporal_filter == "on" or (o.temporal_filter == "auto" and not info.temporal)
                if tf:
                    src = _temporal_filter(clip, info, raw, rp, input_dir, alpha_dir, size, start, end, o)
                    done = [i for i in done if frames.frame_path(src, i).is_file()]
                    entry["temporal_filter"] = {"flow": "raft", "strength": o.temporal_strength}
                pe = _pass_entry("normal", dir="normal", colorspace="linear", bit_depth=16, channels=3,
                                    heuristic=derived, model=info.key, normal_space=o.normal_space,
                                    space=f"tangent ({o.normal_space}, camera-facing card)",
                                    temporal_filter=bool(tf), upscale=o.upscale)
                redo = o.force or o.redo_passes or not _unchanged(m, {"normal": pe})
                _write_pass("normal", _todo([out], [(src, ".png"), *guides], done, redo), len(done),
                            lambda todo: postprocess.normals(src, out, alpha_dir, todo, up, o.normal_space))
            elif pass_name == "depth":
                rng = postprocess.depth_range(rp, alpha_dir, done, rext)
                pe = _pass_entry("depth", dir="depth", colorspace="linear", bit_depth=16, channels=1,
                                    model=info.key, encoding=getattr(adapter, "DEPTH_ENCODING", DEPTH_ENCODING),
                                    raw_range=[round(v, 6) for v in rng], upscale=o.upscale)
                redo = o.force or o.redo_passes or not _unchanged(m, {"depth": pe})   # incl. a new range
                _write_pass("depth", _todo([out], [(rp, rext), *guides], done, redo), len(done),
                            lambda todo: postprocess.depth(rp, out, alpha_dir, todo, up, rng, rext))
            elif pass_name == "albedo":     # a model's albedo replaces the keyed plate
                pe = _pass_entry("albedo", dir="albedo", colorspace="sRGB", bit_depth=16, channels=4,
                                 model=info.key, alpha="straight", upscale=o.upscale)
                redo = o.force or o.redo_passes or not _unchanged(m, {"albedo": pe})
                _write_pass("albedo", _todo([out], [(rp, rext), *guides], done, redo), len(done),
                            lambda todo: postprocess.model_albedo(rp, out, alpha_dir, todo, up, rext))
            elif pass_name in ("roughness", "metallic"):
                pe = _pass_entry(pass_name, dir=pass_name, colorspace="linear", bit_depth=16, channels=1,
                                 model=info.key, upscale=o.upscale)
                redo = o.force or o.redo_passes or not _unchanged(m, {pass_name: pe})
                _write_pass(pass_name, _todo([out], [(rp, rext), *guides], done, redo), len(done),
                            lambda todo: postprocess.model_gray(rp, out, todo, up, pass_name, rext))
            else:
                raise ValueError(f"no postprocess for pass '{pass_name}'")
            m["passes"][pass_name] = {**pe, "frames_done": manifest.count_frames(out)}

        m["models"] = [e for e in m["models"] if e["key"] != info.key] + [entry]
        _sync_licenses(m)
        manifest.log(m, "infer", model=info.key, range=[start, end], **stats)
        manifest.save(clip, m)

    if o.derive:
        _derive(clip, m, start, end, o.derive_options, o.force or o.redo_passes, [mi.key for mi in models])
    print(f"Updated {manifest.path_for(clip)}")


def _sync_licenses(m: dict) -> None:
    """models/licenses = exactly what the current passes were made with (a pass redone with
    another model drops the old one), so the manifest's licenses match the delivered frames."""
    used = [p["model"] for p in m["passes"].values() if p.get("model")]
    if m["passes"].get("normal", {}).get("temporal_filter"):
        used.append("raft")
    used = list(dict.fromkeys(k for k in used if k in registry.MODELS))
    m["models"] = [e for e in m["models"] if e["key"] in used]
    m["licenses"] = [{"model": k, "license": registry.MODELS[k].license, "commercial": registry.MODELS[k].commercial,
                      "note": registry.MODELS[k].license_note} for k in used]


def _temporal_filter(clip: Path, info, raw: Path, src: Path, input_dir: Path, alpha_dir: Path | None,
                     size: tuple[int, int], start: int, end: int, o: InferOptions) -> Path:
    from .models import temporal_filter
    raft = registry.MODELS["raft"]
    if not registry.installed(raft):
        raise FileNotFoundError("--temporal-filter needs the RAFT weights: "
                                "engine/setup_engine.ps1 -Models raft (or pass --temporal-filter off)")
    out = clip / "raw" / f"{info.key}+tf"
    settings = temporal_filter.job_settings(o.temporal_strength)
    source = json.loads((raw / "settings.json").read_text(encoding="utf-8"))
    _prepare_cache(out, {"source": info.key, "source_settings": source, **settings}, o.force, derived=True)
    print(f"Temporal filter (RAFT, strength {o.temporal_strength}) -> {out}")
    stats = temporal_filter.run(Job(model="raft", clip_dir=str(clip), input_dir=str(input_dir), out_dir=str(out),
                                    frames=[start, end], infer_size=list(size), pad_value=0.5,
                                    settings={**settings, "source_dir": str(src),
                                              "alpha_dir": str(alpha_dir) if alpha_dir else ""}))
    _print_stats(stats)
    return out


def _derive(clip: Path, m: dict, start: int, end: int, d: postprocess.DeriveOptions, redo: bool,
            run_models: list[str]) -> None:
    p = m["passes"]
    # albedo/roughness/metallic made by a model of this run stay; otherwise the heuristics replace
    # them (and a model's metallic is dropped), so the delivered passes match this run's models.
    keep = {k for k in ("albedo", "roughness", "metallic") if p.get(k, {}).get("model") in run_models}
    if "metallic" in p and "metallic" not in keep:
        frames.clear_frames(clip / p.pop("metallic")["dir"])
        _sync_licenses(m)
    need = [clip / p[k]["dir"] for k in ("normal", "depth") if k in p]
    idx = [i for i in range(start, end) if all(frames.frame_path(f, i).is_file() for f in need)]
    if not idx:
        return
    bits = max(8, p["src"]["bit_depth"])
    entries = {}
    if "albedo" not in keep:
        entries["albedo"] = _pass_entry("albedo", dir="albedo", colorspace="sRGB", bit_depth=bits, channels=4,
                                        heuristic=d.delight > 0, alpha="straight", delight=d.delight,
                                        edge_color=0.0 if m.get("alpha", {}).get("source") == "lumakey" else d.edge_color)
    if "roughness" not in keep:
        entries["roughness"] = _pass_entry("roughness", dir="roughness", colorspace="linear", bit_depth=16,
                                           channels=1, heuristic=True, base=d.roughness, cavity=d.roughness_cavity)
    if need:
        entries["ao"] = _pass_entry("ao", dir="ao", colorspace="linear", bit_depth=16, channels=1, heuristic=True,
                                    strength=d.ao_strength, inputs=[k for k in ("depth", "normal") if k in p])
    if not entries:
        return
    print(f"Derived maps: {', '.join(entries)} (heuristic" + ("; albedo = keyed plate)" if "albedo" in entries else ")")
          + (f"; {', '.join(sorted(keep))} from the model" if keep else ""))
    for k in ("albedo", "roughness"):
        if k in entries and p.get(k, {}).get("model"):
            frames.clear_frames(clip / p[k]["dir"])     # a model's map from an earlier run
    # Inputs: everything straight_color and the AO read (fg or plate, alpha, depth, normal).
    plate = p.get("plate") or p["src"]
    inputs = [(clip / p[k]["dir"], ".png") for k in ("fg", "alpha", "depth", "normal") if k in p]
    inputs.append((clip / plate["dir"], ".png"))
    redo = redo or not _unchanged(m, entries)
    todo = _todo([clip / k for k in entries], inputs, idx, redo)
    written = list(entries)
    _write_pass("/".join(entries), todo, len(idx), lambda t: postprocess.derive(clip, m, t, d, set(entries)))
    for name, entry in entries.items():
        p[name] = {**entry, "frames_done": manifest.count_frames(clip / entry["dir"])}
    _sync_licenses(m)
    manifest.log(m, "derive", range=[start, end], passes=written, written=len(todo))
    manifest.save(clip, m)
