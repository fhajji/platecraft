"""manifest.json: the record of one clip's frames, passes, models and licenses.

It lives at <frames-dir>/manifest.json, travels with the frames, and is read by
every later step and by the Unreal import script.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from . import __version__

FILENAME = "manifest.json"
FRAME_PATTERN = "frame_%06d.png"
FRAME_GLOB = "frame_*.png"
START_NUMBER = 1


def path_for(clip_dir: Path) -> Path:
    return Path(clip_dir) / FILENAME


def load(clip_dir: Path) -> dict[str, Any] | None:
    p = path_for(clip_dir)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def save(clip_dir: Path, data: dict[str, Any]) -> Path:
    p = path_for(clip_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    data["platecraft"] = __version__
    data["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(p)   # atomic: an interrupted run never leaves a half-written manifest
    return p


def new(clip_dir: Path) -> dict[str, Any]:
    return {
        "platecraft": __version__,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": {},
        "frames": {},
        "alpha": {"source": "none"},
        "passes": {},
        "models": [],
        "licenses": [],
        "history": [],
    }


def add_pass(data: dict, name: str, *, dir: str, colorspace: str, bit_depth: int, channels: int,
             heuristic: bool = False, **extra) -> None:
    """Register an output folder. colorspace: 'sRGB' for color, 'linear' for data maps."""
    data["passes"][name] = {"dir": dir, "pattern": FRAME_PATTERN, "start_number": START_NUMBER,
                            "colorspace": colorspace, "bit_depth": bit_depth, "channels": channels,
                            "heuristic": heuristic, **extra}


def log(data: dict, step: str, **info) -> None:
    data["history"].append({"step": step, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **info})


def resolve_pass_dir(folder: Path) -> tuple[Path, dict | None, str | None]:
    """Map <clip>/<pass> to the folder that really holds that pass's frames.

    Some passes are aliases: with no alpha and no despill, 'plate' points to 'src'.
    Returns (frame folder, manifest or None, note for the user or None).
    """
    folder = Path(folder)
    m = load(folder.parent) or load(folder)
    if count_frames(folder) > 0:
        return folder, m, None
    if m and folder.name in m.get("passes", {}):
        real = folder.parent / m["passes"][folder.name]["dir"]
        if real != folder and count_frames(real) > 0:
            return real, m, f"'{folder.name}' uses the frames in '{real.name}' (no separate folder was needed)"
    raise FileNotFoundError(f"no frames (frame_*.png) in {folder}")


def count_frames(folder: Path) -> int:
    return sum(1 for _ in Path(folder).glob(FRAME_GLOB)) if Path(folder).is_dir() else 0
