"""Everything that talks to ffmpeg / ffprobe."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

from . import paths
from .manifest import FRAME_PATTERN, START_NUMBER

# Transfer characteristics we treat as normal display-referred video.
SDR_TRANSFERS = {"", "unknown", "bt709", "iec61966-2-1", "bt470bg", "smpte170m", "bt2020-10", "bt2020-12", "gamma22"}


class FfmpegError(RuntimeError):
    pass


def _exe(name: str) -> str:
    exe = paths.tool(name)
    if not exe:
        raise FfmpegError(f"{name} not found. Run engine\\setup_engine.ps1 or put {name} on PATH.")
    return str(exe)


def run_ffmpeg(args: list[str], *, quiet: bool = True) -> None:
    cmd = [_exe("ffmpeg"), "-hide_banner", "-nostdin", "-y", "-loglevel", "error" if quiet else "info",
           "-stats", *args]
    p = subprocess.run(cmd, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        tail = "\n".join(l for l in p.stderr.splitlines() if not l.startswith("frame="))[-1500:]
        raise FfmpegError(f"ffmpeg failed ({p.returncode}):\n{tail}\ncommand: {' '.join(cmd)}")


# --------------------------------------------------------------- probing --

@dataclass
class VideoInfo:
    path: str
    codec: str
    width: int
    height: int
    fps: str                 # exact rational, e.g. "24000/1001"
    fps_float: float
    frame_count: int
    duration: float
    pix_fmt: str
    bit_depth: int
    has_alpha: bool
    color_range: str         # "tv" (limited) or "pc" (full)
    color_matrix: str        # bt709, bt601 (smpte170m/bt470bg), bt2020nc ...
    color_transfer: str
    color_primaries: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _pix_fmt_info(pix_fmt: str) -> tuple[int, bool]:
    """(bit depth, has alpha) from an ffmpeg pixel format name."""
    m = re.search(r"(\d+)(le|be)?$", pix_fmt)
    depth = 8
    if m and pix_fmt.startswith(("yuv", "gbr", "gray", "ya", "p0", "p2", "p4", "x2")):
        n = int(m.group(1))
        depth = n if n in (9, 10, 12, 14, 16) else 8
    elif pix_fmt.startswith(("rgb48", "bgr48", "rgba64", "bgra64")):
        depth = 16
    elif pix_fmt in ("p010le", "p010be"):
        depth = 10
    alpha = (pix_fmt.startswith(("yuva", "gbrap", "ya")) or
             any(t in pix_fmt for t in ("rgba", "bgra", "argb", "abgr")))
    return depth, alpha


def probe(path: Path) -> VideoInfo:
    path = Path(path)
    if not path.is_file():
        raise FfmpegError(f"input not found: {path}")
    cmd = [_exe("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_format",
           "-of", "json", str(path)]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise FfmpegError(f"ffprobe failed: {p.stderr.strip()}")
    data = json.loads(p.stdout)
    if not data.get("streams"):
        raise FfmpegError(f"no video stream in {path}")
    s, fmt = data["streams"][0], data.get("format", {})

    rate = s.get("r_frame_rate") or s.get("avg_frame_rate") or "0/1"
    avg = s.get("avg_frame_rate") or rate
    fr = Fraction(rate) if rate != "0/0" else Fraction(avg)
    duration = float(s.get("duration") or fmt.get("duration") or 0)
    nb = int(s.get("nb_frames") or 0)
    if nb <= 0:
        nb = _count_frames(path)
    depth, alpha = _pix_fmt_info(s.get("pix_fmt", ""))
    # ProRes 4444 often reports yuva444p10le; some encoders tag alpha only in a side tag.
    if not alpha and str(s.get("tags", {}).get("alpha_mode", "")) == "1":
        alpha = True

    info = VideoInfo(
        path=str(path), codec=s.get("codec_name", "?"), width=int(s["width"]), height=int(s["height"]),
        fps=f"{fr.numerator}/{fr.denominator}", fps_float=round(float(fr), 6), frame_count=nb,
        duration=duration, pix_fmt=s.get("pix_fmt", "?"), bit_depth=depth, has_alpha=alpha,
        color_range=s.get("color_range", "") or "", color_matrix=s.get("color_space", "") or "",
        color_transfer=s.get("color_transfer", "") or "", color_primaries=s.get("color_primaries", "") or "",
    )
    if Fraction(avg) != fr and Fraction(avg) != 0:
        info.warnings.append(f"variable frame rate (avg {avg} vs nominal {rate}); frames are kept 1:1, "
                             f"timing uses {info.fps}. Convert to constant frame rate first for exact sync.")
    if info.color_transfer not in SDR_TRANSFERS:
        info.warnings.append(f"transfer '{info.color_transfer}' looks like HDR or log footage; "
                             "convert to Rec.709 before extracting or colors will be wrong.")
    return info


def _count_frames(path: Path) -> int:
    p = subprocess.run([_exe("ffprobe"), "-v", "error", "-select_streams", "v:0", "-count_packets",
                        "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    try:
        return int(p.stdout.strip().split(",")[0])
    except ValueError:
        return 0


# ------------------------------------------------------------ extraction --

def _matrix_for(info: VideoInfo) -> str:
    """Color matrix name for the scale filter. Untagged HD is BT.709, untagged SD BT.601."""
    m = info.color_matrix
    if m in ("bt709",):
        return "bt709"
    if m in ("smpte170m", "bt470bg", "bt601"):
        return "bt601"
    if m.startswith("bt2020"):
        return "bt2020"
    return "bt709" if info.height >= 720 else "bt601"


def _is_rgb_source(info: VideoInfo) -> bool:
    return info.pix_fmt.startswith(("rgb", "bgr", "gbr", "argb", "abgr", "gray", "ya", "pal8", "monob"))


def rgb_filter(info: VideoInfo, out_fmt: str) -> str:
    """YUV->RGB with the right matrix and range (ffmpeg would otherwise assume BT.601)."""
    if _is_rgb_source(info):
        return f"format={out_fmt}"
    rng = "pc" if info.color_range == "pc" else "tv"
    return (f"scale=in_color_matrix={_matrix_for(info)}:in_range={rng}:out_range=pc:"
            f"flags=accurate_rnd+full_chroma_int,format={out_fmt}")


def trim_filter(start_frame: int, end_frame: int | None) -> str:
    """Frame-exact trim. end_frame is exclusive; None = to the end."""
    if start_frame <= 0 and end_frame is None:
        return ""
    parts = [f"start_frame={max(start_frame, 0)}"]
    if end_frame is not None:
        parts.append(f"end_frame={end_frame}")
    return f"trim={':'.join(parts)},setpts=PTS-STARTPTS,"


def _png_out(folder: Path) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    # passthrough: one PNG per decoded frame, no duplicates or drops.
    return ["-fps_mode", "passthrough", "-start_number", str(START_NUMBER),
            "-compression_level", "3", str(folder / FRAME_PATTERN)]


def extract_frames(info: VideoInfo, out_rgb: Path, *, bit_depth: int, start_frame: int = 0,
                   end_frame: int | None = None, out_alpha: Path | None = None,
                   key: dict | None = None) -> None:
    """Decode the video once into RGB frames and, optionally, alpha frames.

    out_alpha with key=None extracts the embedded alpha channel.
    out_alpha with key={color, similarity, blend} pulls a basic chroma key.
    """
    rgb_fmt = "rgb48be" if bit_depth == 16 else "rgb24"
    gray_fmt = "gray16be" if bit_depth == 16 else "gray"
    trim = trim_filter(start_frame, end_frame)

    if out_alpha is None:
        graph = f"[0:v]{trim}{rgb_filter(info, rgb_fmt)}[rgb]"
        maps = ["-map", "[rgb]", *_png_out(out_rgb)]
    else:
        if key:
            alpha_chain = (f"format=yuv444p,chromakey=color={key['color']}:similarity={key['similarity']}"
                           f":blend={key['blend']},format=yuva444p,alphaextract,format={gray_fmt}")
        else:
            alpha_chain = f"alphaextract,format={gray_fmt}"
        graph = (f"[0:v]{trim}split=2[a][b];"
                 f"[a]{rgb_filter(info, rgb_fmt)}[rgb];"
                 f"[b]{alpha_chain}[alpha]")
        maps = ["-map", "[rgb]", *_png_out(out_rgb), "-map", "[alpha]", *_png_out(out_alpha)]

    run_ffmpeg(["-i", info.path, "-filter_complex", graph, *maps])


def extract_matte(matte: VideoInfo, out_alpha: Path, *, bit_depth: int, size: tuple[int, int],
                  start_frame: int = 0, end_frame: int | None = None) -> None:
    """A separate matte video: white = foreground. Uses its luma (or its alpha if it has one).
    Scaled to `size` (the plate's width, height) when it differs."""
    gray_fmt = "gray16be" if bit_depth == 16 else "gray"
    trim = trim_filter(start_frame, end_frame)
    if (matte.width, matte.height) != tuple(size):
        trim += f"scale={size[0]}:{size[1]}:flags=bicubic,"
    if matte.has_alpha:
        chain = f"alphaextract,format={gray_fmt}"
    elif _is_rgb_source(matte):
        chain = f"format={gray_fmt}"
    else:
        # Full-range luma; for a YUV matte, Y is exactly the matte value.
        rng = "pc" if matte.color_range == "pc" else "tv"
        chain = f"scale=in_range={rng}:out_range=pc,format={gray_fmt}"
    run_ffmpeg(["-i", matte.path, "-filter_complex", f"[0:v]{trim}{chain}[m]", "-map", "[m]",
                *_png_out(out_alpha)])


# ---------------------------------------------------------------- encode --

ENCODERS = {
    # name: (ffmpeg args, default extension). All are in the LGPL ffmpeg build.
    "prores":     (["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"], ".mov"),
    "prores4444": (["-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuva444p10le"], ".mov"),
    "nvenc":      (["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "19", "-pix_fmt", "yuv420p"], ".mp4"),
    "mpeg4":      (["-c:v", "mpeg4", "-q:v", "3", "-pix_fmt", "yuv420p"], ".mp4"),
}


def encode(folder: Path, output: Path, *, fps: str, codec: str = "prores",
           alpha_dir: Path | None = None) -> Path:
    """Encode a frame folder to a video (lossy: never use it for data passes in UE).
    With alpha_dir, the matte is merged in and the result is a transparent ProRes 4444."""
    if alpha_dir is not None:
        codec = "prores4444"
    if codec not in ENCODERS:
        raise FfmpegError(f"unknown codec '{codec}', choose from {', '.join(ENCODERS)}")
    args, ext = ENCODERS[codec]
    output = Path(output)
    if not output.suffix:
        output = output.with_suffix(ext)
    output.parent.mkdir(parents=True, exist_ok=True)
    seq = ["-framerate", fps, "-start_number", str(START_NUMBER), "-i"]
    inputs = [*seq, str(Path(folder) / FRAME_PATTERN)]
    to_yuv = "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int"
    if alpha_dir is not None:
        inputs += [*seq, str(Path(alpha_dir) / FRAME_PATTERN)]
        graph = ["-filter_complex", f"[1:v]format=gray16le[a];[0:v]format=rgb48le[c];"
                                    f"[c][a]alphamerge,{to_yuv},format=yuva444p10le[v]", "-map", "[v]",
                 "-alpha_bits", "16"]
    else:
        fmt = "yuva444p10le" if codec == "prores4444" else args[args.index("-pix_fmt") + 1]
        graph = ["-vf", f"{to_yuv},format={fmt}"]
    # BT.709 tags so players show the colors as extracted.
    tags = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
    run_ffmpeg([*inputs, *graph, *args, *tags, str(output)])
    return output


def grab_frame(info: VideoInfo, frame: int, out_png: Path) -> Path:
    """Decode a single frame (as 8-bit RGB PNG) - used to sample the key color."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(["-i", info.path, "-filter_complex",
                f"[0:v]{trim_filter(frame, frame + 1)}{rgb_filter(info, 'rgb24')}[o]",
                "-map", "[o]", "-frames:v", "1", str(out_png)])
    return out_png


def parse_time(value: str | None, fps: float) -> int | None:
    """'120' or 'f120' = frame 120; '4.5s', '00:00:04.5', '1:02' = time. Returns a frame index."""
    if value is None or value == "":
        return None
    v = value.strip().lower()
    if v.startswith("f"):
        return int(v[1:])
    if v.endswith("s"):
        return round(float(v[:-1]) * fps)
    if ":" in v:
        secs = 0.0
        for part in v.split(":"):
            secs = secs * 60 + float(part)
        return round(secs * fps)
    if re.fullmatch(r"\d+", v):
        return int(v)
    return round(float(v) * fps)


if __name__ == "__main__":  # quick manual check: python -m platecraft.ffmpeg_io clip.mov
    print(json.dumps(probe(Path(sys.argv[1])).to_dict(), indent=2))
