"""`platecraft doctor`: check GPU, driver, ffmpeg, engine environments and weights.

Uses only the standard library, so it runs even before the engine is installed.
"""
from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass

from . import paths
from .models.registry import MODELS, ModelInfo
from .presets import PRESETS, recommend

OK, WARN, FAIL, SKIP = "OK", "WARN", "FAIL", "SKIP"
MIN_DRIVER_WINDOWS = (551, 61)   # CUDA 12.4 minimum driver on Windows
MIN_DRIVER_LINUX = (550, 54)


@dataclass
class Check:
    group: str
    name: str
    status: str
    detail: str


def _run(cmd: list[str], timeout: int = 30, env: dict | None = None) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env,
                           encoding="utf-8", errors="replace")
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return 127, "", f"not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"


def check_system() -> list[Check]:
    out = [Check("system", "OS", OK, f"{platform.system()} {platform.release()} ({platform.machine()})")]
    v = sys.version_info
    out.append(Check("system", "CLI Python", OK if v >= (3, 10) else FAIL,
                     f"{platform.python_version()} at {sys.executable}"))
    return out


def check_gpu() -> tuple[list[Check], float | None]:
    smi = paths.tool("nvidia-smi")
    if not smi:
        return [Check("gpu", "NVIDIA GPU", FAIL, "nvidia-smi not found: install an NVIDIA driver")], None
    rc, out, err = _run([str(smi), "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
    if rc != 0 or not out.strip():
        return [Check("gpu", "NVIDIA GPU", FAIL, (err or out).strip() or "nvidia-smi failed")], None
    name, mem_mib, driver = [s.strip() for s in out.strip().splitlines()[0].split(",")]
    vram_gb = round(float(mem_mib) / 1024, 1)
    checks = [Check("gpu", "GPU", OK, f"{name}, {vram_gb} GB VRAM")]
    nums = tuple(int(x) for x in re.findall(r"\d+", driver)[:2])
    need = MIN_DRIVER_WINDOWS if paths.IS_WINDOWS else MIN_DRIVER_LINUX
    checks.append(Check("gpu", "Driver", OK if nums >= need else FAIL,
                        f"{driver}" + ("" if nums >= need else f" (CUDA 12.4 needs {need[0]}.{need[1]}+)")))
    if vram_gb < 8:
        checks.append(Check("gpu", "VRAM", WARN, "under 8 GB: NormalCrafter will be slow and low-res"))
    return checks, vram_gb


def check_ffmpeg() -> list[Check]:
    out = []
    for name in ("ffmpeg", "ffprobe"):
        exe = paths.tool(name)
        if not exe:
            out.append(Check("ffmpeg", name, FAIL, "not found in engine/bin or PATH (run setup_engine.ps1)"))
            continue
        rc, so, se = _run([str(exe), "-hide_banner", "-version"])
        if rc != 0:
            out.append(Check("ffmpeg", name, FAIL, (se or so).strip()[:200]))
            continue
        mver = re.search(r"version (\S+)", so)
        first = f"{name} {mver.group(1)}" if mver else "?"
        where = "bundled" if exe.parent == paths.BIN else f"from PATH ({exe})"
        detail = f"{first} [{where}]"
        status = OK
        if name == "ffmpeg":
            if "--enable-gpl" in so:
                status, detail = WARN, detail + " - GPL build (fine to use; the bundled one is LGPL)"
            enc = _run([str(exe), "-hide_banner", "-encoders"])[1]
            missing = [c for c in ("png", "exr", "prores_ks") if not re.search(rf"\s{c}\s", enc)]
            if missing:
                status, detail = WARN, detail + f" - missing encoders: {', '.join(missing)}"
        out.append(Check("ffmpeg", name, status, detail))
    return out


def check_engine() -> list[Check]:
    if not paths.ENGINE.is_dir():
        return [Check("engine", "engine folder", FAIL, f"{paths.ENGINE} missing")]
    out = []
    if paths.ENGINE_JSON.is_file():
        info = json.loads(paths.ENGINE_JSON.read_text(encoding="utf-8-sig"))
        out.append(Check("engine", "setup", OK,
                         f"installed {info.get('installed_at', '?')}, torch {info.get('torch', '?')}, "
                         f"models: {', '.join(info.get('models', []))}"))
        if info.get("weights_failed"):
            out.append(Check("engine", "setup", WARN, f"weights failed during setup: {', '.join(info['weights_failed'])}"))
    else:
        out.append(Check("engine", "setup", FAIL, "engine not installed: run engine\\setup_engine.ps1"))
    py = paths.env_python("core")
    out.append(Check("engine", "core env", OK if py.is_file() else WARN,
                     str(py) if py.is_file() else "missing (needed from build step 2 on)"))
    return out


PROBE_TEMPLATE = r"""
import json, sys
sys.path.insert(0, {repo!r})
extra = {{}}
res = {{"ok": False}}
try:
    import torch
    res.update(torch=torch.__version__, cuda=torch.version.cuda, cuda_available=torch.cuda.is_available())
    if torch.cuda.is_available():
        res["device"] = torch.cuda.get_device_name(0)
        torch.zeros(1, device="cuda").add_(1)   # make sure kernels actually run on this GPU
    try:
        import xformers
        extra["xformers"] = xformers.__version__
    except Exception:
        extra["xformers"] = None
{probe}
    res["ok"] = True
except Exception as e:
    res["error"] = f"{{type(e).__name__}}: {{e}}"
res["extra"] = extra
print("PROBE:" + json.dumps(res))
"""


_probe_cache: dict[tuple[str, str], tuple[int, str, str]] = {}


def _indent(code: str) -> str:
    return "".join("    " + line + "\n" for line in code.strip().splitlines())


def check_model(m: ModelInfo, requested: bool, vram: float | None = None) -> list[Check]:
    g = f"model:{m.key}"
    if not m.probe:   # computed on the CLI side, nothing to install
        return [Check(g, "model", OK, f"built in (needs a {', '.join(m.requires)} model's output)")]
    run = paths.worker_launcher(m.env)
    missing = [w for w in m.weights if not (paths.MODELS / w).exists()]
    if not run.exists():
        st = FAIL if requested else SKIP
        if m.linux_only:
            where = "inside WSL2 or on a Linux PC" if paths.IS_WINDOWS else "on this PC"
            have = "weights present; " if not missing else ""
            return [Check(g, "environment", st, f"{have}needs Linux: run engine/setup_engine.sh {where}")]
        return [Check(g, "environment", st, f"not installed (setup_engine.ps1 -Models {m.key})")]
    if missing and not requested:
        # Environments are shared (vda-s / vda-l), so an env alone doesn't mean this model was installed.
        return [Check(g, "model", SKIP, f"weights not installed (setup_engine.ps1 -Models {m.key})")]
    out = []
    repo = paths.REPOS / m.repo
    if m.repo and not repo.is_dir():
        out.append(Check(g, "code", FAIL, f"{repo} missing"))
    code = PROBE_TEMPLATE.format(repo=run.path(repo), probe=_indent(m.probe))
    key = (m.env, m.probe)
    if key not in _probe_cache:
        _probe_cache[key] = _run(run.command(["-c", code]), timeout=180, env=paths.hf_env())
    rc, so, se = _probe_cache[key]
    line = next((l for l in so.splitlines() if l.startswith("PROBE:")), None)
    if not line:
        out.append(Check(g, "runtime", FAIL, (se or so).strip()[-300:] or f"exit {rc}"))
    else:
        r = json.loads(line[6:])
        base = f"torch {r.get('torch')} / CUDA {r.get('cuda')}"
        if not r.get("ok"):
            out.append(Check(g, "runtime", FAIL, f"{base}: {r.get('error')}"))
        elif not r.get("cuda_available"):
            out.append(Check(g, "runtime", FAIL, f"{base}: CUDA not available (CPU-only torch or driver issue)"))
        else:
            ex = ", ".join(f"{k} {v or 'no'}" for k, v in r.get("extra", {}).items())
            out.append(Check(g, "runtime", OK, f"{base} on {r.get('device')}; {ex}"))
    out.append(Check(g, "weights", FAIL if missing else OK,
                     ("missing: " + ", ".join(missing)) if missing else "present"))
    if m.min_vram_gb and vram is not None and vram < m.min_vram_gb:
        out.append(Check(g, "vram", WARN, f"needs >= {m.min_vram_gb:g} GB VRAM, this GPU has {vram:.0f} GB"))
    if m.warning:
        out.append(Check(g, "license", WARN, m.warning))
    return out


def run(models: list[str] | None = None, as_json: bool = False) -> int:
    requested = set(models or [])
    checks: list[Check] = []
    checks += check_system()
    gpu_checks, vram = check_gpu()
    checks += gpu_checks
    checks += check_ffmpeg()
    checks += check_engine()
    for m in MODELS.values():
        checks += check_model(m, m.key in requested, vram)

    preset = recommend(vram)
    p = PRESETS[preset]
    rec = (f"--preset {preset}  (max_res {p['max_res']}, offload {p['offload']}, "
           f"window {p['window']}, overlap {p['overlap']}, decode_chunk {p['decode_chunk']})")
    failed = any(c.status == FAIL for c in checks)

    if as_json:
        print(json.dumps({"checks": [asdict(c) for c in checks], "vram_gb": vram,
                          "recommended_preset": preset, "ok": not failed}, indent=2))
    else:
        group = None
        for c in checks:
            if c.group != group:
                group = c.group
                print(f"\n{group}")
            print(f"  [{c.status:<4}] {c.name:<12} {c.detail}")
        print(f"\nRecommended: {rec}")
        print("\nResult: " + ("problems found (see FAIL lines)" if failed else "ready"))
    return 1 if failed else 0
