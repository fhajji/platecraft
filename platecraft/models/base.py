"""Model adapter plumbing shared by every model.

An adapter module (platecraft/models/<name>.py) has two halves:
  * core side (CLI env, no torch): `run(job)` hands the job to `spawn_worker`;
  * worker side (the model's own engine env): `python -m platecraft.models.<name> job.json`
    loads the model, reads the plate frames, writes raw pass frames into the job's
    out_dir and reports progress on stdout.

Workers write their raw output at inference resolution; the core side
(postprocess.py) turns that into the final full-resolution pass.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .. import paths
from .registry import ModelInfo

PROGRESS = "PROGRESS:"   # PROGRESS:{"done": 12, "total": 48}
RESULT = "RESULT:"       # RESULT:{...stats...}, last line of a successful worker
WIN_ABS = re.compile(r"^[A-Za-z]:[\\/]")


@dataclass
class Job:
    model: str                      # registry key
    clip_dir: str
    input_dir: str                  # frames the model sees (the plate)
    out_dir: str                    # raw output, inference resolution
    frames: list[int]               # [start, end) 0-based clip frame indices
    infer_size: list[int]           # [w, h] before padding
    pad_value: float                # background value used to pad up to the model's size multiple
    settings: dict = field(default_factory=dict)   # model-specific (preset values etc.)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "Job":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def spawn_worker(info: ModelInfo, module: str, job: Job) -> dict:
    """Run `python -m <module> job.json` in the model's env, relay progress, return its RESULT."""
    run = paths.worker_launcher(info.env)
    if not run.exists():
        setup = ("engine/setup_engine.sh (Linux, or inside WSL2 on Windows)" if info.linux_only
                 else "engine/setup_engine.ps1")
        raise FileNotFoundError(f"engine env '{info.env}' is missing ({run.python}); run {setup}")
    job_file = Path(job.out_dir) / "job.json"
    if run.wsl:   # the worker sees Linux paths; relative ones resolve against the cwd, which wsl.exe maps too
        job = Job(**{**asdict(job), **{k: run.path(getattr(job, k)) for k in ("clip_dir", "input_dir", "out_dir")},
                     "settings": {k: run.path(v) if isinstance(v, str) and WIN_ABS.match(v) else v
                                  for k, v in job.settings.items()}})
    job.save(job_file)
    env = paths.hf_env()
    pythonpath = [run.path(paths.ROOT), run.path(paths.REPOS / info.repo)]
    env["PYTHONPATH"] = (":" if run.wsl else os.pathsep).join(pythonpath)
    env["PYTHONUNBUFFERED"] = "1"
    passed = None
    if run.wsl:
        passed = {k: run.path(env[k]) if k.endswith("_HOME") else env[k]
                  for k in ("HF_HOME", "TORCH_HOME", "HF_HUB_DISABLE_TELEMETRY", "PYTHONPATH", "PYTHONUNBUFFERED")}
    cmd = run.command(["-m", module, run.path(job_file)], passed)
    log_file = Path(job.out_dir) / "worker.log"
    result: dict | None = None
    tail: list[str] = []
    t0 = time.time()
    print(f"  loading {info.name}...", end="", flush=True)
    with open(log_file, "w", encoding="utf-8") as log, \
            subprocess.Popen(cmd, cwd=paths.ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace", bufsize=1) as p:
        try:
            for line in p.stdout:
                log.write(line)
                line = line.rstrip()
                if line.startswith(PROGRESS):
                    pr = json.loads(line[len(PROGRESS):])
                    rate = pr["done_now"] / max(time.time() - t0, 1e-6)
                    speed = f"{rate:.2f} fps incl. loading" if pr["done_now"] else "cached"
                    print(f"\r  {info.key}: {pr['done']}/{pr['total']} frames ({speed})", end="", flush=True)
                elif line.startswith(RESULT):
                    result = json.loads(line[len(RESULT):])
                elif line:
                    tail = (tail + [line])[-40:]
        except KeyboardInterrupt:
            p.terminate()
            raise
    print()
    if p.returncode != 0 or result is None:
        print("\n".join(f"  | {t}" for t in tail), file=sys.stderr)
        raise RuntimeError(f"{info.name} worker failed (exit code {p.returncode}); full log: {log_file}")
    if result.get("note"):
        print(f"  note: {result['note']}")
    return result


# ------------------------------------------------------------ worker side --

def worker_progress(done: int, total: int, done_now: int) -> None:
    print(PROGRESS + json.dumps({"done": done, "total": total, "done_now": done_now}), flush=True)


def worker_result(**stats) -> None:
    print(RESULT + json.dumps(stats), flush=True)


def worker_main(fn) -> None:
    """Entry point for `python -m platecraft.models.<name> job.json`."""
    if len(sys.argv) != 2:
        print(f"usage: python -m <adapter module> job.json", file=sys.stderr)
        sys.exit(2)
    fn(Job.load(Path(sys.argv[1])))


def chunks(start: int, end: int, length: int, overlap: int, min_len: int) -> list[tuple[int, int]]:
    """Split [start, end) into chunks of `length` that overlap by `overlap` frames.

    A short last chunk is extended backwards to at least `min_len` frames (the
    model's own window), so it overlaps its predecessor more.
    """
    if end - start <= 0:
        return []
    out = []
    s = start
    while True:
        e = min(s + length, end)
        out.append((max(0, min(s, e - min_len)), e))
        if e >= end:
            return out
        s = e - overlap
