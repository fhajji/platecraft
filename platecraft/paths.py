"""Locations of the project, the engine and the tools inside it."""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE = Path(os.environ.get("PLATECRAFT_ENGINE", ROOT / "engine")).resolve()
BIN = ENGINE / "bin"
ENVS = ENGINE / "envs"
REPOS = ENGINE / "repos"
MODELS = ENGINE / "models"
ENGINE_JSON = ENGINE / "engine.json"

IS_WINDOWS = sys.platform == "win32"
EXE = ".exe" if IS_WINDOWS else ""


def env_python(env: str) -> Path:
    """Python interpreter of one engine environment."""
    sub = "Scripts" if IS_WINDOWS else "bin"
    return ENVS / env / sub / f"python{EXE}"


@dataclass
class Launcher:
    """How to start a Python worker of one engine env: natively, or inside WSL2 on Windows.

    A Linux-only env (setup_engine.sh run inside WSL2) leaves engine/envs/<env>/wsl.json
    {"distro": ..., "python": "/home/.../bin/python"}; workers then run through wsl.exe and
    every Windows path handed to them goes through `path` (M:\\x -> /mnt/m/x).
    """
    python: str
    wsl: dict | None = None

    def path(self, p: str | os.PathLike) -> str:
        p = str(p)
        if not self.wsl:
            return p
        m = re.match(r"^([A-Za-z]):[\\/]?(.*)$", p)
        if not m:
            return p.replace("\\", "/")
        return f"/mnt/{m.group(1).lower()}/{m.group(2)}".replace("\\", "/").rstrip("/")

    def command(self, args: list[str], env: dict[str, str] | None = None) -> list[str]:
        """argv running `python <args>`; inside WSL, `env` (only the given variables) is passed explicitly."""
        if not self.wsl:
            return [self.python, *args]
        pre = ["wsl.exe", "-d", self.wsl["distro"], "-e", "env"]
        pre += [f"{k}={v}" for k, v in (env or {}).items()]
        return pre + [self.python, *args]

    def exists(self) -> bool:
        return True if self.wsl else Path(self.python).is_file()


def worker_launcher(env: str) -> Launcher:
    native = env_python(env)
    marker = ENVS / env / "wsl.json"
    if IS_WINDOWS and not native.is_file() and marker.is_file():
        info = json.loads(marker.read_text(encoding="utf-8"))
        return Launcher(python=info["python"], wsl=info)
    return Launcher(python=str(native))


def tool(name: str) -> Path | None:
    """Find a bundled tool (engine/bin first), then fall back to PATH."""
    bundled = BIN / f"{name}{EXE}"
    if bundled.is_file():
        return bundled
    found = shutil.which(name)
    return Path(found) if found else None


def hf_env() -> dict[str, str]:
    """Environment for model workers: keep Hugging Face and torchvision files inside the engine."""
    env = dict(os.environ)
    env.setdefault("HF_HOME", str(MODELS / "hf"))
    env.setdefault("TORCH_HOME", str(MODELS / "torch"))
    env.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    return env
