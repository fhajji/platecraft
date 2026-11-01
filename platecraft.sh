#!/usr/bin/env sh
# PlateCraft launcher (Linux): uses the engine's core Python, falls back to the system one.
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="$ROOT/engine/envs/core/bin/python"
[ -x "$PY" ] || PY=python3
exec "$PY" "$ROOT/platecraft.py" "$@"
