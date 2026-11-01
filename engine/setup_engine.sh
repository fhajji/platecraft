#!/usr/bin/env bash
# PlateCraft engine setup for Linux and WSL2: the models that only run on Linux.
#
# Currently: diffusion-renderer (Cosmos DiffusionRenderer, inverse 7B). It needs
# transformer-engine (Linux only), Python 3.10 and >= 20 GB VRAM (~27 GB per 57-frame chunk).
# The environment is built from upstream's own conda file (cosmos-predict1.yaml: Python 3.10,
# CUDA 12.4 toolkit, gcc 12) with a portable micromamba, so nothing is installed system-wide.
#
#   Native Linux:  bash engine/setup_engine.sh            -> engine/envs/cosmos (+ engine/envs/core
#                  for the CLI; run it with ./platecraft.sh). Needs ffmpeg on PATH.
#   WSL2 (Windows: run it in a WSL2 Ubuntu shell from the PlateCraft folder on /mnt/<drive>):
#                  the env goes to ~/.platecraft/envs/cosmos (a conda env on /mnt is very slow
#                  and can't hold all its symlinks); engine/envs/cosmos/wsl.json tells the Windows
#                  CLI to run the worker through wsl.exe. Weights stay in engine/models.
#
# Options: --env-dir DIR   where to build the env
#          --no-core       don't build the CLI env (default on WSL2: the CLI runs on Windows)
#          --skip-weights  don't download the weights (~28 GB; setup_engine.ps1 may have done it)
#          --force         rebuild the env and re-download the code
# Disk: ~15 GB for the env, ~28 GB weights. Safe to re-run: finished steps are skipped.
set -euo pipefail

# Keep in sync with engine/setup_engine.ps1 (checked 2026-11-01).
DR_REPO_URL=https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer
DR_SHA=0f3e2dc435032ecbad654c2fc2153df85384b138
DR_INVERSE_REV=41f63abccfb764b2516ff85b98af529c2662e2df     # nvidia/Diffusion_Renderer_Inverse_Cosmos_7B
DR_TOKENIZER_REV=b6af495317c76f287a4131e9299936b1533f5f9f   # nvidia/Cosmos-Tokenize1-CV8x8x8-720p
TE_VERSION=1.12.0
CORE_PACKAGES=("numpy<2" "opencv-python>=4.9,<5" "pillow>=10" "OpenEXR>=3.2,<4" "tqdm" "huggingface-hub>=0.30" "markdown>=3.5")

ENGINE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IS_WSL=0
grep -qi microsoft /proc/version 2>/dev/null && IS_WSL=1
if [ "$IS_WSL" = 1 ]; then
    LOCAL="$HOME/.platecraft"
    ENV_DIR="$LOCAL/envs/cosmos"
    WITH_CORE=0
else
    LOCAL="$ENGINE"
    ENV_DIR="$ENGINE/envs/cosmos"
    WITH_CORE=1
fi
SKIP_WEIGHTS=0
FORCE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --env-dir) ENV_DIR="$2"; shift 2 ;;
        --no-core) WITH_CORE=0; shift ;;
        --skip-weights) SKIP_WEIGHTS=1; shift ;;
        --force) FORCE=1; shift ;;
        -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

step() { printf '\n\033[36m==> %s\033[0m\n' "$1"; }
ok()   { printf '    \033[32m%s\033[0m\n' "$1"; }
warn() { printf '    \033[33mWARNING: %s\033[0m\n' "$1"; }

REPO="$ENGINE/repos/diffusion-renderer"
WEIGHTS="$ENGINE/models/diffusion-renderer"
MAMBA="$LOCAL/bin/micromamba"
export MAMBA_ROOT_PREFIX="$LOCAL/cache/mamba"
export HF_HOME="$ENGINE/models/hf" HF_HUB_DISABLE_TELEMETRY=1
export PIP_CACHE_DIR="$LOCAL/cache/pip"

echo "PlateCraft engine setup (Linux)"
echo "  engine folder : $ENGINE"
echo "  cosmos env    : $ENV_DIR"
[ "$IS_WSL" = 1 ] && echo "  WSL2 distro   : ${WSL_DISTRO_NAME:-?}"
if command -v nvidia-smi >/dev/null; then
    echo "  GPU           : $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"
else
    warn "nvidia-smi not found: install an NVIDIA driver with CUDA 12.4 support (550+)."
fi
warn "diffusion-renderer: NVIDIA Open Model License. Commercial use is allowed; credit 'Built on NVIDIA Cosmos'"
warn "and do not bypass safety guardrails (that terminates the license)."
for t in curl tar; do command -v "$t" >/dev/null || { echo "$t is required" >&2; exit 1; }; done

step "micromamba"
if [ ! -x "$MAMBA" ]; then
    mkdir -p "$LOCAL/bin"
    curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$LOCAL" bin/micromamba
fi
ok "$("$MAMBA" --version)"

step "DiffusionRenderer code"
if [ "$FORCE" = 1 ] || [ "$(cat "$REPO/.platecraft-sha" 2>/dev/null)" != "$DR_SHA" ]; then
    tmp="$(mktemp -d)"
    curl -fsSL "$DR_REPO_URL/archive/$DR_SHA.tar.gz" | tar -xz -C "$tmp"
    rm -rf "$REPO"
    mkdir -p "$ENGINE/repos"
    mv "$tmp"/*/ "$REPO"
    echo "$DR_SHA" > "$REPO/.platecraft-sha"
    rm -rf "$tmp"
fi
ok "at ${DR_SHA:0:7}"

step "Environment 'cosmos' (conda: Python 3.10, CUDA 12.4 toolkit, torch 2.6, transformer-engine $TE_VERSION)"
PY="$ENV_DIR/bin/python"
[ "$FORCE" = 1 ] && rm -rf "$ENV_DIR"
if [ ! -x "$PY" ]; then
    "$MAMBA" create -y -p "$ENV_DIR" -f "$REPO/cosmos-predict1.yaml"
fi
if ! "$PY" -c "import transformer_engine.pytorch" 2>/dev/null; then
    "$PY" -m pip install -r "$REPO/requirements.txt"
    "$PY" -m pip install "OpenEXR>=3.2,<4"       # PlateCraft's frame I/O in the worker
    # Upstream's install steps: expose the pip CUDA/cuDNN headers to the transformer-engine build.
    site="$ENV_DIR/lib/python3.10/site-packages"
    ln -sf "$site"/nvidia/*/include/* "$ENV_DIR/include/"
    ln -sf "$site"/nvidia/*/include/* "$ENV_DIR/include/python3.10"
    CUDA_HOME="$ENV_DIR" "$PY" -m pip install "transformer-engine[pytorch]==$TE_VERSION"
fi
CUDA_HOME="$ENV_DIR" PYTHONPATH="$REPO" "$PY" - <<'EOF'
import torch, transformer_engine
from cosmos_predict1.diffusion.inference.diffusion_renderer_pipeline import DiffusionRendererPipeline
dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA"
print(f"    torch {torch.__version__} (CUDA {torch.version.cuda}), transformer-engine {transformer_engine.__version__}, {dev}")
EOF
ok "environment ready"

if [ "$IS_WSL" = 1 ]; then
    mkdir -p "$ENGINE/envs/cosmos"
    printf '{"distro": "%s", "python": "%s"}\n' "${WSL_DISTRO_NAME:?not set: run this inside WSL2}" "$PY" \
        > "$ENGINE/envs/cosmos/wsl.json"
    ok "Windows CLI will run the worker in WSL2 ($WSL_DISTRO_NAME): engine/envs/cosmos/wsl.json"
elif [ "$ENV_DIR" != "$ENGINE/envs/cosmos" ]; then
    mkdir -p "$ENGINE/envs"
    ln -sfn "$ENV_DIR" "$ENGINE/envs/cosmos"
fi

if [ "$SKIP_WEIGHTS" = 0 ]; then
    step "Weights: diffusion-renderer (~28 GB)"
    "$PY" - "$WEIGHTS" "$DR_TOKENIZER_REV" "$DR_INVERSE_REV" <<'EOF'
import sys
from huggingface_hub import snapshot_download
target, tok_rev, inv_rev = sys.argv[1:4]
snapshot_download("nvidia/Cosmos-Tokenize1-CV8x8x8-720p", revision=tok_rev, local_dir=f"{target}/Cosmos-Tokenize1-CV8x8x8-720p",
                  allow_patterns=["README.md", "config.json", "encoder.jit", "decoder.jit", "mean_std.pt", "image_mean_std.pt"])
snapshot_download("nvidia/Diffusion_Renderer_Inverse_Cosmos_7B", revision=inv_rev,
                  local_dir=f"{target}/Diffusion_Renderer_Inverse_Cosmos_7B", allow_patterns=["README.md", "config.json", "model.pt"])
EOF
    ok "weights ready"
fi

if [ "$WITH_CORE" = 1 ]; then
    step "Core environment (PlateCraft CLI, Python 3.11)"
    CORE="$ENGINE/envs/core"
    [ -x "$CORE/bin/python" ] || "$MAMBA" create -y -p "$CORE" -c conda-forge python=3.11 pip
    "$CORE/bin/python" -m pip install "${CORE_PACKAGES[@]}"
    ok "core env ready"
    if command -v ffmpeg >/dev/null && command -v ffprobe >/dev/null; then
        ok "ffmpeg: $(ffmpeg -hide_banner -version | head -1)"
    else
        warn "ffmpeg/ffprobe not on PATH: install them (e.g. sudo apt install ffmpeg)."
    fi
fi

step "Done"
if [ "$IS_WSL" = 1 ]; then
    echo "    Next, in Windows:  platecraft.cmd doctor --model diffusion-renderer"
else
    echo "    Next:  ./platecraft.sh doctor --model diffusion-renderer"
fi
