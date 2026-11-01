# PlateCraft engine

Self-contained runtime for the PlateCraft CLI. Nothing here except this file
and `setup_engine.ps1` is committed; the script downloads the rest.

## Install

From the PlateCraft folder, in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File engine\setup_engine.ps1 -HfToken hf_xxx
.\platecraft.cmd doctor
```

NormalCrafter's base model (`stabilityai/stable-video-diffusion-img2vid-xt`)
is gated on Hugging Face. Accept its license on the model page while logged
in, create a read token at https://huggingface.co/settings/tokens and pass it
with `-HfToken` (or set `HF_TOKEN`).

Options: `-Models normalcrafter,vda-s,marigold-n,raft,vda-l,diffusion-renderer` (default: the first four), `-SkipWeights`, `-SkipFfmpeg`,
`-Force` (reinstall), `-Cuda cu124`, `-TorchVersion 2.5.1`.
The disk footprint is roughly 18 GB with the default models (two PyTorch CUDA
environments, which share files through uv's cache, plus the weights).

## Linux-only models (Cosmos DiffusionRenderer)

`diffusion-renderer` (albedo, normal, depth, roughness, metallic from one 7B video model) needs
Linux (transformer-engine) and a GPU with at least 20 GB VRAM (about 27 GB per 57-frame chunk;
the CLI refuses smaller GPUs, `PLATECRAFT_IGNORE_VRAM=1` overrides). On Windows,
`setup_engine.ps1 -Models diffusion-renderer` downloads its code and weights (~28 GB) but cannot
build its environment. Build that with `setup_engine.sh`:

- **Windows + WSL2** (an Ubuntu distro, NVIDIA driver on Windows): in the WSL shell, from the
  PlateCraft folder (`cd /mnt/m/...`), run `bash engine/setup_engine.sh`. The conda env goes to
  `~/.platecraft/envs/cosmos` inside WSL; `engine/envs/cosmos/wsl.json` makes the Windows CLI
  run the model worker through `wsl.exe`. Everything else stays on Windows.
- **Linux PC**: `bash engine/setup_engine.sh` builds `engine/envs/cosmos` and the CLI env
  `engine/envs/core` (needs `ffmpeg` on PATH); run the CLI with `./platecraft.sh`.

The env is built from upstream's `cosmos-predict1.yaml` (Python 3.10, CUDA 12.4 toolkit, gcc 12)
with a portable micromamba, then `requirements.txt` and `transformer-engine[pytorch]==1.12.0`
(compiled; takes a while). About 15 GB. Check with `doctor --model diffusion-renderer`.

## Layout

| Folder | Contents |
| --- | --- |
| `bin/` | `uv.exe`, `ffmpeg.exe`, `ffprobe.exe` (LGPL build); `micromamba` on Linux |
| `python/` | Portable Python 3.11 managed by uv |
| `envs/core` | CLI environment (numpy, OpenCV, OpenEXR) |
| `envs/normalcrafter` | PyTorch + diffusers 0.29.1 for NormalCrafter and Marigold Normals |
| `envs/vda` | PyTorch + torchvision for Video Depth Anything and RAFT |
| `envs/cosmos` | Linux only (`setup_engine.sh`): conda env for Cosmos DiffusionRenderer, or `wsl.json` pointing into WSL2 |
| `repos/` | Model source code, pinned to tested commits |
| `models/` | Weights (`normalcrafter/`, `vda-s/`, `vda-l/`, `marigold-n/`, `diffusion-renderer/`, `torch/` for RAFT) and the Hugging Face cache |
| `engine.json` | What was installed, written by the setup script |

## Licenses

| Component | License | Commercial use |
| --- | --- | --- |
| NormalCrafter | MIT code; weights based on Stable Video Diffusion | Restricted: Stability AI Community License (free under US$1M annual revenue) |
| Video Depth Anything Small | Apache 2.0 | Yes |
| Video Depth Anything Large | CC-BY-NC 4.0 | No |
| Marigold Normals v1-1 | Apache 2.0 code; weights CreativeML Open RAIL++-M (Stable Diffusion 2 base) | Yes, with the license's use restrictions (Attachment A) passed on |
| RAFT (torchvision weights) | BSD-3-Clause | Yes |
| Cosmos DiffusionRenderer (inverse 7B + Cosmos tokenizer) | Apache 2.0 code; weights NVIDIA Open Model License | Yes: credit "Built on NVIDIA Cosmos"; redistributing the weights needs the notice "Licensed by NVIDIA Corporation under the NVIDIA Open Model License"; bypassing a safety guardrail terminates the license |
| ffmpeg (bundled build) | LGPL | Yes, as a separate program |
