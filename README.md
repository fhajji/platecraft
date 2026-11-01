# PlateCraft

Turn keyed video footage into temporally consistent PBR frame sequences (albedo + alpha,
normal, roughness / AO, depth) and play them in Unreal Engine 5.8 as a relightable
"video PBR material" for virtual production and Sequencer cinematics.

```
 ProRes 4444 with alpha, video + matte, or green/blue/black screen
        │
 extract ──► frames, alpha, straight colour, neutral-grey model plate, manifest.json
 infer   ──► normal/ depth/ albedo/ ao/ roughness/   (AI models, guided upscale to source res)
 pack    ──► atlas/frame_%06d.exr                    (2×2 atlas per frame, all linear)
        │
 Unreal  ──► Media Plate card + master material + CineCamera + Level Sequence → Movie Render Queue
```

Every step is resumable: an interrupted run continues where it stopped, and a rerun with
unchanged settings does nothing. Long clips are processed in overlapping chunks with
seamless joins.

Inspired by Beeble's AI relighting tools; this is an independent project, not affiliated with
or endorsed by Beeble.

## Requirements

- Windows 10/11 with an NVIDIA GPU. Every default is tuned for an **RTX 3060 12 GB** with
  32 GB RAM (`--preset auto` picks settings by detected VRAM).
- Unreal Engine 5.8 with the Python Editor Script Plugin and Editor Scripting Utilities
  enabled (no other plugins).
- About 18 GB of disk for the engine (Python environments + model weights), plus room for
  frames: a 421-frame 1080p clip takes a few GB across its passes and atlas.
- A Hugging Face account and read token: NormalCrafter's base model (Stable Video Diffusion)
  is gated.
- Optional, for Cosmos DiffusionRenderer only: Linux or Windows + WSL2 and a GPU with at
  least 20 GB VRAM.

## Install

The engine (portable Python, one virtual environment per model, ffmpeg, weights) is
downloaded into `engine/` and never committed. In PowerShell, from the repository folder:

```powershell
powershell -ExecutionPolicy Bypass -File engine\setup_engine.ps1 -HfToken hf_xxx
.\platecraft.cmd doctor
```

`doctor` checks the GPU, VRAM, CUDA, ffmpeg, environments and weights. Options, the
Linux / WSL2 setup and the engine layout are in [engine/README.md](engine/README.md).

## Usage

Run the CLI through `platecraft.cmd` (Windows) or `platecraft.sh` (Linux), which use the
engine's Python.

```powershell
.\platecraft.cmd process D:\shoot\take03.mov
```

`process` runs the three steps below in one go; each can also be run on its own:

| Command | What it does |
| --- | --- |
| `extract <video>` | Lossless frames, alpha (embedded, `--matte`, `--key black`, or chroma key fallback), premultiplied-alpha detection, neutral-grey plate for the models, `manifest.json`. |
| `infer <clip>` | Runs the model set, upscales to source resolution guided by the plate, converts normals to UE tangent space (DirectX green), derives albedo / AO / roughness. |
| `pack <clip>` | Packs all passes into one 2×2 atlas per frame (EXR or 16-bit PNG) so a single media player keeps them in sync. |
| `encode <dir>` | Preview video of a frame folder (`--alpha` for ProRes 4444). Preview only: data passes are never stored as lossy video. |
| `list-models` | Models, the passes they produce and their licences. |
| `doctor` | Environment check. |

Common options: `--start/--end/--max-frames` to trim, `--model quality|safe`,
`--commercial`, `--preset`, `--force`. Output lands in `work/<clip>/`:

```
work/<clip>/
  manifest.json      fps, frame count, resolution, passes, models, settings, licences, colour spaces
  src/ alpha/ plate/ extracted frames
  normal/ depth/     16-bit PNG
  albedo/            RGBA, straight alpha
  ao/ roughness/     16-bit PNG (heuristic)
  atlas/             frame_000001.exr ...
```

On the RTX 3060, 421 frames of 1080p take about 9 min for NormalCrafter, 1 min for depth,
5 min for derived maps and 3 min for packing.

### Unreal Engine

Add the repository folder to *Project Settings > Plugins > Python > Additional Paths* (or copy
`unreal/Content/Python/` into the project), restart the editor and use
**Tools > PlateCraft > Import Clip...**. The importer creates an Img Media Source, the shared
master material `M_PlateCraft` and a material instance, places a Media Plate card facing the
viewport camera, adds a CineCamera framing it and a Level Sequence with a frame-accurate media
track. Re-importing updates everything in place. The same importer runs from the editor's
Python console:

```
py "<repo>/unreal/Content/Python/platecraft_import.py" <clip folder> [options]
```

For Movie Render Queue renders of keyed plates, set *Anti-aliasing > Spatial Sample Count* to 8
and *Override Anti-Aliasing* to None, so soft matte edges resolve to their true coverage.

## Models and licences

PlateCraft's own code is BSD-licensed, but the models it downloads and runs are not. Every
model's licence is printed when it runs and recorded in the clip's `manifest.json`;
`--commercial` refuses non-commercial models.

| Model | Passes | Commercial use |
| --- | --- | --- |
| NormalCrafter | normal | Restricted: based on Stable Video Diffusion (Stability AI Community License: free under US$1M annual revenue, otherwise an Enterprise licence) |
| Video Depth Anything Small | depth | Yes (Apache 2.0) |
| Video Depth Anything Base / Large | depth | No (CC-BY-NC 4.0) |
| Marigold Normals v1-1 | normal (per frame) | Yes: Apache 2.0 code, weights CreativeML Open RAIL++-M (use restrictions passed on) |
| RAFT optical flow (torchvision) | temporal filter for normals | Yes (BSD-3-Clause) |
| Cosmos DiffusionRenderer | albedo, normal, depth, roughness, metallic | Yes: Apache 2.0 code, weights NVIDIA Open Model License (credit "Built on NVIDIA Cosmos"). Linux / WSL2, ≥ 20 GB VRAM; not yet tested on real hardware |
| ffmpeg | decoding / encoding | LGPL build, run as a separate program |

Model sets: `quality` (default) = NormalCrafter + Video Depth Anything Small;
`safe` = Video Depth Anything Small + Marigold Normals with the RAFT temporal filter
(depth-derived normals if Marigold isn't installed). Check the current licence terms yourself
before commercial use; this table is not legal advice.

## Documentation

[TECHNICAL.md](TECHNICAL.md) explains every algorithm (decoding, alpha handling, keying,
inference, upscaling, derived maps, atlas layout), the Unreal side (material, card mesh,
camera, Sequencer, Movie Render Queue) and gives a practical guide for filmmakers: shooting,
keying and delivery in DaVinci Resolve, and working in Sequencer.

## Repository layout

```
platecraft.cmd / .sh / .py   CLI entry points
platecraft/                  pipeline: extract, infer, pack, postprocess, presets, doctor
platecraft/models/           one adapter per model, each run as a subprocess in its own env
unreal/Content/Python/        UE importer, import dialog, menu registration
engine/                       setup scripts; everything else in it is downloaded, not committed
```

## License

[BSD 2-Clause](LICENSE). Copyright (c) 2026 Farid Hajji. Third-party models and tools keep
their own licences (see above).
