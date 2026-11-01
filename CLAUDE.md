# PlateCraft — Video to PBR material pipeline

Turns video footage (typically chroma-keyed camera recordings) into temporally
consistent PBR frame sequences (albedo+alpha, normal, roughness/AO, depth) and
imports them into Unreal Engine 5.8 as a playable "video PBR material" for
virtual production and Sequencer cinematics. Inspired by Beeble's AI
relighting studio (beta).

Full plan: https://claude.ai/artifact/C6WvyaZyS2aVVjvhdcRRVb
Algorithms + filmmaker/Resolve/Sequencer guide: TECHNICAL.md (keep it in sync when behaviour changes).

## Target environment
- Windows, NVIDIA RTX 3060 **12 GB VRAM**, 32 GB RAM. Every default must run on this.
- Unreal Engine **5.8** (verify Python API names against the installed version;
  5.8 is newer than Claude's training data).
- Use case: keyed footage placed into UE environments, rendered via Sequencer.

## Layout (planned)
```
platecraft.py              # CLI entry point
platecraft/
  cli.py                    # argparse subcommands: process, extract, infer, pack, encode, list-models, doctor
  ffmpeg_io.py              # extract / encode / probe (fps, frame count, color, alpha)
  extract.py                # `extract` command: src/ alpha/ plate/ + manifest
  frames.py                 # PNG/EXR I/O, despill, plate composite, alpha normalisation
  manifest.py               # manifest.json read/write
  doctor.py, presets.py, paths.py
  pipeline.py               # windowing, overlap blending, resume
  postprocess.py            # normal-space conversion, derived maps, guided upscaling, atlas packing
  models/
    base.py                 # ModelAdapter interface: name, passes, license, run(frames) -> {pass: frames}
    registry.py             # --model lookup; add a model by adding one adapter file
    normalcrafter.py
    video_depth_anything.py
    diffusion_renderer.py   # Linux-only env; on Windows the worker runs through WSL2
engine/                     # self-contained runtime, NOT committed (see .gitignore)
  setup_engine.ps1          # installs everything below (Windows)
  setup_engine.sh           # Linux / WSL2: Linux-only model envs (cosmos) + core env on native Linux
  python/                   # portable Python 3.11
  envs/<model>/             # one venv per model; adapters run as subprocess workers
  bin/ffmpeg.exe            # LGPL static build, invoked as an external process
  models/                   # downloaded weights (Hugging Face cache)
unreal/Content/Python/platecraft_import.py, platecraft_ui.py, init_unreal.py
```

## Models and licenses (must be respected in code)
| Model | Passes | 12 GB? | Commercial use |
|---|---|---|---|
| NormalCrafter | normal | yes, ~576–768px, fp16, offload | Restricted: built on Stable Video Diffusion (Stability AI Community License: free under US$1M annual revenue, else Enterprise licence) |
| Video Depth Anything Small | depth | yes | OK (Apache 2.0) |
| Video Depth Anything Base/Large | depth | yes | Non-commercial (CC-BY-NC 4.0) |
| Marigold Normals v1-1 (`marigold-n`) | normal (per frame) | yes, 6.8 GB at 768px | OK: Apache 2.0 code; weights CreativeML Open RAIL++-M (use restrictions passed on) |
| DSINE | normal | - | Non-commercial (Imperial College licence): rejected for `safe` |
| Cosmos DiffusionRenderer (`diffusion-renderer`) | albedo, normal, depth, roughness, metallic | no (20 GB min enforced, ~27 GB video); Linux only | OK: Apache 2.0 code; weights + tokenizer NVIDIA Open Model License (credit "Built on NVIDIA Cosmos"; no guardrail bypass) |

Rules:
- Every adapter declares its license. Restricted models print a warning with the
  license and a safer alternative, and the license is written into manifest.json.
- `--commercial` refuses non-commercial models.
- Commercial-safe set (`--model safe`): VDA Small for depth + Marigold Normals v1-1
  (per frame) + RAFT optical-flow temporal filter (torchvision weights, BSD-3; `--temporal-filter`),
  with depth-derived normals (`depth-normals`) as fallback when Marigold isn't installed.
  Licenses checked 2026-10-29.
- Re-verify licenses/releases before adding or updating a model.
- Default model set is `quality` (NormalCrafter + VDA Small), decided 2026-10-31; `safe` on request.

## Pipeline conventions
- Extraction: lossless PNG frames, keep every frame (no frame dropping), keep fps/timing.
- Keyed footage: accept alpha (ProRes 4444) or a separate matte sequence; basic ffmpeg
  chromakey+despill is a fallback only. Before inference, replace background with
  neutral grey and suppress green spill.
- Long clips: process in overlapping chunks with seamless joins; interrupted runs resume
  automatically from the cache (same settings), `--force` redoes.
- Run inference at reduced resolution, then guided-upscale to source resolution.
- Normals: models output camera-space normals; convert to tangent space with UE's
  DirectX green convention (`--normal-space`).
- Albedo = keyed plate (optional rough de-lighting); roughness = constant/heuristic;
  AO from depth+normals. Label heuristic maps as such.
- Output per pass: `<clip>/<pass>/frame_000001.png` (next to manifest.json); normals 16-bit PNG,
  depth/roughness/AO 16-bit PNG or EXR; albedo+alpha as RGBA. Never 8-bit for data maps.
- `manifest.json`: fps, frame count, resolution, passes, models, settings, licenses,
  color space per pass (sRGB vs linear).
- Presets: `--preset 3060` (768px, fp16, model CPU offload, decode chunk 4); measured numbers
  in presets.py. `--preset auto` (default) picks by detected VRAM.
  `platecraft doctor` checks GPU/VRAM/CUDA/ffmpeg/weights.

## Unreal side
- Pack passes into a 2x2 atlas per frame (albedo/alpha | normal / roughness-AO | depth)
  so a single Img Media Source + single player keeps all passes in sync.
- Play via **Media Plate**; Sequencer Media Track for frame-accurate cinematics.
- Recommended Movie Render Queue settings for keyed plates: in Anti-aliasing, set Spatial Sample Count
  to 8 and set Override Anti-Aliasing to None (measurements under Status).
- `platecraft_import.py`: read manifest -> create Img Media Source (correct fps) ->
  build master material (alpha cutout with soft edges, normal unpack, sRGB/linear
  correction per region, roughness control, optional depth parallax/card bending,
  shadow casting) -> material instance -> place Media Plate -> optional Level
  Sequence with bound media track.
- Use only built-in editor scripting (no plugin dependencies). Optional Editor
  Utility Widget with folder picker + Import button.
- Don't encode data passes to lossy video; MP4 is preview-only.

## Build order
1. Engine setup script + `doctor`
2. ffmpeg extraction with alpha/key handling + manifest
3. NormalCrafter adapter on a short clip; measure speed/VRAM on the 3060
4. VDA Small, derived maps, guided upscale, commercial-safe normal path + temporal filter
5. Atlas packing
6. UE import script (Media Plate + material + Sequencer)
7. Editor Utility Widget

## Status
- Done: step 1 (engine, doctor; verified on the RTX 3060), step 2 (extract, encode),
  step 3 (`infer` with the NormalCrafter adapter; verified on kitty), step 4 (VDA, Marigold,
  depth-normals, RAFT temporal filter, guided upscale, albedo/AO/roughness; verified on kitty),
  step 5 (`pack`; verified on kitty), step 6 (`unreal/Content/Python/platecraft_import.py`; verified
  on kitty in UE 5.8: lit, normals, shadow, bend, Sequencer scrub/play, re-import, MRQ frame-accurate),
  step 7 (`platecraft_ui.py` + `init_unreal.py`: Tools > PlateCraft > Import Clip...).
- Step 8 (beyond the original plan): Cosmos DiffusionRenderer adapter, implemented and downloaded 2026-11-01,
  **untested on real hardware** (needs >= 20 GB VRAM + Linux/WSL2); test plan in its section below.
- Step 7 is a details-panel dialog instead of an Editor Utility Widget: a widget blueprint is a binary
  .uasset that can't be built from a script or diffed. `EditorDialog.show_object_details_view` on a
  Python `@uclass` options object (DirectoryPath property = stock "..." OS folder picker; EditCondition
  works with the Python property names, checked in the UI); OK = Import, invalid input re-shows the dialog. Placement
  default: centred on the viewport camera's view line at 400 cm, facing it. Last options in
  `<project>/Saved/PlateCraft/import_dialog.json`. Menu is registered by init_unreal.py, which UE runs
  from every Python path (incl. Project Settings > Python > Additional Paths) at startup; `work/UETest`
  has the repo folder in AdditionalPaths. show_object_details_view returns False under -unattended.
- Camera (default on; `--no-camera`, dialog "Add Camera"): CineCamera actor `PlateCraftCam_<clip>` (possessable in
  the level like the plate; adding spawnables from scripts is deprecated in 5.8 / needs an open
  Sequencer), 35 mm, f/8, manual focus on the plate, filmback height = 23.76 mm * tile aspect, distance =
  height * focal / sensor_h -> plate edges land exactly on the image edges (checked: NDC +-1.0000).
  A Camera Cuts track is added only if the sequence has none. An existing camera keeps its transform.
  Stale bindings (sequence made in another level) are rebound and cuts that pointed at them retargeted.
  Python pitfall: unreal.Guid structs don't compare by value (`==` False); match bindings by name.
- `pack <clip> [--format exr|png] [--exr-compression zip|piz|none] [--scale 1] [--start/--end/--max-frames]
  [--force]` -> `<clip>/atlas/frame_%06d.exr` + manifest `atlas` (size, tile, per-region rect/channels/
  encoding/source/constant). Layout: TL albedo RGB, TR normal (n*0.5+0.5), BL ORM (R=AO, G=roughness,
  B=metallic 0), BR depth (R=G=B); A = alpha in every quadrant. All values linear (albedo converted from
  sRGB, transparent pixels filled with nearby colour by push-pull so bilinear edges don't go dark;
  depth below alpha 0.5 filled from the subject so a depth-bent card doesn't fold at the silhouette).
  Missing passes -> constants. Resume: frames newer than all their sources are skipped; changed
  settings/sources clear the folder.
- UE 5.8 Img Media (checked in engine source): EXR -> PF_FloatRGBA, 16-bit RGBA PNG -> PF_A16B16G16R16,
  8-bit -> BGRA8. 16-bit PNG is read as linear. EXR reader ignores in-file mips; GPU EXR path needs
  uncompressed + a special "EXR GPU" format, not used.
- kitty atlas: 3840x2160 half EXR ZIP, ~4.5 MB/frame (1.8 GB), 0.34 s/frame packing (OpenEXR library
  writer with 8 threads; cv2's EXR writer was 1.5 s/frame). Max error vs passes 2.4e-4 (half precision).
- UE import (in the editor: `py "<repo>/unreal/Content/Python/platecraft_import.py" <clip> [--height cm]
  [--location x,y,z] [--yaw] [--grid 128] [--depth-amount cm] [--no-sequence] [--no-place] [--rebuild-material] [--no-light-shafts] [--streaming-radius m]`):
  shared `/Game/PlateCraft/M_PlateCraft` (+ metadata VERSION tag; bump VERSION when the graph changes)
  and `SM_PlateCraftCard_CxR` (grid copy of SM_MediaPlateScreen: YZ plane, front -X, U along +Y, V down,
  pivot bottom centre); per clip `IMS_/MI_/LS_<clip>` and Media Plate actor `PlateCraft_<clip>` (updated in place).
  Material (v2): Masked, two-sided, clip 1/3, params AlbedoTint/Brightness, AlphaBlackPoint/WhitePoint
  (levels before the mask: choke a soft matte edge), NormalStrength, FlipGreen, AOStrength,
  RoughnessScale/Offset, MetallicOffset, Specular, DepthAmount/DepthPivot (WPO along the vertex normal;
  0 = flat), static SoftEdges (off = hard cut at alpha 0.5). Quadrant UVs clamped 1/TextureSize inside.
  Soft edges = Custom node (DITHER_HLSL): drawn iff alpha > threshold; threshold = per-pixel interleaved
  gradient noise + k/8, k = StateFrameIndexMod8 XOR a pixel hash -> 8 consecutive samples hit 8 evenly
  spaced thresholds, coverage = alpha. Not the engine's DitherTemporalAA: that is `((x + 2y + frame) mod 5)
  / 6` + a noise texture, i.e. thresholds only over alpha 1/6..5/6 (MRQ render of kitty3: alpha 0.6-0.7 drew
  93%, the soft fur became a solid dark rim). Found on the way: independent noise per sample = visible grain
  (+-0.18 after 8 samples); the same k order in every pixel correlates with MRQ's jitter = h/v line pattern.
- MRQ settings for keyed plates: Anti-aliasing with spatial samples 8 (MRQ advances the frame index per
  sample: FrameIndex = frame * samples + sample) and **Override Anti-Aliasing = None**. Measured coverage
  per alpha band (kitty3 frame 213; estimate = solved blend of nearby plate and sky):
  alpha 0.4-0.5 / 0.5-0.6 / 0.7-0.8: DitherTemporalAA 1 spp 0.74/0.86/0.99; v2 1 spp (TSR) 0.60/0.72/0.90;
  8 spp + TSR 0.60/0.71/0.89 (TSR keeps edges too solid and dotted); 8 spp + AA None 0.43/0.53/0.75.
- UE 5.8 facts found while building it (engine source + editor runs):
  - Media Plate binds its texture to any material parameter named `MediaTexture`. Its ApplyMaterial isn't
    exposed; `set_editor_property("override_materials", [mi])` on the plate's StaticMeshComponent fires
    PostEditChange -> the plate wraps the MI in a MID with the texture (logs a harmless "no MediaTexture
    parameters" warning for MIs). `select_media_source_asset` alone doesn't open the player: call `open()`.
  - The plate's mesh component has bCastDynamicShadow off by default.
  - `StaticMesh.build_from_static_mesh_descriptions(..., fast_build=True)` (default) skips normals -> NaN
    bounds; use fast_build=False and set LOD build settings only after that build (before = crash).
  - A media track on the plate's binding drives the plate's texture (the plate's own player stays at 0);
    the Media Plate track editor would disable autoplay, so the importer does when it makes a sequence.
  - `-ExecutePythonScript` quits after the script unless `unreal.EditorPythonScripting.set_keep_python_script_alive(True)`.
    Test project for this: `work/UETest` (PythonScriptPlugin + EditorScriptingUtilities enabled).
  - Playback (3060): Sequencer's media track waits for every frame. 3840x2160 half EXR (60 ms decode) =
    3-6 editor fps with frames cancelled; 1920x1080 (`pack --scale 0.5`, 15 ms) = 79 fps, real time.
    Plate autoplay at 3840x2160: editor 103 fps (frame drops not measured).
  - Movie Render Queue (verified, 3840x2160 atlas, PIE executor, 1920x1080 PNG): frame-accurate. Test: a
    camera framing the card exactly so a render maps 1:1 onto the albedo tile, then high-pass NCC of each
    render against atlas frames -4..+4 (scratchpad scripts, not in the repo); wherever the subject moves the
    expected frame wins by a wide margin (e.g. 0.74 vs 0.35), the rest are ties between near-duplicate
    source frames. 30 frames ~16 s incl. PIE startup.
  - The media track only blocks on frames while *playing*; scrubbing/stopped (MRQ's first evaluation of a
    shot) doesn't wait unless the track's `synchronous_scrubbing` is on -> blank first frame of every MRQ
    shot (A/B tested). The importer turns it on.
  - MRQ test pitfall: a sequence with several camera-cut tracks / stale camera bindings renders from a
    random one (looked like a media failure). Delete LS_<clip> between test runs.
  - Dark wedges on the wall beside the plate, widening away from the sun = the Directional Light's
    Light Shaft Occlusion (screen-space, depth based: unaffected by Cast Shadow/Lumen; `ShowFlag.LightShafts 0`
    removes them). It's a level-light setting: the importer turns it off only on request
    (`--no-light-shafts`, dialog "Turn Off Light Shaft Occlusion"; default off), on every directional light.
  - World Partition levels: a spatially loaded plate streams into PIE/MRQ from its saved package and
    renders the material's default texture (opaque grey, black when lit) = "black render". The importer
    makes the plate always loaded (`is_spatially_loaded` False). Remote execution for driving the editor
    from outside: Project Settings > Plugins > Python > Enable Remote Execution, then UE's own
    `PythonScriptPlugin/Content/Python/remote_execution.py` (MODE_EXEC_FILE).
  - Black sky/landscape (RGB 0, alpha 0) in PIE/MRQ while the viewport looks fine: the level was made from
    the Open World template and saved, and the editor's in-memory WP actor descriptors still point to
    `/Temp/Untitled_N` (files on disk are fine). PIE streaming skips those actors (no sky, lights, Landscape,
    PlayerStart); only actors saved since appear. Fix: save, then reopen the level. The importer warns
    (`check_world_partition_descriptors`: descriptor `actor_path` starting with /Temp/).
  - Distant terrain missing from PIE/MRQ but in the viewport (editor loads every cell): only cells within the
    grid's loading range of the player stream in (Open World template: 256 m) and MRQ renders with r.HLOD 0.
    Beyond the loaded landscape the Sky Atmosphere's flat grey planet ground shows. The importer gives the
    camera a WorldPartitionStreamingSourceComponent (shape radius, use_grid_loading_range False;
    `--streaming-radius` m, default 3000, dialog "Camera Streaming Radius"); MRQ then matches the viewport
    (verified on kitty frame 213). Adding a component to a level actor: SubobjectDataSubsystem.add_new_subobject.
  - The plate sets Shadow Cache Invalidation Behavior = Always: VSM's Auto only invalidates on transform/WPO,
    not on the video's alpha changing.
- Run the CLI with `platecraft.cmd` (engine core Python), not `platecraft.py` directly.
- `process <input>` = extract + infer + pack with all their options (shared `_add_*_options` helpers in
  cli.py); `--start/--end/--max-frames` trim the extraction, infer/pack then take the whole clip; `--force`
  redoes every step. Models/licenses are checked before extracting. Verified on 12 kitty frames (274 s, mostly
  NormalCrafter loading); a rerun with nothing to do takes ~14 s (model workers still start to check their cache).
- infer skips final-pass frames (normal, depth, albedo/ao/roughness) that are newer than every input frame
  (raw/filtered output, plate, alpha, fg) when the pass's manifest entry (model, settings, depth raw_range)
  is unchanged; a changed entry rewrites the whole range. `--redo-passes` rewrites them anyway (model cache
  kept); `--force` implies it. `frames.write` writes to `<name>.tmp` then renames, so an interrupted
  run can't leave a truncated frame that looks up to date. Verified: rerun, changed setting, touched raw
  frame (only that frame + its atlas redone), --redo-passes.
- Clip folder from `extract` (default `work/<clip>/`): `src/` RGB as decoded, `alpha/`
  (white = foreground, opaque = exactly 1.0), `plate/` (model input: despilled over grey;
  manifest's `plate.dir` points to `src` when no plate was needed), `manifest.json`.
- `--premultiplied auto|yes|no` (embedded alpha or --matte): auto tests 6 frames spread over the clip
  (frames.premultiplied_stats): premultiplied color never exceeds its alpha (< 5% of soft-edge pixels may,
  for codec noise) AND edge brightness / nearby solid brightness tracks alpha rather than 1 (keeps a dark
  straight subject from passing). kitty2: 0.0% over, 0.11 vs 0.56 -> premultiplied; kitty3: 43-52% over ->
  straight. Premultiplied -> `fg/` = src/alpha (straight, src bit depth), plate built from fg/, manifest
  `alpha.premultiplied`. Premultiplication is in the encoded (sRGB/video) values, as Resolve writes it.
  Verified: kitty2 fg/ = kitty3 src/ within 0.001-0.004, plate within 0.0006, albedo within 0.0004.
  Extract settings saved before an option existed count as its default (old clips stay "already extracted").
- `--key black`: luma key for subjects over pure black (hole-filled core, ratio alpha in an
  edge band, decontaminated `fg/` pass = straight color; `src/` stays premultiplied over black).
  `--garbage x0:y0:x1:y1` (repeatable) forces regions transparent, for the black key and the chroma key. `encode <dir> --alpha` -> ProRes 4444.
- YUV->RGB uses the stream's matrix/range (untagged HD = BT.709) with accurate rounding.
- `infer <clip> [--model quality|safe|keys] [--preset] [--max-res] [--start/--end/--max-frames]
  [--normal-space directx|opengl] [--temporal-filter auto|on|off] [--temporal-strength 0.8]
  [--upscale guided|bicubic] [--no-derive --roughness --roughness-cavity --ao-strength --delight
  --edge-color 1] [--commercial] [--force]`. Model output cache: `<clip>/raw/<model>/` (inference res, settings.json,
  worker.log); filtered normals `raw/<model>+tf/`. Final passes `normal/ depth/` + derived
  `albedo/` (RGBA straight, src bit depth) `ao/ roughness/` (16-bit, heuristic).
- `--edge-color` (postprocess.edge_color; skipped for the black key, whose fg/ is fixed already): soft-edge albedo pixels get
  luminance lifted to the nearby solid subject (only brighter) and its chromaticity, fully below alpha 0.5,
  none at 0.98. Why: kitty3's fur tips were 10-25% darker (dark background baked in) and the outermost
  pixels pure red (R/G 8 vs 1.5): 4:2:0 sources smear chroma 1-2 px into the dark background.
- manifest `models`/`licenses` list exactly the models behind the current passes (+ raft if the
  normals were filtered); running another model for a pass replaces the old one.
- Adapters: `platecraft/models/<adapter>.py` (ModelInfo.adapter in registry.py). Core side
  `job_settings(preset)` + `run(job)`; worker side runs as `python -m platecraft.models.<x> job.json`
  in engine/envs/<env> (PYTHONPATH = repo root + engine/repos/<repo>), prints PROGRESS:/RESULT: lines.

## Notes for step 4
- Test clip: `kitty.mp4` (1920x1080, 30 fps, 8-bit HEVC, 421 frames, kitten composited over
  black), extracted to `work/kitty/` with `extract kitty.mp4 --key black --garbage 0:1064:1920:1080`.
  `kitty_keyed.mov` = transparent ProRes 4444 of it.
- `kitty2.mov` / `kitty3.mov` (2026-11-01): Resolve re-exports, ProRes 4444 12-bit with a DeltaKeyer alpha
  (`extract` picks it up as embedded; same alpha in both). kitty2's color is **premultiplied** (edge RGB =
  alpha x fur color; `--premultiplied auto` detects it, see above); kitty3 = same with a
  Fusion "Alpha Divide" after the keyer = straight. Use kitty3. Full clip with the quality set: NormalCrafter
  539 s (0.78 fps; 0.32 fps while Resolve was rendering on the same GPU), VDA 61 s, derive ~5 min, pack 3 min.
- doctor on the user's PC: RTX 3060 12 GB, driver 616.92, torch 2.5.1+cu124, xformers
  0.0.28.post3, diffusers 0.29.1; all models' weights present.
- NormalCrafter (engine/repos/normalcrafter, commit 75af988), measured:
  - Raw output is camera space **+X right, +Y up, +Z toward camera** (OpenGL); DirectX flips G.
    Verified by correlating edge normals with the alpha silhouette's outward direction.
  - The adapter drives the pipeline itself (`_infer_chunk`): each chunk's first window is
    conditioned on the previous chunk's latents, so chunk/resume seams match the model's own
    window seams (an unconditioned restart differs by 2-7 deg; a pixel crossfade doesn't hide it).
  - 3060: 768x432 = 0.96 fps, 7.8 GB reserved; 1024 = 0.42 fps, 11.9 GB (too close to the limit).
- VDA (engine/repos/vda, commit 4f5ae23): adapter streams upstream's `infer_video_depth`
  (32-frame windows, step 22, key frames incl. the run's first frame, scale/shift alignment,
  8-frame crossfade) and keeps `vda_state.npz`; matches upstream within 0.06%, an interrupted
  + resumed run is bit-exact. Raw = float EXR disparity at 924x518; final depth/ is normalised
  clip-wide (0.5/99.5 percentiles of the subject), 1 = nearest. 3060: 7.7 fps, 3.0 GB reserved.
- Marigold (diffusers MarigoldNormalsPipeline in the normalcrafter env, fp16, 4 steps, same noise
  every frame): same axes as NormalCrafter (measured). 3060: 768x432 = 1.42 fps, 6.8 GB reserved.
  Much more fine detail than NormalCrafter, but flickers.
- RAFT filter (vda env, raft_large): recursive, warps the previous output by flow, F/B consistency
  + alpha mask. 4 fps, 1.2 GB. Flow-compensated change/frame on kitty: Marigold raw 0.077,
  filtered 0.029 at the default 0.8 (NormalCrafter 0.025). Metric script idea: warp out[t-1] by
  RAFT flow and compare to out[t] inside alpha (plain frame differences mostly measure motion).
- Guided upscale: fast guided filter with the full-res plate as color guide (closed-form 3x3
  inverse; np.linalg.solve was 6 s/frame), ~0.5 s/frame single-threaded.
- AO: depth "behind neighbourhood" at 3 radii + normal concavity, faded out over h/25 px from
  the silhouette (otherwise every outline reads as a crease).
- Model adapters run in their own env (engine/envs/<env>) as subprocess workers; the CLI
  (core env) has numpy/opencv/OpenEXR but no torch.
- Delivering files to the user's PC: after copying, verify sizes on the device (one copy
  once reported success but left old files).

## Cosmos DiffusionRenderer (added 2026-11-01; NOT run on a real GPU yet)
- Code: nv-tlabs/cosmos-transfer1-diffusion-renderer @ 0f3e2dc (the old cosmos1-diffusion-renderer URL
  serves it) -> engine/repos/diffusion-renderer. Weights (downloaded on the user's PC, MD5s match upstream's
  download script): engine/models/diffusion-renderer/{Diffusion_Renderer_Inverse_Cosmos_7B/model.pt (27 GB),
  Cosmos-Tokenize1-CV8x8x8-720p/{encoder,decoder}.jit + mean_std.pt + image_mean_std.pt}; HF revisions pinned
  in both setup scripts. No T5 / guardrail needed (upstream's inverse pipeline disables both).
- Upstream needs Linux (transformer-engine 1.12, Python 3.10, torch 2.6). `setup_engine.sh` builds the env from
  upstream's cosmos-predict1.yaml with a portable micromamba. WSL2: env in ~/.platecraft/envs/cosmos and
  `engine/envs/cosmos/wsl.json` {distro, python}; `paths.worker_launcher` then runs workers via
  `wsl.exe -d <distro> -e env K=V... python` and maps job/settings paths `M:\x` -> `/mnt/m/x`. Native Linux:
  `./platecraft.sh` + engine/envs/core. The user's PC has no WSL distro (only docker-desktop).
- Adapter: input exactly 1280x704 (latent shape fixed in upstream's config), plate fitted inside + grey padding
  (1920x1080 -> 1248x704); 57-frame chunks (8n+1), overlap 8, 15 steps, guidance 0, seed 1000; one sampling
  run per pass (5 per chunk). `_generate` = upstream generate_video without its uint8 quantisation; offload
  (< 40 GB) moves the DiT to CPU RAM during decode (upstream's offload reloads 27 GB from disk every call).
  Outputs in -1..1: basecolor (assumed sRGB-encoded), normal (axes NOT verified: AXES constant + a silhouette
  check printed after each run, validated on NormalCrafter's kitty normals: +0.48/+0.48, flips detected),
  depth (Marigold-style per-chunk near..far -> -1..1, near = -1) stored as 1 - d, roughness, metallic.
  Chunks on a grid from the run's start; depth fitted (scale/shift, closed form in float64: np.polyfit on ~1e6
  float32 points returned half the true slope) to the previous chunk over the overlap, then a linear crossfade;
  the pre-blend tail is saved to raw/diffusion-renderer/dr_state.npz -> resume after a crash is bit-identical.
  worker.log has a per-chunk depth-fit line.
- `preflight()` (VRAM >= 20 GB unless PLATECRAFT_IGNORE_VRAM=1; env present) runs before extract/infer.
- infer: multi-pass adapters give `raw_pass(raw, pass) -> (dir, ext)`; new final passes albedo (16-bit RGBA,
  sRGB), roughness, metallic (16-bit) via postprocess.model_albedo/model_gray. derive keeps albedo/roughness/
  metallic made by a model of the *current* run; otherwise heuristics replace them and metallic is removed.
  pack puts a metallic pass in ORM B.
- Verified on the 3060 with a stand-in `cosmos_predict1` package (same interface, outputs derived from the
  input, per-chunk depth normalisation; scratchpad only) on 41 kitty frames with 17-frame chunks: infer ->
  derive (AO only) -> pack (B = metallic); crash mid-chunk + resume bit-identical; depth one scale across chunks
  with per-chunk squeezes 0.67..1.11; switching back to vda-s/depth-normals drops the model maps + license.
  Still to check on a big GPU: real VRAM/speed, normal axes, basecolor colour space, seam quality, the TE build.
- Test plan for the big-GPU machine (the user will run it later; >= 24 GB VRAM, Linux or Windows + WSL2 Ubuntu):
  1. Copy/clone the repo; copy `engine/models/diffusion-renderer` (28 GB) to skip the download, verify sizes
     on the device (model.pt = 28,940,280,602 bytes).
  2. `bash engine/setup_engine.sh` (WSL2: from the repo folder on /mnt/<drive>). Watch the transformer-engine
     build; if it fails, try `pip install --no-build-isolation` or a TE wheel matching torch 2.6 / CUDA 12.4.
  3. `doctor --model diffusion-renderer` -> runtime OK with transformer_engine version and the GPU.
  4. `infer work/kitty --model diffusion-renderer --max-frames 57` (one chunk): note load time, s/chunk,
     peak VRAM (worker RESULT), the normal axes check line; flip AXES if it warns. Look at basecolor vs the
     plate (sRGB vs linear: linear would look dark/contrasty).
  5. `--max-frames 120` (3 chunks): look at seams around frames 49-56 and 98-105; worker.log depth-fit lines.
  6. Kill a run mid-chunk and rerun: no "no saved chunk tail" note expected.
  7. `pack` + UE import: metallic in ORM B, albedo from the model; compare lit result with the quality set.
  8. Write measured numbers here and in presets/adapter docstrings; drop the "NOT run on a real GPU" note.

## Repo hygiene
- Never commit engine/ runtime, venvs, weights, extracted frames or outputs.
- `.gitattributes` keeps *.sh LF (bash in WSL2 reads the Windows checkout).
- Paths and model choice are always CLI options; no hard-coded absolute paths.
