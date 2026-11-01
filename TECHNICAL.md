# PlateCraft — technical description and production guide

This document explains what PlateCraft does to your footage, step by step, and why. Part A
covers the processing pipeline (extraction, keying, alpha handling, model inference,
post-processing, atlas packing). Part B covers the Unreal Engine 5.8 side (Img Media, the
master material, the card mesh, camera, Sequencer, Movie Render Queue). Part C is practical
advice for filmmakers: shooting, lighting, keying and delivery in DaVinci Resolve Studio, and
using the result in Sequencer.

All numbers below come from the code (`platecraft/`, `unreal/Content/Python/`) or from
measurements on the test clips (kitty, kitty2, kitty3) on an RTX 3060 12 GB.

---

## Overview

```
 camera / Resolve export
        │  ProRes 4444 with alpha (preferred), or video + matte, or green/blue/black screen
        ▼
 extract ──► src/  alpha/  [fg/]  plate/  manifest.json
        │          (straight colour)   (subject over neutral grey = model input)
        ▼
 infer   ──► raw/<model>/            model output at inference resolution (resume cache)
        │    normal/ depth/          final passes at source resolution (guided upscale)
        │    albedo/ ao/ roughness/  derived maps (heuristic)
        ▼
 pack    ──► atlas/frame_%06d.exr    2×2 atlas per frame, all linear, alpha in every quadrant
        ▼
 Unreal  ──► Img Media Source → Media Plate (card) + M_PlateCraft material
             + CineCamera + Level Sequence (media track, camera cuts) → Movie Render Queue
```

`platecraft.cmd process <input>` runs extract + infer + pack in one go; each step can also
be run alone. Every step is resumable: an interrupted run continues where it stopped, and a
rerun with unchanged settings does nothing.

---

# Part A — Processing pipeline

## A1. Decoding and frame accuracy (`ffmpeg_io.py`)

- **One PNG per decoded frame.** ffmpeg runs with `-fps_mode passthrough`, so no frame is
  dropped or duplicated, even for odd frame rates. Frame `i` of the clip is always
  `frame_{i+1:06d}.png`.
- **Exact frame rate.** The rate is stored as a rational (`24000/1001`, not `23.976`) and
  carried through to the manifest, the Img Media Source and the Level Sequence. Variable frame
  rate input (phones, screen recordings) triggers a warning: frames are kept 1:1, but timing
  uses the nominal rate. Convert VFR footage to constant frame rate first.
- **Frame-exact trimming.** `--start/--end/--max-frames` use ffmpeg's `trim=start_frame=…`
  rather than seeking by time, so the first extracted frame is exactly the frame you asked for.
- **Correct YUV→RGB.** ffmpeg on its own assumes BT.601 for untagged video, which shifts
  colours (greens and skin tones) on HD material. PlateCraft reads the stream's colour
  matrix and range and converts explicitly:
  `scale=in_color_matrix=bt709:in_range=tv:out_range=pc:flags=accurate_rnd+full_chroma_int`.
  Untagged footage ≥ 720 lines is treated as BT.709, smaller as BT.601. `full_chroma_int`
  interpolates 4:2:0/4:2:2 chroma properly instead of duplicating it.
- **Bit depth.** Sources above 8 bits (ProRes, 10-bit HEVC, DNx) are extracted as 16-bit PNG;
  8-bit sources as 8-bit. Data maps written later are always 16-bit.
- **Colour encoding.** Extracted frames are display-referred, sRGB/Rec.709 encoded. HDR
  (PQ/HLG) or log transfer functions trigger a warning: grade/convert to Rec.709 before
  extracting. EXR image sequences are assumed to be linear Rec.709 and converted to sRGB.

## A2. Where the alpha comes from

`extract` picks one alpha source, in this order:

| Mode | Option | How |
|---|---|---|
| matte | `--matte file_or_folder` | separate matte video/sequence, white = foreground (luma, or its alpha channel if it has one); resized to the plate if needed |
| chroma key | `--key green\|blue\|#RRGGBB` | ffmpeg `chromakey` (similarity 0.10, blend 0.08); the key colour is sampled from the first frame when not given. **Fallback only** — a keyer in Resolve/Nuke/AE gives far cleaner edges |
| black luma key | `--key black` | custom key for subjects shot or composited over pure black (see A4) |
| embedded | (auto) | the alpha channel of ProRes 4444 / RGBA sequences |
| none | | no alpha: the models see the full frame |

`--garbage x0:y0:x1:y1` forces a rectangle transparent, e.g. a floor reflection or a light
stand. Coordinates are source pixels, end exclusive (commas also work as separators). Repeat
the option for several rectangles:
`--key black --garbage 0:1064:1920:1080 --garbage 1700:0:1920:200`. It requires `--key` and
works with both the black key and the green/blue chroma key (for the chroma key, the
rectangles are cleared from ffmpeg's alpha before the plate is built). The rectangles are
stored in the manifest (`alpha.key.garbage`).

### Alpha normalisation

ffmpeg widens N-bit alpha to 16 bits by bit-shifting, so "fully opaque" lands at
`(2^N − 1) << (16 − N)`: 65280 for 8-bit, 65520 for 12-bit, instead of 65535. Left alone,
opaque pixels would have alpha 0.99976 and a material clip or blend would let a tiny bit of
background through. `frames.normalize_alpha` rescales every alpha frame through a 65536-entry
lookup table so that opaque is exactly 1.0 and transparent exactly 0.0.

## A3. Premultiplied vs straight alpha (`--premultiplied auto|yes|no`)

### Why it matters

A pixel at a soft edge (hair, motion blur, defocus) is partly subject and partly background.
There are two ways to store it:

- **Straight (unassociated):** RGB = the subject's own colour; alpha says how much of it is
  there. Compositing: `out = rgb·α + bg·(1 − α)`.
- **Premultiplied (associated):** RGB is already multiplied by alpha (as if composited over
  black). Compositing: `out = rgb + bg·(1 − α)`.

If premultiplied footage is treated as straight, every soft edge is multiplied by alpha a
second time: hair and motion-blurred edges go dark, and you get a black fringe around the
subject in Unreal. If straight footage is treated as premultiplied, edges glow. Resolve's
ProRes 4444 export with an alpha keyer was premultiplied in our tests (kitty2); adding a Fusion
**Alpha Divide** after the keyer gave straight colour (kitty3).

PlateCraft works internally with **straight** colour everywhere (the plate composite, the
albedo, the atlas and the UE material all assume it), so premultiplied input is converted.

### Automatic detection

For embedded alpha or a `--matte`, `extract` samples 6 frames spread over the clip and
gathers evidence from their soft-edge pixels (values as encoded, 0..1):

1. **"Brighter than alpha" test.** For premultiplied colour, `max(R,G,B) ≤ α` must hold
   (the colour was multiplied by α). In the band `0.05 < α < 0.9`, count the pixels with
   `max(rgb) > 1.15·α + 0.03` (the margin absorbs codec noise). Premultiplied footage has
   almost none (kitty2: 0.0 %); straight footage has many (kitty3: 43–52 %).
2. **"Brightness tracks alpha" test.** A dark straight subject would also pass test 1, so a
   second test compares each edge pixel's luminance with the luminance of the nearby
   **solid** subject (pixels with α > 0.98, averaged with a normalised Gaussian of
   σ = height/70): `ratio = lum / lum_solid`. For premultiplied colour the ratio follows α;
   for straight colour it stays near 1 (somewhat below, since fur tips are darker). Over
   `0.15 < α < 0.85` it computes
   `d_pre = median|ratio − α|` and `d_str = median|ratio − 1|`.

Verdict: **premultiplied iff fewer than 5 % of edge pixels are brighter than their alpha AND
`d_pre < d_str`.** With fewer than 500 soft-edge pixels there isn't enough evidence and the
footage is assumed straight. On kitty2: `d_pre` 0.11 vs `d_str` 0.56 → premultiplied.
`--premultiplied yes|no` overrides the detection; the verdict and reason are printed and
stored in `manifest.json` (`alpha.premultiplied`).

### Un-premultiplying

When premultiplied, `fg/` is written as `fg = clip(src / α, 0, 1)` (0 where α < 1/1024),
at the source bit depth. `src/` stays untouched. The division happens on the **encoded**
(sRGB/video) values, because that is where Resolve applies the premultiplication.
Verified: kitty2's `fg/` matches kitty3's straight `src/` within 0.001–0.004.

Note that un-premultiplying amplifies noise where α is very small (a value of 2/1023 at
α = 0.01 becomes 0.2). That is harmless: those pixels are almost invisible, and the
edge-colour fix (A8) replaces their colour anyway.

## A4. The black luma key (`--key black`, `frames.luma_key_black`)

For subjects shot against black velvet or already composited over black (the kitty test
clip). Over black, an edge pixel is `fur_colour × α`, so alpha can be recovered from
brightness:

1. **Silhouette:** `v = max(R,G,B) > 0.10`, closed with a 7 px ellipse, garbage rectangles
   removed, connected components smaller than 0.2 % of the frame dropped (specks), enclosed
   holes flood-filled (dark fur and shadows inside the body stay solid).
2. **Core:** the silhouette eroded by `edge` px (default `height/35`, 31 px at 1080p) gets
   α = 1 and its original colour.
3. **Edge band:** the local fur colour `fg_est` is a normalised Gaussian average of the core
   (σ = 0.4·edge), with a wider quarter-resolution estimate for pixels far from the core.
   Then `α = clip(v / max(fg_est), 0, 1)^1.3`, limited to the silhouette dilated by ⅔·edge.
   The exponent 1.3 slightly tightens the falloff so the faint dark halo does not become a
   visible grey veil.
4. **Colour:** core pixels keep their colour; edge pixels with α ≥ 0.6 are un-premultiplied
   (`rgb/α`); fainter ones take `fg_est` (dividing a nearly black pixel by a tiny α would just
   amplify noise).

The result is `alpha/` (true 16-bit) and `fg/` (straight colour). A dark compositing fringe
becomes soft translucency instead of a dark outline.

## A5. The model plate (`plate/`)

Depth and normal models were trained on ordinary photographs. A subject on pure green, black
or transparent pixels confuses them (green spill reads as surface colour; black reads as a
deep hole; a hard alpha edge gets "explained" as a depth step). So the model input is:

```
plate = despill(straight_colour) · α + grey · (1 − α)        grey = 0.5 (--plate-grey)
```

- **Despill** (`--despill auto|green|blue|none`, `--despill-amount`): "average" despill,
  the screen channel is limited to the mean of the other two
  (`G' = G − amount·max(G − (R+B)/2, 0)`). With `auto`, despill runs only for the ffmpeg
  chroma key. **With embedded alpha or a matte, `auto` means no despill** — the footage is
  expected to be despilled already by the keyer (do it in Resolve, see C3).
- Model padding (NormalCrafter pads to multiples of 64, DiffusionRenderer to 1280×704) uses
  the same grey, so the padding blends into the background instead of forming a white or
  black border.
- The plate is only the model's input and the guide for upscaling. The **albedo** is made
  from the straight colour (`fg/` or `src/`), not from the plate; if only the plate exists,
  `postprocess.straight_color` inverts the composite: `fg = (plate − grey·(1−α)) / α`.

## A6. Model inference (`infer.py`, `models/`)

### Resolution and presets

Inference runs at reduced resolution (models are slow and VRAM-bound) and is upscaled back to
source resolution with the plate as guide (A7). `--preset auto` picks by detected VRAM:

| Preset | min VRAM | max long side | NormalCrafter offload | window / overlap / chunk |
|---|---|---|---|---|
| low | 0 | 512 | sequential | 14 / 4 / 48 |
| 3060 | 10 GB | 768 | model | 14 / 4 / 70 |
| mid | 16 GB | 1024 | model | 14 / 4 / 70 |
| high | 22 GB | 1024 | none | 14 / 4 / 70 |

Inference size keeps the aspect ratio, long side ≤ max_res, both sides multiples of 8
(1920×1080 → 768×432). Measured on the 3060 at 768×432: NormalCrafter 0.96 fps, 7.8 GB
reserved; at 1024×576 0.42 fps, 11.9 GB (too close to the limit).

### Model sets

- `quality` (default): **NormalCrafter** (video normals) + **Video Depth Anything Small**.
- `safe` (commercial use): **VDA Small** + **Marigold Normals v1-1** (per frame) + **RAFT
  temporal filter**; depth-derived normals as fallback.
- `diffusion-renderer`: NVIDIA Cosmos DiffusionRenderer (albedo, normal, depth, roughness,
  metallic) — needs ≥ 20 GB VRAM and Linux/WSL2, **not yet tested on real hardware**.

Each model runs as a subprocess worker in its own Python environment and prints
`PROGRESS:`/`RESULT:` lines. Licences are declared per model, printed as warnings for
restricted ones, refused with `--commercial`, and written to `manifest.json` (`licenses`
lists exactly the models behind the delivered passes).

### NormalCrafter (temporally consistent normals)

A video diffusion model (built on Stable Video Diffusion) that predicts camera-space normals
in a single step. Upstream processes a clip in windows of 14 frames overlapping by 4, seeding
each window's overlap with the previous window's latents and blending them linearly.

PlateCraft drives that loop itself (`_infer_chunk`) so long clips can be processed in
**chunks of 70 frames** without visible seams: every chunk's first window is conditioned on
the cached latents of the previous chunk's last 4 frames (`raw/normalcrafter/latents/`, one
small `.npy` per frame), with the same linear merge weights the model uses between its own
windows (`linspace(1, 0, m+2)[1:-1]`). A chunk boundary or a resumed run therefore behaves
exactly like an internal window seam. (An unconditioned restart differed by 2–7°, and a pixel
crossfade did not hide it.) The overlap frames are rewritten with the merged result.

Output axes (measured by correlating edge normals with the silhouette's outward direction):
**+X right, +Y up, +Z toward the camera** (OpenGL convention).

### Video Depth Anything (relative depth)

The adapter streams upstream's `infer_video_depth`: 32-frame windows advancing by 22.
Each new window re-feeds key frames: position 0 is always the **run's first frame** (an
anchor that pins the depth scale for the whole clip), position 1 is frame 12 of the previous
window, positions 2–9 are the previous window's last 8 frames. The new window's depth is
**scale/shift-aligned** to the previous one by least squares over those key frames, then
crossfaded over the 8 overlapping frames. Frames are written as soon as no later window can
change them; the 8 pending frames and the reference depth are kept in `vda_state.npz`, so a
crash and resume is bit-identical to an uninterrupted run (and matches upstream within
0.06 %). Output: float EXR disparity (larger = nearer) at 924×518 for 16:9 input. 7.7 fps,
3.0 GB on the 3060.

### Marigold Normals (per-frame, commercially usable)

diffusers' `MarigoldNormalsPipeline`, fp16, 4 denoising steps, ensemble size 1. **Every frame
starts from the same noise tensor**, which keeps consecutive results as similar as a
per-frame model allows. Same axes as NormalCrafter. More fine detail than NormalCrafter, but
it flickers, hence the temporal filter. 1.42 fps, 6.8 GB.

### RAFT optical-flow temporal filter (`--temporal-filter`, `--temporal-strength 0.8`)

On by default (`auto`) for per-frame normal models. A recursive filter:

```
out[t] = normalize( w · warp(out[t−1], flow_t→t−1) + (1 − w) · raw[t] )
w      = strength · reliable
```

- Flow is computed by RAFT-large (torchvision weights, BSD-3) on the **plate**, in both
  directions.
- `reliable` = forward/backward consistency (following the backward flow and then the
  forward flow must return within `1 px + 5 % of the flow length`) AND α > 0.5 in both
  frames; the mask is softened with a 5×5 box. Where the flow is unreliable
  (disocclusions, edges) the fresh frame is used.
- Detail that moves with the footage is carried along by the warp; detail that changes
  randomly from frame to frame is averaged away. Higher strength = steadier but softer.
  Flow-compensated change per frame on kitty: Marigold raw 0.077 → 0.036 (0.7), 0.029
  (0.8), 0.022 (0.9); NormalCrafter for comparison 0.025.

### Depth-derived normals (`depth-normals`)

The fallback when no normal model is installed. The clip-normalised disparity is blurred
(σ = 1.5 px) and read as a height field whose full range spans 0.3 × the frame width;
normals = `normalize(−∂z/∂x, ∂z/∂y, 1)` (Sobel; image y points down, hence the sign).
Smooth and as stable as the depth, but without fine detail; marked heuristic.

### Cosmos DiffusionRenderer

Inverse renderer predicting basecolor, normal, depth, roughness and metallic. Input is fixed
at 1280×704 (plate fitted inside with grey padding), 57-frame chunks overlapping by 8,
15 steps, seed 1000. Depth per chunk is fitted (scale/shift, closed form in float64) to the
previous chunk over the overlap, then crossfaded. Status: implemented and verified against a
stand-in, untested on a real GPU; see CLAUDE.md for the test plan.

## A7. From raw output to final passes (`postprocess.py`)

### Guided upsampling

Bicubic upscaling of a 768 px normal map to 1920 or 3840 px gives soft, misplaced
silhouettes. Instead PlateCraft uses a **fast guided filter** (He & Sun 2015) with the
full-resolution plate as colour guide:

1. Downscale the plate to inference size: `I_lo`. Model output: `p`.
2. In every (2r+1)² window at low resolution, fit `p ≈ aᵀ·I + b` (a is a 3-vector per output
   channel):
   `a = (Σ_I + ε·Id)⁻¹ · cov(I, p)`, `b = mean(p) − aᵀ·mean(I)`.
   ε regularises the fit: guide contrast below √ε is not transferred (ε = 2·10⁻³ for normals,
   10⁻³ for the others).
3. Average a and b over the window, upsample them bilinearly to full resolution, and apply
   them to the **full-resolution** plate: `q = aᵀ·I_hi + b`.

Because the fit is linear in the guide, the output's edges land exactly where the footage
has them, at full resolution. `r = max(1, round(2·max(w,h)/768))`. The 3×3 matrix inverse is
done in closed form by cofactors (np.linalg on ~300k tiny matrices took 6 s/frame; now
~0.5 s). `--upscale bicubic` turns guidance off.

### Normals

`n = normalize(guided_upsample(raw·2 − 1))`; pixels with α = 0 are set to flat
`(0, 0, 1)`; written as 16-bit PNG `n·0.5 + 0.5`.

**Why camera space is already the right tangent space:** the footage is shown on a card that
faces the camera, with U running right and V running down the image. For such a card the
tangent frame coincides with the camera frame, so converting camera-space normals to tangent
space is only an axis convention: `--normal-space directx` (default, UE) flips green
(G = −Y), `opengl` keeps it. The material has a `FlipGreen` parameter, set automatically from
the manifest.

### Depth

Raw disparity has no absolute scale. PlateCraft normalises it **clip-wide**, never per frame
(per-frame normalisation would pump): the 0.5 and 99.5 percentiles of the subject's disparity
(α > 0.5) over up to 48 sampled frames define `lo..hi`;
`depth = clip((raw − lo)/(hi − lo), 0, 1)`, guided-upscaled, 0 where α = 0. Encoding:
**1 = nearest subject point, 0 = farthest** (relative inverse depth, not metric). The range is
stored in the manifest; a changed range rewrites the whole pass.

### Albedo (= keyed plate, optionally corrected)

PlateCraft does not estimate true albedo with the default models: the albedo is the
straight subject colour, i.e. **lighting from the set is baked in**. Two optional corrections:

- **Edge colour fix** (`--edge-color 1`, default on except for the black key, whose `fg/` is
  already fixed). Two defects observed on kitty3's soft edges:
  - *Brightness:* thin fur keyed off a dark background keeps some of that background
    (edges 10–25 % darker than the fur next to them) → a dark rim over a bright UE backdrop.
  - *Hue:* 4:2:0 sources smear the subject's chroma 1–2 px into the dark background, which
    decodes as saturated red/blue (R/G ratio 8 at α < 0.1 vs 1.5 in the fur).

  In linear light, each edge pixel gets the nearby solid colour's luminance as a gain
  (`gain = clip(lum_solid / lum, 1, 4)` — only brightens), very dark pixels move toward the
  solid luminance, and the chromaticity moves to the nearby solid colour. Both apply fully
  below α = 0.5 and fade to nothing at α = 0.98. "Nearby solid" = normalised Gaussian of the
  α ≥ 0.98 pixels (σ = h/100, with a wider σ = h/25 fallback).
- **De-lighting** (`--delight 0..1`, default 0). Divides the linear colour by its
  low-frequency luminance (masked Gaussian, σ = h/25) and restores the subject's mean:
  `gain = (mean / shade)^amount`, clamped to 0.25..4. It removes broad shading gradients
  (a strong key from one side), not shadows with hard edges or coloured light. Marked
  heuristic.

### Ambient occlusion (heuristic)

Computed at half resolution (it is low-frequency):

```
occ  = Σ_k gain_k · max(blur_k(depth) − depth, 0)       radii h/200, h/80, h/30; gains 6, 4, 2.5
     + max(−div(n_xy), 0) · h/108                        concavity of the normals
occ *= blur(erode(α > 0.5, h/25))                         fade in from the silhouette
ao   = blur(clip(1 − strength·occ, 0, 1), h/720)
```

The depth term finds pixels that lie behind their (masked) neighbourhood at three scales
(recesses between nearer surfaces); the normal term finds creases and folds. The silhouette
fade exists because at the outline depth falls off and normals turn toward the flat
background, which both terms would otherwise read as a crease around the whole subject.

### Roughness (heuristic)

`roughness = base + cavity·(1 − ao)` (defaults 0.7 and 0.15): a constant with slightly rougher
crevices. Metallic is 0 unless a model (DiffusionRenderer) provides it. In UE both are
adjustable per material instance.

### Resume and staleness rules

- `raw/<model>/settings.json` records everything that affects the raw output; frames are only
  reused with identical settings (a mismatch asks for `--force`).
- A final-pass frame is skipped when it is **newer than every input frame** (raw output,
  plate, alpha, fg) and the pass's manifest entry is unchanged.
- Frames are written to `<name>.tmp` and renamed, so a crash cannot leave a truncated frame
  that looks up to date.

## A8. The atlas (`pack.py`)

Unreal's Img Media plays **one** image sequence per player. To keep all passes frame-locked
with no chance of drift, they are packed into a 2×2 atlas per frame:

```
+---------------------------+---------------------------+
| albedo RGB (linear)       | normal RGB, n·0.5+0.5     |
|                           | tangent space, DirectX    |
+---------------------------+---------------------------+
| R = AO, G = roughness,    | depth (R = G = B)         |
| B = metallic              | 1 = nearest               |
+---------------------------+---------------------------+
  A = alpha (straight) in every quadrant
```

- **Everything is linear.** Albedo is converted from sRGB here, so the material needs no
  per-region colour handling and the media texture can be sampled as plain data.
- **Colour bleed under transparent pixels (push-pull).** Straight-alpha colour is black (or
  garbage) where α = 0. Bilinear filtering and mip-mapping at a soft edge would mix that black
  into the visible edge → dark fringe. `bleed_color` builds a pyramid of
  `(colour·known, known)` down to 1×1, then pulls it back up, filling each unknown pixel with
  the weighted average of the nearest known colour. Visible pixels are unchanged.
- **Depth bleed.** Below α = 0.5 the model's depth is background or a mix. It is replaced by
  nearby subject depth the same way, so a depth-bent card does not fold backwards along the
  silhouette.
- **Missing passes become constants** (normal flat, AO 1, roughness 0.7, metallic 0,
  depth 0) and are recorded as such in `manifest.json["atlas"]["regions"]`.
- **Format:** half-float EXR, ZIP compression (default; ~4.5 MB per 3840×2160 frame,
  0.34 s/frame with the OpenEXR library on 8 threads) or 16-bit PNG. Never 8-bit: UE 5.8 Img
  Media keeps EXR as `PF_FloatRGBA` and 16-bit RGBA PNG as `PF_A16B16G16R16`, but 8-bit PNG as
  BGRA8. Max error vs the passes: 2.4·10⁻⁴ (half precision).
- **`--scale 0.5`** halves the quadrant size (1920×1080 atlas for 1080p footage) for
  real-time preview in the editor (see B6).
- **Resume:** frames newer than all their sources are skipped; changed settings or sources
  clear the folder so frames from different runs never mix in one sequence.

---

# Part B — Unreal Engine 5.8 side (`platecraft_import.py`)

Run from **Tools > PlateCraft > Import Clip...** (dialog with folder picker; options are
remembered in `<project>/Saved/PlateCraft/import_dialog.json`) or from the Python console:
`py "<repo>/unreal/Content/Python/platecraft_import.py" <clip> --height 180`.
Only built-in editor scripting is used; no plugins beyond Python Editor Script Plugin and
Editor Scripting Utilities.

## B1. Assets created

| Asset | Shared / per clip | Purpose |
|---|---|---|
| `/Game/PlateCraft/M_PlateCraft` | shared | master material (metadata tag `PlateCraftVersion`; rebuilt when the version changes) |
| `/Game/PlateCraft/SM_PlateCraftCard_CxR` | shared | subdivided card mesh for depth bending |
| `IMS_<clip>` | per clip | Img Media Source on `<clip>/atlas`, exact frame rate |
| `MI_<clip>` | per clip | material instance — **tweak this one** |
| `LS_<clip>` | per clip | Level Sequence with media track + camera cuts |
| `PlateCraft_<clip>` actor | per clip | Media Plate in the level (updated in place on re-import) |
| `PlateCraftCam_<clip>` actor | per clip | CineCamera framing the plate 1:1 |

Re-importing the same clip updates everything in place; an existing camera keeps its
transform and lens.

## B2. Img Media Source

The source points at the first atlas frame, with `frame_rate_override` set to the exact
rational from the manifest (e.g. 24000/1001) and `fill_gaps_in_sequence` on. UE 5.8 reads
EXR as `PF_FloatRGBA` (half float), 16-bit PNG as linear 16-bit. The EXR reader ignores
in-file mips; the GPU EXR path (needs uncompressed files and a special format) is not used.

## B3. Master material `M_PlateCraft`

Blend mode **Masked**, Default Lit, **two-sided**, opacity mask clip 1/3.

**Atlas sampling.** The Media Plate binds its media texture to any parameter named
`MediaTexture`. Each pass is sampled at `clamp(UV, 1/size, 1 − 1/size)·0.5 + offset`
(offset (0,0), (0.5,0), (0,0.5), (0.5,0.5)): the clamp keeps bilinear filtering half an atlas
texel inside its quadrant so it never reads the neighbouring pass.

**Base colour** = albedo · `AlbedoTint` · `AlbedoBrightness`.

**Alpha levels.** `α' = saturate((α − AlphaBlackPoint) / (AlphaWhitePoint − AlphaBlackPoint))`.
Raising the black point chokes a soft or blurred matte edge without re-processing.

**Soft edges — the alpha dither.** A masked material can only draw a pixel or not. To get
soft edges, the mask is dithered and the renderer's anti-aliasing averages the samples:

```hlsl
g = frac(52.9829189 * frac(dot(SvPosition.xy, float2(0.06711056, 0.00583715))))  // interleaved gradient noise
h = hash(pixel)                                                                   // per-pixel scramble
k = (StateFrameIndexMod8 XOR h) & 7
n = frac(g + k/8) * 0.998 + 0.001       // threshold in (0, 1)
mask = Clip + α' − n                    // drawn iff α' > n
```

Over 8 consecutive frame indices, `k` runs through all values 0..7 in every pixel, so the
threshold takes 8 evenly spaced values; the fraction of samples that draw the pixel is α'
to within 1/8. Design notes, all found by measuring renders:

- The engine's own `DitherTemporalAA` spreads its thresholds only over α 1/6..5/6
  (`((x + 2y + frame) mod 5)/6` plus noise): α 0.6–0.7 drew 93 % of samples, turning soft fur
  into a solid dark rim.
- Independent random noise per sample left visible grain (±0.18 after 8 samples) — hence
  stratified thresholds.
- Using the same k order in every pixel correlated with MRQ's sub-pixel jitter and drew
  horizontal/vertical line patterns — hence the per-pixel XOR scramble.
- The [0.001, 0.999] range guarantees α = 0 is never drawn and α = 1 always.

Static switch `SoftEdges` off = hard cut at α 0.5 (cheaper, for previews or opaque subjects).

**Normal:** `n = rgb·2 − 1`, green flipped if `FlipGreen = 1`, `lerp((0,0,1), n, NormalStrength)`,
normalised.

**ORM:** AO = `lerp(1, R, AOStrength)`; roughness = `saturate(G·RoughnessScale + RoughnessOffset)`;
metallic = `saturate(B + MetallicOffset)`; `Specular` constant 0.5.

**Card bend (World Position Offset):** `WPO = VertexNormalWS · (depth − DepthPivot) · DepthAmount`.
`DepthAmount` (cm, default 0 = flat) is the total bend between the farthest and nearest
subject depth; `DepthPivot` (0.5) is the depth that stays on the card plane. The depth is
sampled at mip 0 in the vertex shader. Since depth is relative, choose the amount by eye
(a person: 10–30 cm). The bend gives some parallax and better self-shadowing for small
camera moves and lights at an angle; it is not a 3D reconstruction.

## B4. Card mesh and Media Plate

`SM_PlateCraftCard_CxR` is a grid (default 128 columns, rows from the aspect ratio) copied
from the Media Plate's own screen mesh: YZ plane, front facing −X, U along +Y, V down,
100×100 units, **pivot at the bottom centre** (so `--location` is where the subject stands).
It is built with full normals/tangents (UE's `fast_build` skips them and leaves NaN bounds).

The Media Plate actor `PlateCraft_<clip>` gets:

- the card mesh, scaled to `--height` cm (height of the full video frame, not of the subject)
  and the tile's aspect ratio;
- `MI_<clip>` as override material — the plate wraps it in a dynamic instance and binds its
  media texture (the "no MediaTexture parameters" warning in the log is expected);
- **dynamic shadow casting on** (Media Plate turns it off by default), shadow from the alpha
  silhouette;
- **Shadow Cache Invalidation Behavior = Always**: Virtual Shadow Maps only invalidate on
  transform/WPO changes, not when the video's alpha changes, so cached shadows would show old
  frames;
- **not spatially loaded** (World Partition): a spatially loaded plate is streamed into
  PIE/MRQ from its saved package and renders the material's default texture (grey/black
  plate);
- autoplay + loop only when no Level Sequence is made (otherwise the sequence drives it).

## B5. Camera `PlateCraftCam_<clip>`

A CineCamera that frames the plate's video frame exactly, so a render maps 1:1 onto the
footage:

```
sensor_w = 23.76 mm (UE default)      sensor_h = 23.76 · tile_h / tile_w
focal    = 35 mm                      distance = height · focal / sensor_h
position = plate pivot − distance along the plate's facing direction, + height/2 up
```

f/8 with manual focus on the plate (deep enough that a bent card stays sharp). Checked: the
plate edges land at NDC ±1.0000. It also gets a **World Partition streaming source** (radius
`--streaming-radius`, default 3000 m): PIE/MRQ only stream cells within the player's loading
range (256 m in the Open World template) and MRQ renders with `r.HLOD 0`, so distant terrain
visible in the editor would otherwise be missing from renders.

## B6. Level Sequence `LS_<clip>`

- Display rate = clip rate; playback range = frame count.
- The plate is **possessed** (not spawned) with a **Media track** whose section plays
  `IMS_<clip>` from frame 0, not looping. The media track drives the plate's texture.
- **`synchronous_scrubbing` on:** the media track only waits for decoded frames while
  playing; when stopped/scrubbing (which is how MRQ evaluates the first frame of every shot)
  it would not wait and the first frame renders blank. With it on, every evaluated frame
  blocks until the exact atlas frame is loaded.
- The camera is possessed and a **Camera Cuts** track added if the sequence has none (your own
  cuts are kept). Stale bindings (sequence made in another level) are rebound.

**Playback speed.** The media track waits for every frame. On the 3060, a 3840×2160 half EXR
atlas takes ~60 ms to decode → 3–6 fps in the editor; a 1920×1080 atlas (`pack --scale 0.5`)
~15 ms → real time (79 fps). MRQ renders are frame-accurate either way (verified by matching
each render against atlas frames −4..+4).

## B7. Level hygiene the importer handles or warns about

- **Light Shaft Occlusion** on directional lights is screen-space and depth based: a plate in
  front of the sky darkens wedges of the scene behind it, spreading away from the sun.
  `--no-light-shafts` / dialog option turns it off on every directional light.
- **Open World template levels** saved without reopening keep actor descriptors pointing at
  `/Temp/Untitled_N`; PIE/MRQ skip those actors (black sky, missing landscape). The importer
  warns; fix by saving and reopening the level.

---

# Part C — Practical guide for filmmakers

## C1. What PlateCraft can and cannot do

**It gives you:** a keyed live-action subject on a card in UE that reacts to virtual lights
through its normals, casts a shadow, gets ambient occlusion in creases, can be slightly bent
by depth for small parallax, and plays frame-accurately in Sequencer and Movie Render Queue.

**Limits to plan around:**

- **Set lighting stays in the picture.** The albedo is your footage. UE lighting is applied
  *on top* via normals. Light on set so that the baked lighting agrees with the virtual
  lighting (C2), or use `--delight` for broad gradients.
- **It is a 2.5D card.** The camera can move, but the further it moves from the original
  shooting angle, the more the flatness shows. Comfortable: ±15–20° around the card normal and
  modest dolly moves. `DepthAmount` helps a little.
- **Roughness, AO and metallic are heuristics** (except with DiffusionRenderer). Adjust them in
  the material instance by eye.
- **Shadows on the set floor are lost** with the key; the subject's shadow in UE comes from its
  silhouette on the card. It looks right for lights roughly from the front/side; it gets thin
  as a light approaches the card's plane.
- **Depth is relative**, not metric.

## C2. Shooting and lighting

### Plan the light in the virtual scene first

In virtual production the backgrounds exist before the shoot. Use that:

1. Block the shot in UE first: camera position, height, focal length, the sun/key direction
   and colour of the environment.
2. Recreate on set: same **camera height, tilt, focal length (on the same sensor size) and
   subject distance** as the UE camera. Write them down; you'll enter them on the UE camera
   later (the auto-created camera is a 35 mm starting point).
3. Put your key light on set at the **same direction relative to the camera** as the UE key
   light, with a matching colour temperature. The baked light then agrees with the virtual
   light, and the normals only need to add the variation as the subject moves.

### Subject lighting

- **Soft and fairly flat:** large soft key, generous fill, contrast ratio around 2:1 (key:fill).
  Strong modelling gets doubled — it's in the albedo *and* UE adds its own shading.
  Hard-edged cast shadows on the subject cannot be removed later.
- **Neutral colour:** white light, no coloured gels on the subject unless the virtual scene
  has the same coloured light from the same direction.
- **Rim/back light: subtle.** A rim helps the key and hair separation but is baked in; a strong
  rim looks wrong when the virtual light comes from the front. A slight warm or magenta tint
  in a soft back light helps neutralise green spill on hair.
- **Exposure:** protect skin and highlight detail; nothing clipped on the subject.

### Screen and set

- **Distance:** subject at least 2–3 m in front of the screen. Less spill, no shadows on the
  screen, and the screen drops slightly out of focus (smoother key).
- **Even screen:** light the screen separately and evenly (within about ⅓ stop across the
  frame; check the waveform — the green channel should be a flat line). Expose it around the
  subject's key level or a little below; a hot screen spills more.
- **Green vs blue:** green for most work (more luminance, cleaner key on 4:2:x cameras); blue
  for green wardrobe/props or blond hair where green spill shows.
- **Wardrobe and props:** nothing in the screen colour, avoid glossy/reflective surfaces
  (they mirror the screen), avoid very fine high-contrast patterns (moiré, bad normals).
- **Floor:** if feet are in shot, use a screen-colour floor. Mark the subject's spot; you'll
  put the card's bottom centre at the matching spot in UE.

### Camera settings

- **Codec:** record at least 10-bit 4:2:2 (ProRes 422 HQ, BRAW, ProRes RAW, DNx). 8-bit 4:2:0
  smears chroma into the edges — exactly the fringe the edge-colour fix had to repair on the
  test clip. RAW or 4:4:4 is best.
- **Resolution:** 4K or more helps the key; models run at ~768 px and the guided upscale uses
  your full-resolution plate for edge detail. For UE, a 1080p atlas is usually enough on
  screen-size subjects; keep 4K for close-ups.
- **Shutter:** 180° for natural motion. For fast action, 90–120° gives crisper edges and a
  cleaner key (motion blur becomes partially transparent edges, which survive but are harder
  to key).
- **Stop:** f/4–f/5.6 keeps the subject sharp to the edges; add depth of field in UE.
- **Frame rate:** shoot at the frame rate of the final edit; PlateCraft and UE keep it exactly.
  Avoid variable frame rate (phones).
- **Camera:** a locked-off camera is the native case. A moving camera bakes its motion into the
  plate and needs a matched camera move in UE (matchmoving is outside PlateCraft).

## C3. Keying and delivery in DaVinci Resolve Studio

### Project setup

- Timeline frame rate = shooting frame rate.
- Colour management (e.g. DaVinci YRGB Color Managed) with **output colour space Rec.709**.
  PlateCraft expects display-referred SDR; log or HDR files only get a warning and their
  colours come out wrong. PlateCraft linearises with the **sRGB curve**; for the most exact match choose an sRGB
  output gamma, or accept that Rec.709 Gamma 2.4 renders shadows in UE a little lighter than
  in the Resolve viewer.
- Do the primary grade (white balance, exposure, matching between takes) **before** keying
  and export. Keep it neutral: no creative look, no vignette, no grain, no sharpening — those
  are applied in the final grade after UE (C5).
- Trim to the shot with only a few frames of handles. Every frame is inferred (NormalCrafter
  ~1 fps on a 3060).

### Keying (Fusion page, recommended)

```
MediaIn ─► Delta Keyer ─► (Alpha Divide) ─► MediaOut
```

- **Delta Keyer:** pick the screen colour on a representative screen area; feed a clean plate
  if the screen is uneven; adjust *Matte > Threshold* low/high to get solid core and fully
  transparent background; **keep the soft edges** — don't erode or blur them away. Hair and
  motion blur are what make the result look real, and the UE material reproduces them
  (alpha-true dither). You can always choke the edge later in UE with `AlphaBlackPoint`.
- **Despill in the keyer** (Delta Keyer's spill suppression / Replace mode). PlateCraft does
  *not* despill footage with an embedded alpha.
- **Garbage matte:** polygon mask into the keyer's garbage input for rigs, stands and the
  screen edges.
- **Alpha Divide (optional):** gives straight colour. Without it, Resolve's output was
  premultiplied in our tests; PlateCraft detects that automatically (`--premultiplied auto`),
  so either works. If detection reports something odd (e.g. "too few soft-edge pixels"),
  force it with `--premultiplied yes|no`.

Alternatives: Color page 3D Keyer / qualifier with an alpha output; **Magic Mask** for
footage without a screen (Resolve 21 can render a Magic Mask as an external matte, which you
can pass with `--matte`). Magic Mask edges are less fine than a screen key — fine for a
person at medium distance, weak on hair.

### Export (Deliver page)

| Setting | Value |
|---|---|
| Format / codec | QuickTime, **Apple ProRes 4444** (or 4444 XQ) |
| Export Alpha | **on** |
| Resolution | source resolution |
| Frame rate | timeline = shooting rate |
| Colour tags | Rec.709 (or sRGB gamma, see above) |
| Data levels | Auto |

Alternatives: a 16-bit PNG/TIFF sequence with alpha (pass `--fps`), or an RGB export plus a
separate matte render (`--matte matte.mov`, white = foreground). Avoid H.264/H.265 for the
plate: 8-bit 4:2:0 and no alpha.

### Run PlateCraft

```bash
platecraft.cmd doctor
```

```bash
platecraft.cmd process D:/shoot/take03.mov
```

Useful options:

- `--start 4.5s --end 12s` (or frame numbers) to trim;
- `--model safe --commercial` for commercial projects that can't use NormalCrafter's licence
  (Stability AI Community License: free only under US$1M annual revenue);
- `--delight 0.5` if the set lighting has a strong gradient you want to reduce;
- `--edge-color 0` if the edges of a bright subject on a bright screen are already clean;
- `--format exr` (default) for finals; then `platecraft.cmd pack <clip> --scale 0.5` for
  a fast preview version (packing settings changed → the atlas folder is rebuilt; repack at
  full scale before the final render).

Check `work/<clip>/albedo`, `normal`, `alpha` in an image viewer before going to UE; a
problem in the key is cheaper to fix in Resolve than to hide in UE.

Timing on the 3060 for 421 frames of 1080p (kitty3): NormalCrafter 9 min, VDA 1 min, derived
maps ~5 min, packing 3 min.

## C4. In Unreal: placement, lighting and Sequencer

### Import and placement

1. Open the level, put the viewport camera roughly where the shot will be.
2. **Tools > PlateCraft > Import Clip...**, pick `work/<clip>`. By default the plate is
   placed 400 cm in front of the viewport camera, facing it.
3. Set **Height (cm)** to the real height of the full video frame at the subject's distance (e.g.
   a frame that shows a 180 cm person with some headroom ≈ 220 cm). Wrong scale is the most
   common reason a plate "doesn't sit" in the scene.
4. Move the plate so its **bottom centre** is where the subject's feet touch the ground. If
   the shot was framed with the feet out of frame, lower the plate until the ground line
   matches.
5. Match the camera: set `PlateCraftCam_<clip>` to the on-set height, tilt and focal length, aim it at
   the plate. Keep the plate facing the camera (yaw), within ±15–20°.

### Lighting the plate in UE

- Match the main light's direction and colour to the set key (C2). Use a light coming from the
  **camera side** of the card; lights behind the card (more than ~90° from its normal) light
  the back of a flat card and look wrong.
- Use `NormalStrength` (0.5–1) to set how strongly virtual lights shape the subject, and
  `AOStrength` for creases. If the subject looks double-lit, lower `AlbedoBrightness` slightly
  or reduce the virtual fill rather than pushing normals.
- `RoughnessScale/Offset`: skin ≈ 0.5–0.6, cloth 0.7–0.9; lower values give highlights that
  move with the virtual light.
- **Shadows:** the plate casts a dynamic shadow of the alpha silhouette. Best with the light
  30–60° to the side of the card normal. For grazing light the shadow thins out; a small
  `DepthAmount` (10–30 cm) gives it some volume. Bake-only (static) lights won't see the video:
  use movable lights for anything the plate should shadow.
- **Contact:** add a soft decal or a small blob of darkening under the feet if the shadow
  alone doesn't ground the subject.
- **Edges:** if the matte is too soft or has a halo, raise `AlphaBlackPoint` (0.05–0.15);
  if fine hair is lost, lower it or check the key in Resolve.

### Working in Sequencer

- `LS_<clip>` is ready to play: plate, media track (synchronous scrubbing), camera and camera
  cuts. Use it as a **shot** in your master sequence (Shots/Subsequences track) so each take
  stays self-contained and re-importable.
- **Preview vs final:** a 4K EXR atlas plays at 3–6 fps in the editor because the media track
  waits for every frame. For blocking and timing, pack with `--scale 0.5` (real time); repack
  at full resolution before rendering. The Sequencer timing is identical, only the image
  quality changes.
- **Slip/trim** a take by moving or trimming the media section; the section always shows
  source frames 1:1 at the clip's rate.
- **Animate** the camera (dolly, push-in, small arcs), lights (time-of-day, practical flicker),
  and the plate's transform for small repositioning. Keep camera moves within the angle limits
  above; a slow push-in or a slight lateral move with `DepthAmount` on reads very well.
- **Multiple subjects:** import each take as its own clip; each gets its own plate, media
  track and material instance. Place them at their own distances and heights.
- **Re-import** after reprocessing (new key, different model, repack) updates the source,
  material instance, plate and sequence in place; your camera transform and your own cuts are
  kept.
- **Delete `LS_<clip>` between test variations** if you've duplicated cameras or cut tracks: a
  sequence with several camera-cut tracks or stale bindings renders from an arbitrary one.

### Movie Render Queue

- **Anti-aliasing: Spatial Sample Count 8, Override Anti-Aliasing = None.** The soft-edge dither
  needs 8 samples per frame to average to the true alpha. Measured coverage per alpha band
  (0.4–0.5 / 0.5–0.6 / 0.7–0.8): 8 spp + AA None 0.43 / 0.53 / 0.75 (correct); 8 spp + TSR
  0.60 / 0.71 / 0.89 (edges too solid and dotted).
- Render frame rate = sequence rate (default).
- World Partition: if sky or landscape is black in the render but fine in the viewport, save
  and reopen the level; if distant terrain is missing, raise the camera's streaming radius.
- Output 16-bit PNG or EXR (if you'll grade heavily) for the final grade.

## C5. Finishing back in Resolve

- Import the MRQ image sequence at the project frame rate.
- Grade the whole composite together: the plate and the virtual environment now share one
  camera response, so a single look unifies them.
- Add the things you deliberately left out before export: film grain (Resolve's Film Look
  Creator or Film Grain over the whole frame, so plate and CG share it), halation/glow, lens
  effects, vignette.
- If a take needs a better key, fix it in the Fusion comp, re-export, rerun `process`
  (unchanged steps are skipped), re-import in UE and re-render: the sequence, camera and lights
  stay as they were.

---

## Appendix — file reference

| Folder in `work/<clip>/` | Content | Encoding |
|---|---|---|
| `src/` | decoded frames | sRGB, source bit depth |
| `alpha/` | matte, 1 = foreground | linear, 16-bit (normalised) |
| `fg/` | straight colour (black key or un-premultiplied input) | sRGB |
| `plate/` | model input: despilled subject over grey | sRGB |
| `raw/<model>/` | model output at inference size + settings.json, worker.log | model-specific |
| `normal/` | tangent-space normals | 16-bit, n·0.5+0.5 |
| `depth/` | relative depth, 1 = nearest | 16-bit linear |
| `albedo/` | RGBA, straight alpha | sRGB |
| `ao/`, `roughness/`, `metallic/` | material maps | 16-bit linear |
| `atlas/` | 2×2 packed frames for UE | half EXR / 16-bit PNG, linear |
| `manifest.json` | fps, frame count, resolution, passes, models, settings, licences, colour space per pass, atlas layout | |
