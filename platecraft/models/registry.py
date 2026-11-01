"""Known models, their environments, weights and licenses.

Adding a model = adding one ModelInfo here plus an adapter module
(platecraft/models/<adapter>.py, see base.py) that does the inference.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Commercial-use classes. Every model must declare one.
COMMERCIAL_OK = "ok"
COMMERCIAL_RESTRICTED = "restricted"
COMMERCIAL_NO = "non-commercial"


@dataclass(frozen=True)
class ModelInfo:
    key: str
    name: str
    passes: tuple[str, ...]
    env: str                      # engine/envs/<env>
    repo: str                     # engine/repos/<repo>
    license: str
    commercial: str               # one of the COMMERCIAL_* values
    license_note: str
    weights: tuple[str, ...]      # files/folders relative to engine/models that must exist
    probe: str                    # Python import check run inside the model env (repo on sys.path)
    safer_alternative: str = ""
    adapter: str = ""             # module in platecraft/models; empty = not implemented yet
    sets: tuple[str, ...] = field(default_factory=tuple)  # model sets this belongs to
    temporal: bool = True         # False = per-frame model: --temporal-filter auto smooths its output
    requires: tuple[str, ...] = ()  # passes whose raw output this model is computed from
    linux_only: bool = False      # env exists only on Linux; on Windows it runs through WSL2 (see paths.worker_launcher)
    min_vram_gb: float = 0        # refuse to start below this (PLATECRAFT_IGNORE_VRAM=1 overrides)

    @property
    def warning(self) -> str | None:
        if self.commercial == COMMERCIAL_OK:
            return None
        text = f"{self.name}: {self.license}. {self.license_note}"
        if self.safer_alternative:
            text += f" Commercial-safe alternative: {self.safer_alternative}."
        return text


MODELS: dict[str, ModelInfo] = {m.key: m for m in [
    ModelInfo(
        key="normalcrafter",
        name="NormalCrafter",
        passes=("normal",),
        env="normalcrafter",
        repo="normalcrafter",
        license="MIT code; weights built on Stable Video Diffusion (Stability AI Community License)",
        commercial=COMMERCIAL_RESTRICTED,
        license_note=("Commercial use is free only for organisations under US$1M annual revenue; "
                      "above that a Stability AI Enterprise licence is required."),
        weights=("normalcrafter/normalcrafter/unet", "normalcrafter/normalcrafter/vae",
                 "normalcrafter/svd-xt/model_index.json", "normalcrafter/svd-xt/image_encoder"),
        probe=("import diffusers\n"
               "from normalcrafter.normal_crafter_ppl import NormalCrafterPipeline\n"
               "from normalcrafter.unet import DiffusersUNetSpatioTemporalConditionModelNormalCrafter\n"
               "extra['diffusers'] = diffusers.__version__\n"),
        safer_alternative="the 'safe' set (per-frame normals + temporal filter, or depth-derived normals)",
        adapter="normalcrafter",
        sets=("quality",),
    ),
    ModelInfo(
        key="vda-s",
        name="Video Depth Anything Small",
        passes=("depth",),
        env="vda",
        repo="vda",
        license="Apache 2.0",
        commercial=COMMERCIAL_OK,
        license_note="",
        weights=("vda-s/video_depth_anything_vits.pth",),
        probe="from video_depth_anything.video_depth import VideoDepthAnything\n",
        adapter="video_depth_anything",
        sets=("quality", "safe"),
    ),
    ModelInfo(
        key="vda-l",
        name="Video Depth Anything Large",
        passes=("depth",),
        env="vda",
        repo="vda",
        license="CC-BY-NC 4.0",
        commercial=COMMERCIAL_NO,
        license_note="Non-commercial use only.",
        weights=("vda-l/video_depth_anything_vitl.pth",),
        probe="from video_depth_anything.video_depth import VideoDepthAnything\n",
        safer_alternative="vda-s (Apache 2.0)",
        adapter="video_depth_anything",
    ),
    ModelInfo(
        key="marigold-n",
        name="Marigold Normals v1-1",
        passes=("normal",),
        env="normalcrafter",          # uses the diffusers install of the normalcrafter env
        repo="",
        license="Apache 2.0 code; weights CreativeML Open RAIL++-M (Stable Diffusion 2 base)",
        commercial=COMMERCIAL_OK,
        license_note=("Commercial use allowed; the Open RAIL++-M use restrictions (Attachment A) "
                      "apply and must be passed on with the weights."),
        weights=("marigold-n/model_index.json", "marigold-n/unet/diffusion_pytorch_model.fp16.safetensors"),
        probe="from diffusers import MarigoldNormalsPipeline\n",
        adapter="marigold",
        sets=("safe",),
        temporal=False,
    ),
    ModelInfo(
        key="depth-normals",
        name="Normals from depth",
        passes=("normal",),
        env="core",                   # no model: computed on the CLI side from the depth model's raw output
        repo="",
        license="derived from the depth model's output",
        commercial=COMMERCIAL_OK,
        license_note="",
        weights=(),
        probe="",
        adapter="depth_normals",
        requires=("depth",),
    ),
    ModelInfo(
        key="raft",
        name="RAFT optical flow (torchvision)",
        passes=(),                    # not a pass model: used by --temporal-filter
        env="vda",
        repo="",
        license="BSD-3-Clause",
        commercial=COMMERCIAL_OK,
        license_note="",
        weights=("torch/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth",),
        probe="from torchvision.models.optical_flow import raft_large\n",
        adapter="temporal_filter",
    ),
    ModelInfo(
        key="diffusion-renderer",
        name="Cosmos DiffusionRenderer (inverse, 7B)",
        passes=("albedo", "normal", "depth", "roughness", "metallic"),
        env="cosmos",                 # conda env from upstream's cosmos-predict1.yaml (engine/setup_engine.sh)
        repo="diffusion-renderer",
        license="Apache 2.0 code; weights and tokenizer NVIDIA Open Model License",
        commercial=COMMERCIAL_OK,
        license_note=("Commercial use allowed. Credit 'Built on NVIDIA Cosmos' in the product's docs/about page; "
                      "when redistributing the weights include the notice 'Licensed by NVIDIA Corporation under "
                      "the NVIDIA Open Model License'. Bypassing a safety guardrail terminates the license "
                      "(the inverse renderer ships without one)."),
        weights=("diffusion-renderer/Diffusion_Renderer_Inverse_Cosmos_7B/model.pt",
                 "diffusion-renderer/Cosmos-Tokenize1-CV8x8x8-720p/encoder.jit",
                 "diffusion-renderer/Cosmos-Tokenize1-CV8x8x8-720p/decoder.jit",
                 "diffusion-renderer/Cosmos-Tokenize1-CV8x8x8-720p/mean_std.pt"),
        probe=("import transformer_engine\n"
               "from cosmos_predict1.diffusion.inference.diffusion_renderer_pipeline import DiffusionRendererPipeline\n"
               "extra['transformer_engine'] = transformer_engine.__version__\n"),
        adapter="diffusion_renderer",
        linux_only=True,
        min_vram_gb=20,
    ),
]}

MODEL_SETS = {
    "quality": ("normalcrafter", "vda-s"),
    # Per-frame normals + RAFT temporal filter; depth-derived normals when Marigold isn't installed.
    "safe": ("vda-s", "marigold-n"),
}
SET_FALLBACKS = {"safe": {"marigold-n": "depth-normals"}}


def resolve(spec: str) -> list[ModelInfo]:
    """'quality', 'safe', or a comma-separated list of model keys."""
    keys: list[str] = []
    for part in (p.strip().lower() for p in spec.split(",") if p.strip()):
        if part in MODEL_SETS:
            for k in MODEL_SETS[part]:
                alt = SET_FALLBACKS.get(part, {}).get(k)
                if alt and not installed(MODELS[k]):
                    print(f"Note: {MODELS[k].name} is not installed; the '{part}' set uses {MODELS[alt].name}.")
                    k = alt
                keys.append(k)
        elif part in MODELS and MODELS[part].passes:
            keys.append(part)
        elif part in MODELS:
            raise ValueError(f"'{part}' is not a pass model ({MODELS[part].name} is used by --temporal-filter)")
        else:
            known = ", ".join(list(MODEL_SETS) + [k for k, m in MODELS.items() if m.passes])
            raise ValueError(f"Unknown model or set '{part}'. Known: {known}")
    models = [MODELS[k] for k in dict.fromkeys(keys)]
    # Models computed from another model's output run after it (depth-normals after the depth model).
    return sorted(models, key=lambda m: bool(m.requires))


def installed(m: ModelInfo) -> bool:
    from .. import paths
    return all((paths.MODELS / w).exists() for w in m.weights)


def adapter(m: ModelInfo):
    """Import the model's adapter module (core side: job_settings(preset), run(job))."""
    if not m.adapter:
        raise ValueError(f"{m.name}: no adapter yet (planned in build step 4)")
    import importlib
    return importlib.import_module(f"{__package__}.{m.adapter}")


def check_commercial(models: list[ModelInfo], commercial: bool) -> list[str]:
    """Return warnings; raise if --commercial is set and a model isn't cleared for it."""
    if commercial:
        blocked = [m for m in models if m.commercial == COMMERCIAL_NO]
        if blocked:
            raise PermissionError("--commercial: not allowed for commercial use: " +
                                  "; ".join(m.warning or m.name for m in blocked))
    return [w for m in models if (w := m.warning)]
