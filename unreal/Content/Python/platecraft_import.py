"""Import a PlateCraft clip into Unreal Engine 5.8 as a playable video PBR material.

Reads <clip>/manifest.json (written by `platecraft pack`) and builds:

    /Game/PlateCraft/M_PlateCraft            master material (shared by every clip)
    /Game/PlateCraft/SM_PlateCraftCard_CxR   subdivided card mesh (shared), pivot at bottom centre
    /Game/PlateCraft/<clip>/IMS_<clip>        Img Media Source on <clip>/atlas, fps from the manifest
    /Game/PlateCraft/<clip>/MI_<clip>         material instance (tweak this one)
    /Game/PlateCraft/<clip>/LS_<clip>         Level Sequence with a media track + Camera Cuts (optional)

and places (or updates) a Media Plate actor PlateCraft_<clip> and a CineCamera PlateCraftCam_<clip> framing it
in the open level.

The atlas holds four linear regions (see platecraft/pack.py):
    top-left albedo RGB | top-right normal (n * 0.5 + 0.5, DirectX tangent space)
    bottom-left ORM (R = AO, G = roughness, B = metallic) | bottom-right depth (1 = nearest)
    A = alpha in every quadrant.
One Img Media Source and one player keep all passes frame-locked.

Only built-in editor scripting is used (Python Editor Script Plugin; Media Plate and Img Media
are enabled by default). Run it in the editor:

    py "<repo>/unreal/Content/Python/platecraft_import.py" "D:/work/kitty" --height 120

or put this folder on the editor's Python path (Project Settings > Plugins > Python > Additional
Paths, or copy the files to <project>/Content/Python) and call
`platecraft_import.import_clip("D:/work/kitty")` from the Python console. On that path,
init_unreal.py also adds Tools > PlateCraft > Import Clip... (platecraft_ui.py: folder picker +
options dialog).

Recommended Movie Render Queue settings for keyed plates: in Anti-aliasing, set Spatial Sample
Count to 8 and set Override Anti-Aliasing to None. The material's soft edges are a dither that
those 8 samples average to the true alpha; TSR keeps the edges too solid and dotted.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from fractions import Fraction

import unreal

VERSION = 2  # bump when the master material graph changes; older ones are rebuilt
DEST = "/Game/PlateCraft"
MASTER = "M_PlateCraft"
MEDIA_TEXTURE_PARAM = "MediaTexture"  # the Media Plate binds its texture to this parameter
VERSION_TAG = "PlateCraftVersion"
CLIP = 1.0 / 3.0  # opacity mask clip value (UE default)
# Opacity mask with coverage = alpha: drawn iff alpha > threshold. The threshold is a per-pixel offset
# (interleaved gradient noise) stepped by 1/8 per frame index, so 8 consecutive samples (MRQ advances
# the frame index per spatial/temporal sample; TSR across frames) hit 8 evenly spaced thresholds:
# coverage exact to 1/8. Independent noise per sample left visible grain (8 random samples: +-0.18).
# The step order is scrambled per pixel (frame index XOR a pixel hash): in the same order everywhere
# it correlated with MRQ's sub-pixel jitter sequence and drew horizontal/vertical lines in soft edges.
# Range [0.001, 0.999] so alpha 0 is never and 1 always drawn.
DITHER_HLSL = f"""float g = frac(52.9829189 * frac(dot(Parameters.SvPosition.xy, float2(0.06711056, 0.00583715))));
uint2 ip = uint2(Parameters.SvPosition.xy);
uint h = (ip.x * 73856093u) ^ (ip.y * 19349663u);
h = (h ^ (h >> 13)) * 0x5bd1e995u;
h ^= h >> 15;
uint k = (uint(View.StateFrameIndexMod8) ^ h) & 7u;
float n = frac(g + float(k) * 0.125) * 0.998 + 0.001;
return {CLIP!r} + Alpha - n;"""

# Atlas quadrant offsets in UV (origin top-left), as written by platecraft/pack.py.
REGIONS = {"albedo": (0.0, 0.0), "normal": (0.5, 0.0), "orm": (0.0, 0.5), "depth": (0.5, 0.5)}

MEL = unreal.MaterialEditingLibrary
EAL = unreal.EditorAssetLibrary


def log(msg: str) -> None:
    unreal.log(f"[PlateCraft] {msg}")


def warn(msg: str) -> None:
    unreal.log_warning(f"[PlateCraft] {msg}")


# ---------------------------------------------------------------------------------------------
# Manifest


def load_manifest(clip_dir: str) -> dict:
    path = os.path.join(clip_dir, "manifest.json")
    if not os.path.isfile(path):
        raise RuntimeError(f"no manifest.json in {clip_dir}")
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    atlas = m.get("atlas")
    if not atlas:
        raise RuntimeError(f"{clip_dir} has no atlas yet; run `platecraft pack` first")
    for region, (u, v) in (("albedo", REGIONS["albedo"]), ("normal", REGIONS["normal"]),
                           ("ao", REGIONS["orm"]), ("depth", REGIONS["depth"])):
        rect = atlas["regions"].get(region, {}).get("rect")
        if rect and [round(x, 3) for x in rect[:2]] != [u, v]:
            raise RuntimeError(f"atlas region {region} at {rect}, expected the platecraft 2x2 layout")
    for name, r in atlas["regions"].items():
        if r.get("colorspace", "linear") != "linear":
            warn(f"atlas region {name} is {r['colorspace']}; the material expects linear values")
    done, count = atlas.get("frames_done", 0), atlas.get("frame_count", 0)
    if done < count:
        warn(f"atlas has {done} of {count} frames; the rest will be missing")
    return m


def frame_rate(m: dict) -> tuple[int, int]:
    fps = str(m["atlas"].get("fps") or m["frames"]["fps"])
    f = Fraction(fps) if "/" in fps else Fraction(fps).limit_denominator(1001)
    return f.numerator, f.denominator


# ---------------------------------------------------------------------------------------------
# Asset helpers


def asset_tools() -> unreal.AssetTools:
    return unreal.AssetToolsHelpers.get_asset_tools()


def load_or_create(folder: str, name: str, cls, factory):
    path = f"{folder}/{name}"
    if EAL.does_asset_exist(path):
        asset = EAL.load_asset(path)
        if isinstance(asset, cls):
            return asset, False
        raise RuntimeError(f"{path} exists but is a {asset.get_class().get_name()}, not {cls.__name__}")
    asset = asset_tools().create_asset(name, folder, cls, factory)
    if asset is None:
        raise RuntimeError(f"could not create {path}")
    return asset, True


# ---------------------------------------------------------------------------------------------
# Master material


class Graph:
    """Small helper around MaterialEditingLibrary for building a material graph."""

    def __init__(self, material: unreal.Material):
        self.m = material

    def node(self, cls, x: int, y: int, **props):
        e = MEL.create_material_expression(self.m, cls, x, y)
        for k, v in props.items():
            e.set_editor_property(k, v)
        return e

    def scalar(self, name: str, default: float, x: int, y: int, group: str, desc: str = ""):
        return self.node(unreal.MaterialExpressionScalarParameter, x, y, parameter_name=name,
                         default_value=default, group=group, desc=desc)

    def link(self, src, dst, dst_input: str = "", src_output: str = "") -> None:
        if not MEL.connect_material_expressions(src, src_output, dst, dst_input):
            raise RuntimeError(
                f"cannot connect {src.get_name()}.{src_output!r} -> {dst.get_name()}.{dst_input!r}; "
                f"inputs {MEL.get_material_expression_input_names(dst)}, "
                f"outputs {MEL.get_material_expression_output_names(src)}")

    def out(self, src, prop, src_output: str = "") -> None:
        if not MEL.connect_material_property(src, src_output, prop):
            raise RuntimeError(f"cannot connect {src.get_name()}.{src_output!r} -> {prop}")

    def op(self, cls, a, b, x: int, y: int, a_out: str = "", b_out: str = ""):
        """Binary node (Add, Multiply, ...); a or b may be a float constant."""
        e = self.node(cls, x, y)
        for pin, v, o in (("A", a, a_out), ("B", b, b_out)):
            if isinstance(v, (int, float)):
                e.set_editor_property("const_" + pin.lower(), float(v))
            else:
                self.link(v, e, pin, o)
        return e

    def lerp(self, a, b, alpha, x: int, y: int, a_out: str = "", b_out: str = ""):
        e = self.node(unreal.MaterialExpressionLinearInterpolate, x, y)
        for pin, v, o in (("A", a, a_out), ("B", b, b_out), ("Alpha", alpha, "")):
            if isinstance(v, (int, float)):
                e.set_editor_property("const_" + pin.lower(), float(v))
            else:
                self.link(v, e, pin, o)
        return e


def build_master_material(folder: str = DEST, rebuild: bool = False) -> unreal.Material:
    path = f"{folder}/{MASTER}"
    mat, created = load_or_create(folder, MASTER, unreal.Material, unreal.MaterialFactoryNew())
    if not created and not rebuild and EAL.get_metadata_tag(mat, VERSION_TAG) == str(VERSION):
        return mat
    log(f"building {path} (v{VERSION})")
    MEL.delete_all_material_expressions(mat)
    mat.set_editor_property("blend_mode", unreal.BlendMode.BLEND_MASKED)
    mat.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_DEFAULT_LIT)
    mat.set_editor_property("two_sided", True)
    mat.set_editor_property("opacity_mask_clip_value", CLIP)
    mat.set_editor_property("dithered_lod_transition", False)

    g = Graph(mat)
    color = unreal.MaterialSamplerType.SAMPLERTYPE_COLOR
    tex = g.node(unreal.MaterialExpressionTextureObjectParameter, -2200, 0,
                 parameter_name=MEDIA_TEXTURE_PARAM, sampler_type=color,
                 texture=unreal.load_asset("/MediaPlate/T_MediaPlate"), group="Media",
                 desc="Set by the Media Plate; the 2x2 PlateCraft atlas")

    # Quadrant UVs: clamp the card UV half an atlas texel inside each quadrant so bilinear
    # filtering never reaches the neighbouring pass. Half an atlas texel = 1 / size in quadrant UV.
    uv = g.node(unreal.MaterialExpressionTextureCoordinate, -2200, 300)
    size = g.node(unreal.MaterialExpressionTextureProperty, -2200, 450,
                  property_=unreal.MaterialExposedTextureProperty.TMTM_TEXTURE_SIZE)
    g.link(tex, size, "Tex")
    inset = g.op(unreal.MaterialExpressionDivide, 1.0, size, -1950, 450)
    upper = g.node(unreal.MaterialExpressionOneMinus, -1800, 500)
    g.link(inset, upper)
    clamp = g.node(unreal.MaterialExpressionClamp, -1650, 300)
    g.link(uv, clamp)
    g.link(inset, clamp, "Min")
    g.link(upper, clamp, "Max")
    quarter = g.op(unreal.MaterialExpressionMultiply, clamp, 0.5, -1500, 300)

    samples = {}
    for i, (name, (u, v)) in enumerate(REGIONS.items()):
        y = -300 + i * 450
        off = g.node(unreal.MaterialExpressionConstant2Vector, -1500, y + 150, r=u, g=v)
        quv = g.op(unreal.MaterialExpressionAdd, quarter, off, -1300, y + 100)
        s = g.node(unreal.MaterialExpressionTextureSample, -1100, y, sampler_type=color, desc=name)
        if name == "depth":  # also read in the vertex shader for the card bend
            s.set_editor_property("mip_value_mode", unreal.TextureMipValueMode.TMVM_MIP_LEVEL)
            s.set_editor_property("const_mip_value", 0)
        g.link(tex, s, "Tex")
        g.link(quv, s, "UVs")
        samples[name] = s

    # Base color = albedo * tint * brightness
    tint = g.node(unreal.MaterialExpressionVectorParameter, -800, -500, parameter_name="AlbedoTint",
                  default_value=unreal.LinearColor(1, 1, 1, 1), group="Albedo")
    bright = g.scalar("AlbedoBrightness", 1.0, -800, -350, "Albedo")
    base = g.op(unreal.MaterialExpressionMultiply, samples["albedo"], tint, -550, -400, a_out="RGB", b_out="RGB")
    base = g.op(unreal.MaterialExpressionMultiply, base, bright, -350, -400)
    g.out(base, unreal.MaterialProperty.MP_BASE_COLOR)

    # Opacity mask: alpha, dithered (soft edges under TSR/TAA) or a hard cut at 0.5.
    # The engine's DitherTemporalAA only spreads its thresholds over alpha 1/6..5/6 (measured on a
    # render: alpha 0.7 drew 96% of the pixels), so soft fur turned into a solid rim. Instead the
    # threshold is uniform per pixel, stratified over frames (DITHER_HLSL) -> coverage = alpha.
    soft = g.node(unreal.MaterialExpressionStaticSwitchParameter, -350, -150, parameter_name="SoftEdges",
                  default_value=True, group="Alpha",
                  desc="Dithered alpha: soft edges with temporal AA. Off: hard cut at 0.5")
    # Levels first (black/white point) to choke a soft or blurred matte edge in the editor.
    black = g.scalar("AlphaBlackPoint", 0.0, -1100, -250, "Alpha", "alpha below this = transparent (chokes the edge)")
    white = g.scalar("AlphaWhitePoint", 1.0, -1100, -150, "Alpha", "alpha above this = opaque")
    a_lv = g.op(unreal.MaterialExpressionSubtract, samples["albedo"], black, -950, -250, a_out="A")
    span = g.op(unreal.MaterialExpressionSubtract, white, black, -950, -150)
    span = g.op(unreal.MaterialExpressionMax, span, 1e-3, -850, -150)
    a_lv = g.op(unreal.MaterialExpressionDivide, a_lv, span, -800, -250)
    a_sat = g.node(unreal.MaterialExpressionSaturate, -700, -250)
    g.link(a_lv, a_sat)
    dither = g.node(unreal.MaterialExpressionCustom, -600, -150, description="AlphaDither",
                    output_type=unreal.CustomMaterialOutputType.CMOT_FLOAT1, code=DITHER_HLSL)
    alpha_in = unreal.CustomInput()
    alpha_in.set_editor_property("input_name", "Alpha")
    dither.set_editor_property("inputs", [alpha_in])
    g.link(a_sat, dither, "Alpha")
    g.link(dither, soft, "True")
    hard = g.op(unreal.MaterialExpressionAdd, a_sat, CLIP - 0.5, -600, -50)
    g.link(hard, soft, "False")
    g.out(soft, unreal.MaterialProperty.MP_OPACITY_MASK)

    # Normal: unpack, optional OpenGL green flip, strength, renormalise
    n_rgb = g.op(unreal.MaterialExpressionMultiply, samples["normal"], 2.0, -900, 150, a_out="RGB")
    n_rgb = g.op(unreal.MaterialExpressionSubtract, n_rgb, 1.0, -750, 150)
    flip = g.scalar("FlipGreen", 0.0, -900, 300, "Normal", "1 for OpenGL-convention normals")
    flip_v = g.node(unreal.MaterialExpressionAppendVector, -600, 300)
    sign = g.op(unreal.MaterialExpressionMultiply, flip, -2.0, -750, 300)
    sign = g.op(unreal.MaterialExpressionAdd, sign, 1.0, -650, 380)
    one2 = g.node(unreal.MaterialExpressionConstant, -750, 450, r=1.0)
    g.link(one2, flip_v, "A")
    g.link(sign, flip_v, "B")
    flip_v3 = g.node(unreal.MaterialExpressionAppendVector, -500, 350)
    g.link(flip_v, flip_v3, "A")
    g.link(one2, flip_v3, "B")
    n_rgb = g.op(unreal.MaterialExpressionMultiply, n_rgb, flip_v3, -550, 150)
    strength = g.scalar("NormalStrength", 1.0, -550, 450, "Normal")
    flat = g.node(unreal.MaterialExpressionConstant3Vector, -550, 550, constant=unreal.LinearColor(0, 0, 1, 0))
    n_rgb = g.lerp(flat, n_rgb, strength, -350, 300)
    norm = g.node(unreal.MaterialExpressionNormalize, -200, 300)
    g.link(n_rgb, norm)
    g.out(norm, unreal.MaterialProperty.MP_NORMAL)

    # ORM: AO strength, roughness scale/offset, metallic offset, constant specular
    orm = samples["orm"]
    ao_s = g.scalar("AOStrength", 1.0, -800, 650, "Surface")
    ao = g.lerp(1.0, orm, ao_s, -550, 650, b_out="R")
    g.out(ao, unreal.MaterialProperty.MP_AMBIENT_OCCLUSION)
    r_scale = g.scalar("RoughnessScale", 1.0, -800, 800, "Surface", "roughness = map * scale + offset (map is heuristic)")
    r_off = g.scalar("RoughnessOffset", 0.0, -800, 900, "Surface")
    rough = g.op(unreal.MaterialExpressionMultiply, orm, r_scale, -550, 800, a_out="G")
    rough = g.op(unreal.MaterialExpressionAdd, rough, r_off, -400, 800)
    rough_sat = g.node(unreal.MaterialExpressionSaturate, -250, 800)
    g.link(rough, rough_sat)
    g.out(rough_sat, unreal.MaterialProperty.MP_ROUGHNESS)
    m_off = g.scalar("MetallicOffset", 0.0, -800, 1000, "Surface")
    metal = g.op(unreal.MaterialExpressionAdd, orm, m_off, -550, 1000, a_out="B")
    metal_sat = g.node(unreal.MaterialExpressionSaturate, -400, 1000)
    g.link(metal, metal_sat)
    g.out(metal_sat, unreal.MaterialProperty.MP_METALLIC)
    spec = g.scalar("Specular", 0.5, -400, 1100, "Surface")
    g.out(spec, unreal.MaterialProperty.MP_SPECULAR)

    # Card bend: push vertices along the card normal by (depth - pivot) * amount (cm).
    # Needs the subdivided card mesh; 0 keeps the card flat.
    amount = g.scalar("DepthAmount", 0.0, -800, 1300, "Depth",
                      "Card bend in cm between the farthest and nearest subject depth (0 = flat)")
    pivot = g.scalar("DepthPivot", 0.5, -800, 1400, "Depth", "Depth value that stays on the card plane")
    d = g.op(unreal.MaterialExpressionSubtract, samples["depth"], pivot, -550, 1300, a_out="R")
    d = g.op(unreal.MaterialExpressionMultiply, d, amount, -400, 1300)
    vn = g.node(unreal.MaterialExpressionVertexNormalWS, -400, 1450)
    wpo = g.op(unreal.MaterialExpressionMultiply, vn, d, -250, 1350)
    g.out(wpo, unreal.MaterialProperty.MP_WORLD_POSITION_OFFSET)

    MEL.recompile_material(mat)
    EAL.set_metadata_tag(mat, VERSION_TAG, str(VERSION))
    EAL.save_loaded_asset(mat)
    return mat


# ---------------------------------------------------------------------------------------------
# Card mesh


def build_card_mesh(folder: str, cols: int, rows: int) -> unreal.StaticMesh:
    """A cols x rows grid with the Media Plate screen's orientation (YZ plane, front facing -X,
    U along +Y, V down), 100 x 100 units, pivot at the bottom centre."""
    name = f"SM_PlateCraftCard_{cols}x{rows}"
    path = f"{folder}/{name}"
    if EAL.does_asset_exist(path):
        return EAL.load_asset(path)
    log(f"building {path}")
    mesh = EAL.duplicate_asset("/MediaPlate/SM_MediaPlateScreen", path)
    if mesh is None:
        raise RuntimeError(f"could not create {path}")
    slot = mesh.get_editor_property("static_materials")[0].get_editor_property("material_slot_name")

    desc = mesh.create_static_mesh_description(mesh)
    group = desc.create_polygon_group()
    desc.set_polygon_group_material_slot_name(group, slot)
    inst = []
    for j in range(rows + 1):
        v = j / rows
        for i in range(cols + 1):
            u = i / cols
            vert = desc.create_vertex()
            desc.set_vertex_position(vert, unreal.Vector(0.0, (u - 0.5) * 100.0, (1.0 - v) * 100.0))
            vi = desc.create_vertex_instance(vert)
            desc.set_vertex_instance_uv(vi, unreal.Vector2D(u, v), 0)
            inst.append(vi)
    w = cols + 1
    for j in range(rows):
        for i in range(cols):
            tl, tr = inst[j * w + i], inst[j * w + i + 1]
            bl, br = inst[(j + 1) * w + i], inst[(j + 1) * w + i + 1]
            # same winding as SM_MediaPlateScreen (front face towards -X)
            desc.create_triangle(group, [br, tr, bl])
            desc.create_triangle(group, [bl, tr, tl])
    # full build (fast_build skips normals/tangents); build settings only after the first
    # build, otherwise the editor frees live render data and crashes
    mesh.build_from_static_mesh_descriptions([desc], False, False)
    sub = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
    settings = sub.get_lod_build_settings(mesh, 0)
    settings.set_editor_property("recompute_normals", True)
    settings.set_editor_property("recompute_tangents", True)
    sub.set_lod_build_settings(mesh, 0, settings)
    EAL.save_loaded_asset(mesh)
    return mesh


# ---------------------------------------------------------------------------------------------
# Per-clip assets


def img_media_factory():
    for n in ("ImgMediaSourceFactoryNew", "ImgMediaSourceFactory"):
        cls = getattr(unreal, n, None)
        if cls is not None:
            try:
                f = cls()
                if f.get_editor_property("supported_class") == unreal.ImgMediaSource.static_class():
                    return f
            except Exception:
                pass
    return None


def create_media_source(clip_dir: str, m: dict, folder: str, name: str) -> unreal.ImgMediaSource:
    atlas = m["atlas"]
    first = os.path.join(clip_dir, atlas["dir"], atlas["pattern"] % atlas.get("start_number", 1))
    if not os.path.isfile(first):
        raise RuntimeError(f"first atlas frame missing: {first}")
    src, _ = load_or_create(folder, f"IMS_{name}", unreal.ImgMediaSource, img_media_factory())
    src.set_sequence_path(os.path.abspath(first).replace("\\", "/"))
    num, den = frame_rate(m)
    src.set_editor_property("frame_rate_override", unreal.FrameRate(num, den))
    src.set_editor_property("fill_gaps_in_sequence", True)
    EAL.save_loaded_asset(src)
    return src


def set_mi_params(mi: unreal.MaterialInstanceConstant, m: dict, depth_amount: float | None) -> None:
    regions = m["atlas"]["regions"]
    if regions.get("normal", {}).get("normal_space") == "opengl":
        MEL.set_material_instance_scalar_parameter_value(mi, "FlipGreen", 1.0)
    if depth_amount is not None:
        MEL.set_material_instance_scalar_parameter_value(mi, "DepthAmount", float(depth_amount))


def create_material_instance(m: dict, master, folder: str, name: str, depth_amount):
    mi, created = load_or_create(folder, f"MI_{name}", unreal.MaterialInstanceConstant,
                                 unreal.MaterialInstanceConstantFactoryNew())
    if mi.get_editor_property("parent") != master:
        MEL.set_material_instance_parent(mi, master)
    set_mi_params(mi, m, depth_amount)
    MEL.update_material_instance(mi)
    EAL.save_loaded_asset(mi)
    return mi


def find_actor(label: str):
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
    return next((a for a in actors if isinstance(a, unreal.MediaPlate) and a.get_actor_label() == label), None)


def place_media_plate(m: dict, source, mi, mesh, name: str, location, yaw: float, height: float,
                      autoplay: bool) -> unreal.MediaPlate:
    label = f"PlateCraft_{name}"
    plate = find_actor(label)
    if plate is None:
        plate = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).spawn_actor_from_class(
            unreal.MediaPlate, unreal.Vector(*location), unreal.Rotator(0.0, 0.0, yaw))
        plate.set_actor_label(label)
        log(f"placed Media Plate {label}")
    else:
        log(f"updating Media Plate {label}")
    comp = plate.get_editor_property("media_plate_component")
    smc = plate.get_editor_property("static_mesh_component")

    tile_w, tile_h = m["atlas"]["tile"]
    comp.set_is_aspect_ratio_auto(False)
    smc.set_editor_property("relative_location", unreal.Vector(0, 0, 0))
    smc.set_editor_property("relative_scale3d", unreal.Vector(1.0, height * tile_w / tile_h / 100.0, height / 100.0))
    smc.set_editor_property("static_mesh", mesh)
    # the Media Plate turns dynamic shadows off; static ones stay off (a bake would freeze a frame)
    smc.set_editor_property("cast_dynamic_shadow", True)
    smc.set_cast_shadow(True)
    # the silhouette comes from the video's alpha, which Virtual Shadow Maps can't see change
    # (Auto only invalidates on transform/WPO) -> cached shadows would show old frames
    smc.set_editor_property("shadow_cache_invalidation_behavior", unreal.ShadowCacheInvalidationBehavior.ALWAYS)
    # set_editor_property fires PostEditChange, so the Media Plate wraps the instance in a MID
    # with its media texture bound to the MediaTexture parameter.
    smc.set_editor_property("override_materials", [mi])
    # World Partition: a spatially loaded actor is streamed into PIE/Movie Render Queue from its
    # saved package, and that copy renders the material's default texture instead of the video
    # (opaque grey plate; verified in 5.8). Always loaded actors are copied from the editor world.
    plate.set_editor_property("is_spatially_loaded", False)

    comp.select_media_source_asset(source)
    comp.set_editor_property("play_on_open", autoplay)
    comp.set_editor_property("auto_play", autoplay)
    comp.set_editor_property("loop", True)
    comp.set_enable_audio(False)
    if autoplay:
        comp.open()  # the details panel does this when a source is picked; scripting has to
    return plate


FOCAL_MM = 35.0
SENSOR_W_MM = 23.76  # CineCamera default (16:9 DSLR); the height follows the clip's aspect ratio
APERTURE = 8.0       # deep enough that a depth-bent card stays sharp


def camera_framing(tile_w: int, tile_h: int, location, yaw: float, height: float):
    """Camera transform + filmback that frame the plate's full video frame exactly (1:1 renders).
    The plate's pivot is the bottom centre and its front faces -X rotated by `yaw`."""
    sensor_h = SENSOR_W_MM * tile_h / tile_w
    dist = height * FOCAL_MM / sensor_h  # vertical FOV fits the frame height exactly
    a = math.radians(yaw)
    x, y, z = location
    cam = unreal.Vector(x - math.cos(a) * dist, y - math.sin(a) * dist, z + height / 2.0)
    return cam, unreal.Rotator(0.0, 0.0, yaw), sensor_h, dist


def place_camera(m: dict, name: str, location, yaw: float, height: float) -> unreal.CineCameraActor:
    """CineCamera PlateCraftCam_<name> framing the plate. An existing one keeps its transform and lens."""
    label = f"PlateCraftCam_{name}"
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
    cam = next((a for a in actors if isinstance(a, unreal.CineCameraActor) and a.get_actor_label() == label), None)
    if cam is not None:
        log(f"keeping camera {label}")
        return cam
    tile_w, tile_h = m["atlas"]["tile"]
    loc, rot, sensor_h, dist = camera_framing(tile_w, tile_h, location, yaw, height)
    cam = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).spawn_actor_from_class(
        unreal.CineCameraActor, loc, rot)
    cam.set_actor_label(label)
    cc = cam.get_cine_camera_component()
    fb = cc.get_editor_property("filmback")
    fb.set_editor_property("sensor_width", SENSOR_W_MM)
    fb.set_editor_property("sensor_height", sensor_h)
    cc.set_editor_property("filmback", fb)
    cc.set_editor_property("current_focal_length", FOCAL_MM)
    cc.set_editor_property("current_aperture", APERTURE)
    focus = cc.get_editor_property("focus_settings")
    focus.set_editor_property("focus_method", unreal.CameraFocusMethod.MANUAL)
    focus.set_editor_property("manual_focus_distance", dist)
    cc.set_editor_property("focus_settings", focus)
    log(f"placed camera {label} {dist:.0f} cm in front of the plate ({FOCAL_MM:.0f} mm, frames the clip 1:1)")
    return cam


STREAMING_RADIUS = 300000.0  # cm


def set_streaming_source(cam, radius: float) -> None:
    """World Partition: stream cells within `radius` cm of the camera into PIE / Movie Render Queue.

    The player's own streaming source only loads the grid's loading range (256 m in the Open World
    template) and MRQ renders with r.HLOD 0, so distant terrain the camera sees in the editor (where
    everything is loaded) is missing from renders; the Sky Atmosphere's grey ground shows instead.
    radius 0 removes the source."""
    comp = cam.get_component_by_class(unreal.WorldPartitionStreamingSourceComponent)
    if radius <= 0:
        if comp is not None:
            sds = unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem)
            handles = sds.k2_gather_subobject_data_for_instance(cam)
            sds.k2_delete_subobjects_from_instance(handles[0], [h for h in handles if
                unreal.SubobjectDataBlueprintFunctionLibrary.get_object(
                    unreal.SubobjectDataBlueprintFunctionLibrary.get_data(h)) == comp])
        return
    if comp is None:
        sds = unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem)
        root = sds.k2_gather_subobject_data_for_instance(cam)[0]
        _, fail = sds.add_new_subobject(unreal.AddNewSubobjectParams(
            parent_handle=root, new_class=unreal.WorldPartitionStreamingSourceComponent))
        if str(fail):
            warn(f"could not add a streaming source to {cam.get_actor_label()}: {fail}")
            return
        comp = cam.get_component_by_class(unreal.WorldPartitionStreamingSourceComponent)
    shape = unreal.StreamingSourceShape()
    shape.set_editor_property("use_grid_loading_range", False)
    shape.set_editor_property("radius", radius)
    comp.set_editor_property("shapes", [shape])
    comp.set_editor_property("default_visualizer_loading_range", radius)
    log(f"{cam.get_actor_label()}: World Partition streaming source, radius {radius / 100:.0f} m")


def bind_camera(seq: unreal.LevelSequence, cam, frames: int) -> None:
    """Possess the camera and add a Camera Cuts track on it, unless the sequence already has one."""
    label = cam.get_actor_label()
    cut_tracks = seq.find_tracks_by_type(unreal.MovieSceneCameraCutTrack)
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    binding = next((b for b in seq.get_bindings() if b.get_display_name() == label), None)
    # Cuts on our camera's binding, or on one that no longer exists: retarget them after a rebind.
    # (Guid structs don't compare by value in Python, so match by the resolved binding's name.)
    ours = [s for t in cut_tracks for s in t.get_sections()
            if str(seq.resolve_binding_id(s.get_camera_binding_id()).get_display_name()) in ("", label)]
    if binding is not None and cam not in seq.locate_bound_objects(binding, world):
        binding.remove()  # stale (e.g. the sequence was made in another level)
        binding = None
    if binding is None:
        binding = seq.add_possessable(cam)
        for s in ours:
            s.set_camera_binding_id(seq.get_binding_id(binding))
        if ours:
            log(f"rebound {label} in {seq.get_name()} ({len(ours)} camera cut(s))")
    if cut_tracks:
        return  # keep the user's cuts
    section = seq.add_track(unreal.MovieSceneCameraCutTrack).add_section()
    section.set_range(0, frames)
    section.set_camera_binding_id(seq.get_binding_id(binding))


def create_level_sequence(m: dict, plate, source, folder: str, name: str, camera=None) -> unreal.LevelSequence:
    seq, created = load_or_create(folder, f"LS_{name}", unreal.LevelSequence, unreal.LevelSequenceFactoryNew())
    num, den = frame_rate(m)
    frames = m["atlas"].get("frame_count") or m["frames"]["count"]
    seq.set_display_rate(unreal.FrameRate(num, den))
    seq.set_playback_start(0)
    seq.set_playback_end(frames)
    seq.set_view_range_start(-0.5)
    seq.set_view_range_end(frames * den / num + 0.5)

    # Rebind: drop old bindings of this plate, then possess it with one media track.
    for b in list(seq.get_bindings()):
        if b.get_display_name() == plate.get_actor_label():
            b.remove()
    binding = seq.add_possessable(plate)
    track = binding.add_track(unreal.MovieSceneMediaTrack)
    track.set_display_name("Media")
    # Block on the exact frame when scrubbing/stopped too, not only while playing. Without it the
    # first frame of a Movie Render Queue shot renders before the media arrives (blank plate).
    track.set_editor_property("synchronous_scrubbing", True)
    section = track.add_section()
    section.set_range(0, frames)
    section.set_editor_property("media_source", source)
    section.set_editor_property("looping", False)
    if camera is not None:
        bind_camera(seq, camera, frames)
    EAL.save_loaded_asset(seq)
    return seq


def disable_light_shaft_occlusion() -> int:
    """Turn off Light Shaft Occlusion on every directional light in the level (also ones inside
    Blueprint actors). It's screen-space and depth based: a plate in front of the sky darkens wedges
    of the scene behind it, spreading away from the sun. Returns the number of lights changed."""
    changed = 0
    for actor in unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors():
        for light in actor.get_components_by_class(unreal.DirectionalLightComponent):
            if light.get_editor_property("enable_light_shaft_occlusion"):
                light.set_editor_property("enable_light_shaft_occlusion", False)
                log(f"turned off Light Shaft Occlusion on {actor.get_actor_label()}")
                changed += 1
    return changed


def check_world_partition_descriptors() -> None:
    """Warn when the open World Partition level still carries its template's actor paths.

    A map made from a World Partition template (File > New Level > Open World) and saved keeps
    actor descriptors that point into /Temp/Untitled_N until the level is reopened. PIE and Movie
    Render Queue build their streaming from those descriptors and skip every such actor: sky,
    lights and landscape are missing and the background renders black (alpha 0) while the editor
    viewport looks fine. Only actors saved since (e.g. the plate and camera) appear.
    """
    try:
        descs = unreal.WorldPartitionBlueprintLibrary.get_actor_descs()
    except Exception:
        return  # not a World Partition level
    stale = [d for d in descs if str(d.get_editor_property("actor_path")).startswith("/Temp/")]
    if stale:
        warn(f"{len(stale)} actors of this World Partition level still have template paths (/Temp/...): "
             "PIE and Movie Render Queue will leave them out (black sky/landscape). Save the level, then "
             "reopen it (File > Open Level) before rendering.")


# ---------------------------------------------------------------------------------------------


def import_clip(clip_dir: str, name: str | None = None, dest: str = DEST, location=(0.0, 0.0, 0.0),
                yaw: float = 0.0, height: float = 200.0, grid: int = 128, depth_amount: float | None = None,
                place: bool = True, sequence: bool = True, rebuild_material: bool = False,
                camera: bool = True, no_light_shafts: bool = False,
                streaming_radius: float = STREAMING_RADIUS) -> dict:
    """Import one clip folder. Returns the created/updated assets and actor."""
    clip_dir = os.path.abspath(clip_dir)
    m = load_manifest(clip_dir)
    name = name or os.path.basename(clip_dir.rstrip("/\\"))
    folder = f"{dest}/{name}"
    log(f"importing {clip_dir} -> {folder}")
    for lic in m.get("licenses", []):
        if lic.get("commercial") != "ok":
            warn(f"model {lic['model']}: {lic['license']} ({lic['commercial']}). {lic.get('note', '')}")

    master = build_master_material(dest, rebuild_material)
    tile_w, tile_h = m["atlas"]["tile"]
    rows = max(1, round(grid * tile_h / tile_w))
    mesh = build_card_mesh(dest, grid, rows)
    source = create_media_source(clip_dir, m, folder, name)
    mi = create_material_instance(m, master, folder, name, depth_amount)
    out = {"material": master, "mesh": mesh, "source": source, "instance": mi, "plate": None, "sequence": None,
           "camera": None}
    if place:
        out["plate"] = place_media_plate(m, source, mi, mesh, name, location, yaw, height, autoplay=not sequence)
        if sequence:
            if camera:
                out["camera"] = place_camera(m, name, location, yaw, height)
                set_streaming_source(out["camera"], streaming_radius)
            out["sequence"] = create_level_sequence(m, out["plate"], source, folder, name, out["camera"])
    if no_light_shafts:
        disable_light_shaft_occlusion()
    if place:
        check_world_partition_descriptors()
    num, den = frame_rate(m)
    w, h = m["atlas"]["width"], m["atlas"]["height"]
    log(f"done: {m['atlas']['frame_count']} frames at {num / den:.3f} fps, atlas {w}x{h}"
        + (" (the Media Plate's 'no MediaTexture parameters' warning is expected; it binds the "
           "texture anyway)" if place else ""))
    if sequence and w * h > 1920 * 1080 * 1.1:
        # measured on the 3060: 3840x2160 half EXR = 3-6 editor fps in Sequencer, 1920x1080 = 79
        warn(f"a {w}x{h} atlas decodes too slowly for real-time Sequencer playback in the editor "
             "(the media track waits for every frame). Renders (Movie Render Queue) are frame-accurate "
             "anyway; for real-time preview pack with `--scale 0.5`.")
    if sequence:
        log("Recommended Movie Render Queue settings for keyed plates: in Anti-aliasing, set Spatial "
            "Sample Count to 8 and set Override Anti-Aliasing to None (soft edges average to the true alpha).")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="platecraft_import", description=__doc__.split("\n")[0])
    p.add_argument("clip", help="clip folder with manifest.json and atlas/ (from `platecraft pack`)")
    p.add_argument("--name", help="asset/actor name (default: clip folder name)")
    p.add_argument("--dest", default=DEST, help=f"content folder (default {DEST})")
    p.add_argument("--location", default="0,0,0", help="Media Plate location x,y,z in cm (bottom centre)")
    p.add_argument("--yaw", type=float, default=0.0, help="Media Plate yaw; 0 = front faces -X")
    p.add_argument("--height", type=float, default=200.0, help="height of the full video frame in cm")
    p.add_argument("--grid", type=int, default=128, help="card mesh columns for depth bending")
    p.add_argument("--depth-amount", type=float, help="card bend in cm (material instance DepthAmount)")
    p.add_argument("--no-place", action="store_true", help="only create assets, no Media Plate")
    p.add_argument("--no-sequence", action="store_true",
                   help="no Level Sequence; the Media Plate autoplays and loops instead")
    p.add_argument("--no-camera", action="store_true",
                   help="no CineCamera PlateCraftCam_<name> / Camera Cuts track in the sequence")
    p.add_argument("--rebuild-material", action="store_true", help="rebuild M_PlateCraft")
    p.add_argument("--no-light-shafts", action="store_true",
                   help="turn off Light Shaft Occlusion on the level's directional lights (it darkens "
                        "wedges behind the plate, spreading away from the sun)")
    p.add_argument("--streaming-radius", type=float, default=STREAMING_RADIUS / 100.0,
                   help="World Partition: load everything within this many metres of the camera in PIE/Movie "
                        "Render Queue (default %(default).0f; 0 = player's loading range only)")
    a = p.parse_args(argv)
    loc = tuple(float(x) for x in a.location.split(","))
    import_clip(a.clip, a.name, a.dest, loc, a.yaw, a.height, a.grid, a.depth_amount,
                place=not a.no_place, sequence=not a.no_sequence, rebuild_material=a.rebuild_material,
                camera=not a.no_camera, no_light_shafts=a.no_light_shafts,
                streaming_radius=a.streaming_radius * 100.0)
    return 0


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except SystemExit:
        pass
    except Exception as e:  # keep the editor alive and show the reason
        unreal.log_error(f"[PlateCraft] import failed: {e}")
        raise
