"""PlateCraft import dialog for the Unreal Editor (UE 5.8): Tools > PlateCraft > Import Clip...

A details-panel dialog (EditorDialog.show_object_details_view) on a Python-defined options object:
a folder picker for the clip (the stock DirectoryPath customization opens the OS folder dialog)
plus the importer's options, and OK = Import. Nothing but built-in editor scripting, and no
binary .uasset (an Editor Utility Widget can't be authored from a script or diffed).

The menu entry is registered by init_unreal.py at editor startup when this folder is on the
editor's Python path (Project Settings > Plugins > Python > Additional Paths, or copy both
files to <project>/Content/Python). Without the menu:

    import platecraft_ui; platecraft_ui.show_import_dialog()

The last used options are kept in <project>/Saved/PlateCraft/import_dialog.json.
"""
from __future__ import annotations

import importlib
import json
import math
import os

import unreal

import platecraft_import as sli

MENU = "LevelEditor.MainMenu.Tools"
SECTION = "PlateCraft"
ENTRY = "PlateCraftImportClip"
TITLE = "PlateCraft: Import Clip"
COMMAND = "import platecraft_ui; platecraft_ui.show_import_dialog()"


@unreal.uclass()
class PlateCraftImportOptions(unreal.Object):
    clip_folder = unreal.uproperty(unreal.DirectoryPath, meta={
        "Category": "Clip", "DisplayName": "Clip Folder",
        "ToolTip": "PlateCraft clip folder on disk with manifest.json and atlas/ (from `platecraft pack`, "
                   "e.g. <repo>/work/kitty). Not the Unreal content folder"})
    asset_name = unreal.uproperty(str, meta={
        "Category": "Clip", "DisplayName": "Name",
        "ToolTip": "Asset/actor name. Empty = the clip folder's name"})
    destination = unreal.uproperty(str, meta={
        "Category": "Clip", "DisplayName": "Content Folder",
        "ToolTip": "Content folder for the shared material/mesh; each clip gets a subfolder"})

    place = unreal.uproperty(bool, meta={
        "Category": "Placement", "DisplayName": "Place Media Plate",
        "ToolTip": "Place (or update) the Media Plate actor PlateCraft_<name> in the open level"})
    at_camera = unreal.uproperty(bool, meta={
        "Category": "Placement", "DisplayName": "In Front of Viewport Camera",
        "EditCondition": "place",
        "ToolTip": "Centre the plate on the viewport camera's view line at Distance, upright and facing "
                   "the camera. Off = use Location/Yaw"})
    distance = unreal.uproperty(float, meta={
        "Category": "Placement", "DisplayName": "Distance (cm)", "EditCondition": "place && at_camera",
        "ClampMin": "10", "ToolTip": "Distance from the viewport camera"})
    location = unreal.uproperty(unreal.Vector, meta={
        "Category": "Placement", "EditCondition": "place && !at_camera",
        "ToolTip": "Media Plate location in cm (pivot = bottom centre of the video frame)"})
    yaw = unreal.uproperty(float, meta={
        "Category": "Placement", "EditCondition": "place && !at_camera",
        "ToolTip": "Media Plate yaw in degrees; 0 = front faces -X"})
    height = unreal.uproperty(float, meta={
        "Category": "Placement", "DisplayName": "Height (cm)", "ClampMin": "1",
        "ToolTip": "Height of the full video frame in cm"})

    sequence = unreal.uproperty(bool, meta={
        "Category": "Sequencer", "DisplayName": "Create Level Sequence",
        "EditCondition": "place",
        "ToolTip": "Level Sequence LS_<name> with a media track bound to the plate (frame-accurate in "
                   "Movie Render Queue). Off = the plate autoplays and loops"})
    camera = unreal.uproperty(bool, meta={
        "Category": "Sequencer", "DisplayName": "Add Camera", "EditCondition": "place && sequence",
        "ToolTip": "CineCamera PlateCraftCam_<name> framing the video frame exactly (35 mm, f/8, focus on the plate) "
                   "plus a Camera Cuts track on it. An existing camera / Camera Cuts track is kept"})
    streaming_radius = unreal.uproperty(float, meta={
        "Category": "Sequencer", "DisplayName": "Camera Streaming Radius (m)", "EditCondition": "place && sequence && camera",
        "ClampMin": "0",
        "ToolTip": "World Partition: load everything within this distance of the camera in PIE / Movie Render "
                   "Queue, so distant terrain the camera sees renders. 0 = the player's loading range only"})
    open_sequence = unreal.uproperty(bool, meta={
        "Category": "Sequencer", "DisplayName": "Open in Sequencer", "EditCondition": "place && sequence"})

    grid = unreal.uproperty(int, meta={
        "Category": "Material", "DisplayName": "Card Grid Columns", "ClampMin": "1", "ClampMax": "512",
        "ToolTip": "Card mesh columns for depth bending (rows follow the aspect ratio)"})
    set_depth_amount = unreal.uproperty(bool, meta={
        "Category": "Material", "InlineEditConditionToggle": "true"})
    depth_amount = unreal.uproperty(float, meta={
        "Category": "Material", "DisplayName": "Depth Amount (cm)", "EditCondition": "set_depth_amount",
        "ToolTip": "Card bend in cm (MI DepthAmount). Unticked = keep the instance's current value"})
    rebuild_material = unreal.uproperty(bool, meta={
        "Category": "Material", "DisplayName": "Rebuild Master Material",
        "ToolTip": "Rebuild M_PlateCraft even if its version is current"})

    no_light_shafts = unreal.uproperty(bool, meta={
        "Category": "Lighting", "DisplayName": "Turn Off Light Shaft Occlusion",
        "ToolTip": "Turn off Light Shaft Occlusion on the level's directional lights. It's screen-space: the "
                   "plate in front of the sky darkens wedges of the scene behind it, spreading away from the sun"})


DEFAULTS = {
    "clip_folder": "", "asset_name": "", "destination": sli.DEST, "place": True, "at_camera": True,
    "distance": 400.0, "location": [0.0, 0.0, 0.0], "yaw": 0.0, "height": 200.0, "sequence": True,
    "camera": True, "streaming_radius": sli.STREAMING_RADIUS / 100.0, "open_sequence": True, "grid": 128, "set_depth_amount": False, "depth_amount": 0.0,
    "rebuild_material": False, "no_light_shafts": False,
}
_options = None  # keeps the options object (and the last dialog's values) alive


def _settings_path() -> str:
    return os.path.join(unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_saved_dir()),
                        "PlateCraft", "import_dialog.json")


def _load_settings() -> dict:
    s = dict(DEFAULTS)
    try:
        with open(_settings_path(), encoding="utf-8") as f:
            s.update({k: v for k, v in json.load(f).items() if k in DEFAULTS})
    except (OSError, ValueError):
        pass
    return s


def _save_settings(s: dict) -> None:
    try:
        os.makedirs(os.path.dirname(_settings_path()), exist_ok=True)
        with open(_settings_path(), "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2)
    except OSError as e:
        sli.warn(f"could not save dialog settings: {e}")


def _to_options(s: dict, o: PlateCraftImportOptions) -> None:
    for k, v in s.items():
        if k == "clip_folder":
            v = unreal.DirectoryPath(path=v)
        elif k == "location":
            v = unreal.Vector(*v)
        o.set_editor_property(k, v)


def _from_options(o: PlateCraftImportOptions) -> dict:
    s = {k: o.get_editor_property(k) for k in DEFAULTS}
    s["clip_folder"] = s["clip_folder"].path.strip().strip('"')
    s["location"] = [s["location"].x, s["location"].y, s["location"].z]
    s["asset_name"] = s["asset_name"].strip()
    s["destination"] = s["destination"].strip().rstrip("/") or sli.DEST
    return s


def _message(text: str, error: bool = False) -> None:
    unreal.EditorDialog.show_message(
        TITLE, text, unreal.AppMsgType.OK, unreal.AppReturnType.OK,
        unreal.AppMsgCategory.ERROR if error else unreal.AppMsgCategory.INFO)


def _norm(path: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(path)))


def _validate(s: dict) -> str | None:
    """Check the dialog input; may fix up s["clip_folder"] (atlas/ or a pass folder -> the clip folder)."""
    clip = s["clip_folder"]
    if not clip:
        return "Choose a clip folder (the folder that holds manifest.json and atlas/)."
    if not os.path.isdir(clip):
        return f"Folder not found:\n{clip}"
    parent = os.path.dirname(os.path.normpath(clip))
    if not os.path.isfile(os.path.join(clip, "manifest.json")) and \
            os.path.isfile(os.path.join(parent, "manifest.json")):
        clip = s["clip_folder"] = parent.replace("\\", "/")  # picked atlas/ (or another pass folder)
    content = _norm(unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_content_dir()))
    if _norm(clip) == content or _norm(clip).startswith(content + os.sep):
        return (f"{clip}\nis inside this project's Content folder, which holds the imported Unreal assets."
                "\n\nChoose the PlateCraft clip folder instead: the output of `platecraft extract` / "
                "`pack` that holds manifest.json and atlas/ (e.g. <repo>/work/kitty).")
    try:
        sli.load_manifest(clip)
    except (RuntimeError, ValueError, KeyError) as e:
        return str(e)
    if not s["destination"].startswith("/Game"):
        return f"Content Folder must be under /Game (got {s['destination']})."
    return None


def _camera_placement(distance: float, height: float):
    """Bottom-centre location and yaw that put the plate `distance` in front of the viewport camera,
    centred on the view line and facing the camera (the card's front is -X at yaw 0)."""
    cam_loc, cam_rot = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_level_viewport_camera_info()
    fwd = cam_rot.get_forward_vector()
    flat = unreal.Vector(fwd.x, fwd.y, 0.0)
    if flat.length() < 1e-3:  # looking straight up/down: use the camera's yaw
        flat = unreal.Rotator(0.0, cam_rot.yaw, 0.0).get_forward_vector()
    flat = flat.normal()
    centre = cam_loc + fwd * distance
    loc = (centre.x, centre.y, centre.z - height / 2.0)
    yaw = math.degrees(math.atan2(flat.y, flat.x))  # plate front (-X) faces back along the view
    return loc, yaw


def run_import(s: dict) -> dict:
    """Import with dialog settings `s` (keys as in DEFAULTS). Returns platecraft_import.import_clip's dict."""
    if s["place"] and s["at_camera"]:
        loc, yaw = _camera_placement(s["distance"], s["height"])
    else:
        loc, yaw = tuple(s["location"]), s["yaw"]
    with unreal.ScopedSlowTask(1, f"PlateCraft: importing {os.path.basename(s['clip_folder'])}") as task:
        task.make_dialog(False)
        task.enter_progress_frame(1)
        out = sli.import_clip(
            s["clip_folder"], s["asset_name"] or None, s["destination"], loc, yaw, s["height"], s["grid"],
            s["depth_amount"] if s["set_depth_amount"] else None,
            place=s["place"], sequence=s["place"] and s["sequence"], rebuild_material=s["rebuild_material"],
            camera=s["camera"], no_light_shafts=s["no_light_shafts"],
            streaming_radius=s["streaming_radius"] * 100.0)
    if out["plate"]:
        unreal.get_editor_subsystem(unreal.EditorActorSubsystem).set_selected_level_actors([out["plate"]])
    if out["sequence"] and s["open_sequence"]:
        unreal.LevelSequenceEditorBlueprintLibrary.open_level_sequence(out["sequence"])
    return out


def _summary(s: dict, out: dict) -> str:
    m = sli.load_manifest(s["clip_folder"])
    a = m["atlas"]
    num, den = sli.frame_rate(m)
    clip = os.path.basename(s["clip_folder"].rstrip("/\\"))
    lines = [f"Imported {clip}: {a['frame_count']} frames at "
             f"{num / den:.3f} fps, atlas {a['width']}x{a['height']}.", ""]
    for key, label in (("source", "Media source"), ("instance", "Material instance"), ("sequence", "Sequence")):
        if out.get(key):
            lines.append(f"{label}: {out[key].get_path_name().split('.')[0]}")
    if out.get("plate"):
        lines.append(f"Media Plate: {out['plate'].get_actor_label()} (selected)")
    if out.get("camera"):
        lines.append(f"Camera: {out['camera'].get_actor_label()} (Camera Cuts in the sequence)")
    if out.get("sequence") and not s["open_sequence"]:
        lines += ["", "The plate shows video only while the sequence drives it: open "
                      f"{out['sequence'].get_name()} in Sequencer (the plate stays dark until then)."]
    lic = [l for l in m.get("licenses", []) if l.get("commercial") != "ok"]
    if lic:
        lines += ["", "License notes:"]
        lines += [f"- {l['model']}: {l['license']} ({l['commercial']})" for l in lic]
    if out.get("sequence") and a["width"] * a["height"] > 1920 * 1080 * 1.1:
        lines += ["", f"A {a['width']}x{a['height']} atlas is too slow for real-time Sequencer playback in the "
                      "editor; renders are frame-accurate anyway. For real-time preview use "
                      "`platecraft pack --scale 0.5`."]
    return "\n".join(lines)


def show_import_dialog() -> dict | None:
    """Show the import dialog; re-shows it with an error until the input is valid or cancelled."""
    global _options
    importlib.reload(sli)  # pick up edits to the importer without restarting the editor
    if _options is None:
        _options = unreal.new_object(PlateCraftImportOptions)
        _to_options(_load_settings(), _options)
    view = unreal.EditorDialogLibraryObjectDetailsViewOptions(
        show_object_name=False, allow_search=False, allow_resizing=True, min_width=560, min_height=520,
        value_column_width_ratio=0.55)
    while True:
        if not unreal.EditorDialog.show_object_details_view(TITLE + " (OK = Import)", _options, view):
            return None
        s = _from_options(_options)
        err = _validate(s)
        _options.set_editor_property("clip_folder", unreal.DirectoryPath(path=s["clip_folder"]))
        _save_settings(s)
        if err:
            _message(err, error=True)
            continue
        try:
            out = run_import(s)
        except Exception as e:  # keep the editor alive and show the reason
            unreal.log_error(f"[PlateCraft] import failed: {e}")
            _message(f"Import failed:\n{e}\n\nSee the Output Log for details.", error=True)
            raise
        _message(_summary(s, out))
        return out


def register_menu() -> None:
    """Add Tools > PlateCraft > Import Clip... (safe to call again)."""
    menus = unreal.ToolMenus.get()
    menu = menus.find_menu(MENU)
    if not menu:
        sli.warn(f"menu {MENU} not found; use platecraft_ui.show_import_dialog()")
        return
    menu.add_section(SECTION, SECTION)
    entry = unreal.ToolMenuEntry(name=ENTRY, type=unreal.MultiBlockType.MENU_ENTRY)
    entry.set_label("Import Clip...")
    entry.set_tool_tip("Import a PlateCraft clip folder (manifest.json + atlas/) as a Media Plate with a "
                       "PBR material and a Level Sequence")
    entry.set_string_command(unreal.ToolMenuStringCommandType.PYTHON, "", COMMAND)
    menu.add_menu_entry(SECTION, entry)
    menus.refresh_all_widgets()
