"""Editor startup script: adds Tools > PlateCraft > Import Clip... (see platecraft_ui.py).

Unreal runs every init_unreal.py found on its Python paths at editor startup. If the project already
has its own init_unreal.py in the same folder, call platecraft_ui.register_menu() from that one instead.
"""
import unreal

try:
    import platecraft_ui
    platecraft_ui.register_menu()
except Exception as e:  # never break editor startup
    unreal.log_warning(f"[PlateCraft] could not register the import menu: {e}")
