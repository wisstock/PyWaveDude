# SPDX-License-Identifier: GPL-3.0-or-later
# PyWaveDude: Python port of WaveDude Next (C) Alexander Graef, after Jeroen Vleggaar (Huygens Optics) and 0x23.
"""Dear PyGui front end: the project tree and properties on the left, the pictures, timeline and transport in the
middle, the probe graphs on the right, the status bar at the bottom.

Live editing: an edit of the scene, a source or a part stops playback and a run in progress; when the edit has
finished (the mouse released, a field left) it becomes an undo step and, after a run, a new run starts from step 0
while the timeline stays where it is: the picture there changes when the new run reaches it."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from types import SimpleNamespace

import dearpygui.dearpygui as dpg
import numpy as np

from . import config
from . import project as project_io
from . import ui_theme as th
from .engine import PointSource
from .export import glowing_edges
from .history import UndoHistory
from .parts import PART_TYPES
from .parts import model as pm
from .project import Project, build_engine, paint_scene
from .render import builtin_schemes
from .runner import SimulationRun
from .ui_dialogs import DialogsMixin
from .ui_export import ExportMixin
from .ui_preview import PREVIEW, PreviewMixin
from .ui_probes import ProbesMixin
from .ui_timeline import TimelineMixin
from .ui_tree import TreeMixin

IMAGE_EXT = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff", ".gif")
EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
STATUS_H = 28
SIM_KINDS = ("source", "part", "scene", "sim")


def LEFT_W() -> int:
    return int(config.app("ui.left_pane_width", 400))


def ZOOMS() -> tuple:
    return tuple(float(z) for z in config.app("ui.zoom_choices", (1.0,)))


class App(TreeMixin, PreviewMixin, TimelineMixin, ProbesMixin, ExportMixin, DialogsMixin):
    def __init__(self, image: str | None = None, backend: str = "auto"):
        self.backend = backend
        self.rgb: np.ndarray | None = None
        if image and image.lower().endswith(".wdproj"):
            self.p = project_io.load(image)
            self.rgb = self.p.scene_rgb()
            self.p.sim.backend = backend
        else:
            self.p = self._new_project()
            if image:
                self.rgb = self.p.embed_image(image)
        self.run: SimulationRun | None = None
        self.run_stale = False
        self.selection: list = []
        self.current = None
        self.place_mode = False
        self.place_part: str | None = None
        self._init_tree()
        self._init_preview()
        self._init_timeline()
        self._init_probes()
        self._init_export()
        self._init_dialogs()
        self.history = UndoHistory(self.p)
        self._sim_edit = False
        self._probe_edit = False
        self._scene_edges: np.ndarray | None = None
        self._shown_frame = None
        self._shown_key = None
        self._dirty_view = True
        self._model_version = 0
        self._editor_version = -1
        self._end_edit_now = False
        self._old_textures: list = []
        self._message_text, self._message_level = "", "info"
        self.recent: list[str] = [self.p.path] if self.p.path else []
        self.zoom = float(config.app("ui.zoom", 1.0))
        self._texture_registry = None

    # ================================================================= model
    def _new_project(self) -> Project:
        p = config.new_project()
        p.sim.backend = self.backend
        return p

    def _schemes(self) -> dict:
        return {**builtin_schemes(), **{cs.name: cs for cs in self.p.color_schemes}}

    def _scene_label(self) -> str:
        s = self.p.scene
        name = s.image_name or Path(s.image_path).name
        return f"Scene image: {name}" if s.use_image else "Blank canvas"

    def _edited(self, kind: str) -> None:
        """Something in the project changed by the user's hand (``kind`` of the object, or "view")."""
        self.history.edit_made()
        self._dirty_view = True
        if kind in SIM_KINDS:
            self._sim_edit = True
            if self.run is not None:
                self.run_stale = True
                if self.run.running:
                    self.run.stop()
            if self.playing:
                self._set_playing(False)
        elif kind == "probe":
            self._probe_edit = True

    def _model_changed(self) -> None:
        self._model_version += 1
        self._overlays_dirty = True

    def _message(self, text: str, level: str = "info") -> None:
        self._message_text, self._message_level = text, level

    _status = _message

    # ================================================================= project files
    def _set_project(self, p: Project) -> None:
        """Show another project: every pane rebuilt from it, nothing computed."""
        self.stop()
        p.sim.backend = self.backend
        self.p = p
        self.rgb = p.scene_rgb()
        self.selection, self.current = [], None
        self._editor_obj = None
        self._luts.clear()
        self._graph_key = None
        self._scene_edges = None
        self.history.start(p)
        self._sim_edit = self._probe_edit = False
        self._set_position(0, by_user=False)
        self.view_zoom, self.view_x, self.view_y = 1.0, 0.0, 0.0
        self._layout_dirty = True
        self._refresh_tree()
        self._show_props()
        self._sync_transport()
        self._update_title()
        self._model_changed()
        if p.path:
            self._remember(p.path)

    def _replace_project(self, new: Project) -> None:
        """Undo / redo: the project as it was, the selection carried over by list and position."""
        keep = [(self.kind_of(o), self.index_in(self.list_of(o), o)) for o in self.selection
                if self.list_of(o) is not None]
        cur = self.current
        cur_key = (self.kind_of(cur), self.index_in(self.list_of(cur), cur)) if self.list_of(cur) is not None \
            else self.kind_of(cur)
        new.path = self.p.path
        new.sim.backend = self.p.sim.backend
        self.p = new
        self.rgb = new.scene_rgb()
        lists = {"source": new.sources, "part": new.parts, "probe": new.probes, "scheme": new.color_schemes}
        self.selection = [lists[k][i] for k, i in keep if 0 <= i < len(lists[k])]
        if isinstance(cur_key, tuple):
            k, i = cur_key
            self.current = lists[k][i] if 0 <= i < len(lists[k]) else None
        else:
            self.current = {"project": new, "scene": new.scene, "sim": new.sim, "display": new.display}.get(cur_key)
        self._editor_obj = None
        self._luts.clear()
        self._layout_dirty = True
        self._refresh_tree()
        self._show_props()
        self._sync_transport()
        self._update_title()
        self._model_changed()
        self._dirty_view = True
        if self.run is not None:
            self.run_stale = True
            if self.p.sim.rerun_on_edit:
                self._rerun()

    def _undo(self, *_):
        new = self.history.undo(self.p)
        if new is not None:
            self._replace_project(new)
            self._message("Undo")

    def _redo(self, *_):
        new = self.history.redo(self.p)
        if new is not None:
            self._replace_project(new)
            self._message("Redo")

    def _update_title(self) -> None:
        name = Path(self.p.path).name if self.p.path else "untitled"
        dirty = "*" if self.history.dirty else ""
        dpg.set_viewport_title(f"PyWaveDude — {name}{dirty}")

    def _remember(self, path: str) -> None:
        path = str(path)
        self.recent = [path] + [r for r in self.recent if r != path][:9]
        self._build_recent_menu()

    def _new_cb(self, *_):
        self._set_project(self._new_project())

    def _open_path(self, path: str) -> None:
        try:
            p = project_io.load(path)
        except Exception as exc:
            self._message(f"Cannot open the project: {exc}", "error")
            return
        self._set_project(p)
        self._message(f"Opened {path}")

    def _open_example(self, path: str) -> None:
        self._open_path(path)
        self.p.path = ""                  # an example opens as an unsaved project
        self._update_title()

    def _open_project_cb(self, _s, data) -> None:
        self._open_path(data["file_path_name"])

    def _save_cb(self, *_):
        if not self.p.path:
            dpg.show_item("save_dlg")
            return
        self._save_to(self.p.path)

    def _save_as_cb(self, _s, data) -> None:
        path = data["file_path_name"]
        if not path.lower().endswith(".wdproj"):
            path += ".wdproj"
        self._save_to(path)

    def _save_to(self, path: str) -> None:
        try:
            self.history.end_step(self.p)
            project_io.save(self.p, path)
        except Exception as exc:
            self._message(f"Cannot save: {exc}", "error")
            return
        self.history.mark_saved()
        self._remember(path)
        self._update_title()
        self._message(f"Saved {path}")

    def _open_cb(self, _s, data) -> None:
        try:
            self.rgb = self.p.embed_image(data["file_path_name"])
        except Exception as exc:
            self._message(f"Cannot open the image: {exc}", "error")
            return
        self._editor_obj = None
        self._show_props()
        self._layout_dirty = True
        self._edited("scene")
        self._model_changed()

    def _use_blank(self, *_):
        self.p.scene.use_image = False
        self._editor_obj = None
        self._show_props()
        self._layout_dirty = True
        self._edited("scene")
        self._model_changed()

    def _save_png(self, *_):
        tex = self._tex.get(self._shown_layers()[0])
        if tex is None:
            return
        from PIL import Image
        out = Path.cwd() / f"pywavedude_frame_{int(time.time())}.png"
        Image.fromarray((np.clip(tex[2][..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)).save(out)
        self._message(f"Saved {out}")

    # ================================================================= runs
    def _build_engine(self):
        image = paint_scene(self.p, self.rgb if self.p.scene.use_image else None)
        self._scene_edges = glowing_edges(image) if image is not None else None
        return build_engine(self.p, painted=image)

    def start(self, *_) -> None:
        """Run (F5): compute the simulation from step 0; the timeline returns to 0."""
        if self._ask_memory(self._start_now):
            self._start_now()

    def _rerun(self) -> None:
        """The automatic re-run after an edit: from step 0, the timeline kept where it is."""
        if self._ask_memory(lambda: self._start_now(keep_position=True)):
            self._start_now(keep_position=True)

    def _start_now(self, keep_position: bool = False) -> None:
        if self.run is not None:
            self.run.stop()
        try:
            eng = self._build_engine()
        except Exception as exc:
            self._message(f"Cannot build the engine: {exc}", "error")
            return
        m = self.p.sim
        self.run = SimulationRun(eng, m.steps, m.visualize_every * self._div(), PREVIEW[m.preview],
                                 memory_budget=self.memory_budget(), sampler=self._make_sampler(eng))
        self.run.start()
        if self._graph_probes and config.app("ui.show_probe_graphs", True) and not self._graphs_hidden_by_user:
            self._show_graphs(True, by_user=False)
        self.run_stale = False
        self._sim_edit = self._probe_edit = False
        self._graph_key = None
        if not keep_position:
            self._set_position(0, by_user=False)
            self._shown_key = None
        self._message(f"Run started: {eng.description}")

    def stop(self, *_) -> None:
        """Stop: end the computation and discard it; the timeline returns to 0."""
        if self.run is not None:
            self.run.stop()
        self.run = None
        self.run_stale = False
        self._set_playing(False)
        self._set_position(0, by_user=False)
        self._shown_key = None
        self._dirty_view = True

    # ================================================================= placing new objects
    def _arm(self, what: str, value: str | None = None) -> None:
        """The next canvas click places a new object: "source", "part" (``value`` = kind) or "probe" (kind)."""
        self._disarm()
        if what == "source":
            self.place_mode = True
        elif what == "part":
            self.place_part = value
        else:
            self.place_probe = {"kind": value, "a": None}
        self._place_labels()

    def _disarm(self, *_):
        self.place_mode = False
        self.place_part = None
        self.place_probe = None
        self._place_labels()

    def _place_labels(self) -> None:
        pass                              # the status bar shows what is being placed

    def _placing(self) -> str:
        if self.place_part:
            return self.place_part
        if self.place_mode:
            return "source"
        if self.place_probe:
            return self._probe_button_label()
        return ""

    def _place_at(self, cx: int, cy: int) -> None:
        if self.place_part:
            stem = self.place_part
            cls = PART_TYPES[stem]
            part = config.new_object(cls, name=self._unique_name(self.p.parts, f"{stem} 1"), x=max(0, cx),
                                     y=max(0, cy))
            if hasattr(part, "frequency_scale"):
                part.frequency_scale = self.p.sim.source_frequency_scale
            if hasattr(part, "project_seed"):
                part.project_seed = self.p.seed
            self.p.parts.append(part)
            self._disarm()
            self._added(part)
        elif self.place_probe:
            self._probe_click(cx, cy)
        elif self.place_mode:
            src = config.new_object(PointSource, x=cx, y=cy,
                                    name=self._unique_name(self.p.sources, f"Source {len(self.p.sources) + 1}"))
            self.p.sources.append(src)
            self._disarm()
            self._added(src)

    def _added(self, obj) -> None:
        self._refresh_tree()
        self._select([obj], current=obj)
        self._edited(self.kind_of(obj))
        self._model_changed()

    def _gate_sources(self, gate: pm.LogicGatePart) -> None:
        ports = gate.ports()
        if ports is None:
            return
        added = []
        for label, (px, py) in zip("AB", ports[:2]):
            back = gate.to_canvas(-17.0, 0.0)
            dx, dy = back[0] - gate.x, back[1] - gate.y
            src = PointSource(int(px + dx), int(py + dy), name=f"{gate.name} {label}", level=gate.design_frequency)
            self.p.sources.append(src)
            added.append(src)
        self._refresh_tree()
        self._select(added)
        self._edited("source")
        self._model_changed()

    # ================================================================= clipboard
    def _clip_objects(self) -> list:
        return [o for o in (self.selection or [self.current]) if self.list_of(o) is not None]

    def _copy(self, *_) -> bool:
        objs = self._clip_objects()
        if not objs:
            return False
        dpg.set_clipboard_text(project_io.items_to_xml(objs))
        self._message(f"Copied {len(objs)} object(s)")
        return True

    def _cut(self, *_):
        objs = self._clip_objects()
        if self._copy():
            self._remove_objects(objs)

    def _paste(self, *_, offset: int = 0, text: str | None = None):
        try:
            items = project_io.items_from_xml(text if text is not None else dpg.get_clipboard_text())
        except Exception:
            self._message("The clipboard holds no project objects", "warning")
            return
        added = []
        for attr in ("sources", "parts", "probes", "color_schemes"):
            lst = getattr(self.p, attr)
            at = len(lst)
            if self.current is not None and self.list_of(self.current) is lst:
                at = self.index_in(lst, self.current) + 1
            for obj in items[attr]:
                if attr == "color_schemes":
                    from dataclasses import replace
                    obj = replace(obj, name=self._unique_name(lst, obj.name))
                else:
                    obj.name = self._unique_name(lst, obj.name)
                    if offset:
                        PreviewMixin._translate(obj, copy.copy(obj), offset, offset)
                    if attr == "parts":
                        project_io._attach(self.p, obj)
                lst.insert(at, obj)
                at += 1
                added.append(obj)
        if added:
            self._refresh_tree()
            self._select(added)
            for o in added:
                self._edited(self.kind_of(o))
            self._model_changed()
            self._message(f"Pasted {len(added)} object(s)")

    def _duplicate(self, *_):
        objs = self._clip_objects()
        if objs:
            self._paste(offset=20, text=project_io.items_to_xml(objs))

    def _select_all(self, *_):
        objs = [*self.p.sources, *self.p.parts, *self.p.probes]
        self._select(objs, current=None)

    def _delete(self, *_):
        self._remove_selected()

    # ================================================================= building
    def build(self) -> None:
        dpg.create_context()
        # every callback runs on this thread, between frames (Dear PyGui's own callback thread raced the frame
        # loop: a form rebuilt by a checkbox while the loop read its widgets)
        dpg.configure_app(manual_callback_management=True)
        th.load_fonts(self.zoom)
        th.apply_theme()
        dpg.create_viewport(title="PyWaveDude", width=int(config.app("ui.window_width", 1600)),
                            height=int(config.app("ui.window_height", 980)))
        self._texture_registry = dpg.add_texture_registry()
        with dpg.window(tag="main"):
            self._build_menus()
            with dpg.group(horizontal=True):
                with dpg.child_window(width=int(LEFT_W() * self.zoom), height=-STATUS_H, tag="left", border=False):
                    with dpg.child_window(height=int(config.app("ui.tree_pane_height", 330)), tag="tree_pane",
                                          resizable_y=True):
                        self._build_tree_pane()
                    with dpg.child_window(tag="props_pane"):
                        self._build_props_pane()
                with dpg.child_window(tag="center", width=-1, height=-STATUS_H, no_scrollbar=True):
                    dpg.add_group(tag="pics")
                    self._build_timeline()
                self._build_graph_column()
            with dpg.child_window(height=STATUS_H - 4, tag="statusbar", no_scrollbar=True, border=False):
                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="status_msg")
                    dpg.add_spacer(width=30)
                    dpg.add_text("", tag="status_run", color=(220, 235, 250))
            dpg.bind_item_theme("statusbar", "theme_status")
        self._build_export_window()
        with dpg.file_dialog(tag="proj_dlg", show=False, callback=self._open_project_cb, width=700, height=450):
            dpg.add_file_extension("WaveDude projects{.wdproj}")
        with dpg.file_dialog(tag="save_dlg", show=False, callback=self._save_as_cb, width=700, height=450,
                             default_filename="project"):
            dpg.add_file_extension(".wdproj")
        with dpg.file_dialog(tag="open_dlg", show=False, callback=self._open_cb, width=700, height=450):
            dpg.add_file_extension("Images{" + ",".join(IMAGE_EXT) + "}")
            dpg.add_file_extension(".*")
        with dpg.handler_registry():
            dpg.add_key_press_handler(callback=self._on_key)
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=self._on_click)
            dpg.add_mouse_double_click_handler(button=dpg.mvMouseButton_Left, callback=self._on_double_click)
            dpg.add_mouse_wheel_handler(callback=self._on_wheel)
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Right, callback=self._on_pan_click)
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Middle, callback=self._on_pan_click)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main", True)
        self._refresh_tree()
        self._show_props()
        self._build_recent_menu()
        self._update_title()

    def _build_menus(self) -> None:
        with dpg.menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="New project", shortcut="Ctrl+N", callback=self._new_cb)
                dpg.add_menu_item(label="Open project...", shortcut="Ctrl+O",
                                  callback=lambda: dpg.show_item("proj_dlg"))
                with dpg.menu(label="Recent projects", tag="recent_menu"):
                    pass
                with dpg.menu(label="Examples"):
                    for f in sorted(EXAMPLES.glob("*.wdproj")) if EXAMPLES.exists() else []:
                        dpg.add_menu_item(label=f.stem, user_data=str(f),
                                          callback=lambda s, a, u: self._open_example(u))
                dpg.add_menu_item(label="Save", shortcut="Ctrl+S", callback=self._save_cb)
                dpg.add_menu_item(label="Save as...", shortcut="Ctrl+Shift+S",
                                  callback=lambda: dpg.show_item("save_dlg"))
                dpg.add_separator()
                dpg.add_menu_item(label="Open scene image...", callback=lambda: dpg.show_item("open_dlg"))
                dpg.add_menu_item(label="Blank canvas", callback=self._use_blank)
                dpg.add_separator()
                with dpg.menu(label="Export"):
                    dpg.add_menu_item(label="Field...", callback=lambda: self._show_export("Field"))
                    dpg.add_menu_item(label="Intensity...", callback=lambda: self._show_export("Intensity"))
                    dpg.add_menu_item(label="Both...", callback=lambda: self._show_export("Both"))
                    dpg.add_separator()
                    dpg.add_menu_item(label="Field frame", callback=lambda: self._show_export("Field", True))
                    dpg.add_menu_item(label="Intensity frame", callback=lambda: self._show_export("Intensity", True))
                    dpg.add_separator()
                    dpg.add_menu_item(label="Probe data as CSV", callback=lambda: self._export_csv())
                dpg.add_menu_item(label="Export last", shortcut="Ctrl+E", callback=self._export_last)
                dpg.add_menu_item(label="Save shown picture as PNG", callback=self._save_png)
                dpg.add_separator()
                dpg.add_menu_item(label="Exit", callback=lambda: dpg.stop_dearpygui())
            with dpg.menu(label="Edit"):
                dpg.add_menu_item(label="Undo", shortcut="Ctrl+Z", callback=self._undo)
                dpg.add_menu_item(label="Redo", shortcut="Ctrl+Y", callback=self._redo)
                dpg.add_separator()
                dpg.add_menu_item(label="Cut", shortcut="Ctrl+X", callback=self._cut)
                dpg.add_menu_item(label="Copy", shortcut="Ctrl+C", callback=self._copy)
                dpg.add_menu_item(label="Paste", shortcut="Ctrl+V", callback=self._paste)
                dpg.add_menu_item(label="Duplicate", shortcut="Ctrl+D", callback=self._duplicate)
                dpg.add_menu_item(label="Delete", shortcut="Del", callback=self._delete)
                dpg.add_menu_item(label="Select all", shortcut="Ctrl+A", callback=self._select_all)
            with dpg.menu(label="View"):
                with dpg.menu(label="Panes"):
                    for label, tag, item in (("Project tree", "view_tree", "tree_pane"),
                                             ("Properties", "view_props", "props_pane"),
                                             ("Status bar", "view_status", "statusbar")):
                        dpg.add_menu_item(label=label, tag=tag, check=True, default_value=True, user_data=item,
                                          callback=lambda s, v, u: self._show_pane(u, v))
                    dpg.add_menu_item(label="Probe graphs", tag="view_graphs", check=True, default_value=False,
                                      callback=lambda s, v: self._show_graphs(v))
                with dpg.menu(label="Preview resolution"):
                    for r in PREVIEW:
                        dpg.add_menu_item(label=r, check=True, tag=f"res_{r}", user_data=r,
                                          callback=lambda s, a, u: self._set_sim("preview", u))
                with dpg.menu(label="Preview frames"):
                    for n, label in ((1, "Every frame"), (2, "Every second frame"), (4, "Every fourth frame")):
                        dpg.add_menu_item(label=label, check=True, tag=f"frames_{n}", user_data=n,
                                          callback=lambda s, a, u: self._set_sim("preview_frames", u))
                with dpg.menu(label="UI zoom"):
                    for z in ZOOMS():
                        dpg.add_menu_item(label=f"{z:g}×", check=True, tag=f"zoom_{z}", user_data=z,
                                          callback=lambda s, a, u: self._set_zoom(u))
                dpg.add_menu_item(label="Zoom in", shortcut="Ctrl+=", callback=lambda: self._zoom_at(1.25))
                dpg.add_menu_item(label="Zoom out", shortcut="Ctrl+-", callback=lambda: self._zoom_at(0.8))
                dpg.add_menu_item(label="Whole canvas", shortcut="Ctrl+0", callback=self._zoom_fit)
                dpg.add_separator()
                for label, attr in (("Grid", "grid"), ("Handles in the preview", "handles"),
                                    ("Labels in the preview", "labels"),
                                    ("Graphs on the canvas", "graphs_in_preview")):
                    dpg.add_menu_item(label=label, check=True, tag=f"view_{attr}", user_data=attr,
                                      callback=lambda s, v, u: self._set_display(u, v))
            with dpg.menu(label="Simulation"):
                dpg.add_menu_item(label="Run", shortcut="F5", callback=self.start)
                dpg.add_menu_item(label="Pause / resume run", shortcut="F6", callback=self._toggle_run_pause)
                dpg.add_menu_item(label="Stop", shortcut="Shift+F5", callback=self.stop)
                dpg.add_menu_item(label="Play / Pause", shortcut="Space", callback=self._toggle_play)
                dpg.add_separator()
                with dpg.menu(label="Compute"):
                    from .engine import available_backends
                    for b in ["auto"] + available_backends():
                        dpg.add_menu_item(label=b, check=True, tag=f"backend_{b}", user_data=b,
                                          callback=lambda s, a, u: self._set_sim("backend", u))
                dpg.add_menu_item(label="Re-run automatically after edits", check=True, tag="menu_rerun",
                                  callback=lambda s, v: self._set_sim("rerun_on_edit", v))
                dpg.add_menu_item(label="Separate wavelengths (true colour)", check=True, tag="menu_split",
                                  callback=lambda s, v: self._set_sim("split_wavelengths", v))
                dpg.add_menu_item(label="Warn when a run does not fit in memory", check=True, tag="menu_warn",
                                  default_value=self.warn_memory, callback=lambda s, v: setattr(self, "warn_memory", v))
            with dpg.menu(label="Help"):
                dpg.add_menu_item(label="Manual", shortcut="F1", callback=self._show_manual)
                dpg.add_menu_item(label="About PyWaveDude", callback=self._show_about)

    def _build_recent_menu(self) -> None:
        if not dpg.does_item_exist("recent_menu"):
            return
        dpg.delete_item("recent_menu", children_only=True)
        for r in self.recent:
            dpg.add_menu_item(label=r, parent="recent_menu", user_data=r, callback=lambda s, a, u: self._open_path(u))

    def _sync_menus(self) -> None:
        m, d = self.p.sim, self.p.display
        for r in PREVIEW:
            dpg.set_value(f"res_{r}", m.preview == r)
        for n in (1, 2, 4):
            dpg.set_value(f"frames_{n}", m.preview_frames == n)
        for z in ZOOMS():
            dpg.set_value(f"zoom_{z}", abs(self.zoom - z) < 1e-6)
        from .engine import available_backends
        for b in ["auto"] + available_backends():
            dpg.set_value(f"backend_{b}", m.backend == b)
        dpg.set_value("menu_rerun", m.rerun_on_edit)
        dpg.set_value("menu_split", m.split_wavelengths)
        for attr in ("grid", "handles", "labels", "graphs_in_preview"):
            dpg.set_value(f"view_{attr}", getattr(d, attr))
        self._sync_transport()

    def _set_sim(self, attr: str, value) -> None:
        setattr(self.p.sim, attr, value)
        if self._editor is not None:
            self._editor.refresh()
        self._edited("view" if attr in ("backend", "rerun_on_edit") else "sim")
        self._model_changed()

    def _set_display(self, attr: str, value) -> None:
        setattr(self.p.display, attr, value)
        self._layout_dirty = attr == "grid" or self._layout_dirty
        self._edited("view")
        self._model_changed()

    def _show_pane(self, item: str, on: bool) -> None:
        dpg.configure_item(item, show=on)
        if item == "statusbar":
            for col in ("left", "center", "graph_col"):
                dpg.configure_item(col, height=-STATUS_H if on else -1)
        self._layout_dirty = True

    def _set_zoom(self, z: float) -> None:
        self.zoom = z
        th.set_zoom(z)
        dpg.configure_item("left", width=int(LEFT_W() * z))
        self._layout_dirty = True

    # ================================================================= input
    @staticmethod
    def _command_down() -> bool:
        return any(dpg.is_key_down(k) for k in (dpg.mvKey_LControl, dpg.mvKey_RControl, dpg.mvKey_LWin,
                                                dpg.mvKey_RWin))

    @staticmethod
    def _shift_down() -> bool:
        return dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)

    def _typing(self) -> bool:
        """A text or number field has the keyboard."""
        item = dpg.get_focused_item()
        if item and dpg.does_item_exist(item) and "Input" in dpg.get_item_type(item) and dpg.is_item_active(item):
            return True
        ed = self._editor
        if ed is not None and any(dpg.does_item_exist(w) and dpg.is_item_active(w) for w in list(ed.widgets.values())):
            return True
        return False

    def _on_click(self, *_):
        if dpg.does_item_exist("grad_dl") and dpg.is_item_hovered("grad_dl"):
            self._gradient_press()
            return
        pn = self._hovered_panel()
        if pn is not None:
            self._canvas_press(pn)
            return
        if dpg.does_item_exist("timeline") and dpg.is_item_hovered("timeline"):
            self._timeline_press()
            return
        self._canvas_pick = False

    def _on_double_click(self, *_):
        if dpg.does_item_exist("grad_dl") and dpg.is_item_hovered("grad_dl"):
            self._gradient_press(double=True)

    def _on_key(self, _s, key) -> None:
        cmd, shift = self._command_down(), self._shift_down()
        if cmd:
            action = {dpg.mvKey_N: self._new_cb, dpg.mvKey_O: lambda: dpg.show_item("proj_dlg"),
                      dpg.mvKey_S: (lambda: dpg.show_item("save_dlg")) if shift else self._save_cb,
                      dpg.mvKey_E: self._export_last, dpg.mvKey_Y: self._redo,
                      dpg.mvKey_Plus: lambda: self._zoom_at(1.25), dpg.mvKey_Add: lambda: self._zoom_at(1.25),
                      dpg.mvKey_Minus: lambda: self._zoom_at(0.8), dpg.mvKey_Subtract: lambda: self._zoom_at(0.8),
                      dpg.mvKey_0: self._zoom_fit,
                      dpg.mvKey_Z: self._redo if shift else self._undo}.get(key)
            if action is None and not self._typing():
                action = {dpg.mvKey_X: self._cut, dpg.mvKey_C: self._copy, dpg.mvKey_V: self._paste,
                          dpg.mvKey_D: self._duplicate, dpg.mvKey_A: self._select_all}.get(key)
            if action is not None:
                action()
            return
        if key == dpg.mvKey_F1:
            self._show_manual()
        elif key == dpg.mvKey_F5:
            self.stop() if shift else self.start()
        elif key == dpg.mvKey_F6:
            self._toggle_run_pause()
        elif key == dpg.mvKey_F7:
            self._toggle_play()
        if self._typing():
            return
        if key == dpg.mvKey_Escape:
            if not self._cancel_drag():
                if self._placing():
                    self._disarm()
                else:
                    self._select([], current=None)
            return
        if key == dpg.mvKey_Spacebar:
            self._toggle_play()
        elif key in (dpg.mvKey_Delete, dpg.mvKey_Back):
            if self._canvas_pick and self._drag is None:
                self._remove_selected()
                self._canvas_pick = False
        elif key in (dpg.mvKey_Left, dpg.mvKey_Right, dpg.mvKey_Up, dpg.mvKey_Down):
            if self._canvas_pick:
                n = 10 if shift else 1
                d = {dpg.mvKey_Left: (-n, 0), dpg.mvKey_Right: (n, 0), dpg.mvKey_Up: (0, -n),
                     dpg.mvKey_Down: (0, n)}[key]
                self._nudge(*d)
        elif key == dpg.mvKey_Tab:
            if self._canvas_pick:
                self._cycle(-1 if shift else 1)
        elif key == dpg.mvKey_Comma:
            self._step_frame(-1)
        elif key == dpg.mvKey_Period:
            self._step_frame(1)
        elif key == dpg.mvKey_I:
            self._set_in()
        elif key == dpg.mvKey_O:
            self._set_out()
        elif key == dpg.mvKey_X:
            self._clear_range()
        elif key == dpg.mvKey_L:
            self.loop = not self.loop
            dpg.set_value("loop_chk", self.loop)
        elif key == dpg.mvKey_F:
            self.live = not self.live
            dpg.set_value("live_chk", self.live)

    # ================================================================= frame loop
    def _auto_brightness(self, layer: str | None = None) -> None:
        from .render import auto_brightness
        fr = self._shown_frame
        if fr is None:
            return
        d = self.p.display
        if layer in (None, "Field"):
            d.field_brightness = auto_brightness(fr.field.astype(np.float32), False)
        if layer in (None, "Intensity") and fr.intensity is not None:
            d.intensity_brightness = auto_brightness(fr.intensity.astype(np.float32), True)
        self._edited("view")
        self._model_changed()

    def _frame_to_show(self):
        """The run's frame at the position; while a run has not reached it yet, the picture shown stays."""
        if self.run is not None:
            n = self.run.frame_count()
            idx = self.position // self._div()
            if n and (idx < n or not self.run.running or self._shown_frame is None
                      or self._shown_key[0] == "idle"):
                fr = self.run.frame(min(idx, n - 1))
                return fr, ("run", id(fr))
            if n or self._shown_frame is not None:
                return self._shown_frame, self._shown_key
        s = self.p.scene
        ps = PREVIEW[self.p.sim.preview]
        h, w = -(-s.height // ps), -(-s.width // ps)
        zero = np.zeros((h, w), np.float16)
        return SimpleNamespace(step=0, field=zero, intensity=zero, profiles=None), ("idle", h, w)

    def update(self) -> None:
        now = time.perf_counter()
        self._update_export()
        self._update_drag()
        self._update_pan()
        self._update_timeline_drag()
        self._update_stop_drag()
        # an edit has finished: an undo step, and (after a run) a new run
        if (self.history.pending or self._end_edit_now) and self._drag is None and self._stop_drag is None \
                and not dpg.is_mouse_button_down(dpg.mvMouseButton_Left) and not self._typing():
            self._end_edit_now = False
            self.history.end_step(self.p)
            if (self._sim_edit or self._probe_edit) and self.p.sim.rerun_on_edit and self.run is not None:
                self._rerun()
            self._sim_edit = self._probe_edit = False
            self._update_title()
        self._layout_preview()
        self._tick_playback(now)
        fr, key = self._frame_to_show()
        if key != self._shown_key or self._dirty_view:
            if self.p.display.auto_brightness and key != self._shown_key and key[0] != "idle":
                self._shown_frame = fr
                self._auto_brightness()
            self._shown_frame = fr
            self._shown_key = key
            self._dirty_view = False
            self._show_frame(fr)
            self._overlays_dirty = True
            self._update_graphs(fr)
        if self._overlays_dirty:
            self._redraw_overlays()
        if self._editor is not None and self._editor_version != self._model_version:
            self._editor_version = self._model_version
            self._editor.refresh()
            self._sync_tree_selection()
        self._sync_headers()
        self._redraw_timeline()
        dpg.set_value("pos_text", self._position_text())
        if dpg.does_item_exist("pause_run_btn"):
            label = "Resume run" if self.run is not None and self.run.paused else "Pause run"
            if dpg.get_item_label("pause_run_btn") != label:
                dpg.configure_item("pause_run_btn", label=label)
        self._update_status()
        while self._old_textures:
            dpg.delete_item(self._old_textures.pop())

    def _update_status(self) -> None:
        what = self._placing()
        if what:
            msg, col = f"Click the canvas to place: {what}   (Esc cancels)", (255, 230, 120)
        else:
            col = {"warning": th.WARNING, "error": th.ERROR}.get(self._message_level, (230, 240, 250))
            msg = self._message_text
        dpg.set_value("status_msg", msg)
        dpg.configure_item("status_msg", color=col)
        run = self.run
        if run is None:
            state = "no run: Run (F5) computes the simulation"
        else:
            st = run.stats
            if st.error:
                state = "error: " + st.error
            elif self.run_stale:
                state = "edited since the run"
            elif run.running:
                state = ("paused" if run.paused else "computing") + \
                        f"  step {st.steps_done}/{st.total_steps}  {st.steps_per_second:.0f} steps/s"
            else:
                state = f"computed {st.total_steps} steps"
            if st.frames_dropped:
                state += "  (memory budget reached: history truncated)"
        if self.playing:
            state += ",  playing"
        dpg.set_value("status_run", state)

    def main_loop(self) -> None:
        self._sync_menus()
        last_size = None
        while dpg.is_dearpygui_running():
            size = (tuple(dpg.get_item_rect_size("center")), self._graphs_shown)
            if size != last_size:
                last_size = size
                self._layout_dirty = True
            self.update()
            if self._model_version != getattr(self, "_menus_version", -1):
                self._menus_version = self._model_version
                self._sync_menus()
            dpg.render_dearpygui_frame()
            dpg.run_callbacks(dpg.get_callback_queue())
        self.stop()
        dpg.destroy_context()


def run_app(image: str | None = None, backend: str = "auto") -> None:
    app = App(image, backend)
    app.build()
    app.main_loop()
