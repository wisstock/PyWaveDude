# SPDX-License-Identifier: GPL-3.0-or-later
# PyWaveDude: Python port of WaveDude Next (C) Alexander Graef, after Jeroen Vleggaar (Huygens Optics) and 0x23.
"""The project tree and the properties pane (mixin of the app).

The tree lists the project's settings (Project, Scene, Simulation, Display) and its sources, optical parts, probes
and colour schemes, each row with Enabled, move up, move down and remove; the toolbar adds objects. The properties
pane shows the form of the current object."""

from __future__ import annotations

import random

import dearpygui.dearpygui as dpg

from .engine import PointSource
from .parts import PART_TYPES
from .parts import model as pm
from .probes import Probe
from .render import ColorScheme, builtin_schemes
from .ui_meta import kind_description
from .ui_props import Editor, computed_fields
from . import ui_theme as th

# the Add part menu: groups of kinds, as in the original
PART_MENU = (
    ("Lenses and mirrors", ("Lens", "Fresnel lens", "GRIN lens")),
    ("Shapes", ("Rectangle", "Ellipse", "Polygon", "Triangle", "Outline")),
    ("Waveguides", ("Fiber",)),
    ("Layers", ("Bragg reflector", "Interference filter")),
    ("Diffraction gratings", ("Grating", "Facetted grating", "Diffuser")),
    ("Diffractive optics", ("Zone plate", "Meta lens", "Meta prism", "Meta axicon")),
    ("Phase plates", ("Phase step plate",)),
    ("Logic gates", ("XOR gate", "AND gate", "OR gate")),
)
GROUPS = (("Sources", "sources"), ("Optical parts", "parts"), ("Probes", "probes"), ("Colour schemes", "color_schemes"))
SIM_FIELDS_VIEW = {"backend"}                           # simulation settings that do not change the result


class TreeMixin:
    """Needs from the app: ``self.p``, ``self.selection``, ``self.current``, ``self._edited(kind)``, ``self._arm``,
    ``self._schemes()``, ``self._status``."""

    def _init_tree(self) -> None:
        self._rows: dict[int, tuple] = {}
        self._group_nodes: dict[str, int] = {}
        self._group_open = {g: True for g, _ in GROUPS}
        self._anchor = None
        self._editor: Editor | None = None
        self._editor_obj = None

    # ------------------------------------------------------------ what an object is
    def kind_of(self, obj) -> str:
        p = self.p
        if obj is p:
            return "project"
        if obj is p.scene:
            return "scene"
        if obj is p.sim:
            return "sim"
        if obj is p.display:
            return "display"
        if isinstance(obj, PointSource):
            return "source"
        if isinstance(obj, pm.OpticalPart):
            return "part"
        if isinstance(obj, Probe):
            return "probe"
        if isinstance(obj, ColorScheme):
            return "scheme"
        return ""

    def list_of(self, obj) -> list | None:
        return {"source": self.p.sources, "part": self.p.parts, "probe": self.p.probes,
                "scheme": self.p.color_schemes}.get(self.kind_of(obj))

    @staticmethod
    def index_in(lst, obj) -> int:
        return next((i for i, o in enumerate(lst) if o is obj), -1)

    @staticmethod
    def name_of(obj) -> str:
        if isinstance(obj, pm.OpticalPart):
            return obj.name or obj.stem
        return getattr(obj, "name", "") or "(unnamed)"

    @staticmethod
    def enabled_attr(obj) -> str | None:
        if isinstance(obj, Probe):
            return "active"
        if isinstance(obj, (PointSource, pm.OpticalPart)):
            return "enabled"
        return None

    # ------------------------------------------------------------ building
    def _build_tree_pane(self) -> None:
        with dpg.group(horizontal=True):
            dpg.add_button(label="+ Source", callback=lambda: self._arm("source"))
            with dpg.tooltip(dpg.last_item()):
                dpg.add_text("Add a source: click the canvas where it goes")
            b = dpg.add_button(label="+ Part ▼")
            with dpg.popup(b, mousebutton=dpg.mvMouseButton_Left):
                for group, kinds in PART_MENU:
                    with dpg.menu(label=group):
                        for k in kinds:
                            dpg.add_menu_item(label=k, callback=lambda s, a, u: self._arm("part", u), user_data=k)
                            desc = kind_description(PART_TYPES[k]())
                            if desc:
                                with dpg.tooltip(dpg.last_item()):
                                    dpg.add_text(desc, wrap=360)
            b = dpg.add_button(label="+ Probe ▼")
            from .ui_probes import NEW_PROBES
            with dpg.popup(b, mousebutton=dpg.mvMouseButton_Left):
                for k in NEW_PROBES:
                    dpg.add_menu_item(label=k, callback=lambda s, a, u: self._arm("probe", u), user_data=k)
            for label, tip, cb in (("↑", "Move up (Alt+Up)", lambda: self._move_selected(-1)),
                                   ("↓", "Move down (Alt+Down)", lambda: self._move_selected(1)),
                                   ("×", "Remove (Del)", self._remove_selected)):
                dpg.add_button(label=label, callback=cb, width=22)
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text(tip)
        dpg.add_separator()
        dpg.add_group(tag="tree_body")

    def _build_props_pane(self) -> None:
        dpg.add_text("", tag="props_title", color=(120, 170, 230))
        dpg.add_separator()
        dpg.add_group(tag="props_body")

    def _refresh_tree(self) -> None:
        """Rebuild the rows (after an add, a removal, a move, an undo or a new project)."""
        if not dpg.does_item_exist("tree_body"):
            return
        for g, tag in self._group_nodes.items():
            if dpg.does_item_exist(tag):
                self._group_open[g] = dpg.get_value(tag)
        dpg.delete_item("tree_body", children_only=True)
        self._rows.clear()
        self._group_nodes.clear()
        p = self.p
        dpg.push_container_stack("tree_body")
        try:
            for label, obj in (("Project", p), ("Scene", p.scene), ("Simulation", p.sim), ("Display", p.display)):
                sid = dpg.add_selectable(label=label, callback=self._row_clicked, user_data=obj)
                self._rows[id(obj)] = (sid, None, obj)
            for title, attr in GROUPS:
                lst = getattr(p, attr)
                with dpg.tree_node(label=f"{title}  ({len(lst)})", default_open=self._group_open[title]) as tn:
                    self._group_nodes[title] = tn
                    if attr == "color_schemes":
                        b = dpg.add_button(label="+ New scheme from ▼", small=True)
                        with dpg.popup(b, mousebutton=dpg.mvMouseButton_Left):
                            for name, cs in self._schemes().items():
                                dpg.add_menu_item(label=name, user_data=cs,
                                                  callback=lambda s, a, u: self._add_scheme(u))
                    if not lst:
                        dpg.add_text("(none)", color=th.TEXT_DIM)
                        continue
                    from . import config
                    shown = lst[:int(config.app("ui.tree_max_rows", 2000))]
                    with dpg.table(header_row=False, policy=dpg.mvTable_SizingStretchProp, pad_outerX=False):
                        dpg.add_table_column(width_fixed=True, init_width_or_weight=22)
                        dpg.add_table_column(init_width_or_weight=1.0)
                        for _ in range(3):
                            dpg.add_table_column(width_fixed=True, init_width_or_weight=18)
                        for obj in shown:
                            with dpg.table_row():
                                ea = self.enabled_attr(obj)
                                chk = dpg.add_checkbox(default_value=getattr(obj, ea), callback=self._row_enabled,
                                                       user_data=obj) if ea else dpg.add_text("")
                                sid = dpg.add_selectable(label=self._row_label(obj), span_columns=False,
                                                         callback=self._row_clicked, user_data=obj)
                                for label, cb in (("↑", self._row_up), ("↓", self._row_down), ("×", self._row_remove)):
                                    dpg.add_button(label=label, callback=cb, user_data=obj, small=True)
                                    dpg.bind_item_theme(dpg.last_item(), "theme_small_button")
                                self._rows[id(obj)] = (sid, chk if ea else None, obj)
                    if len(lst) > len(shown):
                        dpg.add_text(f"... {len(lst) - len(shown)} more", color=th.TEXT_DIM)
        finally:
            dpg.pop_container_stack()
        self._sync_tree_selection()

    def _row_label(self, obj) -> str:
        if isinstance(obj, pm.OpticalPart):
            return f"{obj.name or obj.stem}" + ("" if obj.name.startswith(obj.stem) else f"  ({obj.stem})")
        if isinstance(obj, ColorScheme):
            return f"{obj.name}  ({obj.kind or 'any'})"
        return self.name_of(obj)

    def _sync_tree_selection(self) -> None:
        sel = {id(o) for o in self.selection}
        cur = id(self.current) if self.current is not None else None
        for key, (sid, chk, obj) in self._rows.items():
            if dpg.does_item_exist(sid):
                dpg.set_value(sid, key in sel or key == cur)
            if chk is not None and dpg.does_item_exist(chk):
                ea = self.enabled_attr(obj)
                if dpg.get_value(chk) != getattr(obj, ea):
                    dpg.set_value(chk, getattr(obj, ea))

    def _update_row_label(self, obj) -> None:
        row = self._rows.get(id(obj))
        if row and dpg.does_item_exist(row[0]):
            dpg.configure_item(row[0], label=self._row_label(obj))

    # ------------------------------------------------------------ row actions
    def _row_clicked(self, _s, _v, obj) -> None:
        self._canvas_pick = False
        if self._command_down():
            self._toggle_in_selection(obj)
        elif self._shift_down() and self._anchor is not None and self.list_of(obj) is not None \
                and self.list_of(obj) is self.list_of(self._anchor):
            lst = self.list_of(obj)
            a, b = sorted((self.index_in(lst, self._anchor), self.index_in(lst, obj)))
            self._select(lst[a:b + 1], current=obj, anchor=False)
        else:
            self._select([obj] if self.list_of(obj) is not None else [], current=obj)

    def _row_enabled(self, _s, value, obj) -> None:
        targets = self.selection if obj in self._sel_ids() and len(self.selection) > 1 else [obj]
        for o in targets:
            ea = self.enabled_attr(o)
            if ea:
                setattr(o, ea, value)
                self._edited(self.kind_of(o))
        self._sync_tree_selection()
        self._model_changed()

    def _sel_ids(self):
        return [o for o in self.selection]

    def _row_up(self, _s, _a, obj) -> None:
        self._move_objects([obj], -1)

    def _row_down(self, _s, _a, obj) -> None:
        self._move_objects([obj], 1)

    def _row_remove(self, _s, _a, obj) -> None:
        self._remove_objects([obj])

    # ------------------------------------------------------------ selection
    def _select(self, objs, current=None, anchor=True) -> None:
        self.selection = list(objs)
        self.current = current if current is not None else (self.selection[-1] if self.selection else None)
        if anchor:
            self._anchor = self.current
        self._sync_tree_selection()
        self._show_props()
        self._overlays_dirty = True

    def _toggle_in_selection(self, obj) -> None:
        if self.list_of(obj) is None:
            self._select([], current=obj)
            return
        sel = [o for o in self.selection if o is not obj]
        if len(sel) == len(self.selection):
            sel.append(obj)
            cur = obj
        else:
            cur = sel[-1] if sel else None
        self._select(sel, current=cur)

    # ------------------------------------------------------------ list commands
    def _move_objects(self, objs, d: int) -> None:
        lst = self.list_of(objs[0]) if objs else None
        if lst is None:
            return
        idx = sorted(self.index_in(lst, o) for o in objs if self.list_of(o) is lst)
        if not idx or (d < 0 and idx[0] == 0) or (d > 0 and idx[-1] == len(lst) - 1):
            return
        for i in (idx if d < 0 else reversed(idx)):
            lst[i], lst[i + d] = lst[i + d], lst[i]
        self._edited(self.kind_of(objs[0]))
        self._refresh_tree()

    def _move_selected(self, d: int) -> None:
        objs = self.selection or ([self.current] if self.current is not None else [])
        if objs:
            self._move_objects(objs, d)

    def _remove_objects(self, objs) -> None:
        removed = False
        for o in objs:
            lst = self.list_of(o)
            if lst is None:
                continue
            i = self.index_in(lst, o)
            if i >= 0:
                del lst[i]
                removed = True
                self._edited(self.kind_of(o))
        if removed:
            gone = {id(o) for o in objs}
            self.selection = [o for o in self.selection if id(o) not in gone]
            if self.current is not None and id(self.current) in gone:
                self.current = None
            self._refresh_tree()
            self._show_props()
            self._model_changed()

    def _remove_selected(self, *_) -> None:
        objs = self.selection or ([self.current] if self.list_of(self.current) is not None else [])
        self._remove_objects(list(objs))

    def _unique_name(self, lst, base: str) -> str:
        names = {getattr(o, "name", "") for o in lst}
        if base not in names:
            return base
        stem = base.rstrip("0123456789 ")
        k = 2
        while f"{stem} {k}" in names:
            k += 1
        return f"{stem} {k}"

    # ------------------------------------------------------------ properties pane
    def _show_props(self) -> None:
        if not dpg.does_item_exist("props_body"):
            return
        obj = self.current
        if obj is self._editor_obj and self._editor is not None:
            return
        self._editor_obj = obj
        self._editor = None
        dpg.delete_item("props_body", children_only=True)
        kind = self.kind_of(obj) if obj is not None else ""
        titles = {"project": "Project", "scene": "Scene", "sim": "Simulation", "display": "Display",
                  "source": "Source", "part": "Optical part", "probe": "Probe", "scheme": "Colour scheme"}
        dpg.set_value("props_title", titles.get(kind, "Nothing selected") +
                      (f": {self.name_of(obj)}" if kind in ("source", "part", "probe", "scheme") else ""))
        if obj is None:
            n = len(self.selection)
            dpg.add_text(f"{n} objects selected" if n else "Click an object on the canvas or a row of the tree.",
                         parent="props_body", wrap=0, color=th.TEXT_DIM)
            return
        if kind == "project":
            self._project_form()
            return
        if kind == "scheme":
            self._scheme_form(obj)
            return
        hidden = set()
        if kind == "display":
            hidden |= {"export_from", "export_to", "field_brightness", "intensity_brightness"}
        if kind == "scene":
            hidden |= {"image_name", "image_path"}
        self._editor = Editor(obj, "props_body", lambda name: self._prop_changed(obj, name),
                              medium=self._medium, choices=self._field_choices, extras=self._form_extras,
                              hidden=hidden,
                              computed=lambda o: computed_fields(o, self.p.sim.source_frequency_scale))

    def _medium(self) -> float:
        s = self.p.scene
        return 1.0 if s.use_image else s.refractive_index

    def _field_choices(self, obj, name):
        from .engine import available_backends
        kind = self.kind_of(obj)
        if kind == "sim":
            return {"backend": ["auto"] + available_backends(), "preview": ["Full", "Half", "Quarter"],
                    "preview_frames": [1, 2, 4]}.get(name)
        if kind == "scene" and name == "open_edges":
            return ["Open", "Reflective"]
        if kind == "display":
            if name in ("field_scheme", "intensity_scheme"):
                return list(self._schemes())
            return {"layer": ["Field", "Intensity", "Both"], "arrangement": ["Stacked", "Side by side"]}.get(name)
        return None

    def _form_extras(self, obj) -> None:
        if isinstance(obj, pm.LogicGatePart):
            dpg.add_button(label="Put sources on inputs A and B", callback=lambda: self._gate_sources(obj))
        if self.kind_of(obj) == "scene":
            with dpg.group(horizontal=True):
                dpg.add_button(label="Open scene image...", callback=lambda: dpg.show_item("open_dlg"))
                dpg.add_button(label="Blank canvas", callback=self._use_blank)
            dpg.add_text(self._scene_label(), wrap=0, color=th.TEXT_DIM)

    def _prop_changed(self, obj, name: str) -> None:
        kind = self.kind_of(obj)
        if kind in ("source", "part", "probe", "scheme"):
            if hasattr(obj, "normalize") and kind != "part":
                obj.normalize()
            if name == "name":
                self._update_row_label(obj)
                dpg.set_value("props_title", dpg.get_value("props_title").split(":")[0] + f": {self.name_of(obj)}")
            ea = self.enabled_attr(obj)
            if name == ea:
                self._sync_tree_selection()
        if kind == "sim" and name == "source_frequency_scale":
            for part in self.p.parts:
                if hasattr(part, "frequency_scale"):
                    part.frequency_scale = self.p.sim.source_frequency_scale
        if kind == "display":
            if name in ("arrangement", "layer"):
                self._layout_dirty = True
            self._luts.clear()
            self._edited("view")
        elif kind == "sim" and name in SIM_FIELDS_VIEW:
            self._edited("view")
        else:
            self._edited(kind)
        self._model_changed()

    def _project_form(self) -> None:
        p = self.p
        with dpg.table(header_row=False, parent="props_body", policy=dpg.mvTable_SizingStretchProp):
            dpg.add_table_column(init_width_or_weight=0.42)
            dpg.add_table_column(init_width_or_weight=0.58)
            with dpg.table_row():
                dpg.add_text("Random seed")
                with dpg.group(horizontal=True):
                    dpg.add_input_int(default_value=p.seed, width=120, tag="seed_in", step=0,
                                      callback=lambda s, v: self._set_seed(v))
                    dpg.add_button(label="New seed",
                                   callback=lambda: self._set_seed(random.randint(1, 2 ** 31 - 2)))
            with dpg.table_row():
                dpg.add_text("File")
                dpg.add_text(p.path or "(not saved)", wrap=0)
        dpg.add_text("Description", parent="props_body")
        dpg.add_input_text(default_value=p.description, multiline=True, width=-1, height=120, parent="props_body",
                           callback=lambda s, v: (setattr(p, "description", v), self._edited("view")))

    def _set_seed(self, v: int) -> None:
        self.p.seed = int(v)
        for part in self.p.parts:
            if hasattr(part, "project_seed"):
                part.project_seed = self.p.seed
        if dpg.does_item_exist("seed_in"):
            dpg.set_value("seed_in", self.p.seed)
        self._edited("sim")

    def _add_scheme(self, base: ColorScheme | None = None) -> None:
        base = base or self._schemes().get(self.p.display.field_scheme) or next(iter(builtin_schemes().values()))
        cs = ColorScheme(self._unique_name(self.p.color_schemes, base.name + " copy"), base.kind, base.stops)
        self.p.color_schemes.append(cs)
        self._edited("view")
        self._refresh_tree()
        self._select([cs], current=cs)
