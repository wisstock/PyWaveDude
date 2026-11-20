# SPDX-License-Identifier: GPL-3.0-or-later
# PyWaveDude: Python port of WaveDude Next (C) Alexander Graef, after Jeroen Vleggaar (Huygens Optics) and 0x23.
"""Generic Dear PyGui property editor for the dataclass models (parts, sources)."""

from __future__ import annotations

import dataclasses
import math
from enum import Enum
from typing import Callable

import dearpygui.dearpygui as dpg

from .parts import model as m

HIDDEN = {"project_seed", "frequency_scale"}
COMPOSITE_FIELDS = {"size_x", "size_y", "spacing_x", "spacing_y", "array_angle", "random_frequency", "random_phase",
                    "random_off_width", "random_position"}
PULSE_FIELDS = {"pulse_shape", "pulse_width", "off_width"}
WIDTH = 150


def _label(name: str) -> str:
    return name.replace("_", " ").capitalize()


# --------------------------------------------------------------------- list fields as text
def _list_to_text(name: str, value) -> str:
    if name == "points":
        return "; ".join(f"{x},{y}" for x, y in value)
    if name == "layers":
        return "; ".join(f"{l.thickness:g}@{l.refractive_index:g}" for l in value)
    if name == "nodes":
        return "; ".join(f"{n.x},{n.y},{n.in_x},{n.in_y},{n.out_x},{n.out_y}" for n in value)
    return str(value)


def _text_to_list(name: str, text: str):
    items = [t.strip() for t in text.split(";") if t.strip()]
    if name == "points":
        return [tuple(int(float(v)) for v in t.split(",")[:2]) for t in items]
    if name == "layers":
        out = []
        for t in items:
            th, n = t.split("@")
            out.append(m.FilterLayer(float(th), float(n)))
        return out
    if name == "nodes":
        out = []
        for t in items:
            v = [int(float(s)) for s in t.split(",")]
            v += [0] * (6 - len(v))
            out.append(m.FiberNode(*v[:6]))
        return out
    raise ValueError(name)


LIST_HINTS = {
    "points": "x,y; x,y; ...  (part frame, px)",
    "layers": "thickness@index; ...  (front to back)",
    "nodes": "x,y,in_dx,in_dy,out_dx,out_dy; ...",
}


# --------------------------------------------------------------------- read-outs
def readouts(obj, medium: float = 1.0, sources=()) -> list[str]:
    """Derived values shown under the form; ``sources``: the project's enabled sources, for per-source values."""
    from .engine.sources import PointSource
    out = []
    if isinstance(obj, PointSource):
        if obj.composite:
            out.append(f"{obj.size_x * obj.size_y} emitters")
    if isinstance(obj, m.LensPart):
        r1, r2 = obj.fitted_radii()
        out.append(f"Edge thickness {obj.edge_thickness:.1f} px, fitted R1 {r1:g}, R2 {r2:g}")
        fl = obj.focal_lengths(medium)
        if fl is not None:
            where = f" (in n = {medium:g})" if medium != 1.0 else ""
            if math.isinf(fl[0]):
                out.append(f"Focal length: none, no optical power{where}")
            else:
                out.append(f"Focal length {fl[0]:.0f} px, back focal length {fl[1]:.0f} px{where}"
                           + ("  (diverging)" if fl[0] < 0 else ""))
    if isinstance(obj, m.TrianglePart):
        out.append(f"Apex angle {obj.apex_angle:.1f}°")
    if isinstance(obj, m.ZonePlatePart):
        out.append(f"Wavelength {obj.wavelength:.1f} px, {obj.zone_count} zones")
    if isinstance(obj, m.PhaseStepPlatePart):
        out.append(f"Wavelength {obj.wavelength:.1f} px, step height {obj.step_height:.2f} px, "
                   f"zone width {max(2, obj.aperture) / min(max(obj.zones, 1), 64):.0f} px")
    if isinstance(obj, m.MetaSurfacePart):
        out.append(f"Wavelength {obj.wavelength:.1f} px, full-turn height {obj.full_turn_height:.1f} px, "
                   f"{obj.cell_count} cells")
        if isinstance(obj, m.MetaAxiconPart):
            r = math.radians(min(max(obj.cone_angle, 0.1), 89.0))
            out.append(f"Line of focus {max(2, obj.width) / 2 / math.tan(r):.0f} px")
    if isinstance(obj, m.InterferenceFilterPart):
        out.append(f"Stack thickness {obj.stack_thickness:.2f} px")
    if isinstance(obj, m.DiffuserPart):
        p = obj.profile()
        out.append(f"Profile height {max(p) - min(p):.2f} px, {len(obj.components())} ripples")
    if isinstance(obj, m.LogicGatePart):
        d = obj.design
        if d is None:
            out.append("No guided mode: the core index must exceed the cladding's")
        else:
            out.append(f"Wavelength {d.wavelength:.1f} px, core {d.core_width} px, coupler gap {d.gap} px, "
                       f"coupler {d.coupler_length} px, length {d.length} px, port spacing {d.port_spacing} px")
            out.append(f"Outputs: upper {obj.outputs[0]}, lower {obj.outputs[1]}")
    if isinstance(obj, m.DichroicMirrorPart):
        out += _dichroic_readouts(obj, sources)
    if isinstance(obj, m.BraggReflectorPart):
        out.append(f"Length {obj.pairs * (obj.thickness1 + obj.thickness2)} px")
    return out


def _dichroic_readouts(d: m.DichroicMirrorPart, sources) -> list[str]:
    layers = d.layers
    if not layers:
        return ["No stack: check the cut-off frequency and the design angle"]
    a, b = d.band
    hi_n, lo_n = max(l.refractive_index for l in layers), min(l.refractive_index for l in layers)
    quarter = lambda n: max(l.thickness for l in layers if l.refractive_index == n)  # noqa: E731
    lv = lambda lam: _level_for(lam, d.frequency_scale)  # noqa: E731
    out = [f"Cut-off {d.cutoff:.1f} px, band centre {d.centre:.1f} px",
           f"Reflected band {a:.1f}–{b:.1f} px (frequency {lv(b):.1f}–{lv(a):.1f}), peak R "
           f"{d.reflectance(d.centre) * 100:.1f} %",
           f"{len(layers)} layers: high {quarter(hi_n):.2f} px, low {quarter(lo_n):.2f} px, "
           f"stack {d.stack_thickness:.1f} px"]
    thin = min(quarter(hi_n), quarter(lo_n))
    if thin < 2.0:
        out.append(f"Quarter-wave layer {thin:.2f} px: too thin for the grid to resolve; use a longer wavelength")
    for s in sources:
        if not getattr(s, "enabled", True):
            continue
        lam = _lambda_of(s.level, d.frequency_scale)
        r = d.reflectance(lam)
        out.append(f"{s.name or 'Source'} (λ {lam:.1f} px): R {r * 100:.0f} %, T {(1 - r) * 100:.0f} %")
    return out


@dataclasses.dataclass(frozen=True)
class FieldMetaLike:
    label: str
    hint: str = ""


# --------------------------------------------------------------------- computed fields
@dataclasses.dataclass(frozen=True)
class Computed:
    """A field shown in a form but stored in another one: ``get(obj)`` reads it, ``set(obj, v)`` writes the real
    field (named by ``writes``, which is what ``on_change`` hears); shown after the field ``after``."""
    name: str
    label: str
    hint: str
    after: str
    writes: str
    get: Callable
    set: Callable
    fmt: str = "%.2f"


def _lambda_of(level: float, scale: float) -> float:
    from .engine.sources import wavelength
    return wavelength(level, scale)


def _level_for(lam: float, scale: float, lo: float = 0.01, hi: float = 255.0) -> float:
    """The frequency level whose wavelength is ``lam`` pixels (wavelength ∝ 1 / (level · scale))."""
    if not lam > 0:
        raise ValueError(lam)
    return min(hi, max(lo, round(_lambda_of(1.0, scale) / lam, 4)))


REFERENCE_LEVEL = 100.0


def _scale_for(lam: float) -> float:
    """The source frequency scale that gives frequency 100 a wavelength of ``lam`` pixels."""
    if not lam > 0:
        raise ValueError(lam)
    return round(_lambda_of(REFERENCE_LEVEL, 1.0) / lam, 6)


def computed_fields(obj, frequency_scale: float) -> list[Computed]:
    """The wavelength in pixels beside every frequency: of a source, a wavelength-designed part, and of the
    simulation's reference level (the source frequency scale)."""
    from .engine import PointSource
    from .project import SimSettings
    if isinstance(obj, PointSource):
        return [Computed("_wavelength", "Wavelength (px)", "Wavelength of this source in pixels at the simulation's "
                         "frequency scale; typing it sets the frequency.", "level", "level",
                         lambda o: _lambda_of(o.level, frequency_scale),
                         lambda o, v: setattr(o, "level", _level_for(v, frequency_scale)))]
    if isinstance(obj, SimSettings):
        return [Computed("_wavelength", f"Wavelength at frequency {REFERENCE_LEVEL:g} (px)",
                         "The wavelength in pixels of a source at frequency 100; typing it sets the source frequency "
                         "scale, which scales every source's wavelength alike (λ ∝ 1 / (frequency × scale)).",
                         "source_frequency_scale", "source_frequency_scale",
                         lambda o: _lambda_of(REFERENCE_LEVEL, o.source_frequency_scale),
                         lambda o, v: setattr(o, "source_frequency_scale", _scale_for(v)))]
    if hasattr(obj, "design_frequency") and hasattr(obj, "frequency_scale"):
        return [Computed("_wavelength", "Design wavelength (px)", "The wavelength the part is made for, in pixels; "
                         "typing it sets the design frequency.", "design_frequency", "design_frequency",
                         lambda o: _lambda_of(o.design_frequency, o.frequency_scale),
                         lambda o, v: setattr(o, "design_frequency", _level_for(v, o.frequency_scale)))]
    return []


# --------------------------------------------------------------------- editor
LABEL_WIDTH = 0.42          # share of the form's width for the labels


class Editor:
    """The form of one object: a label and a control per field, grouped by category, a hint on every label.

    A control writes its field as you type (no Enter needed) and calls ``on_change(name)``; ``refresh()`` puts
    changed values into the controls that are not being edited, and rebuilds the form only when the set of fields
    shown changes (a source made composite, a pulsed mode chosen)."""

    def __init__(self, obj, parent, on_change: Callable[[str], None], *, medium: Callable[[], float] = lambda: 1.0,
                 choices: Callable[[object, str], list | None] = lambda o, n: None,
                 extras: Callable[[object], None] = lambda o: None, hidden: set = frozenset(),
                 computed: Callable[[object], list] = lambda o: [], sources: Callable[[], list] = lambda: []):
        self.obj, self.parent, self.on_change = obj, parent, on_change
        self.computed, self.sources = computed, sources
        self.medium, self.choices, self.extras, self.hidden = medium, choices, extras, set(hidden)
        self.widgets: dict[str, int] = {}
        self.readout_ids: list[int] = []
        self._shown: tuple = ()
        self.build()

    # ------------------------------------------------------------ which fields
    def visible(self) -> tuple:
        o = self.obj
        out = []
        for f in dataclasses.fields(o):
            n = f.name
            if n in HIDDEN or n in self.hidden or n.startswith("_"):
                continue
            if n in COMPOSITE_FIELDS and not getattr(o, "composite", True):
                continue
            if n in PULSE_FIELDS and getattr(o, "mode", None) is not None and str(getattr(o.mode, "value", o.mode)) \
                    == "Continuous":
                continue
            out.append(n)
        return tuple(out)

    def build(self) -> None:
        from .ui_meta import category_rank, field_meta, kind_description
        o = self.obj
        dpg.delete_item(self.parent, children_only=True)
        self.widgets.clear()
        self.readout_ids.clear()
        self._shown = self.visible()
        dpg.push_container_stack(self.parent)
        try:
            desc = kind_description(o) if isinstance(o, m.OpticalPart) else ""
            if desc:
                with dpg.collapsing_header(label=f"About: {getattr(o, 'stem', '')}", default_open=False):
                    dpg.add_text(desc, wrap=0, color=(170, 175, 185))
            groups: dict[str, list] = {}
            extra = {c.after: c for c in self.computed(o)}
            for n in self._shown:
                fm = field_meta(o, n)
                groups.setdefault(fm.category, []).append((n, fm))
                if n in extra:
                    c = extra[n]
                    groups[fm.category].append((c, FieldMetaLike(c.label, c.hint)))
            for cat in sorted(groups, key=category_rank):
                dpg.add_text(cat, color=(120, 170, 230))
                with dpg.table(header_row=False, borders_innerH=False, policy=dpg.mvTable_SizingStretchProp):
                    dpg.add_table_column(init_width_or_weight=LABEL_WIDTH)
                    dpg.add_table_column(init_width_or_weight=1 - LABEL_WIDTH)
                    for n, fm in groups[cat]:
                        with dpg.table_row():
                            t = dpg.add_text(fm.label)
                            if fm.hint:
                                with dpg.tooltip(t):
                                    dpg.add_text(fm.hint, wrap=360)
                            if isinstance(n, Computed):
                                self.widgets[n.name] = self._computed_control(n)
                            else:
                                self.widgets[n] = self._control(n, getattr(o, n), fm.hint)
            if isinstance(o, m.LensPart):
                dpg.add_button(label="Flip concave / convex", callback=lambda: (o.flip(), self.on_change("r1"),
                                                                                  self.refresh()))
            self.extras(o)
            self.readout_group = dpg.add_group()
            self._readouts()
        finally:
            dpg.pop_container_stack()

    def _computed_control(self, c: Computed):
        def f(_s, v, _u=None):
            try:
                c.set(self.obj, float(v))
            except Exception:
                return
            self.on_change(c.writes)

        return dpg.add_input_float(default_value=c.get(self.obj), width=-1, format=c.fmt, step=0, callback=f)

    def _control(self, name, value, hint):
        o = self

        def cb(convert=lambda v: v):
            def f(_s, v, _u=None):
                try:
                    setattr(o.obj, name, convert(v))
                except Exception:
                    return
                o.on_change(name)
                if o.visible() != o._shown:
                    o.build()
            return f

        opts = self.choices(self.obj, name)
        if opts is not None:
            if isinstance(value, bool):          # a two-way choice shown as words: [True word, False word]
                return dpg.add_combo(opts, default_value=opts[0] if value else opts[1], width=-1,
                                     callback=cb(lambda v: v == opts[0]))
            return dpg.add_combo([str(x) for x in opts], default_value=str(value), width=-1,
                                 callback=cb(lambda v: type(value)(v) if not isinstance(value, str) else v))
        if isinstance(value, bool):
            return dpg.add_checkbox(default_value=value, callback=cb())
        if isinstance(value, Enum):
            from .ui_meta import enum_label
            cls = type(value)
            names = {enum_label(e): e for e in cls}
            return dpg.add_combo(list(names), default_value=enum_label(value), width=-1,
                                 callback=cb(lambda v: names[v]))
        if isinstance(value, int):
            return dpg.add_input_int(default_value=value, width=-1, step=1, step_fast=10, callback=cb())
        if isinstance(value, float):
            return dpg.add_input_float(default_value=value, width=-1, format="%.4g", step=0, callback=cb())
        if isinstance(value, str):
            return dpg.add_input_text(default_value=value, width=-1, callback=cb())
        if isinstance(value, list):
            return dpg.add_input_text(default_value=_list_to_text(name, value), width=-1, hint=LIST_HINTS.get(name, ""),
                                      on_enter=True, callback=cb(lambda t: _text_to_list(name, t)))
        return dpg.add_text(str(value))

    # ------------------------------------------------------------ keep in step with the model
    def refresh(self) -> None:
        if self.visible() != self._shown:
            if not any(dpg.is_item_active(w) for w in self.widgets.values() if dpg.does_item_exist(w)):
                self.build()
            return
        from .ui_meta import enum_label
        o = self.obj
        comp = {c.name: c for c in self.computed(o)}
        for n, w in self.widgets.items():
            if not dpg.does_item_exist(w) or dpg.is_item_active(w):
                continue
            if n in comp:
                v = comp[n].get(o)
                if abs(dpg.get_value(w) - v) > 1e-6 * max(1.0, abs(v)):
                    dpg.set_value(w, v)
                continue
            v = getattr(o, n)
            opts = self.choices(o, n)
            if opts is not None and isinstance(v, bool):
                shown = opts[0] if v else opts[1]
            elif isinstance(v, Enum):
                shown = enum_label(v)
            elif isinstance(v, list):
                shown = _list_to_text(n, v)
            elif opts is not None:
                shown = str(v)
            else:
                shown = v
            if dpg.get_value(w) != shown:
                dpg.set_value(w, shown)
        self._readouts()

    def _readouts(self) -> None:
        lines = readouts(self.obj, self.medium(), self.sources())
        if len(lines) == len(self.readout_ids):
            for t, line in zip(self.readout_ids, lines):
                if dpg.get_value(t) != line:
                    dpg.set_value(t, line)
            return
        dpg.delete_item(self.readout_group, children_only=True)
        self.readout_ids = [dpg.add_text(line, wrap=0, color=(160, 200, 255), parent=self.readout_group)
                            for line in lines]
