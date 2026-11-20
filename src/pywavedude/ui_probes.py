# SPDX-License-Identifier: GPL-3.0-or-later
# PyWaveDude: Python port of WaveDude Next (C) Alexander Graef, after Jeroen Vleggaar (Huygens Optics) and 0x23.
"""Probe UI (mixin of the app): placing new probes by clicks, the graph boxes over the pictures, the probe graph
column beside the preview and the CSV export."""

from __future__ import annotations

import re
from pathlib import Path

import dearpygui.dearpygui as dpg
import numpy as np

from . import config
from .probes import (Probe, ProbeDetector, ProbeKind, ProbeLayer, ProbeMeasurement, ProbeSampler, ProbeScaleMode,
                     write_csv)

NEW_PROBES = {
    "Point (field)": dict(),
    "Point (intensity)": dict(layer=ProbeLayer.INTENSITY, measurement=ProbeMeasurement.VALUE),
    "Line (intensity)": dict(kind=ProbeKind.LINE, layer=ProbeLayer.INTENSITY, measurement=ProbeMeasurement.VALUE),
    "Counting detector": dict(detector=ProbeDetector.COUNTING),
    "Imaging detector": dict(detector=ProbeDetector.IMAGING),
}
MAX_PLOT_POINTS = 4000


class ProbesMixin:
    def _init_probes(self) -> None:
        self.place_probe: dict | None = None     # {"kind": label, "a": (x, y) or None}
        self._graph_probes: list[Probe] = []
        self._graph_key = None
        self._graphs_shown = False
        self._graphs_hidden_by_user = False

    # ------------------------------------------------------------ placing
    def _probe_click(self, cx: int, cy: int) -> bool:
        """A canvas click while a new probe is being placed; True when it was taken."""
        pp = self.place_probe
        if pp is None:
            return False
        kw = NEW_PROBES[pp["kind"]]
        line = kw.get("kind") == ProbeKind.LINE or "detector" in kw
        if line and pp["a"] is None:
            pp["a"] = (cx, cy)
            self._place_labels()
            return True
        ax, ay = pp["a"] if line else (cx, cy)
        name = self._unique_name(self.p.probes, "Detector 1" if "detector" in kw else f"Probe {len(self.p.probes) + 1}")
        pr = config.new_object(Probe, name=name, ax=ax, ay=ay, bx=cx, by=cy, box_x=ax, box_y=ay, **kw)
        pr.normalize()
        self.p.probes.append(pr)
        self.place_probe = None
        self._place_labels()
        self._added(pr)
        return True

    def _probe_button_label(self) -> str:
        pp = self.place_probe
        return "" if pp is None else (f"{pp['kind']}: click point B" if pp["a"] else f"{pp['kind']}")

    # ------------------------------------------------------------ per-wavelength curves
    def _group_curves(self, i: int, step: int):
        """(GraphData, colour) of each wavelength's share of probe ``i``, when the run keeps them and they are on."""
        from .probe_graph import group_graph_data
        from .render import group_colours
        from .engine import wavelength
        run = self.run
        levels = getattr(run.engine, "levels", None) if run is not None else None
        if not levels or not self.p.display.probe_colours:
            return []
        data = group_graph_data(run.collector.traces[i], step)
        if len(data) != len(levels):
            return []
        cols = group_colours([wavelength(lv, self.p.sim.source_frequency_scale) for lv in levels],
                             self.p.display.nm_per_px)
        return [(d, tuple(int(255 * c) for c in col) + (255,)) for d, col in zip(data, cols)]

    # ------------------------------------------------------------ graph boxes over the pictures
    def _draw_probe_graphs(self, pn) -> None:
        """The graph boxes over a panel, laid out as in an export and drawn as vectors."""
        from .probe_graph import (BACKGROUND, BORDER, CURVE, EDGE_PAD, FONT_SIZE, FRAME, GRID, LABEL, LABEL_GAP,
                                  TITLE, ZERO, graph_data, layout, line_height, measure)
        run, fr = self.run, self._shown_frame
        if run is None or run.collector is None or fr is None:
            return
        cw, ch = self._canvas_size()
        k = pn["k"]
        size = FONT_SIZE * k
        lh = line_height()
        ovl = pn["ovl"]
        shown = [pn2["layer"] for pn2 in self._panels]
        profiles = getattr(fr, "profiles", None)
        for i, pr in enumerate(self._graph_probes):
            if not any(q is pr for q in self.p.probes):
                continue
            mine = "Intensity" if pr.layer == ProbeLayer.INTENSITY else "Field"
            if mine in shown and pn["layer"] != mine:
                continue                       # a probe's graph sits on the picture of the layer it measures
            bx, by, bw, bh = pr.box(cw, ch)
            prof = profiles[i] if profiles is not None else None
            groups = self._group_curves(i, fr.step)
            g = layout(pr, graph_data(run.collector.traces[i], fr.step, prof), self.p.sim.steps, bw, bh,
                       [d for d, _ in groups])
            ox, oy = self._S(pn, bx, by)

            def P(x, y):
                return ox + x * k, oy + y * k

            chosen = pr is self.current
            dpg.draw_rectangle(P(0, 0), P(bw, bh), color=(255, 220, 0, 255) if chosen else BORDER, fill=BACKGROUND,
                               parent=ovl)
            if g.has_plot:
                l, t, r, b = g.plot_x, g.plot_y, g.plot_x + g.plot_w, g.plot_y + g.plot_h
                for pos, label, zero in g.y_ticks:
                    dpg.draw_line(P(l, pos), P(r, pos), color=ZERO if zero else GRID, parent=ovl)
                    if label:
                        lx = max(EDGE_PAD, l - LABEL_GAP - measure(label))
                        dpg.draw_text(P(lx, min(max(pos - lh / 2, t - lh / 2), b - lh / 2)), label, color=LABEL,
                                      size=size, parent=ovl)
                for pos, label, _ in g.x_ticks:
                    dpg.draw_line(P(pos, t), P(pos, b), color=GRID, parent=ovl)
                    if label:
                        wl = measure(label)
                        lx = min(max(pos - wl / 2, EDGE_PAD), max(EDGE_PAD, bw - wl - EDGE_PAD))
                        dpg.draw_text(P(lx, b + 2), label, color=LABEL, size=size, parent=ovl)
                dpg.draw_rectangle(P(l, t), P(r, b), color=FRAME, parent=ovl)
                for curve, (_, col) in zip(g.extra, groups):
                    if len(curve) > 1:
                        pts = [P(min(max(x, l), r), min(max(y, t), b)) for x, y in curve]
                        dpg.draw_polyline(pts, color=col, thickness=max(1.0, 0.8 * g.line_width * k), parent=ovl)
                if len(g.curve) > 1:
                    pts = [P(min(max(x, l), r), min(max(y, t), b)) for x, y in g.curve]
                    dpg.draw_polyline(pts, color=CURVE, thickness=max(1.0, g.line_width * k), parent=ovl)
            if g.readout:
                dpg.draw_text(P(bw - measure(g.readout) - 8, EDGE_PAD), g.readout, color=CURVE, size=size, parent=ovl)
            dpg.draw_text(P(EDGE_PAD + 3, EDGE_PAD), g.title, color=TITLE, size=size, parent=ovl)

    # ------------------------------------------------------------ run wiring
    def _make_sampler(self, eng) -> ProbeSampler | None:
        active = [pr for pr in self.p.probes if pr.active]
        self._graph_probes = active
        if not active:
            return None
        return ProbeSampler.build(active, eng.canvas_w, eng.canvas_h, eng.rim, self.p.seed)

    # ------------------------------------------------------------ the graph column
    def _build_graph_column(self) -> None:
        with dpg.child_window(tag="graph_col", width=int(config.app("ui.probe_graph_column", 440)), show=False):
            with dpg.group(horizontal=True):
                dpg.add_text("Probe graphs", color=(120, 170, 230))
                dpg.add_button(label="Hide", callback=lambda: self._show_graphs(False))
            dpg.add_text("Intensity probes of a run with separate wavelengths: a curve per colour beside the total "
                         "(white).", wrap=0, color=(150, 152, 160))
            dpg.add_group(tag="graph_group")

    def _show_graphs(self, on: bool, by_user: bool = True) -> None:
        self._graphs_shown = on
        if by_user:
            self._graphs_hidden_by_user = not on
        dpg.configure_item("graph_col", show=on)
        width = int(config.app("ui.probe_graph_column", 440))
        dpg.configure_item("center", width=-(width + 8) if on else -1)
        for tag in ("view_graphs", "row_graph_col"):
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, on)
        self._graph_key = None
        self._layout_dirty = True

    def _rebuild_graphs(self) -> None:
        dpg.delete_item("graph_group", children_only=True)
        n_groups = len(getattr(self.run.engine, "levels", ()) or ()) if self.run is not None else 0
        for i, pr in enumerate(self._graph_probes):
            log = pr.is_logarithmic
            xlabel = "distance (px)" if pr.shows_profile else "step"
            ylabel = "detections" if pr.is_detector else (pr.measurement.value if pr.layer == ProbeLayer.INTENSITY
                                                          else "field")
            with dpg.plot(label=pr.name, height=180, width=-1, parent="graph_group", tag=f"plot_{i}"):
                dpg.add_plot_axis(dpg.mvXAxis, label=xlabel, tag=f"px_{i}")
                with dpg.plot_axis(dpg.mvYAxis, label=ylabel, tag=f"py_{i}",
                                   scale=dpg.mvPlotScale_Log10 if log else dpg.mvPlotScale_Linear):
                    for k in range(n_groups if self._splits(pr) else 0):
                        dpg.add_line_series([0.0], [0.0], tag=f"pg_{i}_{k}", label=f"wavelength {k + 1}")
                    if pr.shows_profile and pr.detector == ProbeDetector.IMAGING:
                        dpg.add_stair_series([0.0], [0.0], tag=f"ps_{i}")
                    else:
                        dpg.add_line_series([0.0], [0.0], tag=f"ps_{i}", label="total")
                        dpg.bind_item_theme(f"ps_{i}", self._series_theme((240, 240, 245, 255)))

    @staticmethod
    def _splits(pr) -> bool:
        return pr.layer == ProbeLayer.INTENSITY and pr.detector == ProbeDetector.STANDARD

    def _series_theme(self, col):
        cache = self.__dict__.setdefault("_series_themes", {})
        if col not in cache:
            with dpg.theme() as t:
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, col, category=dpg.mvThemeCat_Plots)
            cache[col] = t
        return cache[col]

    def _update_graphs(self, fr) -> None:
        run = self.run
        if run is None or run.collector is None or fr is None or not self._graphs_shown:
            return
        key = (id(run), len(self._graph_probes))
        if key != self._graph_key:
            self._graph_key = key
            self._rebuild_graphs()
        profiles = getattr(fr, "profiles", None)
        for i, pr in enumerate(self._graph_probes):
            if not dpg.does_item_exist(f"ps_{i}"):
                continue
            tr = run.collector.traces[i]
            if pr.shows_profile:
                y = profiles[i] if profiles is not None and profiles[i] is not None else None
                if y is None or len(y) == 0:
                    continue
                x = np.linspace(0.0, tr.length, len(y))
            else:
                v = tr.values()[:fr.step]
                start = 0
                if pr.scale_mode == ProbeScaleMode.RUNNING and len(v) > pr.window:
                    start = len(v) - pr.window
                    v = v[start:]
                stride = max(1, len(v) // MAX_PLOT_POINTS)
                y = v[::stride]
                x = start + np.arange(0, len(v), stride)
                if len(y) == 0:
                    continue
            y = np.asarray(y, np.float64)
            if pr.is_logarithmic:
                top = float(np.max(y)) if len(y) else 1.0
                y = np.maximum(y, max(top, 1e-30) * 10.0 ** (-pr.decades))
            dpg.set_value(f"ps_{i}", [list(map(float, x)), list(map(float, y))])
            for k, (d, col) in enumerate(self._group_curves(i, fr.step)):
                tag = f"pg_{i}_{k}"
                if not dpg.does_item_exist(tag):
                    continue
                if pr.shows_profile:
                    gy = d.profile if d.profile is not None else np.zeros(0)
                    gx = np.linspace(0.0, tr.length, len(gy))
                else:
                    gv = d.values[:d.count]
                    gv = gv[start:] if len(gv) > start else gv[:0]
                    gy = gv[::stride]
                    gx = start + np.arange(0, len(gv), stride)
                dpg.set_value(tag, [list(map(float, gx)), list(map(float, np.asarray(gy, np.float64)))])
                dpg.bind_item_theme(tag, self._series_theme(col))
            dpg.fit_axis_data(f"px_{i}")
            if pr.auto_scale_y:
                dpg.fit_axis_data(f"py_{i}")
            else:
                dpg.set_axis_limits(f"py_{i}", pr.y_min, pr.y_max)

    # ------------------------------------------------------------ CSV
    def _export_csv(self, pr: Probe | None = None) -> None:
        pr = pr if isinstance(pr, Probe) else (self.current if isinstance(self.current, Probe) else None)
        run = self.run
        if pr is None or run is None or run.collector is None or not any(q is pr for q in self._graph_probes):
            self._message("Run the simulation with this probe active first", "warning")
            return
        i = next(j for j, q in enumerate(self._graph_probes) if q is pr)
        name = re.sub(r"[^\w.-]", "_", pr.name.strip()) or "probe"
        path = Path.cwd() / f"pywavedude_{name}.csv"
        rows = write_csv(path, pr, run.collector.traces[i])
        self._message(f"Wrote {rows} rows to {path}")
