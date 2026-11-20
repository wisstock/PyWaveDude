# SPDX-License-Identifier: GPL-3.0-or-later
# PyWaveDude: Python port of WaveDude Next (C) Alexander Graef, after Jeroen Vleggaar (Huygens Optics) and 0x23.
"""Export dialog for the app (mixin): videos, sequences, OpenEXR, and the shown frame as a still."""

from __future__ import annotations

from pathlib import Path

import dearpygui.dearpygui as dpg

from .export import (DATA, EXTENSIONS, STILLS, ExportFormat, ExportJob, ExportLayer, ExportSpec, available_formats,
                     ffmpeg_path)
from . import config
from .canvas_edit import part_center
from .render import builtin_schemes

LAYER_CHOICES = ("Field", "Intensity", "Both")


class ExportMixin:
    """Needs ``self.p`` (sim, display), ``self._build_engine``, ``self._scene_edges``, ``self._shown_frame``,
    ``self._message``."""

    def _init_export(self) -> None:
        self.export_job: ExportJob | None = None
        self._last_export: dict | None = None

    def _build_export_window(self) -> None:
        fmts = [f.value for f in available_formats()]
        with dpg.window(label="Export", tag="export_win", width=460, height=420, pos=(420, 120), show=False):
            if ffmpeg_path() is None:
                dpg.add_text("MP4 and MPEG-1 need ffmpeg on the PATH; not found.", color=(255, 200, 120), wrap=430)
            want = ExportFormat[str(config.app("export.format", "MP4"))].value
            dpg.add_combo(fmts, label="Format", tag="exp_fmt", default_value=want
                          if want in fmts else ExportFormat.AVI.value, width=220,
                          callback=self._export_fmt_changed)
            dpg.add_combo(LAYER_CHOICES, label="Layers", tag="exp_layers",
                          default_value=str(config.app("export.layers", "Both")), width=220)
            dpg.add_input_text(label="File", tag="exp_path", width=320)
            dpg.add_input_int(label="Frame rate", tag="exp_fps", default_value=self.p.display.fps, min_value=1, max_value=240,
                              min_clamped=True, max_clamped=True, width=150)
            dpg.add_input_int(label="Quality (MP4, MPEG, JPEG)", tag="exp_quality", default_value=self.p.display.quality, min_value=1,
                              max_value=100, min_clamped=True, max_clamped=True, width=150)
            dpg.add_input_int(label="From step", tag="exp_from", default_value=0, min_value=0, min_clamped=True,
                              width=150)
            dpg.add_input_int(label="To step (0 = last)", tag="exp_to", default_value=0, min_value=0,
                              min_clamped=True, width=150)
            dpg.add_checkbox(label="Part outlines in export", tag="exp_edges",
                             default_value=bool(config.app("export.outlines", True)))
            d = self.p.display
            dpg.add_checkbox(label="Probe graphs in export", tag="exp_graphs", default_value=d.probe_graphs_in_export,
                             callback=lambda _s, v: setattr(self.p.display, "probe_graphs_in_export", v))
            dpg.add_checkbox(label="Probe markers in export", tag="exp_pmarkers", default_value=d.probe_markers_in_export,
                             callback=lambda _s, v: setattr(self.p.display, "probe_markers_in_export", v))
            dpg.add_checkbox(label="Source markers in export", tag="exp_smarkers",
                             default_value=d.source_markers_in_export,
                             callback=lambda _s, v: setattr(self.p.display, "source_markers_in_export", v))
            dpg.add_checkbox(label="Labels in export", tag="exp_labels", default_value=d.labels_in_export,
                             callback=lambda _s, v: setattr(self.p.display, "labels_in_export", v))
            with dpg.group(horizontal=True):
                dpg.add_button(label="Export video", callback=lambda: self._start_export(still=False))
                dpg.add_button(label="Export shown frame", callback=lambda: self._start_export(still=True))
                dpg.add_button(label="Cancel", callback=self._cancel_export)
            dpg.add_progress_bar(tag="exp_progress", width=-1)
            dpg.add_text("", tag="exp_status", wrap=430)
        self._export_fmt_changed()

    def _show_export(self, layers: str | None = None):
        """The export window, with the layers preset (File > Export > Field / Intensity / Both)."""
        if isinstance(layers, str) and layers in LAYER_CHOICES:
            dpg.set_value("exp_layers", layers)
        d = self.p.display
        dpg.set_value("exp_from", d.export_from)
        dpg.set_value("exp_to", d.export_to)
        dpg.show_item("export_win")
        dpg.focus_item("export_win")

    def _export_last(self, *_):
        """The session's last run export again: same layers, format and file, no dialog."""
        last = self._last_export
        if last is None:
            self._message("No export yet in this session", "warning")
            return
        for tag, v in last.items():
            dpg.set_value(tag, v)
        self._start_export(still=False)

    def _fmt(self) -> ExportFormat:
        return ExportFormat(dpg.get_value("exp_fmt"))

    def _export_fmt_changed(self, *_):
        ext = EXTENSIONS[self._fmt()]
        cur = dpg.get_value("exp_path") or str(Path.cwd() / str(config.app("export.file", "pywavedude_export")))
        dpg.set_value("exp_path", str(Path(cur).with_suffix(ext)))

    def _export_layers(self) -> list[ExportLayer]:
        d = self.p.display
        choice = dpg.get_value("exp_layers")
        gain = d.outline_gain if dpg.get_value("exp_edges") and d.outline else 0.0
        schemes = self._schemes()
        f = ExportLayer(False, schemes[d.field_scheme].lut(d.field_black), d.field_brightness, gain)
        i = ExportLayer(True, schemes[d.intensity_scheme].lut(d.intensity_black), d.intensity_brightness, gain,
                        d.true_colour and self.p.sim.split_wavelengths, d.nm_per_px, self.p.sim.source_frequency_scale,
                        d.intensity_black)
        return {"Field": [f], "Intensity": [i], "Both": [f, i]}[choice]

    def _start_export(self, still: bool) -> None:
        if self.export_job is not None and self.export_job.running:
            dpg.set_value("exp_status", "An export is already running")
            return
        fmt = self._fmt()
        if still and fmt not in STILLS:
            fmt = ExportFormat.PNG
        m = self.p.sim
        last = dpg.get_value("exp_to") or m.steps
        first = dpg.get_value("exp_from")
        if still:
            fr = self._shown_frame
            first = last = fr.step if fr is not None else 0
        path = Path(dpg.get_value("exp_path")).with_suffix(EXTENSIONS[fmt])
        if still:
            path = path.with_name(f"{path.stem}_step{first}{path.suffix}")
        try:
            eng = self._build_engine()
        except Exception as exc:
            dpg.set_value("exp_status", f"Cannot build the engine: {exc}")
            return
        if not still:
            self._last_export = {t: dpg.get_value(t) for t in ("exp_fmt", "exp_layers", "exp_path", "exp_fps",
                                                                  "exp_quality", "exp_from", "exp_to", "exp_edges")}
        spec = ExportSpec(fmt, path, self._export_layers(), first, min(last, m.steps), m.visualize_every,
                          dpg.get_value("exp_fps"), dpg.get_value("exp_quality"), still, self._export_decor(eng))
        self.export_job = ExportJob(lambda: eng, spec, self._scene_edges)
        self.export_job.start()

    def _export_decor(self, eng=None) -> dict:
        d = self.p.display
        active = [pr for pr in self.p.probes if pr.active]
        sources = [(e[0], e[1]) for s in self.p.sources if s.enabled for e in s.elements(self.p.seed)] \
            if d.source_markers_in_export else []
        labels = []
        if d.labels_in_export:
            labels += [(*part_center(pt), self.name_of(pt)) for pt in self.p.parts if pt.enabled]
            labels += [(s.x, s.y, s.name) for s in self.p.sources if s.enabled]
            if not d.probe_graphs_in_export:            # with graphs, a probe's name is its graph's title
                labels += [(pr.ax, pr.ay, pr.name) for pr in active]
        return dict(probes=active, numbers=[self.p.probes.index(pr) + 1 for pr in active],
                    graphs=d.probe_graphs_in_export, markers=d.probe_markers_in_export, sources=sources,
                    labels=labels, group_colours=self._export_group_colours(eng),
                    seed=self.p.seed, total_steps=self.p.sim.steps)

    def _export_group_colours(self, eng) -> list:
        """RGBA per wavelength for intensity probes' per-colour curves (separate wavelengths and the switch on)."""
        from .engine import wavelength
        from .render import group_colours
        levels = getattr(eng, "levels", None)
        if not levels or not self.p.display.probe_colours:
            return []
        cols = group_colours([wavelength(lv, self.p.sim.source_frequency_scale) for lv in levels],
                             self.p.display.nm_per_px)
        return [tuple(int(255 * c) for c in col) + (255,) for col in cols]

    def _cancel_export(self, *_):
        if self.export_job is not None:
            self.export_job.cancel()

    def _update_export(self) -> None:
        job = self.export_job
        if job is None or not dpg.does_item_exist("exp_progress"):
            return
        pr = job.progress
        total = max(1, pr.frames_total)
        dpg.set_value("exp_progress", pr.frames_done / total)
        if pr.finished:
            if pr.error:
                msg = f"Failed: {pr.error}"
            elif pr.cancelled:
                msg = "Cancelled"
            else:
                msg = f"Wrote {pr.frames_done} frame(s) in {pr.seconds:.1f} s: " + ", ".join(pr.outputs)
            dpg.set_value("exp_status", msg)
        else:
            dpg.set_value("exp_status", f"Simulating step {pr.steps_done}, frame {pr.frames_done}/{pr.frames_total}")
