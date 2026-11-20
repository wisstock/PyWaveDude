# SPDX-License-Identifier: GPL-3.0-or-later
# PyWaveDude: Python port of WaveDude Next (C) Alexander Graef, after Jeroen Vleggaar (Huygens Optics) and 0x23.
"""The timeline and the transport controls (mixin of the app).

The position is a frame of the project (frame k = the state after k × Visualize every steps); the run of a preview
that keeps every n-th frame only shows the nearest kept frame. The in and out points are the export range (the
Display settings' From step and To step), so they are saved with the project."""

from __future__ import annotations

import dearpygui.dearpygui as dpg

from . import config
from . import ui_theme as th
from .probe_graph import nice_ticks



class TimelineMixin:
    TIMELINE_H = 42

    def _init_timeline(self) -> None:
        self.position = 0
        self.playing = False
        self.loop = bool(config.app("playback.loop", False))
        self.live = bool(config.app("playback.live", False))
        self.speed = float(config.app("playback.speed", 1.0))
        self._play_acc = 0.0
        self._last_tick = 0.0
        self._tl_key = None

    # ------------------------------------------------------------ frames
    def _last_frame(self) -> int:
        m = self.p.sim
        return max(0, m.steps // max(1, m.visualize_every))

    def _range(self) -> tuple[int, int]:
        d, e = self.p.display, max(1, self.p.sim.visualize_every)
        last = self._last_frame()
        a = min(last, d.export_from // e)
        b = min(last, d.export_to // e) if d.export_to > 0 else last
        return a, max(a, b)

    def _computed_frames(self) -> int:
        """How far the run's frames reach, in project frames."""
        if self.run is None:
            return -1
        n = self.run.frame_count()
        return (n - 1) * self._div() if n else -1

    def _div(self) -> int:
        return max(1, int(self.p.sim.preview_frames))

    def _set_position(self, pos: int, by_user: bool = True) -> None:
        self.position = min(max(0, int(pos)), self._last_frame())
        if by_user:
            self.live = False
            if dpg.does_item_exist("live_chk"):
                dpg.set_value("live_chk", False)
        self._tl_key = None

    # ------------------------------------------------------------ building
    def _build_timeline(self) -> None:
        dpg.add_drawlist(400, self.TIMELINE_H, tag="timeline")
        with dpg.group(horizontal=True, tag="transport"):
            b = dpg.add_button(label="Run", callback=self.start, width=50)
            dpg.bind_item_theme(b, "theme_accent_button")
            self._tip(b, "Run the simulation from step 0 (F5)")
            self._tip(dpg.add_button(label="Pause run", tag="pause_run_btn", callback=self._toggle_run_pause),
                      "Pause or resume the computation (F6)")
            self._tip(dpg.add_button(label="Stop", callback=self.stop), "Discard the run (Shift+F5)")
            dpg.add_spacer(width=6)
            self._tip(dpg.add_button(label="▶ Play", tag="play_btn", width=70, callback=self._toggle_play),
                      "Play or pause the computed frames (Space)")
            self._tip(dpg.add_button(label="◀", callback=lambda: self._step_frame(-1)), "One frame back (,)")
            self._tip(dpg.add_button(label="▶", callback=lambda: self._step_frame(1)), "One frame forward (.)")
            dpg.add_spacer(width=6)
            self._tip(dpg.add_button(label="In", callback=self._set_in), "Start the export range here (I)")
            self._tip(dpg.add_button(label="Out", callback=self._set_out), "End the export range here (O)")
            self._tip(dpg.add_button(label="Clear", callback=self._clear_range), "Export the whole run (X)")
            self._tip(dpg.add_checkbox(label="Loop", tag="loop_chk", default_value=self.loop,
                                       callback=lambda s, v: setattr(self, "loop", v)),
                      "Loop playback over the export range (L)")
            self._tip(dpg.add_checkbox(label="Live", tag="live_chk", default_value=self.live,
                                       callback=lambda s, v: setattr(self, "live", v)),
                      "Follow the newest frame while the run computes (F); scrubbing ends it")
            dpg.add_spacer(width=6)
            dpg.add_text("", tag="pos_text")
            dpg.add_spacer(width=6)
            dpg.add_text("Speed", color=th.TEXT_DIM)
            speeds = [f"{float(v):g}" for v in config.app("playback.speed_choices", (1,))]
            dpg.add_combo(speeds, default_value=f"{self.speed:g}", width=60, tag="speed_combo",
                          callback=lambda s, v: setattr(self, "speed", float(v)))
            dpg.add_spacer(width=6)
            self._tip(dpg.add_checkbox(label="Field", tag="show_field", callback=self._layers_toggled),
                      "Show the field picture")
            self._tip(dpg.add_checkbox(label="Intensity", tag="show_int", callback=self._layers_toggled),
                      "Show the intensity picture")
            self._tip(dpg.add_combo(("Stacked", "Side by side"), width=110, tag="arr_combo",
                                    callback=self._arrangement_changed), "Pictures one above the other or side by side")
        with dpg.group(horizontal=True, tag="view_row"):
            self._tip(dpg.add_checkbox(label="Grid", tag="row_grid", callback=lambda s, v: self._row_set("grid", v)),
                      "Grid lines over the pictures (never exported)")
            dpg.add_text("step", color=th.TEXT_DIM)
            self._tip(dpg.add_input_int(tag="row_grid_step", width=90, min_value=2, max_value=4096, min_clamped=True,
                                        max_clamped=True, step=10, callback=lambda s, v: self._row_set("grid_step", v)),
                      "Distance of the grid lines in canvas pixels")
            self._tip(dpg.add_checkbox(label="Snap", tag="row_snap",
                                       callback=lambda s, v: self._row_set("grid_snap", v)),
                      "Centres of moved and placed objects stick to grid lines and nodes")
            dpg.add_text("opacity", color=th.TEXT_DIM)
            dpg.add_slider_float(tag="row_grid_op", width=80, min_value=0.0, max_value=1.0, format="%.2f",
                                 callback=lambda s, v: self._row_set("grid_opacity", v))
            dpg.add_spacer(width=14)
            self._tip(dpg.add_checkbox(label="Graphs on the canvas", tag="row_graphs",
                                       callback=lambda s, v: self._row_set("graphs_in_preview", v)),
                      "Probe graph boxes over the pictures")
            self._tip(dpg.add_checkbox(label="Graph column", tag="row_graph_col",
                                       callback=lambda s, v: self._show_graphs(v)),
                      "The probe graphs in a column at the right edge of the window")
            dpg.add_spacer(width=14)
            self._tip(dpg.add_checkbox(label="Separate wavelengths", tag="row_split",
                                       callback=lambda s, v: self._set_sim("split_wavelengths", v)),
                      "One simulation per source wavelength: the intensity per colour (needs a new run)")
            self._tip(dpg.add_checkbox(label="True colour", tag="row_true",
                                       callback=lambda s, v: self._row_set("true_colour", v)),
                      "Intensity of each wavelength in its own colour, added like light (with separate wavelengths); "
                      "off: the intensity scheme")
        self._sync_transport()

    def _row_set(self, attr: str, value) -> None:
        self._set_display(attr, value)
        if attr.startswith("grid"):
            self._layout_dirty = True

    @staticmethod
    def _tip(item, text):
        with dpg.tooltip(item):
            dpg.add_text(text)
        return item

    def _sync_transport(self) -> None:
        d = self.p.display
        if dpg.does_item_exist("show_field"):
            dpg.set_value("show_field", d.layer in ("Field", "Both"))
            dpg.set_value("show_int", d.layer in ("Intensity", "Both"))
            dpg.set_value("arr_combo", d.arrangement)
        if dpg.does_item_exist("row_grid"):
            for tag, v in (("row_grid", d.grid), ("row_grid_step", d.grid_step), ("row_snap", d.grid_snap),
                           ("row_grid_op", d.grid_opacity), ("row_graphs", d.graphs_in_preview),
                           ("row_graph_col", self._graphs_shown), ("row_split", self.p.sim.split_wavelengths),
                           ("row_true", d.true_colour)):
                if not dpg.is_item_active(tag) and dpg.get_value(tag) != v:
                    dpg.set_value(tag, v)

    def _layers_toggled(self, *_):
        f, i = dpg.get_value("show_field"), dpg.get_value("show_int")
        if not f and not i:
            f = True
            dpg.set_value("show_field", True)
        self.p.display.layer = "Both" if f and i else ("Field" if f else "Intensity")
        self._layout_dirty = True
        self._edited("view")
        self._model_changed()

    def _arrangement_changed(self, _s, v):
        self.p.display.arrangement = v
        self._layout_dirty = True
        self._edited("view")
        self._model_changed()

    # ------------------------------------------------------------ commands
    def _toggle_run_pause(self, *_):
        if self.run is not None and self.run.running:
            self.run.toggle_pause()

    def _set_playing(self, on: bool) -> None:
        self.playing = on
        self._play_acc = 0.0
        if dpg.does_item_exist("play_btn"):
            dpg.configure_item("play_btn", label="II Pause" if on else "▶ Play")

    def _toggle_play(self, *_):
        """Play / Pause: the animation of the computed frames only; the computation goes on regardless."""
        if self.run is None:
            self._message("Run the simulation first (F5)", "warning")
            return
        if not self.playing:
            a, b = self._range() if self.loop else (0, self._last_frame())
            if self.position >= min(b, max(0, self._computed_frames())) and not self.run.running:
                self.position = a        # at the end of a finished run: play from the start
        self._set_playing(not self.playing)

    def _step_frame(self, d: int) -> None:
        self._set_position(self.position + d * self._div())

    def _set_in(self, *_):
        e = max(1, self.p.sim.visualize_every)
        self.p.display.export_from = self.position * e
        if self.p.display.export_to and self.p.display.export_to < self.p.display.export_from:
            self.p.display.export_to = 0
        self._edited("view")
        self._tl_key = None

    def _set_out(self, *_):
        e = max(1, self.p.sim.visualize_every)
        self.p.display.export_to = max(self.position * e, self.p.display.export_from)
        self._edited("view")
        self._tl_key = None

    def _clear_range(self, *_):
        self.p.display.export_from = self.p.display.export_to = 0
        self._edited("view")
        self._tl_key = None

    # ------------------------------------------------------------ playback clock
    def _tick_playback(self, now: float) -> None:
        dt, self._last_tick = now - self._last_tick, now
        if self.run is None:
            return
        computed = self._computed_frames()
        if self.live and self.run.running and computed >= 0:
            self.position = computed
            return
        if not self.playing:
            return
        self._play_acc += min(dt, 0.25) * max(1, self.p.display.fps) * self.speed * self._div() \
            / max(1, self._div())
        steps = int(self._play_acc)
        if steps <= 0:
            return
        self._play_acc -= steps
        a, b = self._range() if self.loop else (0, self._last_frame())
        nxt = self.position + steps * self._div()
        if nxt > min(b, computed):
            if nxt > b and self.loop:
                nxt = a
            elif self.run.running:
                nxt = min(b, computed)  # the run is waited for
            else:
                nxt = min(b, computed)
                self._set_playing(False)
        self.position = max(0, nxt)

    # ------------------------------------------------------------ drawing and the mouse
    def _tl_geom(self):
        w, _ = dpg.get_item_rect_size("timeline")
        last = max(1, self._last_frame())
        x0, x1 = 10.0, max(20.0, w - 10.0)
        return x0, x1, last

    def _redraw_timeline(self) -> None:
        if not dpg.does_item_exist("timeline"):
            return
        cw, _ = dpg.get_item_rect_size("center")
        width = max(100, cw - 16)
        if dpg.get_item_width("timeline") != width:
            dpg.configure_item("timeline", width=width)
        x0, x1, last = self._tl_geom()
        a, b = self._range()
        key = (width, last, self.position, self._computed_frames(), a, b, self.p.display.export_to > 0 or a > 0)
        if key == self._tl_key:
            return
        self._tl_key = key
        dpg.delete_item("timeline", children_only=True)
        X = lambda f: x0 + (x1 - x0) * f / last  # noqa: E731
        H = self.TIMELINE_H
        dpg.draw_rectangle((x0, 18), (x1, H - 4), fill=(45, 46, 52, 255), color=(70, 72, 80, 255), parent="timeline")
        comp = self._computed_frames()
        if comp >= 0:
            dpg.draw_rectangle((x0, 18), (X(min(comp, last)), H - 4), fill=(0, 90, 150, 140), color=(0, 0, 0, 0),
                               parent="timeline")
        if key[-1]:
            dpg.draw_rectangle((X(a), 18), (X(b), H - 4), fill=(255, 255, 255, 25), color=(0, 0, 0, 0),
                               parent="timeline")
            for f, sgn in ((a, 1), (b, -1)):
                x = X(f)
                dpg.draw_polyline([(x + 5 * sgn, 18), (x, 18), (x, H - 4), (x + 5 * sgn, H - 4)],
                                  color=(255, 200, 60, 255), thickness=2, parent="timeline")
        target = max(2, int((x1 - x0) / 70))
        for f in nice_ticks(0, last, target):
            if f < 0 or f > last:
                continue
            x = X(f)
            dpg.draw_line((x, 14), (x, 18), color=(150, 150, 160, 255), parent="timeline")
            dpg.draw_text((x - 8, 1), f"{int(f)}", size=12, color=th.TEXT_DIM, parent="timeline")
        x = X(self.position)
        dpg.draw_line((x, 14), (x, H - 4), color=(255, 255, 255, 255), thickness=1.5, parent="timeline")
        dpg.draw_triangle((x - 5, 10), (x + 5, 10), (x, 17), color=(255, 255, 255, 255), fill=(255, 255, 255, 255),
                          parent="timeline")

    def _timeline_press(self) -> None:
        mx, _ = dpg.get_drawing_mouse_pos()
        x0, x1, last = self._tl_geom()
        f = round((mx - x0) / (x1 - x0) * last)
        a, b = self._range()
        X = lambda q: x0 + (x1 - x0) * q / last  # noqa: E731
        has_range = self.p.display.export_to > 0 or a > 0
        mode = "scrub"
        if has_range and abs(mx - X(a)) < 6:
            mode = "in"
        elif has_range and abs(mx - X(b)) < 6:
            mode = "out"
        self._tl_drag = mode
        self._timeline_drag_to(f)

    def _timeline_drag_to(self, f: int) -> None:
        mode = getattr(self, "_tl_drag", None)
        f = min(max(0, f), self._last_frame())
        e = max(1, self.p.sim.visualize_every)
        if mode == "scrub":
            self._set_position(f)
        elif mode == "in":
            self.p.display.export_from = min(f * e, self.p.display.export_to or self.p.sim.steps)
            self._edited("view")
        elif mode == "out":
            self.p.display.export_to = max(f * e, self.p.display.export_from)
            self._edited("view")
        self._tl_key = None

    def _update_timeline_drag(self) -> None:
        if getattr(self, "_tl_drag", None) is None:
            return
        if not dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
            self._tl_drag = None
            return
        mx = dpg.get_mouse_pos(local=False)[0] - dpg.get_item_rect_min("timeline")[0]
        x0, x1, last = self._tl_geom()
        self._timeline_drag_to(round((mx - x0) / (x1 - x0) * last))

    def _position_text(self) -> str:
        e = max(1, self.p.sim.visualize_every)
        return f"frame {self.position} / {self._last_frame()}   step {self.position * e}"
