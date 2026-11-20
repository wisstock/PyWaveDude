# SPDX-License-Identifier: GPL-3.0-or-later
# PyWaveDude: Python port of WaveDude Next (C) Alexander Graef, after Jeroen Vleggaar (Huygens Optics) and 0x23.
"""About, the manual (the README rendered), the memory warning and the colour scheme form (mixin of the app)."""

from __future__ import annotations

import os
import platform
import re
from dataclasses import replace
from pathlib import Path

import dearpygui.dearpygui as dpg

from . import ui_theme as th
from .render import ColorScheme

README = Path(__file__).resolve().parents[2] / "README.md"


def physical_memory() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 8 << 30


def system_info() -> str:
    lines = [f"PyWaveDude {_version('pywavedude')}", f"Python {platform.python_version()} on "
             f"{platform.system()} {platform.release()} ({platform.machine()})",
             f"Processor threads: {os.cpu_count()}", f"Memory: {physical_memory() / 2 ** 30:.1f} GB"]
    for pkg in ("numpy", "numba", "dearpygui", "pillow", "mlx"):
        v = _version(pkg)
        if v:
            lines.append(f"{pkg} {v}")
    try:
        from .engine import gpu_mlx
        if gpu_mlx.available():
            import mlx.core as mx
            info = (mx.device_info if hasattr(mx, "device_info") else mx.metal.device_info)()
            lines.append(f"GPU: {info.get('device_name', 'Metal')} ({info.get('memory_size', 0) / 2 ** 30:.0f} GB)")
        else:
            lines.append("GPU: none (CPU engine)")
    except Exception as exc:
        lines.append(f"GPU: unavailable ({exc})")
    from .export import ffmpeg_path
    lines.append(f"ffmpeg: {ffmpeg_path() or 'not found'}")
    return "\n".join(lines)


def _version(pkg: str) -> str:
    from importlib import metadata
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return ""


def store_bytes(width: int, height: int, steps: int, every: int, preview_step: int, frames_div: int,
                planes: int = 2, groups: int = 0) -> int:
    """Bytes the frames of a preview run take (float16 planes)."""
    h, w = -(-height // preview_step), -(-width // preview_step)
    frames = steps // max(1, every * frames_div) + 1
    return frames * h * w * 2 * (planes + groups)


class DialogsMixin:
    def _init_dialogs(self) -> None:
        from . import config
        self.warn_memory = bool(config.app("run.warn_memory", True))
        self._stop_sel = None
        self._stop_drag = None

    # ------------------------------------------------------------ About
    def _show_about(self, *_):
        if dpg.does_item_exist("about_win"):
            dpg.delete_item("about_win")
        info = system_info()
        with dpg.window(label="About PyWaveDude", tag="about_win", modal=True, width=520, height=360, pos=(380, 160)):
            t = dpg.add_text("PyWaveDude", color=(120, 170, 230))
            dpg.bind_item_font(t, th.font("heading"))
            dpg.add_text("A 2D wave optics simulator: Python port of WaveDude Next by Alexander Graef, after Wave Dude "
                         "2D by Jeroen Vleggaar (Huygens Optics).", wrap=490)
            dpg.add_text("License: GPL-3.0-or-later. Font: Inter (SIL Open Font License 1.1).", wrap=490,
                         color=th.TEXT_DIM)
            dpg.add_separator()
            dpg.add_input_text(default_value=info, multiline=True, readonly=True, width=-1, height=170)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Copy system information", callback=lambda: dpg.set_clipboard_text(info))
                dpg.add_button(label="Close", callback=lambda: dpg.delete_item("about_win"))

    # ------------------------------------------------------------ Manual
    def _show_manual(self, *_):
        if dpg.does_item_exist("manual_win"):
            dpg.show_item("manual_win")
            dpg.focus_item("manual_win")
            return
        text = README.read_text(encoding="utf-8") if README.exists() else "README.md not found."
        with dpg.window(label="Manual", tag="manual_win", width=760, height=720, pos=(200, 60)):
            with dpg.group(horizontal=True):
                with dpg.child_window(width=200, tag="manual_toc"):
                    pass
                with dpg.child_window(tag="manual_body"):
                    pass
        self._render_markdown(text)

    def _render_markdown(self, text: str) -> None:
        body, toc = "manual_body", "manual_toc"
        in_code = False
        table: list[list[str]] = []

        def flush_table():
            if not table:
                return
            cols = max(len(r) for r in table)
            with dpg.table(parent=body, header_row=True, borders_innerH=True, borders_outerH=True,
                           borders_innerV=True, borders_outerV=True, policy=dpg.mvTable_SizingStretchProp):
                for c in range(cols):
                    dpg.add_table_column(label=_inline(table[0][c]) if c < len(table[0]) else "")
                for row in table[1:]:
                    with dpg.table_row():
                        for c in range(cols):
                            dpg.add_text(_inline(row[c]) if c < len(row) else "", wrap=0)
            table.clear()

        for line in text.splitlines():
            if line.startswith("```"):
                flush_table()
                in_code = not in_code
                continue
            if in_code:
                dpg.add_text("    " + line, parent=body, color=(170, 210, 160))
                continue
            if line.startswith("|"):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    table.append(cells)
                continue
            flush_table()
            m = re.match(r"^(#{1,4})\s+(.*)", line)
            if m:
                level, title = len(m.group(1)), _inline(m.group(2))
                h = dpg.add_text(title, parent=body, color=(120, 170, 230) if level > 1 else (230, 230, 235))
                if level <= 2:
                    dpg.bind_item_font(h, th.font("heading"))
                dpg.add_button(label=("" if level <= 2 else "   ") + title, parent=toc, small=True,
                               callback=lambda s, a, u: dpg.set_y_scroll(body, dpg.get_item_pos(u)[1]), user_data=h)
                dpg.bind_item_theme(dpg.last_item(), "theme_small_button")
                continue
            if not line.strip():
                dpg.add_spacer(height=4, parent=body)
                continue
            dpg.add_text(_inline(line), parent=body, wrap=0)
        flush_table()

    # ------------------------------------------------------------ memory warning
    def _memory_need(self) -> tuple[int, int]:
        s, m = self.p.scene, self.p.sim
        from .ui_preview import PREVIEW
        groups = 0
        if m.split_wavelengths:
            groups = len({src.level for src in self.p.sources if src.enabled}) or 1
        need = store_bytes(s.width, s.height, m.steps, m.visualize_every, PREVIEW[m.preview], m.preview_frames,
                           groups=groups)
        return need, self.memory_budget()

    @staticmethod
    def memory_budget() -> int:
        from . import config
        return max(1 << 30, int(physical_memory() * float(config.app("run.memory_share", 0.5))))

    def _ask_memory(self, then) -> bool:
        """True: go on now. False: a warning is shown and ``then()`` runs when the user chooses Run."""
        if not self.warn_memory:
            return True
        need, budget = self._memory_need()
        if need <= budget:
            return True
        share = self.memory_budget() / physical_memory()
        if dpg.does_item_exist("mem_win"):
            dpg.delete_item("mem_win")
        m = self.p.sim
        from .ui_preview import PREVIEW
        levers = []
        if PREVIEW[m.preview] < 4:
            levers.append(("Half the resolution", "res"))
        if m.preview_frames < 4:
            levers.append(("Half the frames (keep every second one)", "frames"))

        def result():
            res = PREVIEW[m.preview] * (2 if dpg.get_value("mem_res") else 1) if dpg.does_item_exist("mem_res") \
                else PREVIEW[m.preview]
            fr = m.preview_frames * (2 if dpg.does_item_exist("mem_frames") and dpg.get_value("mem_frames") else 1)
            s = self.p.scene
            return store_bytes(s.width, s.height, m.steps, m.visualize_every, min(res, 4), min(fr, 4))

        def update(*_):
            dpg.set_value("mem_result", f"With these: {result() / 2 ** 30:.2f} GB "
                          f"({'fits' if result() <= budget else 'still more than the budget; the oldest frames go'})")

        def run(*_):
            names = {1: "Full", 2: "Half", 4: "Quarter"}
            if dpg.does_item_exist("mem_res") and dpg.get_value("mem_res"):
                m.preview = names[min(4, PREVIEW[m.preview] * 2)]
            if dpg.does_item_exist("mem_frames") and dpg.get_value("mem_frames"):
                m.preview_frames = min(4, m.preview_frames * 2)
            if dpg.get_value("mem_never"):
                self.warn_memory = False
                if dpg.does_item_exist("menu_warn"):
                    dpg.set_value("menu_warn", False)
            dpg.delete_item("mem_win")
            self._model_changed()
            then()

        with dpg.window(label="The run does not fit in memory", tag="mem_win", modal=True, width=480, pos=(420, 220),
                        no_resize=True):
            dpg.add_text(f"The run's frames need {need / 2 ** 30:.2f} GB; the budget is {budget / 2 ** 30:.2f} GB "
                         f"({share:.0%} of the memory). Beyond it only the newest frame is kept.", wrap=460)
            for i, (label, tag) in enumerate(levers):
                dpg.add_checkbox(label=label, tag=f"mem_{tag}", callback=update)
            dpg.add_text("", tag="mem_result", color=th.TEXT_DIM)
            dpg.add_checkbox(label="Do not warn me again", tag="mem_never")
            with dpg.group(horizontal=True):
                dpg.add_button(label="Run", callback=run, width=80)
                dpg.add_button(label="Cancel", width=80, callback=lambda: dpg.delete_item("mem_win"))
        # the fewest ticks that fit, the cache-friendly ones first
        for tag in ("frames", "res"):
            if result() > budget and dpg.does_item_exist(f"mem_{tag}"):
                dpg.set_value(f"mem_{tag}", True)
        update()
        return False

    # ------------------------------------------------------------ colour scheme form
    GRAD_H = 26
    MARK_H = 12

    def _scheme_form(self, cs: ColorScheme) -> None:
        self._stop_sel = 0 if self._stop_sel is None or self._stop_sel >= len(cs.stops) else self._stop_sel
        with dpg.table(header_row=False, parent="props_body", policy=dpg.mvTable_SizingStretchProp):
            dpg.add_table_column(init_width_or_weight=0.35)
            dpg.add_table_column(init_width_or_weight=0.65)
            with dpg.table_row():
                dpg.add_text("Name")
                dpg.add_input_text(default_value=cs.name, width=-1, on_enter=True,
                                   callback=lambda s, v: self._scheme_rename(v))
            with dpg.table_row():
                dpg.add_text("Kind")
                dpg.add_combo(("", "Field", "Intensity"), default_value=cs.kind, width=-1,
                              callback=lambda s, v: self._scheme_set(replace(self.current, kind=v)))
        dpg.add_text("Gradient: click to add a stop, drag a marker to move it, drag it away or double-click it to "
                     "remove it.", parent="props_body", wrap=0, color=th.TEXT_DIM)
        dpg.add_drawlist(300, self.GRAD_H + self.MARK_H + 4, parent="props_body", tag="grad_dl")
        with dpg.group(horizontal=True, parent="props_body"):
            dpg.add_text("Stop")
            dpg.add_input_float(tag="stop_pos", width=90, step=0, format="%.3f",
                                callback=lambda s, v: self._stop_edit(pos=v))
            dpg.add_color_edit(tag="stop_col", no_alpha=True, width=150,
                               callback=lambda s, v: self._stop_edit(col=v))
            dpg.add_button(label="Remove stop", callback=lambda: self._stop_remove(self._stop_sel))
        dpg.add_text("Used by the Display settings' Field scheme / Intensity scheme.", parent="props_body",
                     color=th.TEXT_DIM, wrap=0)
        self._draw_gradient()

    def _scheme_set(self, new: ColorScheme) -> None:
        old = self.current
        i = self.index_in(self.p.color_schemes, old)
        if i < 0:
            return
        self.p.color_schemes[i] = new
        self.current = new
        self.selection = [new if o is old else o for o in self.selection]
        self._editor_obj = new
        self._rows[id(new)] = self._rows.pop(id(old), (None, None, new))
        self._update_row_label(new)
        self._luts.clear()
        self._edited("view")
        self._model_changed()
        self._draw_gradient()

    def _scheme_rename(self, name: str) -> None:
        name = name.strip()
        old = self.current
        if not name or name == old.name or any(c.name == name for c in self.p.color_schemes):
            return
        d = self.p.display
        if d.field_scheme == old.name:
            d.field_scheme = name
        if d.intensity_scheme == old.name:
            d.intensity_scheme = name
        self._scheme_set(replace(old, name=name))

    def _grad_geom(self):
        w = max(120, dpg.get_item_rect_size("props_body")[0] - 16)
        if dpg.does_item_exist("grad_dl") and dpg.get_item_width("grad_dl") != w:
            dpg.configure_item("grad_dl", width=w)
        return 8.0, w - 8.0

    def _draw_gradient(self) -> None:
        cs = self.current
        if not isinstance(cs, ColorScheme) or not dpg.does_item_exist("grad_dl"):
            return
        dpg.delete_item("grad_dl", children_only=True)
        x0, x1 = self._grad_geom()
        lut = cs.lut(0.0, 256)
        n = 128
        for i in range(n):
            c = tuple(int(v * 255) for v in lut[int(i / (n - 1) * 255)][:3])
            dpg.draw_rectangle((x0 + (x1 - x0) * i / n, 2), (x0 + (x1 - x0) * (i + 1) / n + 1, self.GRAD_H),
                               fill=c, color=c, parent="grad_dl")
        dpg.draw_rectangle((x0, 2), (x1, self.GRAD_H), color=th.BORDER, parent="grad_dl")
        for i, (pos, col) in enumerate(cs.stops):
            x = x0 + (x1 - x0) * pos
            y = self.GRAD_H + 2
            sel = i == self._stop_sel
            dpg.draw_triangle((x, y), (x - 6, y + self.MARK_H), (x + 6, y + self.MARK_H),
                              color=th.ACCENT_HOVER if sel else (200, 200, 205),
                              fill=th.ACCENT if sel else (*col, 255), parent="grad_dl")
        if self._stop_sel is not None and self._stop_sel < len(cs.stops):
            pos, col = cs.stops[self._stop_sel]
            if not dpg.is_item_active("stop_pos"):
                dpg.set_value("stop_pos", pos)
            dpg.set_value("stop_col", [*col, 255])

    def _set_stops(self, stops) -> None:
        stops = sorted(stops, key=lambda t: t[0])
        self._scheme_set(replace(self.current, stops=tuple(stops)))

    def _stop_edit(self, pos=None, col=None) -> None:
        cs = self.current
        if not isinstance(cs, ColorScheme) or self._stop_sel is None:
            return
        stops = list(cs.stops)
        p0, c0 = stops[self._stop_sel]
        if pos is not None:
            p0 = min(1.0, max(0.0, round(float(pos), 3)))
        if col is not None:
            c0 = tuple(int(round(v)) for v in col[:3])
        stops[self._stop_sel] = (p0, c0)
        moved = stops[self._stop_sel]
        stops.sort(key=lambda t: t[0])
        self._stop_sel = next(i for i, s in enumerate(stops) if s is moved)
        self._set_stops(stops)

    def _stop_remove(self, i) -> None:
        cs = self.current
        if not isinstance(cs, ColorScheme) or i is None or len(cs.stops) <= 1:
            return
        stops = [s for j, s in enumerate(cs.stops) if j != i]
        self._stop_sel = min(i, len(stops) - 1)
        self._set_stops(stops)

    def _gradient_press(self, double: bool = False) -> None:
        cs = self.current
        if not isinstance(cs, ColorScheme):
            return
        mx, my = dpg.get_drawing_mouse_pos()
        x0, x1 = self._grad_geom()
        pos = min(1.0, max(0.0, (mx - x0) / (x1 - x0)))
        near = min(range(len(cs.stops)), key=lambda i: abs(cs.stops[i][0] - pos), default=None)
        if near is not None and abs(cs.stops[near][0] - pos) * (x1 - x0) <= 7 and my >= self.GRAD_H - 2:
            if double:
                self._stop_remove(near)
                return
            self._stop_sel = near
            self._stop_drag = dict(i=near, y0=dpg.get_mouse_pos(local=False)[1])
            self._draw_gradient()
            return
        if double:
            return
        lut = cs.lut(0.0, 1024)
        c = tuple(int(v * 255) for v in lut[int(pos * 1023)][:3])
        stops = list(cs.stops) + [(round(pos, 3), c)]
        stops.sort(key=lambda t: t[0])
        self._stop_sel = next(i for i, s in enumerate(stops) if s == (round(pos, 3), c))
        self._stop_drag = dict(i=self._stop_sel, y0=dpg.get_mouse_pos(local=False)[1])
        self._set_stops(stops)

    def _update_stop_drag(self) -> None:
        dr = self._stop_drag
        if dr is None:
            return
        cs = self.current
        if not isinstance(cs, ColorScheme):
            self._stop_drag = None
            return
        gx, gy = dpg.get_mouse_pos(local=False)
        away = abs(gy - dr["y0"]) > 24
        if not dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
            self._stop_drag = None
            if away:
                self._stop_remove(dr["i"])
            return
        x0, x1 = self._grad_geom()
        lx = gx - dpg.get_item_rect_min("grad_dl")[0]
        pos = min(1.0, max(0.0, round((lx - x0) / (x1 - x0), 3)))
        if self._stop_sel is not None and abs(cs.stops[self._stop_sel][0] - pos) > 1e-9:
            self._stop_edit(pos=pos)
            dr["i"] = self._stop_sel


def _inline(s: str) -> str:
    """Markdown inline markup removed: links, emphasis and code."""
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    s = s.replace("`", "")
    return s
