# SPDX-License-Identifier: GPL-3.0-or-later
# PyWaveDude: Python port of WaveDude Next (C) Alexander Graef, after Jeroen Vleggaar (Huygens Optics) and 0x23.
"""The preview pictures (mixin of the app): a panel per layer with its header (title, step, brightness, Auto),
the picture, the grid and the overlays, and everything the mouse and the keys do on them.

Coordinates: canvas pixels for the model; a panel shows the canvas at ``k`` screen pixels per canvas pixel."""

from __future__ import annotations

import copy
import math

import dearpygui.dearpygui as dpg
import numpy as np

from . import config
from . import handles as hd
from . import ui_theme as th
from .canvas_edit import part_center, part_outline, pick, probe_center, snap_point
from .engine import PointSource
from .parts import model as pm
from .probes import Probe, ProbeKind
from .render import colorize, colour_mix, group_colours, wavelength_nm
from .engine import wavelength

PREVIEW = {"Full": 1, "Half": 2, "Quarter": 4}
HEADER_H = 28


def PICK() -> float:                 # screen pixels: how close a click must be to a marker
    return float(config.app("canvas.pick_radius", 8))


def SNAP() -> float:                 # screen pixels: how close a centre must come to a grid line
    return float(config.app("canvas.snap_radius", 8))


def LEVER() -> float:                # screen pixels beyond the outline
    return float(config.app("canvas.lever_length", 28))


class PreviewMixin:
    def _init_preview(self) -> None:
        self._panels: list[dict] = []
        self._layout_key = None
        self._layout_dirty = True
        self._overlays_dirty = True
        self._tex: dict[str, tuple] = {}         # layer -> (texture id, shape, buffer)
        self._drag: dict | None = None
        self._canvas_pick = False
        self._luts: dict = {}
        self._dots_key: dict = {}
        self.view_zoom = 1.0          # 1 = the whole canvas fits the panel
        self.view_x = self.view_y = 0.0   # canvas pixel at the panel's top-left corner
        self._pan: dict | None = None

    # ------------------------------------------------------------ sizes and panels
    def _canvas_size(self) -> tuple[int, int]:
        return self.p.scene.width, self.p.scene.height

    def _shown_layers(self) -> list[str]:
        d = self.p.display
        return {"Field": ["Field"], "Intensity": ["Intensity"], "Both": ["Field", "Intensity"]}.get(d.layer, ["Field"])

    def _layout_preview(self) -> None:
        """Panels sized to fit the centre pane; rebuilt when the layers or the arrangement change."""
        if not dpg.does_item_exist("pics"):
            return
        cw_, ch_ = dpg.get_item_rect_size("center")
        if cw_ <= 0:
            return
        avail_w, avail_h = max(80, cw_ - 14), max(80, ch_ - self.TIMELINE_H - 104)
        layers = self._shown_layers()
        side = self.p.display.arrangement == "Side by side" and len(layers) > 1
        cw, ch = self._canvas_size()
        n = len(layers)
        if side:
            k = min((avail_w - 8 * (n - 1)) / n / cw, (avail_h - HEADER_H) / ch)
        else:
            k = min(avail_w / cw, (avail_h - n * HEADER_H - 6 * (n - 1)) / n / ch)
        k = max(k, 0.01)
        w, h = max(8, int(cw * k)), max(8, int(ch * k))
        key = (tuple(layers), side, w, h, cw, ch)
        if key == self._layout_key and not self._layout_dirty:
            return
        if self._layout_key is not None and self._layout_key[4:] != key[4:]:
            self.view_zoom, self.view_x, self.view_y = 1.0, 0.0, 0.0     # another canvas: the whole of it
        rebuild = self._layout_key is None or key[:2] != self._layout_key[:2] or self._layout_dirty
        self._layout_key = key
        self._layout_dirty = False
        if rebuild:
            dpg.delete_item("pics", children_only=True)
            self._panels = []
            holder = dpg.add_group(horizontal=side, parent="pics")
            for layer in layers:
                with dpg.group(parent=holder):
                    with dpg.group(horizontal=True):
                        title = dpg.add_text(layer, color=(200, 205, 215))
                        dpg.bind_item_font(title, th.font("body"))
                        attr = "field_brightness" if layer == "Field" else "intensity_brightness"
                        dpg.add_text("  Brightness", color=th.TEXT_DIM)
                        b = dpg.add_input_float(default_value=getattr(self.p.display, attr), width=90, step=0,
                                                format="%.4g", callback=lambda s, v, u: self._set_brightness(u, v),
                                                user_data=attr)
                        dpg.add_button(label="Auto", callback=lambda s, a, u: self._auto_brightness(u),
                                       user_data=layer)
                        legend = dpg.add_group(horizontal=True)
                    dl = dpg.add_drawlist(w + 2, h + 2)
                    img = dpg.add_draw_layer(parent=dl)
                    grid = dpg.add_draw_layer(parent=dl)
                    dots = dpg.add_draw_layer(parent=dl)
                    ovl = dpg.add_draw_layer(parent=dl)
                self._panels.append(dict(layer=layer, dl=dl, img=img, grid=grid, dots=dots, ovl=ovl, title=title,
                                         bright=b, attr=attr, legend=legend, kf=k, k=k, w=w, h=h, ox=1.0, oy=1.0,
                                         vx=0.0, vy=0.0))
        else:
            for pn in self._panels:
                dpg.configure_item(pn["dl"], width=w + 2, height=h + 2)
                pn.update(kf=k, w=w, h=h)
        self._apply_view()

    # ------------------------------------------------------------ zoom and pan
    def _apply_view(self) -> None:
        """Every panel shows the same part of the canvas: ``view_zoom`` × the fit, from (view_x, view_y)."""
        cw, ch = self._canvas_size()
        self.view_zoom = min(max(self.view_zoom, 1.0), float(config.app("canvas.max_zoom", 32)))
        if self._panels:
            pn = self._panels[0]
            vw, vh = pn["w"] / (pn["kf"] * self.view_zoom), pn["h"] / (pn["kf"] * self.view_zoom)
            self.view_x = min(max(self.view_x, 0.0), max(0.0, cw - vw))
            self.view_y = min(max(self.view_y, 0.0), max(0.0, ch - vh))
        for pn in self._panels:
            pn.update(k=pn["kf"] * self.view_zoom, vx=self.view_x, vy=self.view_y)
        self._dots_key.clear()
        self._redraw_all()
        if dpg.does_item_exist("zoom_text"):
            dpg.set_value("zoom_text", f"{self.view_zoom * 100:.0f} %")

    def _zoom_at(self, factor: float, pn=None, at=None) -> None:
        """Zoom by ``factor`` keeping the canvas point ``at`` (default: the view's centre) where it is."""
        pn = pn or (self._panels[0] if self._panels else None)
        if pn is None:
            return
        if at is None:
            at = (pn["vx"] + pn["w"] / pn["k"] / 2, pn["vy"] + pn["h"] / pn["k"] / 2)
        sx, sy = (at[0] - pn["vx"]) * pn["k"], (at[1] - pn["vy"]) * pn["k"]     # its place in the panel
        self.view_zoom *= factor
        self.view_zoom = min(max(self.view_zoom, 1.0), float(config.app("canvas.max_zoom", 32)))
        k = pn["kf"] * self.view_zoom
        self.view_x, self.view_y = at[0] - sx / k, at[1] - sy / k
        self._apply_view()

    def _zoom_fit(self, *_) -> None:
        self.view_zoom, self.view_x, self.view_y = 1.0, 0.0, 0.0
        self._apply_view()

    def _on_wheel(self, _s, delta) -> None:
        pn = self._hovered_panel()
        if pn is None or not delta:
            return
        self._zoom_at(float(config.app("canvas.zoom_step", 1.2)) ** float(delta), pn, self._mouse_canvas(pn))

    def _start_pan(self, pn) -> None:
        self._pan = dict(mouse=tuple(dpg.get_mouse_pos(local=False)), v0=(self.view_x, self.view_y), k=pn["k"])

    def _update_pan(self) -> None:
        pan = self._pan
        if pan is None:
            return
        if not any(dpg.is_mouse_button_down(b) for b in (dpg.mvMouseButton_Left, dpg.mvMouseButton_Right,
                                                        dpg.mvMouseButton_Middle)):
            self._pan = None
            return
        mx, my = dpg.get_mouse_pos(local=False)
        x = pan["v0"][0] - (mx - pan["mouse"][0]) / pan["k"]
        y = pan["v0"][1] - (my - pan["mouse"][1]) / pan["k"]
        if (x, y) != (self.view_x, self.view_y):
            self.view_x, self.view_y = x, y
            self._apply_view()

    def _on_pan_click(self, *_) -> None:
        pn = self._hovered_panel()
        if pn is not None:
            self._start_pan(pn)

    def _set_brightness(self, attr: str, v: float) -> None:
        setattr(self.p.display, attr, max(1e-6, float(v)))
        self._edited("view")
        self._model_changed()

    def _sync_headers(self) -> None:
        fr = self._shown_frame
        step = fr.step if fr is not None else 0
        for pn in self._panels:
            label = f"{pn['layer']}   step {step}"
            if dpg.get_value(pn["title"]) != label:
                dpg.set_value(pn["title"], label)
            if not dpg.is_item_active(pn["bright"]):
                v = getattr(self.p.display, pn["attr"])
                if abs(dpg.get_value(pn["bright"]) - v) > 1e-9 * max(1, abs(v)):
                    dpg.set_value(pn["bright"], v)

    # ------------------------------------------------------------ pictures
    def _lut(self, name: str, black: float):
        key = (name, round(black, 4))
        if key not in self._luts:
            schemes = self._schemes()
            self._luts[key] = (schemes.get(name) or schemes["Gray scales"]).lut(black)
        return self._luts[key]

    def _true_colour(self, fr) -> bool:
        return self.p.display.true_colour and getattr(fr, "groups", None) is not None and bool(fr.levels)

    def _group_lambdas(self, fr) -> list[float]:
        return [wavelength(lv, self.p.sim.source_frequency_scale) for lv in fr.levels]

    def _compose_layer(self, layer: str, fr) -> np.ndarray:
        d = self.p.display
        if layer == "Field" or fr.intensity is None:
            vals = fr.field if layer == "Field" else np.zeros_like(fr.field)
            rgba = colorize(vals.astype(np.float32), self._lut(d.field_scheme, d.field_black), d.field_brightness,
                            False)
        elif self._true_colour(fr):
            rgba = colour_mix(fr.groups, group_colours(self._group_lambdas(fr), d.nm_per_px),
                              d.intensity_brightness, d.intensity_black)
        else:
            rgba = colorize(fr.intensity.astype(np.float32), self._lut(d.intensity_scheme, d.intensity_black),
                            d.intensity_brightness, True)
        edges = getattr(self, "_scene_edges", None)
        if d.outline and edges is not None and d.outline_gain > 0:
            ps = max(1, round(self.p.scene.width / max(1, fr.field.shape[1])))
            e = edges[::ps, ::ps][:rgba.shape[0], :rgba.shape[1]]
            if e.shape == rgba.shape[:2]:
                rgba[..., :3] = np.minimum(1.0, rgba[..., :3] + e.astype(np.float32) * (d.outline_gain / 255.0))
        return rgba

    def _upload(self, layer: str, rgba: np.ndarray) -> None:
        h, w = rgba.shape[:2]
        cur = self._tex.get(layer)
        if cur is None or cur[1] != (h, w):
            buf = np.ascontiguousarray(rgba, dtype=np.float32)
            tid = dpg.add_raw_texture(w, h, buf, format=dpg.mvFormat_Float_rgba, parent=self._texture_registry)
            if cur is not None:
                self._old_textures.append(cur[0])
            self._tex[layer] = (tid, (h, w), buf)
            self._dots_key.clear()
            self._redraw_all()
        else:
            np.copyto(cur[2], rgba)
            dpg.set_value(cur[0], cur[2])

    def _show_frame(self, fr) -> None:
        for layer in ("Field", "Intensity"):
            if layer in self._shown_layers():
                self._upload(layer, self._compose_layer(layer, fr))
        self._update_legend(fr)

    def _update_legend(self, fr) -> None:
        tc = fr is not None and self._true_colour(fr)
        key = (tc, tuple(fr.levels) if tc else (), self.p.display.nm_per_px)
        if getattr(self, "_legend_key", None) == key:
            return
        self._legend_key = key
        for pn in self._panels:
            dpg.delete_item(pn["legend"], children_only=True)
            if pn["layer"] != "Intensity" or not tc:
                continue
            lams = self._group_lambdas(fr)
            cols = group_colours(lams, self.p.display.nm_per_px)
            nms = wavelength_nm(lams, self.p.display.nm_per_px)
            for lam, nm, col in zip(lams, nms, cols):
                rgb = tuple(int(255 * c) for c in col)
                dpg.add_text("■", color=rgb, parent=pn["legend"])
                dpg.add_text(f"{lam:.1f} px" + (f" ({nm:.0f} nm)" if self.p.display.nm_per_px > 0 else ""),
                             parent=pn["legend"])

    # ------------------------------------------------------------ drawing
    def _edge_colour(self):
        s = self.p.scene
        if s.damping_edge < 0:
            return (40, 90, 255, 255)         # a damping rim outside the canvas
        if not s.open_edges:
            return (255, 40, 40, 255)         # a reflective wall
        return (90, 90, 96, 255)

    def _redraw_all(self) -> None:
        for pn in self._panels:
            dpg.delete_item(pn["img"], children_only=True)
            tex = self._tex.get(pn["layer"])
            x0, y0, w, h = pn["ox"], pn["oy"], pn["w"], pn["h"]
            cw, ch = self._canvas_size()
            u0, v0 = pn["vx"] / cw, pn["vy"] / ch                 # the part of the canvas in view
            u1, v1 = u0 + w / pn["k"] / cw, v0 + h / pn["k"] / ch
            if tex is not None:
                dpg.draw_image(tex[0], (x0, y0), (x0 + w, y0 + h), uv_min=(u0, v0), uv_max=(u1, v1),
                               parent=pn["img"])
            else:
                dpg.draw_rectangle((x0, y0), (x0 + w, y0 + h), fill=(0, 0, 0, 255), color=(0, 0, 0, 0),
                                   parent=pn["img"])
            col = self._edge_colour()                            # the canvas edges that are in view
            eps = 1e-6
            if u0 <= eps:
                dpg.draw_line((x0 - 1, y0 - 1), (x0 - 1, y0 + h), color=col, parent=pn["img"])
            if v0 <= eps:
                dpg.draw_line((x0 - 1, y0 - 1), (x0 + w, y0 - 1), color=col, parent=pn["img"])
            if u1 >= 1 - eps:
                dpg.draw_line((x0 + w, y0 - 1), (x0 + w, y0 + h), color=col, parent=pn["img"])
            if v1 >= 1 - eps:
                dpg.draw_line((x0 - 1, y0 + h), (x0 + w, y0 + h), color=col, parent=pn["img"])
            self._draw_grid(pn)
        self._overlays_dirty = True

    def _draw_grid(self, pn) -> None:
        dpg.delete_item(pn["grid"], children_only=True)
        d = self.p.display
        k, step = pn["k"], d.grid_step
        if not d.grid or step <= 0 or step * k < 4 or d.grid_opacity <= 0:
            return
        col = (255, 255, 255, int(round(255 * min(1.0, d.grid_opacity))))
        cw, ch = self._canvas_size()
        x0, y0, vx, vy = pn["ox"], pn["oy"], pn["vx"], pn["vy"]
        first_x, last_x = int(math.ceil(vx / step)) * step, min(cw, vx + pn["w"] / k)
        first_y, last_y = int(math.ceil(vy / step)) * step, min(ch, vy + pn["h"] / k)
        for g in range(first_x, int(last_x) + 1, step):
            dpg.draw_line((x0 + (g - vx) * k, y0), (x0 + (g - vx) * k, y0 + pn["h"]), color=col, parent=pn["grid"])
        for g in range(first_y, int(last_y) + 1, step):
            dpg.draw_line((x0, y0 + (g - vy) * k), (x0 + pn["w"], y0 + (g - vy) * k), color=col, parent=pn["grid"])

    def _S(self, pn, x, y):
        """Canvas pixels to the panel's drawing coordinates (zoomed and panned)."""
        return pn["ox"] + (x - pn["vx"]) * pn["k"], pn["oy"] + (y - pn["vy"]) * pn["k"]

    def _redraw_overlays(self) -> None:
        self._overlays_dirty = False
        d = self.p.display
        sel = {id(o) for o in self.selection} | ({id(self.current)} if self.current is not None else set())
        for pn in self._panels:
            ovl = pn["ovl"]
            dpg.delete_item(ovl, children_only=True)
            self._draw_dots(pn, sel)
            if d.graphs_in_preview and self._shown_frame is not None:
                self._draw_probe_graphs(pn)
            if not d.markers:
                continue
            for part in self.p.parts:
                chosen = id(part) in sel
                if not (d.handles or chosen):
                    continue
                pts = part_outline(part)
                if len(pts) < 2:
                    continue
                if len(pts) > 400:
                    pts = pts[::len(pts) // 400 + 1] + [pts[-1]]
                line = [self._S(pn, x, y) for x, y in pts] + ([self._S(pn, *pts[0])] if part.outline_closed else [])
                col = th.SELECTED if chosen else ((180, 220, 255, 150) if part.enabled else (128, 128, 128, 90))
                dpg.draw_polyline(line, color=col, thickness=1.6 if chosen else 1.0, parent=ovl)
                cx, cy = self._S(pn, *part_center(part))
                self._cross(ovl, cx, cy, col, chosen)
                if d.labels or chosen:
                    self._label(ovl, cx + 8, cy + 6, self.name_of(part), col)
            for s in self.p.sources:
                chosen = id(s) in sel
                if not (d.handles or chosen):
                    continue
                col = th.SELECTED if chosen else ((*th.HANDLE[:3], 230) if s.enabled else (128, 128, 128, 110))
                cx, cy = self._S(pn, s.x, s.y)
                dpg.draw_circle((cx, cy), 6, color=th.HALO, thickness=3.5, parent=ovl)
                dpg.draw_circle((cx, cy), 6, color=col, thickness=1.5, parent=ovl)
                dpg.draw_circle((cx, cy), 1.5, color=col, fill=col, parent=ovl)
                if d.labels or chosen:
                    self._label(ovl, cx + 9, cy + 6, s.name, col)
            for pr in self.p.probes:
                chosen = id(pr) in sel
                if not (d.handles or chosen):
                    continue
                if pn["layer"] == "Field" and pr.layer.value == "Intensity" and len(self._panels) > 1 and not chosen:
                    continue
                if pn["layer"] == "Intensity" and pr.layer.value == "Field" and len(self._panels) > 1 and not chosen:
                    continue
                col = th.SELECTED if chosen else ((90, 255, 200, 230) if pr.active else (128, 128, 128, 120))
                a = self._S(pn, pr.ax, pr.ay)
                if pr.kind == ProbeKind.LINE:
                    b = self._S(pn, pr.bx, pr.by)
                    dpg.draw_line(a, b, color=th.HALO, thickness=3.5, parent=ovl)
                    dpg.draw_line(a, b, color=col, thickness=1.5, parent=ovl)
                    for q in (a, b):
                        dpg.draw_rectangle((q[0] - 4, q[1] - 4), (q[0] + 4, q[1] + 4), color=col, parent=ovl)
                else:
                    dpg.draw_rectangle((a[0] - 5, a[1] - 5), (a[0] + 5, a[1] + 5), color=th.HALO, thickness=3,
                                       parent=ovl)
                    dpg.draw_rectangle((a[0] - 5, a[1] - 5), (a[0] + 5, a[1] + 5), color=col, thickness=1.5,
                                       parent=ovl)
                if d.labels or chosen:
                    self._label(ovl, a[0] + 8, a[1] + 6, pr.name, col)
            self._draw_handles(pn)
            if self._drag is not None and self._drag["mode"] == "marquee":
                (x0, y0), (x1, y1) = self._drag["start_c"], self._drag["now_c"]
                dpg.draw_rectangle(self._S(pn, x0, y0), self._S(pn, x1, y1), color=th.ACCENT_HOVER,
                                   fill=(0, 122, 204, 40), parent=ovl)

    def _cross(self, layer, x, y, col, big) -> None:
        r = 5 if big else 3
        dpg.draw_line((x - r, y), (x + r, y), color=col, parent=layer)
        dpg.draw_line((x, y - r), (x, y + r), color=col, parent=layer)

    def _label(self, layer, x, y, text, col) -> None:
        dpg.draw_text((x + 1, y + 1), text, color=(0, 0, 0, 200), size=13, parent=layer)
        dpg.draw_text((x, y), text, color=col, size=13, parent=layer)

    def _draw_dots(self, pn, sel) -> None:
        """Composite sources' emitters, in a layer of their own redrawn only when they change."""
        key = (pn["k"], tuple((id(s), s.x, s.y, s.composite, s.size_x, s.size_y, s.spacing_x, s.spacing_y,
                               s.array_angle, s.random_position, s.enabled, id(s) in sel)
                              for s in self.p.sources if s.composite), self.p.display.markers, self.p.seed)
        if self._dots_key.get(pn["dl"]) == key:
            return
        self._dots_key[pn["dl"]] = key
        dpg.delete_item(pn["dots"], children_only=True)
        if not self.p.display.markers:
            return
        for s in self.p.sources:
            if not s.composite:
                continue
            col = th.SELECTED if id(s) in sel else (255, 255, 255, 170 if s.enabled else 70)
            el = s.elements(self.p.seed)
            stride = max(1, len(el) // 3000)
            for ex, ey, *_ in el[::stride]:
                dpg.draw_circle(self._S(pn, ex, ey), 1.4, color=col, fill=col, parent=pn["dots"])

    def _handles_of(self, obj) -> list[tuple[str, float, float]]:
        k = self._panels[0]["k"] if self._panels else 1.0
        cw, ch = self._canvas_size()
        if isinstance(obj, pm.OpticalPart):
            return hd.part_handles(obj, LEVER() / k)
        if isinstance(obj, PointSource):
            return hd.source_handles(obj, LEVER() / k)
        if isinstance(obj, Probe):
            return hd.probe_handles(obj, cw, ch, self.p.display.graphs_in_preview and obj in self._graph_probes)
        return []

    def _draw_handles(self, pn) -> None:
        obj = self.current
        if obj is None or self.kind_of(obj) not in ("source", "part", "probe"):
            return
        ovl = pn["ovl"]
        centre = None
        if isinstance(obj, pm.OpticalPart):
            centre = (obj.x, obj.y)
        elif isinstance(obj, PointSource):
            centre = (obj.x, obj.y)
        for key, x, y in self._handles_of(obj):
            sx, sy = self._S(pn, x, y)
            if key == "rotate" and centre is not None:
                dpg.draw_line(self._S(pn, *centre), (sx, sy), color=(255, 220, 0, 160), parent=ovl)
                dpg.draw_circle((sx, sy), 5, color=th.HALO, thickness=3, parent=ovl)
                dpg.draw_circle((sx, sy), 5, color=th.SELECTED, fill=(40, 40, 40, 220), parent=ovl)
            elif key in ("r1", "r2"):
                dpg.draw_quad((sx, sy - 6), (sx + 6, sy), (sx, sy + 6), (sx - 6, sy), color=th.SELECTED,
                              fill=(40, 40, 40, 220), parent=ovl)
            elif key in ("size", "boxsize"):
                dpg.draw_rectangle((sx - 5, sy - 5), (sx + 5, sy + 5), color=th.SELECTED, fill=(40, 40, 40, 220),
                                   parent=ovl)
            elif key.startswith(("in", "out")):
                n = obj.nodes[int(key.lstrip("inout"))]
                dpg.draw_line(self._S(pn, *obj.to_canvas(n.x, n.y)), (sx, sy), color=(255, 220, 0, 120), parent=ovl)
                dpg.draw_circle((sx, sy), 4, color=th.SELECTED, parent=ovl)
            else:
                dpg.draw_rectangle((sx - 4, sy - 4), (sx + 4, sy + 4), color=th.SELECTED, parent=ovl)

    # ------------------------------------------------------------ the mouse on the pictures
    def _hovered_panel(self):
        for pn in self._panels:
            if dpg.does_item_exist(pn["dl"]) and dpg.is_item_hovered(pn["dl"]):
                return pn
        return None

    def _mouse_canvas(self, pn) -> tuple[float, float]:
        mx, my = dpg.get_drawing_mouse_pos()
        return (mx - pn["ox"]) / pn["k"] + pn["vx"], (my - pn["oy"]) / pn["k"] + pn["vy"]

    def _snapped(self, x: float, y: float, pn=None) -> tuple[float, float]:
        d = self.p.display
        if d.grid and d.grid_snap and self._panels:
            k = (pn or self._panels[0])["k"]
            return snap_point(x, y, d.grid_step, SNAP() / k)
        return x, y

    def _center_of(self, obj) -> tuple[float, float]:
        if isinstance(obj, PointSource):
            return float(obj.x), float(obj.y)
        if isinstance(obj, pm.OpticalPart):
            return part_center(obj)
        return probe_center(obj)

    def _canvas_press(self, pn) -> None:
        if dpg.is_key_down(dpg.mvKey_LAlt) or dpg.is_key_down(dpg.mvKey_RAlt):
            self._start_pan(pn)                  # Alt + drag pans, as the right and middle buttons do
            return
        fx, fy = self._mouse_canvas(pn)
        k = pn["k"]
        if self._placing():
            cx, cy = (int(round(v)) for v in self._snapped(fx, fy, pn))
            self._place_at(cx, cy)
            return
        self._canvas_pick = True
        mouse = tuple(dpg.get_mouse_pos(local=False))
        base = dict(pn=pn, start_c=(fx, fy), now_c=(fx, fy), mouse=mouse, moved=False)
        cur = self.current
        if cur is not None and self.kind_of(cur) in ("source", "part", "probe"):
            for key, x, y in self._handles_of(cur):
                if math.hypot(x - fx, y - fy) * k <= PICK():
                    self._drag = dict(base, mode="handle", obj=cur, key=key, grab=(x - fx, y - fy),
                                      saved=[(cur, copy.deepcopy(cur))])
                    return
        hit = pick(fx, fy, self.p.sources, self.p.probes, [], PICK() / k)
        if hit is None and self.p.display.graphs_in_preview:
            cw, ch = self._canvas_size()
            for pr in reversed(self._graph_probes):
                if any(q is pr for q in self.p.probes) and hd.box_hit(pr, cw, ch, fx, fy):
                    self._select([pr], current=pr)
                    bx, by, _, _ = pr.box(cw, ch)
                    self._drag = dict(base, mode="box", obj=pr, box0=(bx, by), saved=[(pr, copy.deepcopy(pr))])
                    return
        if hit is None:
            hit = pick(fx, fy, [], [], self.p.parts, PICK() / k)
        if hit is None:
            self._drag = dict(base, mode="marquee", add=self._shift_down(), toggle=self._command_down())
            return
        obj = {"source": self.p.sources, "probe": self.p.probes, "part": self.p.parts}[hit[0]][hit[1]]
        if self._command_down():
            self._toggle_in_selection(obj)
            return
        if self._shift_down():
            if not any(o is obj for o in self.selection):
                self._select(self.selection + [obj], current=obj)
        elif not any(o is obj for o in self.selection):
            self._select([obj], current=obj)
        else:
            self.current = obj
            self._show_props()
        objs = [o for o in self.selection if self.kind_of(o) in ("source", "part", "probe")]
        c0 = self._center_of(obj)
        self._drag = dict(base, mode="move", obj=obj, objs=objs, grab=(c0[0] - fx, c0[1] - fy), c0=c0,
                          saved=[(o, copy.deepcopy(o)) for o in objs],
                          centres=[self._center_of(o) for o in objs])

    def _update_drag(self) -> None:
        dr = self._drag
        if dr is None:
            return
        if not dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
            self._finish_drag()
            return
        pn = dr["pn"]
        mx, my = dpg.get_mouse_pos(local=False)
        fx = dr["start_c"][0] + (mx - dr["mouse"][0]) / pn["k"]
        fy = dr["start_c"][1] + (my - dr["mouse"][1]) / pn["k"]
        if not dr["moved"] and abs(mx - dr["mouse"][0]) + abs(my - dr["mouse"][1]) < 3:
            return
        dr["moved"] = True
        dr["now_c"] = (fx, fy)
        mode = dr["mode"]
        if mode == "marquee":
            self._overlays_dirty = True
            return
        if mode == "move":
            tx, ty = self._snapped(fx + dr["grab"][0], fy + dr["grab"][1], pn)
            dx, dy = tx - dr["c0"][0], ty - dr["c0"][1]
            cw, ch = self._canvas_size()
            xs, ys = [c[0] for c in dr["centres"]], [c[1] for c in dr["centres"]]
            dx = min(max(dx, -min(xs)), cw - 1 - max(xs))
            dy = min(max(dy, -min(ys)), ch - 1 - max(ys))
            dx, dy = int(round(dx)), int(round(dy))
            for obj, saved in dr["saved"]:
                self._translate(obj, saved, dx, dy)
            kind = "probe" if all(self.kind_of(o) == "probe" for o in dr["objs"]) else "sim"
        elif mode == "box":
            pr = dr["obj"]
            pr.box_x = int(round(dr["box0"][0] + fx - dr["start_c"][0]))
            pr.box_y = int(round(dr["box0"][1] + fy - dr["start_c"][1]))
            kind = "view"
        else:
            obj, key = dr["obj"], dr["key"]
            x, y = fx + dr["grab"][0], fy + dr["grab"][1]
            if key not in ("rotate", "size", "r1", "r2", "boxsize"):
                x, y = self._snapped(x, y, pn)
            if isinstance(obj, pm.OpticalPart):
                hd.drag_part_handle(obj, key, x, y, self._shift_down())
                kind = "sim"
            elif isinstance(obj, PointSource):
                hd.drag_source_handle(obj, key, x, y, self._shift_down())
                kind = "sim"
            else:
                cw, ch = self._canvas_size()
                hd.drag_probe_handle(obj, key, x, y, cw, ch)
                obj.normalize()
                kind = "view" if key == "boxsize" else "probe"
        self._edited(kind)
        self._model_changed()

    @staticmethod
    def _translate(obj, saved, dx: int, dy: int) -> None:
        if isinstance(obj, (PointSource, pm.OpticalPart)):
            obj.x, obj.y = saved.x + dx, saved.y + dy
        elif isinstance(obj, Probe):
            obj.ax, obj.ay, obj.bx, obj.by = saved.ax + dx, saved.ay + dy, saved.bx + dx, saved.by + dy
            if (saved.box_x, saved.box_y) == (saved.ax, saved.ay):
                obj.box_x, obj.box_y = obj.ax, obj.ay

    def _finish_drag(self) -> None:
        dr, self._drag = self._drag, None
        if dr["mode"] == "marquee":
            (x0, y0), (x1, y1) = dr["start_c"], dr["now_c"]
            if not dr["moved"]:
                if not (dr["add"] or dr["toggle"]):
                    self._select([], current=None)
                return
            lo_x, hi_x, lo_y, hi_y = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
            inside = [o for o in [*self.p.sources, *self.p.parts, *self.p.probes]
                      if lo_x <= self._center_of(o)[0] <= hi_x and lo_y <= self._center_of(o)[1] <= hi_y]
            if dr["toggle"]:
                ids, ins = {id(o) for o in self.selection}, {id(o) for o in inside}
                sel = [o for o in self.selection if id(o) not in ins] + [o for o in inside if id(o) not in ids]
            elif dr["add"]:
                ids = {id(o) for o in self.selection}
                sel = self.selection + [o for o in inside if id(o) not in ids]
            else:
                sel = inside
            self._select(sel, current=sel[-1] if len(sel) == 1 else None)
            return
        if dr["moved"]:
            self._end_edit_now = True
        self._overlays_dirty = True

    def _cancel_drag(self) -> bool:
        """Esc during a drag: everything back where it was."""
        dr, self._drag = self._drag, None
        if dr is None:
            return False
        for obj, saved in dr.get("saved", []):
            obj.__dict__.update(copy.deepcopy(saved).__dict__)
        self._edited("sim")
        self._model_changed()
        return True

    # ------------------------------------------------------------ keys on the pictures
    def _nudge(self, dx: int, dy: int) -> None:
        objs = [o for o in self.selection if self.kind_of(o) in ("source", "part", "probe")]
        if not objs and self.current is not None and self.kind_of(self.current) in ("source", "part", "probe"):
            objs = [self.current]
        if not objs:
            return
        for o in objs:
            self._translate(o, copy.copy(o), dx, dy)
        self._edited("probe" if all(self.kind_of(o) == "probe" for o in objs) else "sim")
        self._model_changed()

    def _cycle(self, d: int) -> None:
        items = [*self.p.sources, *self.p.parts, *self.p.probes]
        if not items:
            return
        i = next((j for j, o in enumerate(items) if o is self.current), -1)
        obj = items[(i + d) % len(items)]
        self._select([obj], current=obj)
