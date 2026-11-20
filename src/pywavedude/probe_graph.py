# SPDX-License-Identifier: GPL-3.0-or-later
# PyWaveDude: Python port of WaveDude Next (C) Alexander Graef, after Jeroen Vleggaar (Huygens Optics) and 0x23.
"""Probe graph boxes and markers (port of ``Probes/ProbeGraph.cs``, ``ProbeGraphDrawing.cs``, ``Export/ProbeMarkers.cs``).

``layout`` turns a probe and its data at one step into a drawing in box coordinates: the value and step ranges
(auto-fitted with an 8 % margin, logarithmic over N decades), the ticks that fit, and the curve (one point per
value, or each pixel column's min and max when there are more values than columns). Two painters use it: Pillow,
into exported frames, and Dear PyGui, over the preview (``ui_probes``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .probes import Probe, ProbeKind, ProbeMeasurement, ProbeScaleMode

EDGE_PAD = 3.0
LABEL_GAP = 4.0
RIGHT_MARGIN = 8.0
BARE_MARGIN = 6.0
X_LABEL_SPACING = 8.0
MIN_PLOT_W, MIN_PLOT_H = 20.0, 15.0
MIN_VISIBLE_W, MIN_VISIBLE_H = 40, 30
MARGIN = 0.08
FONT_SIZE = 11

BACKGROUND = (14, 16, 22, 235)
BORDER = (200, 200, 210, 170)
GRID = (255, 255, 255, 46)
ZERO = (255, 255, 255, 120)
FRAME = (200, 200, 210, 130)
LABEL = (225, 225, 230, 225)
TITLE = (240, 240, 245, 240)
CURVE = (255, 208, 64, 255)


@lru_cache(maxsize=8)
def font(size: int = FONT_SIZE, bold: bool = False):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - Pillow without FreeType
        return ImageFont.load_default()


def measure(text: str, size: int = FONT_SIZE) -> float:
    return float(font(size).getlength(text))


def line_height(size: int = FONT_SIZE) -> float:
    a, d = font(size).getmetrics()
    return float(a + d)


# ====================================================================== ticks and labels
def nice_ticks(lo: float, hi: float, target: int) -> list[float]:
    span = hi - lo
    if not (span > 0) or math.isinf(span):
        return []
    raw = span / max(1, target)
    mag = 10.0 ** math.floor(math.log10(raw))
    m = raw / mag
    step = (1.0 if m < 1.5 else 2.0 if m < 3.0 else 5.0 if m < 7.0 else 10.0) * mag
    t = math.ceil(lo / step) * step
    out = []
    while t <= hi + step * 1e-9 and len(out) < 40:
        out.append(0.0 if abs(t) < step * 1e-9 else t)
        t += step
    return out


def log_ticks(lo: float, hi: float, target: int) -> list[float]:
    if not (lo > 0) or not (hi > lo) or math.isinf(hi):
        return []
    first = math.ceil(math.log10(lo) - 1e-9)
    last = math.floor(math.log10(hi) + 1e-9)
    if last - first < 1:
        return [t for t in nice_ticks(lo, hi, target) if t > 0]
    every = 10
    for c in (1, 2, 3, 5):
        if (last - first) // c + 1 <= target + 1:
            every = c
            break
    return [10.0 ** e if e >= 0 else 1.0 / 10.0 ** -e for e in range(first, last + 1) if e % every == 0]


def format_tick(v: float) -> str:
    if v == 0:
        return "0"
    s = format(v, ".3g").upper()
    if "E" in s:
        mant, exp = s.split("E")
        s = f"{mant}E{exp[0]}{exp[1:].lstrip('0').rjust(2, '0')}"
    return s


def format_log_tick(v: float) -> str:
    e = round(math.log10(v))
    if abs(v / 10.0 ** e - 1.0) > 1e-6 or abs(e) <= 2:
        return format_tick(v)
    return f"1e{e}"


def format_position(v: float, span: float) -> str:
    if span >= 10:
        return str(int(round(v)))
    s = f"{v:.1f}".rstrip("0").rstrip(".")
    return s or "0"


# ====================================================================== ranges
def _extent(values, first, last):
    if values is None:
        return None
    seg = np.asarray(values[max(0, first):last + 1], np.float64)
    seg = seg[np.isfinite(seg)]
    return (float(seg.min()), float(seg.max())) if seg.size else None


def _fixed(p: Probe):
    lo, hi = p.y_min, p.y_max
    return (lo, hi) if hi > lo else (-1.0, 1.0)


def _auto(p: Probe, lo, hi, ymin, ymax):
    mag = max(abs(hi), abs(lo))
    if mag <= 0:
        return ymin, ymax
    if p.measurement == ProbeMeasurement.AMPLITUDE or lo < 0:
        ymax = mag * (1 + MARGIN)
        return -ymax, ymax
    return 0.0, hi * (1 + MARGIN)


def _log(p: Probe, fitted, hi, ymin, ymax):
    dec = min(max(p.decades, 1), 12)
    if fitted and hi > 0:
        top = math.log10(hi) + MARGIN * dec
        return 10.0 ** (top - dec), 10.0 ** top
    ymin, ymax = _fixed(p)
    if not ymax > 0:
        ymax = 1.0
    if p.auto_scale_y or not ymin > 0 or not ymin < ymax:
        ymin = ymax / 10.0 ** dec
    return ymin, ymax


@dataclass
class GraphData:
    values: np.ndarray | None = None      # the series, value i at step i + 1
    count: int = 0
    profile: np.ndarray | None = None
    profile_length: float = 0.0
    steps_sampled: int = 0


def ranges(p: Probe, d: GraphData, total_steps: int):
    if p.shows_profile:
        xmin, xmax = 0.0, (d.profile_length if d.profile is not None and d.profile_length > 0 else 1.0)
        ymin, ymax = _fixed(p)
        fitted, hi = False, 0.0
        if p.auto_scale_y and d.profile is not None:
            e = _extent(d.profile, 0, len(d.profile) - 1)
            if e:
                fitted, hi = True, e[1]
                ymin, ymax = _auto(p, e[0], e[1], ymin, ymax)
    else:
        n, start = (d.count if d.values is not None else 0), max(0, p.start_after)
        if p.scale_mode == ProbeScaleMode.FULL:
            xmin = start
            xmax = max(n, start + 1.0) if p.auto_scale_x else max(total_steps, start + 1.0)
        else:
            win = max(10, p.window)
            xmax = max(n, start + 1.0) if p.auto_scale_x else max(n, start + win)
            xmin = max(start, xmax - win)
        if xmax <= xmin:
            xmax = xmin + 1.0
        ymin, ymax = _fixed(p)
        fitted, hi = False, 0.0
        if p.auto_scale_y and n > 0:
            e = _extent(d.values, max(0, math.floor(xmin) - 1), min(n - 1, math.ceil(xmax)))
            if e:
                fitted, hi = True, e[1]
                ymin, ymax = _auto(p, e[0], e[1], ymin, ymax)
    if p.is_logarithmic:
        ymin, ymax = _log(p, fitted, hi, ymin, ymax)
    return xmin, xmax, ymin, ymax


# ====================================================================== layout
@dataclass
class GraphLayout:
    width: float
    height: float
    has_plot: bool = False
    line_width: float = 1.2
    plot_x: float = 0.0
    plot_y: float = 0.0
    plot_w: float = 0.0
    plot_h: float = 0.0
    y_ticks: list = field(default_factory=list)   # (pixel, label, is_zero)
    x_ticks: list = field(default_factory=list)
    curve: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    extra: list = field(default_factory=list)        # curves of the extra data (per wavelength), same axes
    title: str = ""
    readout: str = ""


def layout(p: Probe, d: GraphData, total_steps: int, width: float, height: float, extra=()) -> GraphLayout:
    """``extra``: more data (a wavelength's share each) drawn on the axes fitted to ``d``."""
    lh = line_height()
    visible = width >= MIN_VISIBLE_W and height >= MIN_VISIBLE_H
    lw = max(1.2, height / 110.0)
    title_h = lh + 2 * EDGE_PAD
    is_line = p.shows_profile
    started = p.start_after <= 0 or d.steps_sampled >= p.start_after
    shown = GraphData() if is_line and not started else d
    xmin, xmax, ymin, ymax = ranges(p, shown if is_line else d, total_steps)
    log = p.is_logarithmic
    yvals_of = (lambda t: log_ticks(ymin, ymax, t)) if log else (lambda t: nice_ticks(ymin, ymax, t))
    ylabel = format_log_tick if log else format_tick
    yvals = yvals_of(4)
    widest = max((measure(ylabel(t)) for t in yvals), default=0.0)
    px = EDGE_PAD + widest + LABEL_GAP
    py = title_h
    pw = width - px - RIGHT_MARGIN
    ph = height - py - (lh + LABEL_GAP)
    show_labels = visible and pw >= MIN_PLOT_W and ph >= MIN_PLOT_H
    if not show_labels:
        px = BARE_MARGIN
        pw = width - px - BARE_MARGIN
        ph = height - py - BARE_MARGIN
    g = GraphLayout(width, height, line_width=lw, plot_x=px, plot_y=py, plot_w=max(0, pw), plot_h=max(0, ph),
                    title=p.name)
    if not (visible and pw >= MIN_PLOT_W and ph >= MIN_PLOT_H):
        return g
    g.has_plot = True
    if show_labels:
        for t in (3, 2):
            if (len(yvals) - 1) * lh * 1.1 <= ph:
                break
            yvals = yvals_of(t)
    lo, hi = (math.log10(ymin), math.log10(ymax)) if log else (ymin, ymax)

    def to_px(v):
        v = np.asarray(v, np.float64)
        if log:
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(v > 0, np.maximum(np.log10(np.where(v > 0, v, 1.0)), lo), lo)
        else:
            t = v
        return np.clip(py + ph - (t - lo) / (hi - lo) * ph, -30000.0, 30000.0)

    g.y_ticks = [(float(to_px(t)), ylabel(t) if show_labels else "", t == 0.0) for t in yvals]
    for target in (5, 3, 2):
        ticks, total = [], 0.0
        for t in nice_ticks(xmin, xmax, target):
            label = format_position(t, xmax - xmin) if show_labels else ""
            total += measure(label) + X_LABEL_SPACING
            ticks.append((px + (t - xmin) / (xmax - xmin) * pw, label, t == 0.0))
        g.x_ticks = ticks
        if total - X_LABEL_SPACING <= pw:
            break
    cols = int(pw)

    def curve_of(dd: GraphData) -> np.ndarray:
        if is_line:
            prof = dd.profile if started else None
            if prof is None or not len(prof):
                return np.zeros((0, 2))
            n = len(prof)
            if n <= cols * 2:
                xs = px + (np.arange(n) / (n - 1) if n > 1 else np.zeros(1)) * pw
                ok = np.isfinite(prof)
                return np.stack([xs[ok], to_px(np.asarray(prof)[ok])], 1)
            return _bins(prof, 0, n, n - 1, cols, px, to_px)
        if dd.values is None or dd.count <= 0:
            return np.zeros((0, 2))
        first = max(0, math.floor(xmin) - 1)
        last = min(dd.count - 1, math.ceil(xmax))
        if last < first:
            return np.zeros((0, 2))
        n = last - first + 1
        if n <= cols * 2:
            idx = np.arange(first, last + 1)
            v = np.asarray(dd.values[first:last + 1], np.float64)
            ok = np.isfinite(v)
            xs = px + (idx + 1 - xmin) * (pw / (xmax - xmin))
            return np.stack([xs[ok], to_px(v[ok])], 1)
        return _bins(dd.values, first, n, last, cols, px, to_px)

    g.curve = curve_of(shown if is_line else d)
    g.extra = [curve_of(e) for e in extra]
    if is_line:
        prof = shown.profile
        if prof is not None and len(prof):
            e = _extent(prof, 0, len(prof) - 1)
            g.readout = "max " + format_tick(e[1]) if e else ""
    elif d.values is not None and d.count > 0:
        lv = float(d.values[d.count - 1])
        g.readout = "= " + format_tick(lv) if math.isfinite(lv) else ""
    return g


def _bins(values, first, visible, last, cols, px, to_px):
    pts = []
    v = np.asarray(values, np.float64)
    for c in range(cols):
        a = first + c * visible // cols
        b = max(a, first + (c + 1) * visible // cols - 1)
        seg = v[a:min(b, last) + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            x = px + c + 0.5
            pts += [(x, float(to_px(seg.min()))), (x, float(to_px(seg.max())))]
    return np.array(pts).reshape(-1, 2)


def graph_data(trace, step: int, profile) -> GraphData:
    """A probe's data as shown at ``step`` (series up to it, the frame's profile)."""
    values = trace.values()
    return GraphData(values, min(step, len(values)), profile, trace.length, min(trace.steps_sampled, step))


def group_graph_data(trace, step: int) -> list[GraphData]:
    """Each wavelength's share of a probe at ``step`` (a run with separate wavelengths), or none."""
    out = []
    if trace.group_series:
        for g in range(len(trace.group_series)):
            v = trace.group_values(g)
            out.append(GraphData(v, min(step, len(v)), None, trace.length, min(trace.steps_sampled, step)))
    elif trace.group_profiles:
        k = max((i for i, s in enumerate(trace.profile_steps) if s <= step), default=None)
        if k is not None and k < len(trace.group_profiles):
            for row in trace.group_profiles[k]:
                out.append(GraphData(None, 0, row, trace.length, min(trace.steps_sampled, step)))
    return out


# ====================================================================== Pillow painter (exports)
def draw_graph(img, g: GraphLayout, ox: int, oy: int) -> None:
    """Draw a laid-out graph box into a Pillow RGBA image at (ox, oy)."""
    from PIL import Image, ImageDraw
    w, h = int(round(g.width)), int(round(g.height))
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(layer)
    f = font()
    dr.rectangle([0, 0, w - 1, h - 1], fill=BACKGROUND, outline=BORDER)
    if g.has_plot:
        left, top, right, bottom = g.plot_x, g.plot_y, g.plot_x + g.plot_w, g.plot_y + g.plot_h
        lh = line_height()
        for pos, label, zero in g.y_ticks:
            y = round(pos)
            dr.line([(left, y), (right, y)], fill=ZERO if zero else GRID)
            if label:
                lx = max(EDGE_PAD, left - LABEL_GAP - measure(label))
                dr.text((lx, min(max(pos - lh / 2, top - lh / 2), bottom - lh / 2)), label, font=f, fill=LABEL)
        for pos, label, _ in g.x_ticks:
            x = round(pos)
            dr.line([(x, top), (x, bottom)], fill=GRID)
            if label:
                lw_ = measure(label)
                lx = min(max(pos - lw_ / 2, EDGE_PAD), max(EDGE_PAD, w - lw_ - EDGE_PAD))
                dr.text((lx, bottom + 2), label, font=f, fill=LABEL)
        dr.rectangle([left, top, right, bottom], outline=FRAME)
        if len(g.curve):
            clip = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            cd = ImageDraw.Draw(clip)
            pts = [tuple(p) for p in g.curve]
            if len(pts) == 1:
                x, y = pts[0]
                cd.ellipse([x - 2, y - 2, x + 2, y + 2], fill=CURVE)
            else:
                cd.line(pts, fill=CURVE, width=max(1, round(g.line_width)), joint="curve")
            box = (int(left), int(top), int(math.ceil(right)) + 1, int(math.ceil(bottom)) + 1)
            layer.alpha_composite(clip.crop(box), dest=box[:2])
    rw = 0.0
    if g.readout:
        rw = measure(g.readout)
        dr.text((w - rw - 8, EDGE_PAD), g.readout, font=f, fill=CURVE)
    title, room = g.title, max(10.0, w - rw - 16)
    while title and measure(title) > room:
        title = title[:-2] + "…" if len(title) > 1 else ""
    dr.text((EDGE_PAD + 3, EDGE_PAD), title, font=f, fill=TITLE)
    img.alpha_composite(layer, (ox, oy))


# ====================================================================== markers (exports)
def _arm(h):
    return max(6, h // 120)


def _marker_font(h):
    return font(int(max(12, h / 60)))


def draw_probe_marker(dr, p: Probe, number: int, h: int) -> None:
    arm = _arm(h)
    line = p.kind == ProbeKind.LINE
    for colour, width in (((0, 0, 0, 255), 5), ((255, 255, 255, 255), 3)):
        if line:
            dr.line([(p.ax, p.ay), (p.bx, p.by)], fill=colour, width=width)
        for x, y in [(p.ax, p.ay)] + ([(p.bx, p.by)] if line else []):
            dr.line([(x - arm, y), (x + arm, y)], fill=colour, width=width)
            dr.line([(x, y - arm), (x, y + arm)], fill=colour, width=width)
    f = _marker_font(h)
    for (x, y), label in ([((p.ax, p.ay), f"{number}A"), ((p.bx, p.by), f"{number}B")] if line
                          else [((p.ax, p.ay), str(number))]):
        dr.text((x + arm + 2, y - arm - round(f.size * 1.4)), label, font=f, fill=(255, 255, 255, 255),
                stroke_width=2, stroke_fill=(0, 0, 0, 255))


def draw_source_marker(dr, x: int, y: int, h: int) -> None:
    r = _arm(h)
    for colour, width in (((0, 0, 0, 255), 4), ((255, 255, 255, 255), 2)):
        dr.ellipse([x - r, y - r, x + r, y + r], outline=colour, width=width)


def decorate(rgb: np.ndarray, *, probes=(), numbers=(), traces=(), profiles=None, step: int = 0,
             total_steps: int = 0, graphs: bool = True, probe_markers: bool = True, sources=()) -> np.ndarray:
    """An exported frame with the probe graph boxes, probe markers and source markers drawn in."""
    from PIL import Image, ImageDraw
    if not ((graphs or probe_markers) and probes) and not sources:
        return rgb
    h, w = rgb.shape[:2]
    img = Image.fromarray(rgb).convert("RGBA")
    dr = ImageDraw.Draw(img)
    for x, y in sources:
        draw_source_marker(dr, x, y, h)
    if probe_markers:
        for p, n in zip(probes, numbers):
            draw_probe_marker(dr, p, n, h)
    if graphs:
        for i, p in enumerate(probes):
            bx, by, bw, bh = p.box(w, h)
            prof = profiles[i] if profiles is not None and i < len(profiles) else None
            draw_graph(img, layout(p, graph_data(traces[i], step, prof), total_steps, bw, bh), bx, by)
    return np.asarray(img.convert("RGB"))
