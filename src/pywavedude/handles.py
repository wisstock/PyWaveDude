# SPDX-License-Identifier: GPL-3.0-or-later
# PyWaveDude: Python port of WaveDude Next (C) Alexander Graef, after Jeroen Vleggaar (Huygens Optics) and 0x23.
"""Handles of the selected object on the pictures, without the GUI: where they are and what dragging them does.

All positions are canvas pixels. Keys: "rotate" (the lever), "size" (a corner grip), "r1"/"r2" (a lens's surfaces
at its upper rim), "pt<i>" (an outline point), "node<i>"/"in<i>"/"out<i>" (a fibre node and its Bézier handles),
"a"/"b" (a probe's ends), "box"/"boxsize" (a probe's graph box)."""

from __future__ import annotations

import math

from .parts import model as pm
from .probes import Probe, ProbeBoxSize, ProbeKind

# parts with a corner grip: (field along the part's x axis, field across it); the grip sets both to twice the
# distance of the pointer from the centre
SIZE_FIELDS = {
    pm.LensPart: ("thickness", "diameter"), pm.GrinLensPart: ("length", "diameter"),
    pm.ZonePlatePart: ("thickness", "diameter"), pm.PhaseStepPlatePart: ("base_thickness", "aperture"),
    pm.RectanglePart: ("length", "width"), pm.EllipsePart: ("length", "width"), pm.DiffuserPart: ("length", "width"),
    pm.InterferenceFilterPart: ("length", "width"), pm.FacettedGratingPart: ("length", "width"),
    pm.MetaSurfacePart: ("length", "width"), pm.BraggReflectorPart: (None, "width"),
}


def _size_fields(part):
    for cls in type(part).__mro__:
        if cls in SIZE_FIELDS:
            return SIZE_FIELDS[cls]
    return None


def to_frame(part, X: float, Y: float) -> tuple[float, float]:
    """Canvas point to the part's own frame (inverse of ``to_canvas``)."""
    r = math.radians(part.angle)
    c, s = math.cos(r), math.sin(r)
    dx, dy = X - part.x, Y - part.y
    return dx * c - dy * s, dx * s + dy * c


def extent(part) -> float:
    pts = part.outline()
    return max((math.hypot(px, py) for px, py in pts), default=20.0)


def _sag(r: float, y: float) -> float:
    if r == 0 or abs(y) > abs(r):
        return 0.0
    return r - math.copysign(math.sqrt(r * r - y * y), r)


def part_handles(part, lever: float) -> list[tuple[str, float, float]]:
    """The handles of a selected part; ``lever``: how far beyond the outline the rotation lever reaches."""
    out = [("rotate",) + part.to_canvas(extent(part) + lever, 0.0)]
    sf = _size_fields(part)
    if sf is not None:
        lx = getattr(part, sf[0]) / 2.0 if sf[0] else 0.0
        out.append(("size",) + part.to_canvas(lx, getattr(part, sf[1]) / 2.0))
    if isinstance(part, pm.LensPart) and part.zone_count <= 1:
        r1, r2 = part.fitted_radii()
        h = max(2, part.diameter) / 2.0
        v = max(1, part.thickness) / 2.0
        out.append(("r1",) + part.to_canvas(-v + _sag(r1, h), -h))
        out.append(("r2",) + part.to_canvas(v + _sag(r2, h), -h))
    if isinstance(part, pm.OutlinePart):
        out += [(f"pt{i}",) + part.to_canvas(px, py) for i, (px, py) in enumerate(part.points)]
    if isinstance(part, pm.FiberPart):
        for i, n in enumerate(part.nodes):
            out.append((f"node{i}",) + part.to_canvas(n.x, n.y))
            if i > 0:
                out.append((f"in{i}",) + part.to_canvas(n.x + n.in_x, n.y + n.in_y))
            if i < len(part.nodes) - 1:
                out.append((f"out{i}",) + part.to_canvas(n.x + n.out_x, n.y + n.out_y))
    return out


def lever_angle(cx: float, cy: float, X: float, Y: float, snap: bool = False) -> float:
    """The angle (degrees, counter-clockwise on the screen) that points the lever from (cx, cy) at (X, Y)."""
    a = math.degrees(math.atan2(-(Y - cy), X - cx))
    if snap:
        a = round(a / 15.0) * 15.0
    return round(a, 2) if abs(a) != 180.0 else 180.0


def drag_part_handle(part, key: str, X: float, Y: float, snap: bool = False) -> None:
    if key == "rotate":
        part.angle = lever_angle(part.x, part.y, X, Y, snap)
        return
    fx, fy = to_frame(part, X, Y)
    if key == "size":
        lf, wf = _size_fields(part)
        if lf:
            setattr(part, lf, max(1, int(round(2 * abs(fx)))))
        setattr(part, wf, max(2, int(round(2 * abs(fy)))))
    elif key in ("r1", "r2"):
        h = max(2, part.diameter) / 2.0
        v = max(1, part.thickness) / 2.0
        s = fx + v if key == "r1" else fx - v
        r = 0.0 if abs(s) < 0.5 else (h * h + s * s) / (2 * s)
        setattr(part, key, float(round(r)))
    elif key.startswith("pt"):
        i = int(key[2:])
        part.points[i] = (int(round(fx)), int(round(fy)))
    elif key.startswith(("node", "in", "out")):
        name = key.rstrip("0123456789")
        n = part.nodes[int(key[len(name):])]
        if name == "node":
            n.x, n.y = int(round(fx)), int(round(fy))
        else:
            dx, dy = int(round(fx - n.x)), int(round(fy - n.y))
            mine, other = ("in", "out") if name == "in" else ("out", "in")
            setattr(n, f"{mine}_x", dx)
            setattr(n, f"{mine}_y", dy)
            if n.smooth:                       # a smooth node keeps its two handles on one line
                ox, oy = getattr(n, f"{other}_x"), getattr(n, f"{other}_y")
                length, d = math.hypot(ox, oy), math.hypot(dx, dy)
                if length > 0 and d > 0:
                    setattr(n, f"{other}_x", int(round(-dx / d * length)))
                    setattr(n, f"{other}_y", int(round(-dy / d * length)))


def source_handles(src, lever: float) -> list[tuple[str, float, float]]:
    if not src.composite:
        return []
    reach = math.hypot((src.size_x - 1) * src.spacing_x, (src.size_y - 1) * src.spacing_y) / 2.0 + lever
    r = math.radians(src.array_angle)
    return [("rotate", src.x + reach * math.cos(r), src.y - reach * math.sin(r))]


def drag_source_handle(src, key: str, X: float, Y: float, snap: bool = False) -> None:
    if key == "rotate":
        src.array_angle = lever_angle(src.x, src.y, X, Y, snap)


def probe_handles(pr: Probe, canvas_w: int, canvas_h: int, graphs: bool) -> list[tuple[str, float, float]]:
    out = [("a", float(pr.ax), float(pr.ay))]
    if pr.kind == ProbeKind.LINE:
        out.append(("b", float(pr.bx), float(pr.by)))
    if graphs:
        bx, by, bw, bh = pr.box(canvas_w, canvas_h)
        out.append(("boxsize", float(bx + bw), float(by + bh)))
    return out


def drag_probe_handle(pr: Probe, key: str, X: float, Y: float, canvas_w: int, canvas_h: int) -> None:
    x, y = int(round(X)), int(round(Y))
    if key == "a":
        follow = (pr.box_x, pr.box_y) == (pr.ax, pr.ay)
        pr.ax, pr.ay = x, y
        if follow:
            pr.box_x, pr.box_y = x, y
    elif key == "b":
        pr.bx, pr.by = x, y
    elif key == "boxsize":
        bx, by, _, _ = pr.box(canvas_w, canvas_h)
        pr.box_size = ProbeBoxSize.CUSTOM
        pr.box_width, pr.box_height = max(60, x - bx), max(30, y - by)


def box_hit(pr: Probe, canvas_w: int, canvas_h: int, X: float, Y: float) -> bool:
    bx, by, bw, bh = pr.box(canvas_w, canvas_h)
    return bx <= X <= bx + bw and by <= Y <= by + bh
