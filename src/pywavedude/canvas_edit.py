# SPDX-License-Identifier: GPL-3.0-or-later
# PyWaveDude: Python port of WaveDude Next (C) Alexander Graef, after Jeroen Vleggaar (Huygens Optics) and 0x23.
"""Canvas editing without the GUI: picking an element under the mouse, geometric centres, grid snapping."""

from __future__ import annotations

from .probes import ProbeKind


def snap(v: float, step: int, tol: float) -> float:
    """``v`` moved onto the nearest grid line when it lies within ``tol`` of it."""
    if step <= 0:
        return v
    g = round(v / step) * step
    return g if abs(g - v) <= tol else v


def snap_point(x: float, y: float, step: int, tol: float) -> tuple[float, float]:
    """Each axis snaps on its own: near a line the point follows the line, near a node it sits on the node."""
    return snap(x, step, tol), snap(y, step, tol)


def part_outline(part) -> list[tuple[float, float]]:
    return [part.to_canvas(px, py) for px, py in part.outline()]


def part_center(part) -> tuple[float, float]:
    """Geometric centre of a part: the middle of its outline's bounding box (its origin without an outline)."""
    pts = part_outline(part)
    if not pts:
        return float(part.x), float(part.y)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


def probe_center(pr) -> tuple[float, float]:
    if pr.kind == ProbeKind.LINE:
        return (pr.ax + pr.bx) / 2, (pr.ay + pr.by) / 2
    return float(pr.ax), float(pr.ay)


def _inside(pts, x, y) -> bool:
    inside = False
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _seg_dist(px, py, ax, ay, bx, by) -> float:
    dx, dy = bx - ax, by - ay
    t = 0.0 if dx == dy == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return ((px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2) ** 0.5


def _polyline_dist(pts, x, y, closed) -> float:
    segs = list(zip(pts, pts[1:] + (pts[:1] if closed else [])))
    return min((_seg_dist(x, y, *a, *b) for a, b in segs), default=float("inf"))


def _area(pts) -> float:
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]))) / 2


def pick(x: float, y: float, sources, probes, parts, tol: float) -> tuple[str, int] | None:
    """The element under canvas point (x, y): ("source" | "probe" | "part", index), or None.

    Markers within ``tol`` canvas pixels win (nearest first); then the smallest part containing the point;
    then a part whose outline passes within ``tol``."""
    best, best_d = None, tol
    for i, s in enumerate(sources):
        d = ((s.x - x) ** 2 + (s.y - y) ** 2) ** 0.5
        if d <= best_d:
            best, best_d = ("source", i), d
    for i, pr in enumerate(probes):
        if pr.kind == ProbeKind.LINE:
            d = _seg_dist(x, y, pr.ax, pr.ay, pr.bx, pr.by)
        else:
            d = ((pr.ax - x) ** 2 + (pr.ay - y) ** 2) ** 0.5
        if d <= best_d:
            best, best_d = ("probe", i), d
    if best is not None:
        return best
    inside = []
    near, near_d = None, tol
    for i, part in enumerate(parts):
        pts = part_outline(part)
        if len(pts) < 2:
            continue
        if part.outline_closed and len(pts) > 2 and _inside(pts, x, y):
            inside.append((_area(pts), i))
        d = _polyline_dist(pts, x, y, part.outline_closed)
        if d <= near_d:
            near, near_d = i, d
    if inside:
        return "part", min(inside)[1]
    return ("part", near) if near is not None else None
