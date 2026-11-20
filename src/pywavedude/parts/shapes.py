# SPDX-License-Identifier: GPL-3.0-or-later
# PyWaveDude: Python port of WaveDude Next (C) Alexander Graef, after Jeroen Vleggaar (Huygens Optics) and 0x23.
"""Per-pixel geometry of the parts (port of ``Parts/PartShape.cs``), vectorised over NumPy arrays.

Every shape answers, in its part's frame: the signed depth (inside ≥ 0; a distance to the outline where
``depth_is_distance``), the distance to an inner edge (fibre core, filter layer), the body index, the face
(front/back/sides, for selective coating) and the absorption share. Heavy distance searches run in Numba.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from .._parallel import PARALLEL_LOCK
from . import model as m

FRONT, BACK, SIDES = 0, 1, 2
COS45 = 0.70710678118654752
INF = np.inf


def graded_profile(t):
    t = np.clip(t, 0.0, 1.0)
    return (1.0 - np.cos(np.pi * t)) / 2.0


def absorption_profile(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t


def guard_index(n: float) -> float:
    return min(max(n, m.MIN_INDEX), m.MAX_INDEX) if math.isfinite(n) else m.MIN_INDEX


def guard_absorption(a):
    return min(max(int(a), 0), m.MAX_ABSORPTION)


def guard_ramp(r):
    return min(max(int(r), 1), m.MAX_ABSORPTION_RAMP)


def guard_coating(c):
    return min(max(int(c), 0), m.MAX_COATING)


def segment_distance(ax, ay, bx, by, x, y):
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 > 0:
        t = np.clip(((x - ax) * dx + (y - ay) * dy) / l2, 0.0, 1.0)
    else:
        t = 0.0
    return np.hypot(ax + t * dx - x, ay + t * dy - y)


def polygon_depth(vertices, x, y, with_nearest=False):
    inside = np.zeros(np.shape(x), bool)
    nearest = np.full(np.shape(x), INF)
    which = np.zeros(np.shape(x), np.int32) if with_nearest else None
    n = len(vertices)
    for i in range(n):
        ax, ay = vertices[i - 1]
        bx, by = vertices[i]
        if by != ay:  # a horizontal edge never straddles the ray
            straddle = (ay > y) != (by > y)
            cross = ax + (y - ay) / (by - ay) * (bx - ax)
            inside ^= straddle & (x < cross)
        d = segment_distance(ax, ay, bx, by, x, y)
        if with_nearest:
            closer = d < nearest
            which[closer] = i
        nearest = np.minimum(nearest, d)
    depth = np.where(inside, nearest, -nearest)
    return (depth, which) if with_nearest else depth


def rectangle_depth(hl, hw, x, y):
    dx = np.abs(x) - hl
    dy = np.abs(y) - hw
    inside = (dx <= 0) & (dy <= 0)
    out = -np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0))
    return np.where(inside, -np.maximum(dx, dy), out)


# ====================================================================== base
class PartShape:
    is_mirror = False
    coating = 0
    absorption = 0
    absorption_ramp = 1
    paints_index = True
    has_inner_edges = False
    depth_is_distance = False
    extent = 0.0

    def __init__(self, part: m.OpticalPart, front=None, back=None, sides=None):
        self.cx, self.cy = float(part.x), float(part.y)
        r = math.radians(part.angle)
        self.cos, self.sin = math.cos(r), math.sin(r)
        if isinstance(part, m.CoatedPart):
            self.profile = part.coating_profile
            self.front, self.back, self.sides = part.front_coated, part.back_coated, part.sides_coated
        else:
            self.profile = m.CoatingProfile.LAYER
            self.front = self.back = self.sides = True
        if front is not None:
            self.front, self.back, self.sides = front, back, sides

    # ------------------------------------------------------------ frame
    def to_frame(self, cx, cy):
        ox, oy = cx - self.cx, cy - self.cy
        return ox * self.cos - oy * self.sin, ox * self.sin + oy * self.cos

    def bounds(self, width, height):
        e = self.extent + 1.0
        return (max(0, math.floor(self.cx - e)), max(0, math.floor(self.cy - e)),
                min(width - 1, math.ceil(self.cx + e)), min(height - 1, math.ceil(self.cy + e)))

    # ------------------------------------------------------------ per-shape
    def depth(self, x, y):
        raise NotImplementedError

    def body_index(self, x, y):
        raise NotImplementedError

    def inner_edge(self, x, y):
        return np.zeros(np.shape(x)) if self.has_inner_edges else np.full(np.shape(x), INF)

    def depths(self, x, y):
        return self.depth(x, y), self.inner_edge(x, y)

    def face(self, x, y):
        return np.full(np.shape(x), FRONT, np.int8)

    @property
    def all_coated(self):
        return self.front and self.back and self.sides

    def coated(self, x, y):
        if self.all_coated:
            return np.ones(np.shape(x), bool)
        f = self.face(x, y)
        flags = np.array([self.front, self.back, self.sides])
        return flags[f]

    def band_index(self, depth, index, outside):
        if self.profile == m.CoatingProfile.GRADED:
            return outside + (index - outside) * graded_profile(depth / self.coating)
        return np.round(np.sqrt(index), 2)

    def index(self, x, y, outside):
        """Index at frame points; NaN where outside the part."""
        depth = self.depth(x, y)
        if self.is_mirror:
            return np.where(depth >= 0, 0.0, np.nan)
        idx = np.broadcast_to(self.body_index(x, y), np.shape(x)).astype(float)
        if self.coating > 0:
            band = (depth < self.coating)
            if band.any():
                band &= self.coated(x, y)
                idx = np.where(band, self.band_index(depth, idx, outside), idx)
        return np.where(depth >= 0, idx, np.nan)

    def absorption_share(self, x, y, depth, ramp):
        return absorption_profile(depth / ramp if ramp > 0 else 1.0)


# ====================================================================== lens
def _sag_v(radius, half):
    if radius == 0.0:
        return np.zeros(np.shape(half))
    a = abs(radius)
    h = np.minimum(a, half)
    return a - np.sqrt(a * a - h * h)


def _surface_sag_v(radius, y):
    if radius == 0.0:
        return np.zeros(np.shape(y))
    return radius - np.copysign(np.sqrt(np.maximum(0.0, radius * radius - y * y)), radius)


def _zone_bound_v(radius, half, zones, division, k):
    k = np.clip(k, 0, zones)
    if division == m.ZoneDivision.EQUAL_WIDTH or radius == 0.0:
        return half * k / zones
    a = abs(radius)
    along = a - m.sag(radius, half) * k / zones
    out = np.sqrt(np.maximum(0.0, a * a - along * along))
    return np.where(k == zones, half, out)


def _zone_index_v(radius, half, zones, division, across):
    if division == m.ZoneDivision.EQUAL_WIDTH or radius == 0.0 or m.sag(radius, half) <= 0.0:
        share = across / half
    else:
        share = _sag_v(radius, np.minimum(across, half)) / m.sag(radius, half)
    return np.clip((share * zones).astype(np.int64), 0, zones - 1)


def _surface_depth(x, y, vertex, radius, side):
    if radius == 0.0:
        return (x - vertex) * -side
    centre = vertex + radius
    along = (x - centre) * -side
    dist = np.hypot(x - centre, y)
    mag = abs(radius)
    if radius * side < 0.0:
        return np.where(along > 0.0, mag, mag - dist)
    return np.where(along < 0.0, -INF, dist - mag)


class LensShape(PartShape):
    def __init__(self, lens: m.LensPart):
        super().__init__(lens)
        self.half = max(1, lens.diameter) / 2.0
        self.v1 = -max(1, lens.thickness) / 2.0
        self.v2 = -self.v1
        self.zones = max(1, lens.zone_count)
        self.division = lens.zone_division
        self.r1, self.r2 = m.fit_radii(lens.diameter, lens.thickness, lens.r1, lens.r2, self.zones, self.division)
        self.n = guard_index(lens.refractive_index)
        self.is_mirror = lens.mirror
        self.coating = 0 if lens.mirror else guard_coating(lens.coating)
        self.absorption = 0 if lens.mirror else guard_absorption(lens.absorption)
        self.absorption_ramp = guard_ramp(lens.absorption_ramp)
        reach = self.v2 + max(self._sag(self.r1), self._sag(self.r2))
        self.extent = math.hypot(reach, self.half)

    def _sag(self, r):
        return m.zone_sag(r, self.half, self.zones, self.division)

    def _zoned(self, x, y, vertex, radius, side):
        if self.zones == 1 or radius == 0.0:
            return _surface_depth(x, y, vertex, radius, side)
        h, z, dv = self.half, self.zones, self.division
        across = np.abs(y)
        zone = _zone_index_v(radius, h, z, dv, across)
        inner = _zone_bound_v(radius, h, z, dv, zone)
        shift = _surface_sag_v(radius, inner)
        depth = _surface_depth(x, y, vertex - shift, radius, side)
        into = (x - vertex) * -side
        if radius * side < 0.0:
            reach = (shift - _surface_sag_v(radius, _zone_bound_v(radius, h, z, dv, zone - 1))) * -side
            along = across - inner
            past = into - reach
            step = np.where(past <= 0.0, along, np.hypot(along, past))
            depth = np.where(zone >= 1, np.minimum(depth, step), depth)
        else:
            along = _zone_bound_v(radius, h, z, dv, zone + 1) - across
            step = np.where(into <= 0.0, along, np.hypot(along, into))
            depth = np.where(zone <= z - 2, np.minimum(depth, step), depth)
        return depth

    def depth(self, x, y):
        return np.minimum(np.minimum(self._zoned(x, y, self.v1, self.r1, -1), self._zoned(x, y, self.v2, self.r2, 1)),
                          self.half - np.abs(y))

    def face(self, x, y):
        first = self._zoned(x, y, self.v1, self.r1, -1)
        second = self._zoned(x, y, self.v2, self.r2, 1)
        rim = self.half - np.abs(y)
        return np.where((first <= second) & (first <= rim), FRONT, np.where(second <= rim, BACK, SIDES)).astype(np.int8)

    def body_index(self, x, y):
        return self.n


# ====================================================================== polygons
class VertexShape(PartShape):
    depth_is_distance = True

    def __init__(self, part: m.ShapePart, vertices=None, front=None, back=None, sides=None):
        super().__init__(part, front, back, sides)
        self.v = vertices if vertices is not None else part.vertices()
        self.n = guard_index(part.refractive_index)
        self.is_mirror = part.mirror
        self.coating = 0 if part.mirror else guard_coating(part.coating)
        self.absorption = 0 if part.mirror else guard_absorption(part.absorption)
        self.absorption_ramp = guard_ramp(part.absorption_ramp)
        self.extent = max(math.hypot(px, py) for px, py in self.v)
        faces = []
        for i in range(len(self.v)):
            ax, ay = self.v[i - 1]
            bx, by = self.v[i]
            dx, dy = bx - ax, by - ay
            L = math.hypot(dx, dy)
            if L == 0:
                faces.append(SIDES)
                continue
            nx, ny = dy / L, -dx / L
            if polygon_depth(self.v, np.array((ax + bx) / 2 + nx * 0.01), np.array((ay + by) / 2 + ny * 0.01)) > 0:
                nx = -nx
            faces.append(FRONT if nx < -COS45 else BACK if nx > COS45 else SIDES)
        self.faces = np.array(faces, np.int8)

    def depth(self, x, y):
        return polygon_depth(self.v, x, y)

    def face(self, x, y):
        _, which = polygon_depth(self.v, x, y, with_nearest=True)
        return self.faces[which]

    def body_index(self, x, y):
        return self.n


class RectShape(PartShape):
    """A plain slab: filter and meta-surface plates (coated on the back only)."""
    depth_is_distance = True

    def __init__(self, part: m.ShapePart, length, width):
        super().__init__(part, False, part.back_coated, False)
        self.hl, self.hw = max(1, length) / 2.0, max(1, width) / 2.0
        self.n = guard_index(part.refractive_index)
        self.is_mirror = part.mirror
        self.coating = 0 if part.mirror else guard_coating(part.coating)
        self.absorption = 0 if part.mirror else guard_absorption(part.absorption)
        self.absorption_ramp = guard_ramp(part.absorption_ramp)
        self.extent = math.hypot(self.hl, self.hw)

    def depth(self, x, y):
        return rectangle_depth(self.hl, self.hw, x, y)

    def face(self, x, y):
        back, front, side = self.hl - x, self.hl + x, self.hw - np.abs(y)
        return np.where((back <= front) & (back <= side), BACK, np.where(front <= side, FRONT, SIDES)).astype(np.int8)

    def body_index(self, x, y):
        return self.n


class FilterStackShape(PartShape):
    depth_is_distance = True

    def __init__(self, f: m.InterferenceFilterPart):
        super().__init__(f)
        layers = [l for l in f.layers if l.guarded_thickness > 0]
        self.inner = -max(1, f.length) / 2.0
        self.outer = self.inner - sum(l.guarded_thickness for l in layers)
        self.hw = max(1, f.width) / 2.0
        b = [self.outer]
        for l in layers:
            b.append(b[-1] + l.guarded_thickness)
        b[-1] = self.inner
        self.bounds_x = np.array(b)
        self.indices = np.array([guard_index(l.refractive_index) for l in layers])
        self.has_inner_edges = len(layers) > 1
        r = max(abs(self.outer), abs(self.inner))
        self.extent = math.hypot(r, self.hw)

    def depth(self, x, y):
        return np.minimum(np.minimum(x - self.outer, self.inner - x), self.hw - np.abs(y))

    def inner_edge(self, x, y):
        d = np.full(np.shape(x), INF)
        for bx in self.bounds_x[1:-1]:
            d = np.minimum(d, np.abs(x - bx))
        return d

    def body_index(self, x, y):
        i = np.searchsorted(self.bounds_x, x, side="right") - 1
        return self.indices[np.clip(i, 0, len(self.indices) - 1)]


class GratingShape(PartShape):
    depth_is_distance = True

    def __init__(self, g: m.SimpleGratingPart):
        super().__init__(g)
        self.cols, self.rows = max(1, g.columns), max(1, g.rows)
        self.cs, self.rs = g.spacing(g.column_spacing), g.spacing(g.row_spacing)
        self.r = g.guarded_diameter / 2.0
        self.n = guard_index(g.refractive_index)
        self.is_mirror = g.mirror
        self.absorption = 0 if g.mirror else guard_absorption(g.absorption)
        self.absorption_ramp = guard_ramp(g.absorption_ramp)
        self.extent = math.hypot((self.cols - 1) / 2 * self.cs + self.r, (self.rows - 1) / 2 * self.rs + self.r)

    @staticmethod
    def _nearest(v, count, spacing):
        half = (count - 1) / 2.0
        return (np.clip(np.round(v / spacing + half), 0, count - 1) - half) * spacing

    def depth(self, x, y):
        return self.r - np.hypot(x - self._nearest(x, self.cols, self.cs), y - self._nearest(y, self.rows, self.rs))

    def body_index(self, x, y):
        return self.n


class MetaPatternShape(PartShape):
    depth_is_distance = True

    def __init__(self, p: m.MetaSurfacePart):
        super().__init__(p, p.front_coated, False, False)
        self.fills = np.array(p.fill_factors())
        self.pitch = p.guarded_pitch
        self.hw = max(2, p.width) / 2.0
        self.inner_x = -max(1, p.length) / 2.0
        self.outer_x = self.inner_x - p.guarded_height
        self.n = guard_index(p.pattern_index)
        self.first = p.cell_centre(0)
        self.coating = 0 if p.mirror else guard_coating(p.coating)
        r = max(abs(self.outer_x), abs(self.inner_x))
        self.extent = math.hypot(r, self.hw)

    @property
    def is_empty(self):
        return bool(np.all(self.fills * self.pitch < 0.01))

    def depth(self, x, y):
        cell = np.round((y - self.first) / self.pitch).astype(np.int64)
        depth = np.full(np.shape(x), -INF)
        hl = (self.inner_x - self.outer_x) / 2.0
        cxm = (self.outer_x + self.inner_x) / 2.0
        nf = len(self.fills)
        for k in (-1, 0, 1):
            c = cell + k
            valid = (c >= 0) & (c < nf)
            half = np.where(valid, self.fills[np.clip(c, 0, nf - 1)] * self.pitch / 2.0, 0.0)
            ok = valid & (half >= 0.005)
            d = rectangle_depth(hl, half, x - cxm, y - (self.first + c * self.pitch))
            depth = np.maximum(depth, np.where(ok, d, -INF))
        box = -np.maximum(np.maximum(self.outer_x - x, x - self.inner_x), np.abs(y) - self.hw)
        return np.where(np.isneginf(depth), np.minimum(box, -1.0), depth)

    def face(self, x, y):
        nf = len(self.fills)
        cell = np.clip(np.round((y - self.first) / self.pitch).astype(np.int64), 0, nf - 1)
        cy = self.first + cell * self.pitch
        half = self.fills[cell] * self.pitch / 2.0
        top, foot, flank = x - self.outer_x, self.inner_x - x, half - np.abs(y - cy)
        return np.where((top <= foot) & (top <= flank), FRONT, np.where(foot <= flank, BACK, SIDES)).astype(np.int8)

    def body_index(self, x, y):
        return self.n


class BraggShape(PartShape):
    has_inner_edges = True
    depth_is_distance = True

    def __init__(self, b: m.BraggReflectorPart):
        super().__init__(b)
        self.t1 = max(1, b.thickness1)
        self.period = self.t1 + max(1, b.thickness2)
        self.hl = min(max(b.pairs, 1), 200) * self.period / 2.0
        self.hw = max(1, b.width) / 2.0
        self.n1, self.n2 = guard_index(b.index1), guard_index(b.index2)
        self.extent = math.hypot(self.hl, self.hw)

    def _along(self, x):
        return np.fmod(x + self.hl, self.period)  # C# % keeps the sign of the dividend

    def inner_edge(self, x, y):
        a = self._along(x)
        return np.minimum(a, np.minimum(np.abs(a - self.t1), self.period - a))

    def depth(self, x, y):
        return np.minimum(self.hl - np.abs(x), self.hw - np.abs(y))

    def body_index(self, x, y):
        return np.where(self._along(x) < self.t1, self.n1, self.n2)


class GrinShape(PartShape):
    depth_is_distance = True

    def __init__(self, g: m.GrinLensPart):
        super().__init__(g)
        self.hl, self.hd = max(1, g.length) / 2.0, max(1, g.diameter) / 2.0
        self.nc, self.ne = guard_index(g.centre_index), guard_index(g.edge_index)
        self.coating = guard_coating(g.coating)
        self.extent = math.hypot(self.hl, self.hd)

    def depth(self, x, y):
        return np.minimum(self.hl - np.abs(x), self.hd - np.abs(y))

    def face(self, x, y):
        ends = self.hl - np.abs(x) <= self.hd - np.abs(y)
        return np.where(ends, np.where(x < 0, FRONT, BACK), SIDES).astype(np.int8)

    def body_index(self, x, y):
        t = y / self.hd
        return self.nc - (self.nc - self.ne) * t * t


class ZonePlateShape(PartShape):
    paints_index = False
    depth_is_distance = True

    def __init__(self, p: m.ZonePlatePart):
        super().__init__(p)
        self.plate = p
        self.ht = max(1, p.thickness) / 2.0
        self.hd = max(2, p.diameter) / 2.0
        self.ohw = p.outer_half_width
        self.ramp = min(max(0, p.coating), self.ht)
        self.absorption = m.ZonePlatePart.BLOCK_ABSORPTION
        self.absorption_ramp = max(1, math.ceil(self.ramp))
        self.extent = math.hypot(self.ht, self.ohw)
        lam, f = p.wavelength, max(1, p.focal_length)
        self._a, self._b = lam * lam / 4.0, lam * f

    def depth(self, x, y):
        return np.minimum(self.ht - np.abs(x), self.ohw - np.abs(y))

    def body_index(self, x, y):
        return 1.0

    def _block(self, y):
        r = np.abs(y)
        a, b = self._a, self._b
        zone = np.floor((-b + np.sqrt(b * b + 4 * a * r * r)) / (2 * a)).astype(np.int64)
        return ((zone % 2) == 1) != self.plate.inverse

    def absorption_share(self, x, y, depth, ramp):
        across = np.abs(y)
        clear = (across > self.ohw) | ((across <= self.hd) & ~self._block(y))
        inside = self.ht - np.abs(x)
        share = absorption_profile(inside / self.ramp) if self.ramp > 0 else np.ones(np.shape(x))
        return np.where(clear, 0.0, share)


# ====================================================================== profiled plates (diffuser, facetted)
@njit(parallel=True, cache=True, nogil=True)
def _profiled_nearest(xs, ys, front, hl, hw, out_d, out_f):
    last = front.shape[0] - 1
    for p in prange(xs.shape[0]):
        x, y = xs[p], ys[p]
        best = _sd(hl, -hw, hl, hw, x, y)
        face = 1
        d = _sd(front[0], -hw, hl, -hw, x, y)
        if d < best:
            best, face = d, 2
        d = _sd(front[last], hw, hl, hw, x, y)
        if d < best:
            best, face = d, 2
        start = min(max(int(np.floor(y + hw)), 0), last - 1)
        k = start
        while k >= 0:
            if y - (k + 1 - hw) >= best:
                break
            d = _sd(front[k], k - hw, front[k + 1], k + 1 - hw, x, y)
            if d < best:
                best, face = d, 0
            k -= 1
        k = start + 1
        while k < last:
            if k - hw - y >= best:
                break
            d = _sd(front[k], k - hw, front[k + 1], k + 1 - hw, x, y)
            if d < best:
                best, face = d, 0
            k += 1
        out_d[p] = best
        out_f[p] = face


@njit(inline="always")
def _sd(ax, ay, bx, by, x, y):
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    t = 0.0
    if l2 > 0.0:
        t = min(max(((x - ax) * dx + (y - ay) * dy) / l2, 0.0), 1.0)
    px, py = ax + t * dx - x, ay + t * dy - y
    return math.sqrt(px * px + py * py)


class ProfiledShape(PartShape):
    depth_is_distance = True

    def __init__(self, part, front_flags):
        super().__init__(part, *front_flags)
        self.front_face = np.array(part.front_face(), np.float64)
        self.hl, self.hw = max(1, part.length) / 2.0, max(2, part.width) / 2.0
        self.n = guard_index(part.refractive_index)
        self.is_mirror = part.mirror
        self.coating = 0 if part.mirror else guard_coating(part.coating)
        self.absorption = 0 if part.mirror else guard_absorption(part.absorption)
        self.absorption_ramp = guard_ramp(part.absorption_ramp)
        reach = max(self.hl, float(np.abs(self.front_face).max()))
        self.extent = math.hypot(reach, self.hw)

    def _nearest(self, x, y):
        xs = np.ascontiguousarray(np.broadcast_to(x, np.broadcast(x, y).shape), np.float64).ravel()
        ys = np.ascontiguousarray(np.broadcast_to(y, np.broadcast(x, y).shape), np.float64).ravel()
        d, f = np.empty_like(xs), np.empty(xs.shape, np.int8)
        with PARALLEL_LOCK:
            _profiled_nearest(xs, ys, self.front_face, self.hl, self.hw, d, f)
        shape = np.broadcast(x, y).shape
        return d.reshape(shape), f.reshape(shape)

    def _front_at(self, y):
        pos = y + self.hw
        i = np.clip(np.floor(pos).astype(np.int64), 0, len(self.front_face) - 2)
        return self.front_face[i] + (pos - i) * (self.front_face[i + 1] - self.front_face[i])

    def depth(self, x, y):
        d, _ = self._nearest(x, y)
        inside = (np.abs(y) < self.hw) & (x < self.hl) & (x > self._front_at(y))
        return np.where(inside, d, -d)

    def face(self, x, y):
        return self._nearest(x, y)[1]

    def body_index(self, x, y):
        return self.n


# ====================================================================== fibre
_MARGIN = 2.0


@njit(parallel=True, cache=True, nogil=True)
def _fiber_eval(xs, ys, path, R, flat, st, et, start_zone, end_zone, out_depth, out_path, out_s, out_e):
    n = path.shape[0]
    last = n - 1
    reach = R + _MARGIN
    for p in prange(xs.shape[0]):
        x, y = xs[p], ys[p]
        s_raw = (x - path[0, 0]) * st[0] + (y - path[0, 1]) * st[1]
        e_raw = (path[last, 0] - x) * et[0] + (path[last, 1] - y) * et[1]
        sd = s_raw if flat else np.inf
        ed = e_raw if flat else np.inf
        best = -np.inf
        nearest = np.inf
        any_chord = False
        for i in range(1, n):
            ax, ay = path[i - 1, 0], path[i - 1, 1]
            bx, by = path[i, 0], path[i, 1]
            if x < min(ax, bx) - reach or x > max(ax, bx) + reach or y < min(ay, by) - reach or y > max(ay, by) + reach:
                continue
            any_chord = True
            dx, dy = bx - ax, by - ay
            l2 = dx * dx + dy * dy
            length = math.sqrt(l2)
            along = ((x - ax) * dx + (y - ay) * dy) / length
            seg = _sd(ax, ay, bx, by, x, y)
            nearest = min(nearest, seg)
            if (along >= 0.0 and along <= length) or (i == 1 and sd < 0.0) or (i == last and ed < 0.0):
                depth = R - abs((x - ax) * dy - (y - ay) * dx) / length
            else:
                depth = R - seg
            if i <= start_zone:
                depth = min(depth, sd)
            if i >= end_zone:
                depth = min(depth, ed)
            best = max(best, depth)
        if not any_chord:
            out_depth[p] = -_MARGIN
            out_path[p] = np.inf
        else:
            out_depth[p] = max(best, -_MARGIN)
            out_path[p] = nearest
        out_s[p] = s_raw
        out_e[p] = e_raw


def _unit(x, y):
    L = math.hypot(x, y)
    return (x / L, y / L) if L > 0 else None


class FiberShape(PartShape):
    depth_is_distance = True

    def __init__(self, f: m.FiberPart, path):
        super().__init__(f)
        self.path = np.array(path, np.float64)
        p = self.path
        self.flat = f.ends == m.FiberEnds.FLAT
        a, b = f.nodes[0], f.nodes[-1]
        self.st = np.array(_unit(a.out_x, a.out_y) or _unit(*(p[1] - p[0])) or (1.0, 0.0))
        self.et = np.array(_unit(-b.in_x, -b.in_y) or _unit(*(p[-1] - p[-2])) or (1.0, 0.0))
        self.core_r = max(1, f.core_diameter) / 2.0
        self.outer_r = self.core_r + max(0, f.cladding_thickness)
        self.n_core = guard_index(f.core_index)
        self.n_clad = guard_index(f.cladding_index) if f.cladding_thickness > 0 else self.n_core
        self.has_inner_edges = f.cladding_thickness > 0
        self.coating = guard_coating(f.coating)
        self.absorption = guard_absorption(f.absorption) if f.cladding_thickness > 0 else 0
        self.band = min(guard_ramp(f.absorption_ramp), max(0, f.cladding_thickness) / 2.0)
        self.absorption_ramp = max(1, math.ceil(self.band))
        self.extent = float(np.hypot(p[:, 0], p[:, 1]).max()) + self.outer_r
        chord = np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))
        self.start_zone, arc = 1, 0.0
        for i in range(1, len(p)):
            if arc > 2 * self.outer_r:
                break
            self.start_zone = i
            arc += chord[i - 1]
        self.end_zone, arc = len(p) - 1, 0.0
        for i in range(len(p) - 1, 0, -1):
            if arc > 2 * self.outer_r:
                break
            self.end_zone = i
            arc += chord[i - 1]
        self._cache = None

    def _evaluate(self, x, y):
        key = (id(x), id(y))
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        shape = np.broadcast(x, y).shape
        xs = np.ascontiguousarray(np.broadcast_to(x, shape), np.float64).ravel()
        ys = np.ascontiguousarray(np.broadcast_to(y, shape), np.float64).ravel()
        outs = [np.empty_like(xs) for _ in range(4)]
        with PARALLEL_LOCK:
            _fiber_eval(xs, ys, self.path, self.outer_r, self.flat, self.st, self.et, self.start_zone,
                        self.end_zone, *outs)
        res = tuple(o.reshape(shape) for o in outs)
        self._cache = (key, res, x, y)  # keep x, y alive so their ids stay valid
        return res

    def _face_of(self, sd, ed, path):
        if not self.flat:
            return np.where(sd < 0, FRONT, np.where(ed < 0, BACK, SIDES)).astype(np.int8)
        side = self.outer_r - path
        return np.where((sd <= ed) & (sd <= side), FRONT, np.where(ed <= side, BACK, SIDES)).astype(np.int8)

    def depth(self, x, y):
        return self._evaluate(x, y)[0]

    def depths(self, x, y):
        d, path, _, _ = self._evaluate(x, y)
        return d, (np.abs(path - self.core_r) if self.has_inner_edges else np.full(np.shape(d), INF))

    def face(self, x, y):
        _, path, sd, ed = self._evaluate(x, y)
        return self._face_of(sd, ed, path)

    def body_index(self, x, y):
        return np.where(self._evaluate(x, y)[1] <= self.core_r, self.n_core, self.n_clad)

    def absorption_share(self, x, y, depth, ramp):
        if self.band <= 0:
            return np.zeros(np.shape(x))
        r = self._evaluate(x, y)[1]
        inner = self.outer_r - 2.0 * self.band
        t = (r - inner) / self.band
        share = absorption_profile(np.where(t <= 1.0, t, 2.0 - t))
        return np.where((r <= inner) | (r >= self.outer_r), 0.0, share)


# ====================================================================== logic gates
class LogicGateShape(PartShape):
    has_inner_edges = True

    def __init__(self, gate: m.LogicGatePart, design):
        super().__init__(gate)
        self.d = design
        self.hl = design.length / 2.0
        self.core = design.core_width / 2.0
        self.outer = self.core + design.cladding_thickness + design.absorber_width
        self.phase = design.phase_section
        self.end_coating = guard_coating(gate.coating)
        self.input_coated, self.output_coated = gate.input_coated, gate.output_coated
        self.xor_active, self.sum_active = gate.upper_output_active, gate.lower_output_active
        from .gates import ABSORPTION
        self.absorption = ABSORPTION
        self.absorption_ramp = max(1, math.ceil(design.absorber_width))
        self.extent = math.hypot(self.hl, design.half_width)

    def _nearest(self, x, y):
        off, slope = self.d.arm_offset_and_slope(x)
        return np.abs(np.abs(y) - off) / np.sqrt(1.0 + slope * slope), y > 0.0

    def depth(self, x, y):
        return np.minimum(self.hl - np.abs(x), self.outer - self._nearest(x, y)[0])

    def absorption_share(self, x, y, depth, ramp):
        dist, arm_b = self._nearest(x, y)
        inside = self.outer - dist
        side = absorption_profile(1.0 - inside / self.d.absorber_width)
        active = np.where(arm_b, self.sum_active, self.xor_active)
        output = np.where(active, 0.0, absorption_profile((x - self.d.output_ramp_start) / self.d.output_ramp_length))
        return np.where((inside < 0.0) | (np.abs(x) > self.hl), 0.0, np.maximum(side, output))

    def body_index(self, x, y):
        dist, arm_b = self._nearest(x, y)
        d = self.d
        in_phase = (arm_b != d.phase_on_arm_a) & (x >= self.phase[0]) & (x < self.phase[1])
        idx = np.where(dist > self.core, d.cladding_index, np.where(in_phase, d.phase_index, d.core_index))
        coated = np.where(x < 0.0, self.input_coated, self.output_coated)
        band = coated & (self.end_coating > 0) & (self.hl - np.abs(x) < self.end_coating)
        return np.where(band, np.round(np.sqrt(idx), 2), idx)


# ====================================================================== dispatch
def pieces_of(part: m.OpticalPart) -> list[PartShape]:
    """The shapes a part paints, in painting order (port of ``PartShape.PiecesOf``)."""
    if isinstance(part, m.InterferenceFilterPart):
        plate = RectShape(part, part.length, part.width)
        if not any(l.guarded_thickness > 0 for l in part.layers):
            return [plate]
        stack = FilterStackShape(part)
        return [stack, plate] if plate.is_mirror else [plate, stack]
    if isinstance(part, m.MetaSurfacePart):
        plate = RectShape(part, part.length, part.width)
        pattern = MetaPatternShape(part)
        if pattern.is_empty:
            return [plate]
        return [pattern, plate] if plate.is_mirror else [plate, pattern]
    if isinstance(part, m.LensPart):
        return [LensShape(part)]
    if isinstance(part, m.DiffuserPart):
        return [ProfiledShape(part, (part.front_coated, part.back_coated, part.sides_coated))]
    if isinstance(part, m.FacettedGratingPart):
        return [ProfiledShape(part, (False, part.back_coated, False))]
    if isinstance(part, m.SimpleGratingPart):
        return [GratingShape(part)]
    if isinstance(part, m.ShapePart):
        v = part.vertices()
        return [VertexShape(part, v)] if len(v) >= 3 else []
    if isinstance(part, m.FiberPart):
        path = part.flatten()
        return [FiberShape(part, path)] if len(path) >= 2 else []
    if isinstance(part, m.BraggReflectorPart):
        return [BraggShape(part)]
    if isinstance(part, m.GrinLensPart):
        return [GrinShape(part)]
    if isinstance(part, m.LogicGatePart):
        design = part.design
        return [LogicGateShape(part, design)] if design is not None else []
    if isinstance(part, m.ZonePlatePart):
        lam = part.wavelength
        return [ZonePlateShape(part)] if math.isfinite(lam) and lam > 0 else []
    return []
