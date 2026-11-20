# SPDX-License-Identifier: GPL-3.0-or-later
# PyWaveDude: Python port of WaveDude Next (C) Alexander Graef, after Jeroen Vleggaar (Huygens Optics) and 0x23.
"""Optical part definitions (port of ``Model/*Part.cs``): the parameters and the derived geometry.

Coordinates of a part's own frame: x along its axis (0° points right), y across it, centre at (``x``, ``y``) on
the canvas, ``angle`` counter-clockwise on the screen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from ..rng import Pcg32, stable_hash

MIN_INDEX, MAX_INDEX = 1.0, 2.55
MAX_COATING = 50
MAX_ABSORPTION = 255
MAX_ABSORPTION_RAMP = 400
DEFAULT_ABSORPTION_RAMP = 40


def wavelength_of(level: float, frequency_scale: float, wave_speed: float = 1.0) -> float:
    cps = level / 255.0 * 0.5 * frequency_scale / (2.0 * math.pi)
    return wave_speed / cps if cps > 0 else math.nan


class CoatingProfile(str, Enum):
    LAYER = "Layer"
    GRADED = "Graded"


class ZoneDivision(str, Enum):
    EQUAL_WIDTH = "EqualWidth"
    EQUAL_DEPTH = "EqualDepth"


class TriangleType(str, Enum):
    EQUILATERAL = "Equilateral"
    RIGHT_ISOSCELES = "RightIsosceles"
    THIRTY_SIXTY_NINETY = "ThirtySixtyNinety"
    ISOSCELES = "Isosceles"
    CUSTOM = "Custom"


class FacetPattern(str, Enum):
    SAWTOOTH = "Sawtooth"
    SQUARE_WAVE = "SquareWave"
    SINUSOIDAL = "Sinusoidal"


class MetaLensType(str, Enum):
    CONVERGING = "Converging"
    DIVERGING = "Diverging"


class FiberEnds(str, Enum):
    FLAT = "Flat"
    ROUND = "Round"


# ====================================================================== bases
@dataclass
class OpticalPart:
    name: str = ""
    enabled: bool = True
    x: int = 0
    y: int = 0
    angle: float = 0.0

    stem = "Part"

    def outline(self) -> list[tuple[float, float]]:
        """Closed outline in the part frame, for drawing on the canvas."""
        return []

    outline_closed = True

    def to_canvas(self, fx: float, fy: float) -> tuple[float, float]:
        r = math.radians(self.angle)
        c, s = math.cos(r), math.sin(r)
        return self.x + fx * c + fy * s, self.y + fy * c - fx * s


@dataclass
class CoatedPart(OpticalPart):
    coating: int = 0
    coating_profile: CoatingProfile = CoatingProfile.LAYER
    front_coated: bool = True
    back_coated: bool = True
    sides_coated: bool = True


@dataclass
class ShapePart(CoatedPart):
    mirror: bool = False
    refractive_index: float = 1.5
    absorption: int = 0
    absorption_ramp: int = DEFAULT_ABSORPTION_RAMP

    def vertices(self) -> list[tuple[float, float]]:
        raise NotImplementedError

    def outline(self):
        return self.vertices()


def _rect(l, w):
    return [(-l, -w), (l, -w), (l, w), (-l, w)]


# ====================================================================== lenses
def sag(radius: float, half: float) -> float:
    if radius == 0.0:
        return 0.0
    a = abs(radius)
    h = min(a, half)
    return a - math.sqrt(a * a - h * h)


def surface_sag(radius: float, y: float) -> float:
    if radius == 0.0:
        return 0.0
    return radius - math.copysign(math.sqrt(max(0.0, radius * radius - y * y)), radius)


def zone_sag(radius, half, zones, division=ZoneDivision.EQUAL_WIDTH) -> float:
    if zones <= 1:
        return sag(radius, half)
    if division == ZoneDivision.EQUAL_DEPTH:
        return sag(radius, half) / zones
    return sag(radius, half) - sag(radius, half * (zones - 1) / zones)


def zone_bound(radius, half, zones, division, k) -> float:
    zones = max(1, zones)
    k = min(max(k, 0), zones)
    if division == ZoneDivision.EQUAL_WIDTH or radius == 0.0 or k == zones:
        return half * k / zones
    a = abs(radius)
    along = a - sag(radius, half) * k / zones
    return math.sqrt(max(0.0, a * a - along * along))


def _radius_of_zone_sag(half, zones, division, s) -> float:
    if zones <= 1 or division == ZoneDivision.EQUAL_DEPTH:
        whole = min(half, s * max(1, zones))
        return (half * half + whole * whole) / (2.0 * whole)
    if s >= zone_sag(half, half, zones):
        return half
    lo, hi = half, half * 2.0
    while zone_sag(hi, half, zones) > s:
        hi *= 2.0
    for _ in range(100):
        if hi - lo <= 1e-9 * hi:
            break
        mid = (lo + hi) / 2.0
        if zone_sag(mid, half, zones) > s:
            lo = mid
        else:
            hi = mid
    return hi


def _guard_radius(r, half):
    if not math.isfinite(r) or r == 0.0:
        return 0.0
    return math.copysign(half, r) if abs(r) < half else r


def fit_radii(diameter, thickness, r1, r2, zones=1, division=ZoneDivision.EQUAL_WIDTH):
    """Radii raised (made flatter) where both convex sags would not fit the centre thickness."""
    half = max(2, diameter) / 2.0
    zones = max(1, zones)
    r1, r2 = _guard_radius(r1, half), _guard_radius(r2, half)
    s1, s2 = zone_sag(r1, half, zones, division), zone_sag(r2, half, zones, division)
    convex = (s1 if r1 > 0 else 0.0) + (s2 if r2 < 0 else 0.0)
    budget = max(1, thickness) + (s1 if r1 < 0 else 0.0) + (s2 if r2 > 0 else 0.0)
    if convex > budget + 1e-9:
        scale = budget / convex
        if r1 > 0:
            r1 = math.ceil(_radius_of_zone_sag(half, zones, division, s1 * scale))
        if r2 < 0:
            r2 = -math.ceil(_radius_of_zone_sag(half, zones, division, s2 * scale))
    return r1, r2


@dataclass
class LensPart(CoatedPart):
    mirror: bool = False
    refractive_index: float = 1.5
    absorption: int = 0
    absorption_ramp: int = DEFAULT_ABSORPTION_RAMP
    diameter: int = 200
    r1: float = 300.0
    r2: float = -300.0
    thickness: int = 40

    stem = "Lens"

    @property
    def zone_count(self) -> int:
        return 1

    @property
    def zone_division(self) -> ZoneDivision:
        return ZoneDivision.EQUAL_WIDTH

    def fitted_radii(self):
        return fit_radii(self.diameter, self.thickness, self.r1, self.r2, self.zone_count, self.zone_division)

    def focal_lengths(self, medium: float = 1.0) -> tuple[float, float] | None:
        """Effective and back focal length in px from the (fitted) radii, thick-lens lensmaker's formula in a
        medium of index ``medium``; +inf for no power, None for a mirror. A Fresnel lens counts as thin (its zones
        collapse the thickness). Light travels along +x: R1 is the front surface, R2 the back one."""
        if self.mirror:
            return None
        r1, r2 = self.fitted_radii()
        c1, c2 = (1.0 / r1 if r1 else 0.0), (1.0 / r2 if r2 else 0.0)
        n = self.refractive_index / max(medium, 1e-9)
        d = max(1, self.thickness) if self.zone_count <= 1 else 0.0
        power = (n - 1.0) * (c1 - c2 + (n - 1.0) * d * c1 * c2 / n)
        if abs(power) < 1e-12:
            return math.inf, math.inf
        f = 1.0 / power
        return f, f * (1.0 - (n - 1.0) * d * c1 / n)

    def flip(self) -> None:
        """Concave ⇄ convex: flips the sign of both radii."""
        self.r1, self.r2 = -self.r1, -self.r2

    @property
    def edge_thickness(self) -> float:
        r1, r2 = self.fitted_radii()
        half = max(2, self.diameter) / 2.0
        z, dv = self.zone_count, self.zone_division
        return max(1, self.thickness) - math.copysign(zone_sag(r1, half, z, dv), r1) + math.copysign(
            zone_sag(r2, half, z, dv), r2)

    def outline(self):
        r1, r2 = self.fitted_radii()
        half = max(2, self.diameter) / 2.0
        v = max(1, self.thickness) / 2.0
        if self.zone_count <= 1:
            n = 32
            first = [(-v + surface_sag(r1, -half + 2 * half * i / n), -half + 2 * half * i / n) for i in range(n + 1)]
            second = [(v + surface_sag(r2, -half + 2 * half * i / n), -half + 2 * half * i / n) for i in range(n + 1)]
            return first + second[::-1]
        return self._profile(-v, r1, half) + self._profile(v, r2, half)[::-1]

    def _profile(self, vertex, radius, half):
        zones, dv = self.zone_count, self.zone_division
        steps = max(2, 32 // zones)
        pts = []
        for k in range(-zones, zones):
            zone = -k - 1 if k < 0 else k
            inner = zone_bound(radius, half, zones, dv, zone)
            outer = zone_bound(radius, half, zones, dv, zone + 1)
            shift = surface_sag(radius, inner)
            a, b = (-outer, -inner) if k < 0 else (inner, outer)
            for i in range(steps + 1):
                y = a + (b - a) * i / steps
                pts.append((vertex + surface_sag(radius, y) - shift, y))
        return pts


@dataclass
class FresnelLensPart(LensPart):
    diameter: int = 500
    r1: float = 1500.0
    r2: float = -1500.0
    thickness: int = 20
    zones: int = 4
    division: ZoneDivision = ZoneDivision.EQUAL_WIDTH

    stem = "Fresnel lens"

    @property
    def zone_count(self) -> int:
        return min(max(self.zones, 1), 100)

    @property
    def zone_division(self) -> ZoneDivision:
        return self.division


@dataclass
class GrinLensPart(CoatedPart):
    length: int = 100
    diameter: int = 100
    centre_index: float = 1.8
    edge_index: float = 1.5

    stem = "GRIN lens"

    def outline(self):
        return _rect(max(1, self.length) / 2.0, max(1, self.diameter) / 2.0)


# ====================================================================== shapes
@dataclass
class PolygonPart(ShapePart):
    sides: int = 6
    side: int = 60

    stem = "Polygon"

    def vertices(self):
        n = min(max(self.sides, 3), 64)
        r = max(2, self.side) / (2.0 * math.sin(math.pi / n))
        return [(r * math.cos(math.pi + (2 * k + 1) * math.pi / n), r * math.sin(math.pi + (2 * k + 1) * math.pi / n))
                for k in range(n)]


@dataclass
class RectanglePart(ShapePart):
    length: int = 60
    width: int = 60

    stem = "Rectangle"

    def vertices(self):
        return _rect(max(1, self.length) / 2.0, max(1, self.width) / 2.0)


@dataclass
class EllipsePart(ShapePart):
    length: int = 60
    width: int = 60

    stem = "Ellipse"

    def vertices(self):
        a, b = max(1, self.length) / 2.0, max(1, self.width) / 2.0
        h = (a - b) ** 2 / (a + b) ** 2
        circ = math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))
        n = min(max(round(circ / 3.0), 24), 180)
        return [(a * math.cos(2 * math.pi * k / n), b * math.sin(2 * math.pi * k / n)) for k in range(n)]


@dataclass
class TrianglePart(ShapePart):
    base: int = 60
    type: TriangleType = TriangleType.EQUILATERAL
    angle_a: float = 60.0
    angle_b: float = 60.0

    stem = "Triangle"

    def effective_angles(self):
        a = min(max(self.angle_a, 1.0), 178.0) if math.isfinite(self.angle_a) else 60.0
        b = min(max(self.angle_b, 1.0), 178.0) if math.isfinite(self.angle_b) else 60.0
        t = self.type
        if t == TriangleType.EQUILATERAL:
            return 60.0, 60.0
        if t == TriangleType.RIGHT_ISOSCELES:
            return 45.0, 45.0
        if t == TriangleType.THIRTY_SIXTY_NINETY:
            return 90.0, 30.0
        if t == TriangleType.ISOSCELES:
            a = min(a, 89.0)
            return a, a
        return a, min(b, 179.0 - a)

    @property
    def apex_angle(self) -> float:
        a, b = self.effective_angles()
        return 180.0 - a - b

    def vertices(self):
        a, b = self.effective_angles()
        al, be = math.radians(a), math.radians(b)
        half = max(2, self.base) / 2.0
        height = 2.0 * half * math.sin(al) * math.sin(be) / math.sin(al + be)
        apex_y = -half + height * math.cos(al) / math.sin(al)
        return [(-height / 2.0, -half), (-height / 2.0, half), (height / 2.0, apex_y)]


@dataclass
class OutlinePart(ShapePart):
    points: list[tuple[int, int]] = field(default_factory=lambda: [(-40, -30), (40, -30), (0, 40)])

    stem = "Outline"

    def vertices(self):
        return [(float(px), float(py)) for px, py in self.points]


# ====================================================================== plates
@dataclass
class FilterLayer:
    thickness: float = 2.0
    refractive_index: float = 1.5

    @property
    def guarded_thickness(self) -> float:
        return min(max(self.thickness, 0.0), 1000.0) if math.isfinite(self.thickness) else 0.0


def _default_layers():
    q = lambda n: FilterLayer(round(16.0 / (4.0 * n), 2), n)  # noqa: E731
    return [q(2.0) if i % 2 == 0 else q(1.5) for i in range(11)]


@dataclass
class InterferenceFilterPart(ShapePart):
    length: int = 50
    width: int = 500
    layers: list[FilterLayer] = field(default_factory=_default_layers)

    stem = "Interference filter"

    @property
    def stack_thickness(self) -> float:
        return sum(l.guarded_thickness for l in self.layers)

    def vertices(self):
        back = max(1, self.length) / 2.0
        front = -back - self.stack_thickness
        w = max(1, self.width) / 2.0
        return [(front, -w), (back, -w), (back, w), (front, w)]


@dataclass
class BraggReflectorPart(OpticalPart):
    pairs: int = 10
    thickness1: int = 10
    thickness2: int = 10
    index1: float = 1.5
    index2: float = 2.0
    width: int = 200

    stem = "Bragg reflector"

    def outline(self):
        l = min(max(self.pairs, 1), 200) * (max(1, self.thickness1) + max(1, self.thickness2)) / 2.0
        return _rect(l, max(1, self.width) / 2.0)


@dataclass
class SimpleGratingPart(ShapePart):
    mirror: bool = True
    columns: int = 1
    rows: int = 31
    column_spacing: int = 16
    row_spacing: int = 16
    diameter: int = 2

    stem = "Grating"

    @staticmethod
    def spacing(s: int) -> int:
        return max(2, s)

    @property
    def guarded_diameter(self) -> int:
        mx = max(1, min(self.spacing(self.column_spacing), self.spacing(self.row_spacing)) // 2)
        return min(max(self.diameter, 1), mx)

    def vertices(self):
        r = self.guarded_diameter / 2.0
        l = (max(1, self.columns) - 1) / 2.0 * self.spacing(self.column_spacing) + r
        w = (max(1, self.rows) - 1) / 2.0 * self.spacing(self.row_spacing) + r
        return _rect(l, w)


def _front_face(profile, length):
    half = max(1, length) / 2.0
    return [min(-half + p, half - 1.0) for p in profile]


def _profiled_vertices(front, length, width):
    hw, hl = max(2, width) / 2.0, max(1, length) / 2.0
    return [(f, -hw + i) for i, f in enumerate(front)] + [(hl, hw), (hl, -hw)]


@dataclass
class DiffuserPart(ShapePart):
    front_coated: bool = False
    sides_coated: bool = False
    length: int = 50
    width: int = 500
    center_wavelength: float = 16.0
    amplitude: float = 1.0
    sigma: float = 6.0
    project_seed: int = 0

    stem = "Diffuser"

    def components(self):
        c = min(max(self.center_wavelength, 2.0), 1e4)
        a = min(max(self.amplitude, 0.0), 1e4)
        s = min(max(self.sigma, 0.0), 1e4)
        if s <= 0:
            return [(c, a, 0)]
        out = []
        for k in range(-6, 7):
            wl = c + k * s / 2
            if wl < 2.0:
                continue
            out.append((wl, a * math.exp(-(wl - c) ** 2 / (2 * s * s)), k))
        return out

    def profile(self) -> list[float]:
        width = max(2, self.width)
        rng = Pcg32(self.project_seed & 0xFFFFFFFF, stable_hash(self.name or "") ^ 0x44494646)
        phases = [2.0 * math.pi * rng.next_double() for _ in range(13)]
        prof = [0.0] * (width + 1)
        half = width / 2.0
        for wl, amp, idx in self.components():
            ph, k = phases[idx + 6], 2.0 * math.pi / wl
            for i in range(width + 1):
                prof[i] += amp * math.sin(k * (i - half) + ph)
        return prof

    def front_face(self):
        return _front_face(self.profile(), self.length)

    def vertices(self):
        return _profiled_vertices(self.front_face(), self.length, self.width)


@dataclass
class FacettedGratingPart(ShapePart):
    length: int = 50
    width: int = 500
    pattern: FacetPattern = FacetPattern.SAWTOOTH
    pitch: float = 32.0
    amplitude: float = 4.0

    stem = "Facetted grating"

    def profile(self) -> list[float]:
        width = max(2, self.width)
        pitch = min(max(self.pitch, 2.0), 1e4)
        amp = min(max(self.amplitude, 0.0), 1e4)
        half = width / 2.0
        out = []
        for i in range(width + 1):
            t = (i - half) / pitch
            t -= math.floor(t)
            if self.pattern == FacetPattern.SQUARE_WAVE:
                v = -1.0 if t < 0.5 else 1.0
            elif self.pattern == FacetPattern.SINUSOIDAL:
                v = math.sin(2 * math.pi * t)
            else:
                v = 2.0 * t - 1.0
            out.append(amp * v)
        return out

    def front_face(self):
        return _front_face(self.profile(), self.length)

    def vertices(self):
        return _profiled_vertices(self.front_face(), self.length, self.width)


# ====================================================================== designed for a frequency
@dataclass
class ZonePlatePart(OpticalPart):
    diameter: int = 800
    thickness: int = 32
    coating: int = 0
    design_frequency: float = 50.0
    focal_length: int = 600
    inverse: bool = False
    mount: bool = False
    mount_width: int = 200
    frequency_scale: float = 2.0

    stem = "Zone plate"
    BLOCK_ABSORPTION = 96

    @property
    def wavelength(self) -> float:
        return wavelength_of(self.design_frequency, self.frequency_scale)

    @property
    def outer_half_width(self) -> float:
        return max(2, self.diameter) / 2.0 + (max(1, self.mount_width) if self.mount else 0)

    def zone_at(self, r):
        lam, f = self.wavelength, max(1, self.focal_length)
        if not (math.isfinite(lam) and lam > 0):
            return None
        a, b = lam * lam / 4.0, lam * f
        return math.floor((-b + math.sqrt(b * b + 4 * a * r * r)) / (2 * a))

    @property
    def zone_count(self) -> int:
        z = self.zone_at(max(2, self.diameter) / 2.0)
        return 0 if z is None else z + 1

    def outline(self):
        return _rect(max(1, self.thickness) / 2.0, self.outer_half_width)


@dataclass
class PhaseStepPlatePart(ShapePart):
    """The 2D counterpart of a vortex phase plate: a slab whose aperture is split across the beam into zones
    that alternate between two thicknesses, so neighbouring zones delay the wave by ``phase_step`` degrees.

    With 2 zones and 180° the transmitted beam is antisymmetric: its focus has a dark zero on the axis between
    two bright lobes (the 2D doughnut used for STED depletion). More zones give higher-order patterns. Light
    travels along the part's x axis; the flat face is the back (+x)."""
    aperture: int = 400
    base_thickness: int = 8
    zones: int = 2
    phase_step: float = 180.0
    design_frequency: float = 50.0
    medium_index: float = 1.0
    frequency_scale: float = 2.0

    stem = "Phase step plate"

    @property
    def wavelength(self) -> float:
        return wavelength_of(self.design_frequency, self.frequency_scale)

    @property
    def step_height(self) -> float:
        """Extra thickness of the delayed zones: (n - n_medium) · h = phase/360 · λ."""
        dn = self.refractive_index - self.medium_index
        lam = self.wavelength
        if not (math.isfinite(lam) and lam > 0) or dn <= 0:
            return 0.0
        return min(abs(self.phase_step) / 360.0 * lam / dn, 4.0 * lam)

    def vertices(self):
        hw = max(2, self.aperture) / 2.0
        back = max(1, self.base_thickness) / 2.0
        n = min(max(self.zones, 1), 64)
        step = self.step_height
        pts = [(back, -hw), (back, hw)]
        for k in reversed(range(n)):                  # along the front face, from +y down to -y
            front = -back - (step if k % 2 else 0.0)
            hi, lo = -hw + 2 * hw * (k + 1) / n, -hw + 2 * hw * k / n
            pts += [(front, hi), (front, lo)]
        return pts


@dataclass
class MetaSurfacePart(ShapePart):
    sides_coated: bool = False
    length: int = 50
    width: int = 500
    design_frequency: float = 50.0
    pitch: float = 4.0
    height: float = 32.0
    pattern_index: float = 2.0
    frequency_scale: float = 2.0

    @property
    def wavelength(self) -> float:
        return wavelength_of(self.design_frequency, self.frequency_scale)

    @property
    def has_wavelength(self) -> bool:
        return math.isfinite(self.wavelength) and self.wavelength > 0

    @property
    def guarded_pitch(self) -> float:
        return min(max(self.pitch, 1.0), max(2, self.width))

    @property
    def guarded_height(self) -> float:
        return min(max(self.height, 1.0), 1e4)

    @property
    def guarded_pattern_index(self) -> float:
        return min(max(self.pattern_index, MIN_INDEX), MAX_INDEX)

    @property
    def full_turn_height(self) -> float:
        return self.wavelength / max(1e-9, self.guarded_pattern_index - 1.0)

    @property
    def cell_count(self) -> int:
        return max(1, math.floor(max(2, self.width) / self.guarded_pitch))

    def cell_centre(self, cell: int) -> float:
        return (cell - (self.cell_count - 1) / 2.0) * self.guarded_pitch

    @staticmethod
    def wrap(p: float) -> float:
        return p - 2.0 * math.pi * math.floor(p / (2.0 * math.pi))

    def delay(self, y: float) -> float:
        raise NotImplementedError

    def fill_factors(self) -> list[float]:
        n = self.cell_count
        if not self.has_wavelength:
            return [0.0] * n
        lam, h, n2 = self.wavelength, self.guarded_height, self.guarded_pattern_index ** 2
        out = []
        for j in range(n):
            asked = self.delay(self.cell_centre(j))
            d = asked if 0.0 <= asked <= 2 * math.pi else self.wrap(asked)
            eff = 1.0 + d / (2 * math.pi) * lam / h
            out.append(min(max((eff * eff - 1.0) / (n2 - 1.0), 0.0), 1.0) if n2 > 1 else 0.0)
        return out

    def vertices(self):
        back = max(1, self.length) / 2.0
        front = -back - self.guarded_height
        w = max(2, self.width) / 2.0
        return [(front, -w), (back, -w), (back, w), (front, w)]


@dataclass
class MetaLensPart(MetaSurfacePart):
    focal_length: int = 500
    lens_type: MetaLensType = MetaLensType.CONVERGING

    stem = "Meta lens"

    def lens_phase(self, y):
        if not self.has_wavelength:
            return 0.0
        f = max(1, self.focal_length)
        return 2 * math.pi / self.wavelength * (math.sqrt(f * f + y * y) - f)

    def delay(self, y):
        w = self.wrap(self.lens_phase(y))
        return w if self.lens_type == MetaLensType.DIVERGING else 2 * math.pi - w


@dataclass
class MetaPrismPart(MetaSurfacePart):
    deflection: float = 20.0

    stem = "Meta prism"

    def delay(self, y):
        s = math.sin(math.radians(min(max(self.deflection, -89.0), 89.0)))
        return -2 * math.pi / self.wavelength * y * s


@dataclass
class MetaAxiconPart(MetaSurfacePart):
    cone_angle: float = 10.0

    stem = "Meta axicon"

    def delay(self, y):
        r = math.radians(min(max(self.cone_angle, 0.1), 89.0))
        return 2 * math.pi - self.wrap(2 * math.pi / self.wavelength * abs(y) * math.sin(r))


# ====================================================================== fibre
@dataclass
class FiberNode:
    x: int = 0
    y: int = 0
    in_x: int = 0
    in_y: int = 0
    out_x: int = 0
    out_y: int = 0
    smooth: bool = True


def _default_path():
    t = round(2.0 * 100 / 3.0)
    return [FiberNode(-100, 0, out_x=t), FiberNode(100, 0, in_x=-t)]


def _seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    t = min(max(((px - ax) * dx + (py - ay) * dy) / l2, 0.0), 1.0) if l2 > 0 else 0.0
    return math.hypot(ax + t * dx - px, ay + t * dy - py)


@dataclass
class FiberPart(CoatedPart):
    core_index: float = 1.5
    core_diameter: int = 20
    cladding_index: float = 1.45
    cladding_thickness: int = 10
    absorption: int = 0
    absorption_ramp: int = 10
    ends: FiberEnds = FiberEnds.FLAT
    nodes: list[FiberNode] = field(default_factory=_default_path)

    stem = "Fiber"
    outline_closed = False

    def flatten(self) -> list[tuple[float, float]]:
        nodes = self.nodes
        if len(nodes) < 2:
            return []
        tol = 0.05
        path = [(float(nodes[0].x), float(nodes[0].y))]
        for a, b in zip(nodes, nodes[1:]):
            x0, y0, x3, y3 = a.x, a.y, b.x, b.y
            x1, y1 = a.x + a.out_x, a.y + a.out_y
            x2, y2 = b.x + b.in_x, b.y + b.in_y
            steps = 1
            if _seg_dist(x1, y1, x0, y0, x3, y3) > tol or _seg_dist(x2, y2, x0, y0, x3, y3) > tol:
                second = max(math.hypot(x0 - 2 * x1 + x2, y0 - 2 * y1 + y2), math.hypot(x1 - 2 * x2 + x3, y1 - 2 * y2 + y3))
                steps = min(max(math.ceil(math.sqrt(0.75 * second / tol)), 1), 512)
            for k in range(1, steps + 1):
                t = k / steps
                u = 1 - t
                w0, w1, w2, w3 = u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t
                p = (w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3, w0 * y0 + w1 * y1 + w2 * y2 + w3 * y3)
                if p != path[-1]:
                    path.append(p)
        return path if len(path) >= 2 else []

    def outline(self):
        return self.flatten()


# ====================================================================== logic gates
@dataclass
class LogicGatePart(OpticalPart):
    """Waveguide coupler gate; inputs A (y < 0) and B (y > 0) on the left, outputs on the right (``LogicGatePart.cs``)."""

    design_frequency: float = 50.0
    core_index: float = 2.55
    cladding_index: float = 1.45
    port_spacing: int = 2
    coating: int = 6
    input_coated: bool = True
    output_coated: bool = True
    upper_output_active: bool = True
    lower_output_active: bool = True
    frequency_scale: float = 2.0

    phase_shift = 0.0
    outputs = ("", "")

    @property
    def wavelength(self) -> float:
        return wavelength_of(self.design_frequency, self.frequency_scale)

    @property
    def design(self):
        from .gates import design_for
        lam = self.wavelength
        if not (math.isfinite(lam) and lam > 0):
            return None
        norm = lambda n: round(min(max(n, MIN_INDEX), MAX_INDEX), 2)  # noqa: E731
        return design_for(round(min(max(lam, 6.0), 200.0), 3), norm(self.core_index), norm(self.cladding_index),
                          min(max(self.port_spacing, 2), 2000), self.phase_shift)

    def outline(self):
        d = self.design
        if d is None:
            return []
        steps = 64
        outer = d.core_width / 2.0 + d.cladding_thickness + d.absorber_width
        xs = [d.left + d.length * i / steps for i in range(steps + 1)]
        offs = [float(o) + outer for o in d.arm_offset_and_slope(xs)[0]]
        return [(x, -o) for x, o in zip(xs, offs)] + [(x, o) for x, o in zip(xs[::-1], offs[::-1])]

    def ports(self):
        """Canvas pixels of input A, input B, output upper, output lower."""
        d = self.design
        if d is None:
            return None
        h = d.port_spacing / 2.0
        at = lambda fx, fy: tuple(math.floor(v) for v in self.to_canvas(fx, fy))  # noqa: E731
        return [at(d.input_x, -h), at(d.input_x, h), at(d.output_x, -h), at(d.output_x, h)]


@dataclass
class XorGatePart(LogicGatePart):
    stem = "XOR gate"
    phase_shift = 90.0
    outputs = ("XOR", "AND")


@dataclass
class AndGatePart(LogicGatePart):
    stem = "AND gate"
    phase_shift = -90.0
    outputs = ("AND", "XOR")


@dataclass
class OrGatePart(LogicGatePart):
    stem = "OR gate"
    phase_shift = 30.0
    outputs = ("OR", "remainder")


PART_TYPES: dict[str, type[OpticalPart]] = {
    cls.stem: cls for cls in (
        LensPart, FresnelLensPart, PolygonPart, RectanglePart, EllipsePart, TrianglePart, OutlinePart, FiberPart,
        BraggReflectorPart, GrinLensPart, ZonePlatePart, PhaseStepPlatePart, DiffuserPart, InterferenceFilterPart, SimpleGratingPart,
        FacettedGratingPart, MetaLensPart, MetaPrismPart, MetaAxiconPart, XorGatePart, AndGatePart, OrGatePart)
}
