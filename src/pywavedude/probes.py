# SPDX-License-Identifier: GPL-3.0-or-later
# PyWaveDude: Python port of WaveDude Next (C) Alexander Graef, after Jeroen Vleggaar (Huygens Optics) and 0x23.
"""Probes (port of ``Model/Probe.cs`` and ``Probes/``): what they measure, where, and the detections of detectors.

* Point probe: one value per step, the field (amplitude) or the intensity (value, or its running integral).
* Line probe A→B: a profile per frame along a Bresenham line, of the field, the intensity, or the intensity
  integrated over the steps since ``start_after``.
* Counting detector: Poisson detections of the whole line's integrated intensity × sensitivity, per step.
* Imaging detector: Poisson detections per pixel, summed over bins of ``bin_size`` pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from .rng import Pcg32, stable_hash


class ProbeLayer(str, Enum):
    FIELD = "Field"
    INTENSITY = "Intensity"


class ProbeMeasurement(str, Enum):
    AMPLITUDE = "Amplitude"
    VALUE = "Value"
    INTEGRATED_VALUE = "IntegratedValue"


class ProbeKind(str, Enum):
    POINT = "Point"
    LINE = "Line"


class ProbeDetector(str, Enum):
    STANDARD = "Standard"
    COUNTING = "Counting"
    IMAGING = "Imaging"


class ProbeScaleMode(str, Enum):
    RUNNING = "Running"
    FULL = "Full"


class ProbeBoxSize(str, Enum):
    TINY = "Tiny"
    SMALL = "Small"
    MEDIUM = "Medium"
    LARGE = "Large"
    EXTRA_LARGE = "ExtraLarge"
    CUSTOM = "Custom"


BOX_SIZES = {ProbeBoxSize.TINY: (200, 100), ProbeBoxSize.SMALL: (300, 150), ProbeBoxSize.MEDIUM: (400, 200),
             ProbeBoxSize.LARGE: (500, 250), ProbeBoxSize.EXTRA_LARGE: (700, 350)}


class ProbeValueScale(str, Enum):
    LINEAR = "Linear"
    LOGARITHMIC = "Logarithmic"


@dataclass
class Probe:
    name: str = ""
    active: bool = True
    layer: ProbeLayer = ProbeLayer.FIELD
    measurement: ProbeMeasurement = ProbeMeasurement.AMPLITUDE
    kind: ProbeKind = ProbeKind.POINT
    detector: ProbeDetector = ProbeDetector.STANDARD
    ax: int = 0
    ay: int = 0
    bx: int = 0
    by: int = 0
    start_after: int = 0
    sensitivity: float = 20.0
    bin_size: int = 4
    box_size: ProbeBoxSize = ProbeBoxSize.MEDIUM
    box_x: int = 0                 # graph box, canvas pixels (the original's default: at point A)
    box_y: int = 0
    box_width: int = 400
    box_height: int = 200
    scale_mode: ProbeScaleMode = ProbeScaleMode.RUNNING
    window: int = 500
    auto_scale_x: bool = True
    auto_scale_y: bool = True
    y_min: float = -1.0
    y_max: float = 1.0
    y_scale: ProbeValueScale = ProbeValueScale.LINEAR
    decades: int = 6

    @property
    def is_detector(self) -> bool:
        return self.detector != ProbeDetector.STANDARD

    @property
    def shows_profile(self) -> bool:
        return self.kind == ProbeKind.LINE and self.detector != ProbeDetector.COUNTING

    @property
    def is_logarithmic(self) -> bool:
        return self.y_scale == ProbeValueScale.LOGARITHMIC and self.layer == ProbeLayer.INTENSITY

    def normalize(self) -> None:
        """The original's consistency rules: a detector is an integrated intensity line, the field has amplitude only."""
        if self.is_detector:
            self.kind, self.layer, self.measurement = ProbeKind.LINE, ProbeLayer.INTENSITY, ProbeMeasurement.INTEGRATED_VALUE
        self.sensitivity = min(max(self.sensitivity, 0.0), 1e6) if math.isfinite(self.sensitivity) else 20.0
        self.bin_size = min(max(self.bin_size, 1), 20)
        if self.layer == ProbeLayer.FIELD:
            self.measurement, self.y_scale = ProbeMeasurement.AMPLITUDE, ProbeValueScale.LINEAR
        elif self.measurement == ProbeMeasurement.AMPLITUDE:
            self.measurement = ProbeMeasurement.VALUE
        if self.y_max <= self.y_min:
            self.y_min, self.y_max = -1.0, 1.0
        self.window = min(max(self.window, 10), 1_000_000)
        self.start_after = max(0, self.start_after)
        self.decades = min(max(self.decades, 1), 12)

    def box(self, w: int, h: int) -> tuple[int, int, int, int]:
        """The graph box in a ``w`` × ``h`` canvas: shrunk to fit, kept inside (port of ``ProbeGraph.BoxInCanvas``)."""
        if self.box_size == ProbeBoxSize.CUSTOM:
            bw, bh = min(max(self.box_width, 60), 4096), min(max(self.box_height, 30), 4096)
        else:
            bw, bh = BOX_SIZES[self.box_size]
        scale = 1.0
        if bw > w:
            scale = min(scale, w / bw)
        if bh > h:
            scale = min(scale, h / bh)
        bw, bh = max(1, round(bw * scale)), max(1, round(bh * scale))
        return max(0, min(w - bw, self.box_x)), max(0, min(h - bh, self.box_y)), bw, bh

    def anchors(self):
        return [(self.ax, self.ay), (self.bx, self.by)] if self.kind == ProbeKind.LINE else [(self.ax, self.ay)]


# ====================================================================== sampler
def line_pixels(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Bresenham, A to B inclusive, as ``ProbeSampler.LinePixels``."""
    out = []
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx + dy
    while True:
        out.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return out


@dataclass
class ProbeSampler:
    """Flat indices into the engines' padded buffers for each active probe, plus how to read them."""

    probes: list[Probe]
    pixels: list[np.ndarray]
    is_profile: np.ndarray
    use_intensity: np.ndarray
    cumulative: np.ndarray
    integrate: np.ndarray
    lengths: np.ndarray
    start_after: np.ndarray
    seed: int = 0

    @classmethod
    def build(cls, probes: list[Probe], w: int, h: int, rim: int = 0, seed: int = 0) -> "ProbeSampler":
        stride = w + 2 + 2 * rim
        shift = rim * stride + rim
        clamp = lambda v, mx: min(max(v, 0), mx)  # noqa: E731
        pix, prof, inten, cum, integ, lens, start = [], [], [], [], [], [], []
        for p in probes:
            p.normalize()
            ax, ay = clamp(p.ax, w - 1), clamp(p.ay, h - 1)
            integrated = p.measurement == ProbeMeasurement.INTEGRATED_VALUE or p.is_detector
            if p.kind == ProbeKind.LINE or p.is_detector:
                bx, by = clamp(p.bx, w - 1), clamp(p.by, h - 1)
                pts = line_pixels(ax, ay, bx, by)
                pix.append(np.array([(y + 1) * stride + x + 1 + shift for x, y in pts], np.int64))
                prof.append(True)
                integ.append(integrated)
                cum.append(False)
                lens.append(math.hypot(bx - ax, by - ay))
            else:
                pix.append(np.array([(ay + 1) * stride + ax + 1 + shift], np.int64))
                prof.append(False)
                integ.append(False)
                cum.append(integrated)
                lens.append(0.0)
            inten.append(p.measurement != ProbeMeasurement.AMPLITUDE or p.is_detector)
            start.append(p.start_after)
        b = lambda v: np.array(v, bool)  # noqa: E731
        return cls(list(probes), pix, b(prof), b(inten), b(cum), b(integ), np.array(lens), np.array(start, np.int64),
                   seed)

    @property
    def count(self) -> int:
        return len(self.pixels)

    @property
    def needs_intensity(self) -> bool:
        return bool(self.use_intensity.any())


# ====================================================================== detections
class PoissonProcess:
    WALK_LIMIT = 100_000.0

    def __init__(self, seed: int, stream: int):
        self._rng = Pcg32(seed, stream)
        self._next = self._exponential()
        self.count = 0

    def _exponential(self) -> float:
        return -math.log(1.0 - self._rng.next_double())

    def _gaussian(self) -> float:
        u = 1.0 - self._rng.next_double()
        v = self._rng.next_double()
        return math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v)

    def count_up_to(self, position: float) -> int:
        if not math.isfinite(position):
            return self.count
        if position - self._next > self.WALK_LIMIT:
            span = position - self._next - self.WALK_LIMIT / 2.0
            jumped = round(span + math.sqrt(span) * self._gaussian())
            self.count += int(min(max(jumped, 0.0), 2 ** 31 - 1 - self.count))
            self._next += span
        while self._next <= position:
            self.count += 1
            self._next += self._exponential()
        return self.count


def _stream(name_hash: int, pixel: int) -> int:
    return ((name_hash << 24) ^ pixel) & ((1 << 64) - 1)


@dataclass
class ProbeTrace:
    series: list[np.ndarray] = field(default_factory=list)   # chunks, one value per step
    profiles: list[np.ndarray] = field(default_factory=list)  # one per collected frame
    profile_steps: list[int] = field(default_factory=list)
    length: float = 0.0
    steps_sampled: int = 0
    group_series: list = field(default_factory=list)    # per wavelength: chunks (separate-wavelength runs)
    group_profiles: list = field(default_factory=list)  # per collected frame: (wavelengths, length)

    def values(self) -> np.ndarray:
        return np.concatenate(self.series) if self.series else np.zeros(0, np.float32)

    def group_values(self, g: int) -> np.ndarray:
        return np.concatenate(self.group_series[g]) if self.group_series[g] else np.zeros(0, np.float32)


class ProbeCollector:
    """Turns the engine's raw samples into traces: running integrals, detections (port of ``ProbeCollector.cs``)."""

    def __init__(self, sampler: ProbeSampler):
        self.s = sampler
        n = sampler.count
        self.traces = [ProbeTrace(length=float(sampler.lengths[i])) for i in range(n)]
        self._totals = np.zeros(n)
        self._last_steps = 0
        hashes = [stable_hash(p.name or "") for p in sampler.probes]
        self._hash = hashes
        self._counters = [PoissonProcess(sampler.seed & 0xFFFFFFFF, _stream(hashes[i], 0))
                          if sampler.probes[i].detector == ProbeDetector.COUNTING else None for i in range(n)]
        self._counted_exp = np.zeros(n)
        self._counted_steps = np.zeros(n, np.int64)
        self._imagers: list[list[PoissonProcess] | None] = [None] * n
        self._gtotals = None

    def _split(self, i: int) -> bool:
        """Kept per wavelength too: intensity probes that are no detectors."""
        p = self.s.probes[i]
        return p.layer == ProbeLayer.INTENSITY and p.detector == ProbeDetector.STANDARD

    def collect(self, engine, steps_done: int) -> None:
        samples = engine.take_probe_samples()           # (steps, n) float32, NaN for profiles
        if samples is None or samples.shape[0] == 0:
            self._last_steps = steps_done
            return
        self._collect_groups(getattr(engine, "group_samples", None))
        start = self._last_steps
        for i in range(self.s.count):
            if self.s.is_profile[i]:
                continue
            v = samples[:, i].astype(np.float64)
            if self.s.cumulative[i]:
                # sample j is step start + j + 1; before "start after" the total is held at zero (a prefix,
                # since steps only increase), purging what arrived earlier
                step = start + np.arange(1, len(v) + 1)
                before = step < self.s.start_after[i]
                base = 0.0 if before.any() else self._totals[i]
                v = np.where(before, 0.0, v)
                c = np.cumsum(v) + base
                self._totals[i] = c[-1]
                v = c
            self.traces[i].series.append(v.astype(np.float32))
            self.traces[i].steps_sampled = steps_done
        self._last_steps = steps_done

    def _collect_groups(self, groups) -> None:
        if not groups or any(g is None for g in groups):
            return
        start = self._last_steps
        if self._gtotals is None:
            self._gtotals = np.zeros((len(groups), self.s.count))
        for i in range(self.s.count):
            if self.s.is_profile[i] or not self._split(i):
                continue
            tr = self.traces[i]
            if not tr.group_series:
                tr.group_series = [[] for _ in groups]
            for k, samples in enumerate(groups):
                v = samples[:, i].astype(np.float64)
                if self.s.cumulative[i]:
                    step = start + np.arange(1, len(v) + 1)
                    before = step < self.s.start_after[i]
                    base = 0.0 if before.any() else self._gtotals[k, i]
                    c = np.cumsum(np.where(before, 0.0, v)) + base
                    self._gtotals[k, i] = c[-1]
                    v = c
                tr.group_series[k].append(v.astype(np.float32))

    def collect_profiles(self, engine, steps_done: int) -> list[np.ndarray | None]:
        """Profiles of this frame (detector counts for imaging, None for point probes and counting detectors)."""
        profiles = engine.read_probe_profiles()
        groups = getattr(engine, "group_profiles", None)
        if groups:
            for i in range(self.s.count):
                if self.s.is_profile[i] and self._split(i) and all(g[i] is not None for g in groups):
                    self.traces[i].group_profiles.append(np.stack([g[i] for g in groups]))
        out: list[np.ndarray | None] = [None] * self.s.count
        for i in range(self.s.count):
            p = profiles[i]
            if not self.s.is_profile[i] or p is None:
                continue
            det = self.s.probes[i].detector
            if det == ProbeDetector.COUNTING:
                self._count(i, p, steps_done)
                continue
            if det == ProbeDetector.IMAGING:
                p = self._image(i, p)
            self.traces[i].profiles.append(p)
            self.traces[i].profile_steps.append(steps_done)
            self.traces[i].steps_sampled = steps_done
            out[i] = p
        self._last_steps = steps_done
        return out

    def _count(self, i, profile, steps_done):
        counter = self._counters[i]
        steps = steps_done - int(self._counted_steps[i])
        if steps <= 0:
            return
        total = float(np.sum(profile, dtype=np.float64))
        frm = self._counted_exp[i]
        exp = max(frm, total * self.s.probes[i].sensitivity)
        vals = np.empty(steps, np.float32)
        before = counter.count
        for k in range(steps):
            up = counter.count_up_to(frm + (exp - frm) * (k + 1) / steps)
            vals[k] = up - before
            before = up
        self._counted_exp[i] = exp
        self._counted_steps[i] = steps_done
        self.traces[i].series.append(vals)
        self.traces[i].steps_sampled = steps_done

    def _image(self, i, profile):
        n = len(profile)
        if self._imagers[i] is None:
            self._imagers[i] = [PoissonProcess(self.s.seed & 0xFFFFFFFF, _stream(self._hash[i], k)) for k in range(n)]
        im = self._imagers[i]
        sens = self.s.probes[i].sensitivity
        b = self.s.probes[i].bin_size
        counts = np.zeros(n, np.float32)
        px = min(n, len(im))
        for s0 in range(0, px, b):
            e = min(px, s0 + b)
            tot = sum(im[k].count_up_to(float(profile[k]) * sens) for k in range(s0, e))
            counts[s0:e] = tot
        return counts


# ====================================================================== CSV
def write_csv(path: Path, probe: Probe, trace: ProbeTrace) -> int:
    """Semicolon CSV as ``ProbeCsv.cs``: a row per step (series) or per frame (profiles); returns the row count."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        if not probe.shows_profile:
            col = "detections" if probe.is_detector else {
                ProbeMeasurement.AMPLITUDE: "amplitude", ProbeMeasurement.VALUE: "value",
                ProbeMeasurement.INTEGRATED_VALUE: "integrated value"}[probe.measurement]
            f.write(f"step;{col}\n")
            v = trace.values()
            for i, x in enumerate(v):
                f.write(f"{i};{float(x)!r}\n")
            return len(v)
        profs = trace.profiles
        n = len(profs[0]) if profs else 0
        dist = [f"{k * trace.length / (n - 1) if n > 1 else 0:.3f}".rstrip("0").rstrip(".") for k in range(n)]
        f.write("frame;step" + "".join(";" + d for d in dist) + "\n")
        for k, (p, st) in enumerate(zip(profs, trace.profile_steps)):
            f.write(f"{k};{st}" + "".join(f";{float(x)!r}" for x in p) + "\n")
        return len(profs)
