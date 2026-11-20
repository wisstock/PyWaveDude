# SPDX-License-Identifier: GPL-3.0-or-later
# PyWaveDude: Python port of WaveDude Next (C) Alexander Graef, after Jeroen Vleggaar (Huygens Optics) and 0x23.
"""Colour schemes (``.wdscheme``) and field/intensity → RGBA mapping (port of ``ColorLut.cs``)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from numba import njit, prange

from ._parallel import PARALLEL_LOCK

SCHEME_DIR = Path(__file__).parent / "schemes"
LUT_SIZE = 1024


@dataclass(frozen=True)
class ColorScheme:
    name: str
    kind: str                      # "Field" or "Intensity"
    stops: tuple[tuple[float, tuple[int, int, int]], ...]

    @classmethod
    def load(cls, path: Path) -> "ColorScheme":
        return cls.from_element(ET.parse(path).getroot(), path.stem)

    def to_element(self) -> ET.Element:
        el = ET.Element("ColorScheme", {"Name": self.name})
        if self.kind:
            el.set("Kind", self.kind)
        for pos, (r, g, b) in self.stops:
            ET.SubElement(el, "Stop", {"At": f"{pos:g}", "Color": f"#{r:02X}{g:02X}{b:02X}"})
        return el

    @classmethod
    def from_element(cls, root: ET.Element, fallback: str = "Scheme") -> "ColorScheme":
        stops = []
        for s in root.findall("Stop"):
            c = s.get("Color").lstrip("#")
            stops.append((float(s.get("At")), (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))))
        stops.sort(key=lambda t: t[0])
        return cls(root.get("Name", fallback), root.get("Kind", ""), tuple(stops))

    def lut(self, black_level: float = 0.0, size: int = LUT_SIZE) -> np.ndarray:
        """``(size, 4)`` float32 RGBA in 0..1, position i/(size-1), with the black level lift."""
        pos = np.array([p for p, _ in self.stops])
        col = np.array([c for _, c in self.stops], np.float64)
        x = np.linspace(0.0, 1.0, size)
        rgb = np.stack([np.interp(x, pos, col[:, k]) for k in range(3)], axis=1) / 255.0
        rgb = np.clip(rgb * (1.0 - black_level) + black_level, 0.0, 1.0)
        return np.concatenate([rgb, np.ones((size, 1))], axis=1).astype(np.float32)


@lru_cache(maxsize=1)
def builtin_schemes() -> dict[str, ColorScheme]:
    return {s.name: s for s in (ColorScheme.load(p) for p in sorted(SCHEME_DIR.glob("*.wdscheme")))}


@njit(parallel=True, cache=True, nogil=True)
def _map(values, lut, brightness, is_intensity, step, out):
    h, w = out.shape[0], out.shape[1]
    n = lut.shape[0] - 1
    for y in prange(h):
        for x in range(w):
            v = values[y * step, x * step]
            if is_intensity:
                p = np.sqrt(v) * brightness if v > 0 else 0.0
                p = min(p, 1.0)
            else:
                s = v * brightness
                if s != s:
                    p = 0.5
                else:
                    p = (min(max(s, -1.0), 1.0) + 1.0) * 0.5
            i = int(p * n + 0.5)
            for c in range(4):
                out[y, x, c] = lut[i, c]


def colorize(values: np.ndarray, lut: np.ndarray, brightness: float, intensity: bool, step: int = 1,
             out: np.ndarray | None = None) -> np.ndarray:
    """Map a field (−1..1 → 0..1) or intensity (√I) plane through ``lut``; ``step`` subsamples for previews."""
    h, w = -(-values.shape[0] // step), -(-values.shape[1] // step)
    if out is None or out.shape != (h, w, 4):
        out = np.empty((h, w, 4), np.float32)
    with PARALLEL_LOCK:
        _map(values, lut, np.float32(brightness), intensity, step, out)
    return out


# ---------------------------------------------------------------- true colour (one hue per wavelength)
def spectrum_rgb(nm: float) -> tuple[float, float, float]:
    """Linear RGB 0..1 of a monochromatic colour (piecewise approximation of the visible spectrum, 380–780 nm)."""
    nm = min(max(float(nm), 380.0), 780.0)
    if nm < 440:
        r, g, b = (440 - nm) / 60, 0.0, 1.0
    elif nm < 490:
        r, g, b = 0.0, (nm - 440) / 50, 1.0
    elif nm < 510:
        r, g, b = 0.0, 1.0, (510 - nm) / 20
    elif nm < 580:
        r, g, b = (nm - 510) / 70, 1.0, 0.0
    elif nm < 645:
        r, g, b = 1.0, (645 - nm) / 65, 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0
    fade = 0.3 + 0.7 * (nm - 380) / 40 if nm < 420 else (0.3 + 0.7 * (780 - nm) / 80 if nm > 700 else 1.0)
    return r * fade, g * fade, b * fade


def wavelength_nm(lambdas_px, nm_per_px: float = 0.0) -> list[float]:
    """Display wavelengths: ``nm_per_px`` × λ, or (0) the shortest at 450 nm and the longest at 650 nm."""
    lam = [float(v) for v in lambdas_px]
    if nm_per_px > 0:
        return [v * nm_per_px for v in lam]
    lo, hi = min(lam), max(lam)
    return [550.0 if hi - lo < 1e-9 else 450.0 + 200.0 * (v - lo) / (hi - lo) for v in lam]


def group_colours(lambdas_px, nm_per_px: float = 0.0) -> np.ndarray:
    """``(groups, 3)`` float32 colours, one per wavelength."""
    return np.array([spectrum_rgb(nm) for nm in wavelength_nm(lambdas_px, nm_per_px)], np.float32).reshape(-1, 3)


@njit(parallel=True, cache=True, nogil=True)
def _mix(groups, colours, brightness, black, out):
    g, h, w = groups.shape
    for y in prange(h):
        for x in range(w):
            r = gg = b = 0.0
            for k in range(g):
                v = groups[k, y, x]
                p = min(np.sqrt(v) * brightness, 1.0) if v > 0 else 0.0
                r += colours[k, 0] * p
                gg += colours[k, 1] * p
                b += colours[k, 2] * p
            out[y, x, 0] = min(r, 1.0) * (1.0 - black) + black
            out[y, x, 1] = min(gg, 1.0) * (1.0 - black) + black
            out[y, x, 2] = min(b, 1.0) * (1.0 - black) + black
            out[y, x, 3] = 1.0


def colour_mix(groups: np.ndarray, colours: np.ndarray, brightness: float, black: float = 0.0) -> np.ndarray:
    """Intensity per wavelength ``(groups, h, w)``, each through a black-to-its-colour ramp (√I, as the intensity
    schemes), added up: overlapping colours mix like light."""
    out = np.empty(groups.shape[1:] + (4,), np.float32)
    with PARALLEL_LOCK:
        _mix(np.ascontiguousarray(groups, dtype=np.float32), np.ascontiguousarray(colours, dtype=np.float32), np.float32(brightness),
             np.float32(black), out)
    return out


def auto_brightness(values: np.ndarray, intensity: bool) -> float:
    """Gain that maps the brightest 0.5 % of the samples to the top of the scheme."""
    a = np.sqrt(np.maximum(values, 0)) if intensity else np.abs(values)
    sample = a[::4, ::4].ravel()
    top = float(np.quantile(sample, 0.995)) if sample.size else 0.0
    return 1.0 / top if top > 1e-12 else 1.0
