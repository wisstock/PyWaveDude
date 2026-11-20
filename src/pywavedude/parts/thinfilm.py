# SPDX-License-Identifier: GPL-3.0-or-later
# PyWaveDude: Python port of WaveDude Next (C) Alexander Graef, after Jeroen Vleggaar (Huygens Optics) and 0x23.
"""Thin-film optics for designed layer stacks: the quarter-wave thickness, the reflection band of a quarter-wave
stack and the transfer-matrix reflectance.

Units: pixels and time steps; a wave in vacuum (index 1) moves 1 px per step, so a source of vacuum wavelength λ
has angular frequency ω = 2π / λ per step. The 2D scalar wave is the s-polarised (TE) case of thin-film optics."""

from __future__ import annotations

import cmath
import math


def cos_inside(n: float, n0: float, angle: float) -> complex:
    """cos θ in a layer of index ``n`` for a ray at ``angle`` degrees in index ``n0`` (Snell; complex past TIR)."""
    s = n0 * math.sin(math.radians(angle)) / n
    return cmath.sqrt(1.0 - s * s)


def quarter_wave(n: float, lam: float, n0: float, angle: float) -> float:
    """Thickness (px) giving a 90° phase across the layer at ``angle`` for vacuum wavelength ``lam``."""
    c = cos_inside(n, n0, angle).real
    return lam / (4 * n * c) if c > 0 and lam > 0 else math.nan


def band_half_width(n_high: float, n_low: float, n0: float, angle: float) -> float:
    """Half width Δg of a quarter-wave stack's first reflection band in g = λ0 / λ (edges at λ0 / (1 ± Δg))."""
    eh = n_high * cos_inside(n_high, n0, angle).real
    el = n_low * cos_inside(n_low, n0, angle).real
    return 2 / math.pi * math.asin(abs(eh - el) / (eh + el)) if eh + el > 0 else 0.0


def reflectance(layers: list[tuple[float, float]], lam: float, n0: float, n_sub: float, angle: float) -> float:
    """Power reflectance of ``layers`` ((thickness px, index), first facing the incident medium ``n0``) on a
    substrate ``n_sub``, s-polarised, characteristic-matrix method."""
    eta0 = n0 * cos_inside(n0, n0, angle)
    eta_s = n_sub * cos_inside(n_sub, n0, angle)
    m11, m12, m21, m22 = 1 + 0j, 0j, 0j, 1 + 0j
    for d, n in layers:
        c = cos_inside(n, n0, angle)
        eta = n * c
        delta = 2 * math.pi * n * d * c / lam
        cd, sd = cmath.cos(delta), cmath.sin(delta)
        a11, a12, a21, a22 = cd, 1j * sd / eta, 1j * eta * sd, cd
        m11, m12, m21, m22 = m11 * a11 + m12 * a21, m11 * a12 + m12 * a22, m21 * a11 + m22 * a21, \
            m21 * a12 + m22 * a22
    b, c = m11 + m12 * eta_s, m21 + m22 * eta_s
    r = (eta0 * b - c) / (eta0 * b + c)
    return float(abs(r) ** 2)
