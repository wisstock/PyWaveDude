# SPDX-License-Identifier: GPL-3.0-or-later
# PyWaveDude: Python port of WaveDude Next (C) Alexander Graef, after Jeroen Vleggaar (Huygens Optics) and 0x23.
"""Optical logic gates: the directional-coupler design (port of ``Parts/LogicGateDesign.cs`` and
``Parts/WaveguideModes.cs``).

Two single-mode waveguides (ports A below, B above, in the part frame y < 0 / y > 0) run in, one arm gets a phase
section, S-bends bring them to a coupled spacing, a straight coupler makes a 50:50 splitter, and they bend apart to
the outputs. The guided modes are those of the simulator's own discrete operator, so the design fits the grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numba import njit

from ..engine.base import compute_substeps

PHASE_INDEX_STEP = 0.1
BEND_LOSS_EXPONENT = 11.0
MIN_ABSORBER = 0.5
ABSORPTION = 64
BEND_SHARE_LIMIT = 0.6
COUPLING_EXCESS = 0.07
BEND_SAMPLES = 24
SUPERSAMPLE = 4
MAX_INDEX = 2.55


# ====================================================================== discrete slab modes
def omega(wavelength: float) -> float:
    sub = compute_substeps(1.0)
    s = math.sin(2.0 * math.pi / wavelength / (2.0 * sub))
    return 4.0 * sub * sub * s * s


def _mu(beta: float) -> float:
    return math.cos(2.0 * beta) / 6.0 - 8.0 * math.cos(beta) / 3.0 + 2.5


def beta_of(mu: float) -> float:
    lo, hi = 0.0, math.pi
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if _mu(mid) < mu:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


@njit(cache=True)
def _largest_mu(index, om, odd):
    """Largest eigenvalue of L_y + diag(Ω n²) of the chosen symmetry, by inverse iteration (Cholesky, pentadiagonal)."""
    n = index.shape[0]
    pot = om * index * index
    top = pot.max()
    sigma = top + 1e-9
    l0 = np.zeros(n)
    l1 = np.zeros(n)
    l2 = np.zeros(n)
    near, far = -4.0 / 3.0, 1.0 / 12.0
    for j in range(n):
        if j >= 2:
            l2[j] = far / l0[j - 2]
        if j >= 1:
            s = near - (l2[j] * l1[j - 1] if j >= 2 else 0.0)
            l1[j] = s / l0[j - 1]
        l0[j] = math.sqrt(sigma + 2.5 - pot[j] - l1[j] * l1[j] - l2[j] * l2[j])
    x = np.empty(n)
    for j in range(n):
        t = (j + 0.5) / n - 0.5
        x[j] = t if odd else 1.0 - t * t
    mu = np.nan
    for _ in range(400):
        for j in range(n // 2):
            a, b = x[j], x[n - 1 - j]
            x[j] = (a - b) / 2.0 if odd else (a + b) / 2.0
            x[n - 1 - j] = -x[j] if odd else x[j]
        if n % 2 == 1 and odd:
            x[n // 2] = 0.0
        x /= math.sqrt((x * x).sum())
        r = 0.0
        for j in range(n):
            am2 = x[j - 2] if j >= 2 else 0.0
            ap2 = x[j + 2] if j + 2 < n else 0.0
            am1 = x[j - 1] if j >= 1 else 0.0
            ap1 = x[j + 1] if j + 1 < n else 0.0
            r += x[j] * (-(am2 + ap2) / 12.0 + 4.0 * (am1 + ap1) / 3.0 - 2.5 * x[j] + pot[j] * x[j])
        if abs(r - mu) < 1e-15:
            mu = r
            break
        mu = r
        for j in range(n):
            s = x[j]
            if j >= 1:
                s -= l1[j] * x[j - 1]
            if j >= 2:
                s -= l2[j] * x[j - 2]
            x[j] = s / l0[j]
        for j in range(n - 1, -1, -1):
            s = x[j]
            if j + 1 < n:
                s -= l1[j + 1] * x[j + 1]
            if j + 2 < n:
                s -= l2[j + 2] * x[j + 2]
            x[j] = s / l0[j]
    edge = max(pot[0], pot[n - 1])
    return mu if mu > edge else np.nan


def mode_beta(index: np.ndarray, om: float, odd: bool) -> float | None:
    mu = _largest_mu(np.ascontiguousarray(index, np.float64), om, odd)
    return None if math.isnan(mu) else beta_of(mu)


def rows(first: int, count: int, profile) -> np.ndarray:
    """Index per grid row as the painter would supersample it (4 samples, rounded to 0.01)."""
    y = first + np.arange(count)[:, None] + (np.arange(SUPERSAMPLE)[None, :] + 0.5) / SUPERSAMPLE
    return np.round(profile(y).mean(axis=1), 2)


def _single_beta(wl, om, core, clad, width, cladding):
    reach = cladding if cladding > 0 else 2.0 * wl
    half = math.ceil(width / 2.0 + reach + 8)
    r = rows(-half, 2 * half, lambda y: np.where(np.abs(y) <= width / 2.0, core,
                                                np.where(np.abs(y) <= width / 2.0 + reach, clad, 1.0)))
    return mode_beta(r, om, False)


def _kappa_at(distance, om, core, clad, width, cladding):
    c, outer = width / 2.0, width / 2.0 + cladding
    half = math.ceil(distance / 2.0 + outer + 8)

    def prof(y):
        nearest = np.minimum(np.abs(y - distance / 2.0), np.abs(y + distance / 2.0))
        return np.where(nearest <= c, core, np.where(nearest <= outer, clad, 1.0))

    r = rows(-half, 2 * half, prof)
    even, odd = mode_beta(r, om, False), mode_beta(r, om, True)
    return max(0.0, (even - odd) / 2.0) if even is not None and odd is not None else 0.0


def bend_share(t):
    return t - np.sin(2.0 * np.pi * t) / (2.0 * np.pi)


def _bend_kappa(coupled, spacing, bend_length, om, core, clad, width, outer):
    total = 0.0
    for i in range(BEND_SAMPLES + 1):
        t = i / BEND_SAMPLES
        d = spacing - (spacing - coupled) * bend_share(t)
        w = 1.0 if i in (0, BEND_SAMPLES) else (4.0 if i % 2 == 1 else 2.0)
        total += w * _kappa_at(d, om, core, clad, width, outer)
    return total * bend_length / (3.0 * BEND_SAMPLES)


# ====================================================================== design
@dataclass(frozen=True)
class GateDesign:
    wavelength: float
    core_index: float
    cladding_index: float
    core_width: int
    cladding_thickness: int
    port_spacing: int
    coupled_spacing: int
    input_length: int
    lead_length: int
    phase_region_length: int
    phase_length: int
    phase_shift: float
    bend_length: int
    coupler_length: int
    output_length: int
    output_ramp_length: int
    phase_on_arm_a: bool
    phase_index: float
    beta: float
    kappa: float
    bend_coupling: float

    @property
    def absorber_width(self) -> float:
        return self.port_spacing / 2.0 - self.core_width / 2.0 - self.cladding_thickness

    @property
    def gap(self) -> int:
        return self.coupled_spacing - self.core_width

    @property
    def length(self) -> int:
        return (self.input_length + self.lead_length + self.phase_region_length + self.lead_length
                + 2 * self.bend_length + self.coupler_length + self.output_length)

    @property
    def left(self) -> float:
        return -self.length / 2.0

    @property
    def input_x(self) -> float:
        return self.left + self.input_length

    @property
    def output_ramp_start(self) -> float:
        return self.length / 2.0 - self.output_ramp_length

    @property
    def output_x(self) -> float:
        return self.length / 2.0 - self.output_length + round(self.wavelength / 2.0)

    @property
    def half_width(self) -> float:
        return float(self.port_spacing)

    @property
    def phase_section(self) -> tuple[float, float]:
        s = self.input_x + self.lead_length
        return s, s + self.phase_length

    def arm_offset_and_slope(self, x):
        """Arm centre |y| and its slope along x (vectorised)."""
        x = np.asarray(x, np.float64)
        far, near = self.port_spacing / 2.0, self.coupled_spacing / 2.0
        bend_in = self.input_x + 2 * self.lead_length + self.phase_region_length
        cstart = bend_in + self.bend_length
        bend_out = cstart + self.coupler_length
        L = self.bend_length
        first = x < cstart
        t = np.where(first, (x - bend_in) / L, 1.0 - (x - bend_out) / L)
        sign = np.where(first, 1.0, -1.0)
        off = far - (far - near) * bend_share(t)
        slope = -sign * (far - near) * (1.0 - np.cos(2 * np.pi * t)) / L
        straight_far = (x <= bend_in) | (x >= bend_out + L)
        coupler = (x >= cstart) & (x <= bend_out)
        off = np.where(straight_far, far, np.where(coupler, near, off))
        slope = np.where(straight_far | coupler, 0.0, slope)
        return off, slope


@lru_cache(maxsize=64)
def design_for(wavelength: float, core: float, clad: float, port_spacing: int, phase_shift: float) -> GateDesign | None:
    if not (wavelength > 0.0) or not (core > clad):
        return None
    om = omega(wavelength)
    aperture = math.sqrt(core * core - clad * clad)
    width = max(2, math.floor(0.9 * wavelength / (2.0 * aperture)))
    beta = _single_beta(wavelength, om, core, clad, width, 0)
    if beta is None:
        return None
    mu = _mu(beta)
    gamma = math.sqrt(max(1e-9, mu - om * clad * clad))
    cladding = max(2, math.ceil(3.0 / gamma))
    absorber = max(2, math.ceil(MIN_ABSORBER * wavelength))
    outer = cladding + absorber
    lowered = core + PHASE_INDEX_STEP > MAX_INDEX + 1e-9
    b_leads = phase_shift >= 0.0
    phase_on_a = b_leads == lowered
    phase_index = round(core - PHASE_INDEX_STEP if lowered else core + PHASE_INDEX_STEP, 2)
    single = _single_beta(wavelength, om, core, clad, width, outer) or beta
    phase_beta = (_single_beta(wavelength, om, phase_index, clad, width, outer) or single) if phase_index > clad else single
    step = abs(phase_beta - single)
    if step < 1e-6:
        return None
    region = max(1, round(math.pi / 2.0 / step))
    phase_len = min(region, round(abs(phase_shift) * math.pi / 180.0 / step))
    radius = BEND_LOSS_EXPONENT * 1.5 * single * single / gamma ** 3
    gap = max(2, round(1.0 / gamma))
    gap += (width + gap) % 2
    while True:
        coupled = width + gap
        spacing = max(port_spacing, max(coupled + width + 2 * cladding + 2, math.ceil(width + 2.0 * (cladding + absorber))))
        spacing += spacing % 2
        bend_length = max(1, math.ceil(math.sqrt(2.0 * math.pi * radius * (spacing - coupled) / 2.0)))
        kappa = _kappa_at(coupled, om, core, clad, width, outer)
        bend = _bend_kappa(coupled, spacing, bend_length, om, core, clad, width, outer)
        if 2.0 * bend <= BEND_SHARE_LIMIT * math.pi / 4.0 or gap >= 2 * cladding:
            break
        gap += 2
    coupler = max(0, round((math.pi / 4.0 - COUPLING_EXCESS - 2.0 * bend) / kappa)) if kappa > 0 else 0
    return GateDesign(wavelength, core, clad, width, cladding, spacing, coupled, round(wavelength / 2.0),
                      round(wavelength), region, phase_len, phase_shift, bend_length, coupler,
                      round(3.0 * wavelength), round(2.0 * wavelength), phase_on_a, phase_index, single, kappa,
                      2.0 * bend)
