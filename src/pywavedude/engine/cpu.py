# SPDX-License-Identifier: GPL-3.0-or-later
# PyWaveDude: Python port of WaveDude Next (C) Alexander Graef, after Jeroen Vleggaar (Huygens Optics) and 0x23.
"""Multithreaded CPU engine: the stencil JIT-compiled by Numba, rows spread over all cores with ``prange``."""

from __future__ import annotations

import numpy as np
from numba import get_num_threads, njit, prange

from . import base
from .base import EngineOptions, WaveEngine
from .scene import SceneData
from .._parallel import PARALLEL_LOCK
from .sources import SOURCE_OPACITY, Emitter

_F = np.float32
CORNER = _F(base.KERNEL_CORNER)
EDGE = _F(base.KERNEL_EDGE)
NEAR = _F(base.FOURTH_NEAR)
FAR = _F(base.FOURTH_FAR)
CENTER4 = _F(base.FOURTH_CENTER)
BLEND = _F(base.INTENSITY_BLEND)
ONE = _F(1.0)
EDGE_SPEED = _F(base.EDGE_SPEED_FACTOR)
OPACITY = _F(SOURCE_OPACITY)


@njit(inline="always")
def _lap9(u, y, x):
    return (CORNER * (u[y - 1, x - 1] + u[y - 1, x + 1] + u[y + 1, x - 1] + u[y + 1, x + 1])
            + EDGE * (u[y, x - 1] + u[y, x + 1] + u[y - 1, x] + u[y + 1, x]) - u[y, x])


@njit(inline="always")
def _lap4(u, y, x):
    return (FAR * (u[y, x - 2] + u[y, x + 2] + u[y - 2, x] + u[y + 2, x])
            + NEAR * (u[y, x - 1] + u[y, x + 1] + u[y - 1, x] + u[y + 1, x]) + CENTER4 * u[y, x])


@njit(inline="always")
def _finish(u, up, c2, d, s1, s2, out, track, mult, alpha, scale, y, x, lap):
    c = u[y, x]
    nxt = c + (c - up[y, x]) * d[y, x] + lap * c2[y, x] * mult
    up[y, x] = nxt
    if track:
        a1 = s1[y, x] * alpha + nxt * nxt * (ONE - alpha)
        s1[y, x] = a1
        a2 = s2[y, x] * alpha + a1 * (ONE - alpha)
        s2[y, x] = a2
        out[y, x] = (a1 * (ONE - BLEND) + a2 * BLEND) * scale


@njit(parallel=True, cache=True, nogil=True)
def step_kernel(u, up, c2, d, s1, s2, out, track, use4th, mult, alpha, scale):
    """One leapfrog (sub-)step: writes the new field into ``up`` (which then becomes current)."""
    h = u.shape[0] - 2
    w = u.shape[1] - 2
    for y in prange(1, h + 1):
        if use4th and 2 <= y <= h - 1:
            _finish(u, up, c2, d, s1, s2, out, track, mult, alpha, scale, y, 1, _lap9(u, y, 1))
            for x in range(2, w):
                _finish(u, up, c2, d, s1, s2, out, track, mult, alpha, scale, y, x, _lap4(u, y, x))
            _finish(u, up, c2, d, s1, s2, out, track, mult, alpha, scale, y, w, _lap9(u, y, w))
        else:
            for x in range(1, w + 1):
                _finish(u, up, c2, d, s1, s2, out, track, mult, alpha, scale, y, x, _lap9(u, y, x))


@njit(inline="always")
def _absorb(old, nxt, c2, mult, ry, rx, iy, ix):
    s = np.sqrt(c2[iy, ix] * mult) * EDGE_SPEED
    k = (s - ONE) / (s + ONE)
    nxt[ry, rx] = old[iy, ix] + k * (nxt[iy, ix] - old[ry, rx])


@njit(cache=True, nogil=True)
def open_edges_kernel(old, nxt, c2, mult):
    """Mur first-order absorbing condition on the one-cell border ring."""
    h = old.shape[0] - 2
    w = old.shape[1] - 2
    for x in range(0, w + 2):
        ix = min(max(x, 1), w)
        _absorb(old, nxt, c2, mult, 0, x, 1, ix)
        _absorb(old, nxt, c2, mult, h + 1, x, h, ix)
    for y in range(1, h + 1):
        _absorb(old, nxt, c2, mult, y, 0, y, 1)
        _absorb(old, nxt, c2, mult, y, w + 1, y, w)


@njit(cache=True, nogil=True)
def apply_sources_kernel(u_flat, idx, values):
    for i in range(idx.shape[0]):
        v = values[i]
        if v == v:  # NaN = detached emitter, leave the cell alone
            j = idx[i]
            u_flat[j] = u_flat[j] * OPACITY + v * (ONE - OPACITY)


class CpuEngine(WaveEngine):
    name = "CPU"

    def __init__(self, scene: SceneData, options: EngineOptions, emitters: list[Emitter] | None = None):
        super().__init__(scene, options, emitters)
        shape = self.c2_host.shape
        self.c2, self.d = self.c2_host, self.d_host
        self.u = np.zeros(shape, np.float32)
        self.up = np.zeros(shape, np.float32)
        self.track = options.track_intensity
        n = shape if self.track else (1, 1)
        self.s1, self.s2, self.out = (np.zeros(n, np.float32) for _ in range(3))
        self.use4th = options.fourth_order
        self.mult = _F(self.c2_mult)
        self.scale = _F(options.intensity_scale)

    @property
    def description(self) -> str:
        return super().description + f", {get_num_threads()} threads"

    def _clear(self) -> None:
        for a in (self.u, self.up, self.s1, self.s2, self.out):
            a.fill(0)

    def _advance(self, values) -> None:
        if values is not None:
            apply_sources_kernel(self.u.reshape(-1), self.driver.index, values)
        a = _F(self.alpha)
        for sub in range(1, self.substeps + 1):
            track = self.track and sub == self.substeps
            with PARALLEL_LOCK:
                step_kernel(self.u, self.up, self.c2, self.d, self.s1, self.s2, self.out,
                            track, self.use4th, self.mult, a, self.scale)
            if self.open_edges:
                open_edges_kernel(self.u, self.up, self.c2, self.mult)
            self.u, self.up = self.up, self.u

    def _flat_field(self):
        return self.u.reshape(-1)

    def _flat_intensity(self):
        return self.out.reshape(-1) if self.track else self.u.reshape(-1)

    def field(self) -> np.ndarray:
        return self._interior(self.u).copy()

    def intensity(self) -> np.ndarray | None:
        return self._interior(self.out).copy() if self.track else None
