# SPDX-License-Identifier: GPL-3.0-or-later
# PyWaveDude: Python port of WaveDude Next (C) Alexander Graef, after Jeroen Vleggaar (Huygens Optics) and 0x23.
"""Backend-neutral part of the wave engine (port of the set-up half of ``WaveSimulator.cs``).

Units: grid spacing and time step are 1, so c is in cells per step and c² = 1/n². The scheme:
``u' = u + (u - u_prev) * d + laplacian(u) * c²`` with a 9-point kernel (corner 0.066, edge 0.184, centre -1)
or a 4th-order 5-point cross (with sub-stepping for stability), Mur first-order open edges, and an intensity
filtered by a blended two-stage one-pole low-pass.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .scene import SceneData
from .sources import Emitter, SourceDriver

KERNEL_CORNER = 0.066
KERNEL_EDGE = 0.184
FOURTH_NEAR = 4.0 / 3.0
FOURTH_FAR = -1.0 / 12.0
FOURTH_CENTER = -5.0
FOURTH_LAMBDA_MAX = 32.0 / 3.0
SUBSTEP_TARGET_LIMIT = 2.0
EDGE_SPEED_FACTOR = 0.56213877
INTENSITY_BLEND = 0.5
NOMINAL_DECAY_LEVEL = 100


@dataclass
class EngineOptions:
    source_frequency_scale: float = 2.0
    source_amplitude: float = 1.0          # image-source amplitude (Scene.ImageSourceScale)
    track_intensity: bool = True
    fourth_order: bool = True
    time_decay: float = 10.0
    intensity_scale: float = 1.0


def compute_substeps(max_c2: float) -> int:
    if max_c2 <= 0:
        return 1
    return max(1, int(round(math.ceil(math.sqrt(max_c2 * FOURTH_LAMBDA_MAX / SUBSTEP_TARGET_LIMIT)))))


def intensity_retention(time_decay: float, lowest_level: float) -> float:
    tau = max(1.0, time_decay * NOMINAL_DECAY_LEVEL / lowest_level)
    return float(np.float32(math.exp(-1.0 / tau)))


class WaveEngine(ABC):
    """Common set-up; subclasses implement ``_advance`` and the buffer accessors."""

    name = "base"

    def __init__(self, scene: SceneData, options: EngineOptions, emitters: list[Emitter] | None = None):
        self.options = options
        self.scene = scene
        h, w = scene.height, scene.width
        self.h, self.w = h, w
        self.stride = w + 2
        self.rim = scene.rim
        self.canvas_w, self.canvas_h = scene.canvas_width, scene.canvas_height
        self.open_edges = scene.open_edges
        c2 = np.zeros((h + 2, w + 2), np.float32)
        d = np.zeros((h + 2, w + 2), np.float32)
        c2[1:-1, 1:-1] = scene.c2
        d[1:-1, 1:-1] = scene.d
        self.substeps = compute_substeps(float(c2.max())) if options.fourth_order else 1
        if self.substeps > 1:
            d = np.power(d, 1.0 / self.substeps).astype(np.float32)
            self.c2_mult = np.float32(1.0 / (self.substeps * self.substeps))
        else:
            self.c2_mult = np.float32(1.0)
        self.c2_host, self.d_host = c2, d
        self.time = 0.0
        self._sampler = None
        self._series: list = []
        self.set_emitters(emitters or [])

    # ------------------------------------------------------------------ sources
    def set_emitters(self, emitters: list[Emitter]) -> None:
        sc = self.scene
        inside = [e for e in emitters if 0 <= e.x < self.canvas_w and 0 <= e.y < self.canvas_h]
        e_idx = np.array([(e.y + 1 + self.rim) * self.stride + e.x + 1 + self.rim for e in inside], np.int64)
        img_idx = (sc.source_y + 1) * self.stride + sc.source_x + 1
        self.driver = SourceDriver(img_idx, sc.source_level, inside, e_idx,
                                   self.options.source_frequency_scale, self.options.source_amplitude)
        levels = [int(lv) for lv in np.unique(sc.source_level) if lv > 0]
        lowest = float(levels[0]) if levels else 0.0
        for e in inside:
            if e.level > 0 and (lowest == 0 or e.level < lowest):
                lowest = e.level
        self.alpha = intensity_retention(self.options.time_decay, lowest or NOMINAL_DECAY_LEVEL)
        self._upload_sources()

    def _upload_sources(self) -> None:
        """Hook for backends that keep the source index on the device."""

    # ------------------------------------------------------------------ stepping
    def step(self, n: int = 1) -> None:
        for _ in range(n):
            values = self.driver.compute(self.time) if len(self.driver) else None
            self._advance(values)
            self.time += 1.0
            if self._sampler is not None:
                self._sample_probes()

    def reset(self) -> None:
        self.time = 0.0
        self.driver.reset()
        self._clear()
        self.set_probes(self._sampler)

    # ------------------------------------------------------------------ probes (port of SampleProbes & co.)
    def set_probes(self, sampler) -> None:
        """Install a ``probes.ProbeSampler`` (or None): point values every step, integrated line totals on the device."""
        self._series = []
        if sampler is None or sampler.count == 0:
            self._sampler = None
            return
        self._sampler = sampler
        prof = sampler.is_profile
        self._has_points = bool((~prof).any())
        pt = np.array([0 if prof[i] else sampler.pixels[i][0] for i in range(sampler.count)], np.int64)
        self._pt_idx = self._dev(pt)
        self._pt_int = self._dev(sampler.use_intensity & ~prof)
        integ = [i for i in range(sampler.count) if sampler.integrate[i]]
        self._int_slices = {}
        if integ:
            off, idx, start = 0, [], []
            for i in integ:
                n = len(sampler.pixels[i])
                self._int_slices[i] = (off, off + n)
                idx.append(sampler.pixels[i])
                start.append(np.full(n, sampler.start_after[i], np.int64))
                off += n
            self._int_idx = self._dev(np.concatenate(idx))
            self._int_start = self._dev(np.concatenate(start))
            self._totals = self._dev(np.zeros(off, np.float32))
        else:
            self._int_idx = None

    def _sample_probes(self) -> None:
        xp, s = self._xp, self._sampler
        if self._has_points:
            f = self._flat_field()[self._pt_idx]
            i = self._gather_intensity(self._pt_idx) if s.needs_intensity else f
            self._series.append(xp.where(self._pt_int, i, f))
        if self._int_idx is not None:
            step = int(round(self.time))
            add = self._gather_intensity(self._int_idx)
            self._totals = self._totals + xp.where(self._int_start <= step, add, 0.0)

    def _probe_pending(self) -> list:
        """Lazy probe arrays a GPU backend should evaluate with its state."""
        if self._sampler is None:
            return []
        return self._series[-8:] + ([self._totals] if self._int_idx is not None else [])

    def take_probe_samples(self) -> np.ndarray | None:
        """The point values since the last call, ``(steps, probes)`` float32 (profile columns are meaningless)."""
        if self._sampler is None or not self._series:
            return None
        out = self._host(self._xp.stack(self._series)).astype(np.float32)
        self._series = []
        return out

    def read_probe_profiles(self) -> list:
        s = self._sampler
        if s is None:
            return []
        out = []
        totals = self._host(self._totals) if self._int_idx is not None else None
        for i in range(s.count):
            if not s.is_profile[i]:
                out.append(None)
            elif s.integrate[i]:
                a, b = self._int_slices[i]
                out.append(totals[a:b].astype(np.float32))
            else:
                px = self._dev(s.pixels[i])
                vals = self._gather_intensity(px) if s.use_intensity[i] else self._flat_field()[px]
                out.append(self._host(vals).astype(np.float32))
        return out

    # backend hooks for the probes: array namespace, device upload, flat current buffers, host download
    _xp = np

    def _dev(self, a: np.ndarray):
        return a

    def _host(self, a) -> np.ndarray:
        return np.asarray(a)

    def _flat_field(self):
        raise NotImplementedError

    def _flat_intensity(self):
        raise NotImplementedError

    def _gather_intensity(self, idx):
        return self._flat_intensity()[idx]

    @property
    def description(self) -> str:
        stencil = f"4th-order, {self.substeps} sub-step(s)" if self.options.fourth_order else "9-point"
        return f"{self.name}: grid {self.w}×{self.h}, {stencil}"

    def _interior(self, a: np.ndarray) -> np.ndarray:
        r = 1 + self.rim
        return a[r:r + self.canvas_h, r:r + self.canvas_w]

    @abstractmethod
    def _advance(self, source_values: np.ndarray | None) -> None: ...

    @abstractmethod
    def _clear(self) -> None: ...

    @abstractmethod
    def field(self) -> np.ndarray:
        """Canvas field as a float32 host array (copy)."""

    @abstractmethod
    def intensity(self) -> np.ndarray | None:
        """Canvas intensity as a float32 host array (copy), or None when not tracked."""

    def preview(self, step: int = 1) -> tuple[np.ndarray, np.ndarray | None]:
        """Field and intensity subsampled by ``step``, as float16 host arrays (what the frame store keeps)."""
        f = self.field()[::step, ::step].astype(np.float16)
        i = self.intensity()
        return f, (i[::step, ::step].astype(np.float16) if i is not None else None)

    def preview_start(self, step: int = 1):
        """Start capturing a preview; returns a function that yields ``preview(step)``'s arrays. A GPU backend
        copies them out while it goes on stepping."""
        result = self.preview(step)
        return lambda: result

    def sync(self) -> None:
        """Block until queued device work is done (GPU backends)."""
