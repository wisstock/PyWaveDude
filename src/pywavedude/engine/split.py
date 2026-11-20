# SPDX-License-Identifier: GPL-3.0-or-later
# PyWaveDude: Python port of WaveDude Next (C) Alexander Graef, after Jeroen Vleggaar (Huygens Optics) and 0x23.
"""One simulation per wavelength, run side by side on the same scene.

The medium is linear, so the field of all sources is the sum of the fields of each wavelength group, and waves
of different frequencies do not interfere on average: the intensity is the sum of the groups' intensities. Each
group keeps its own intensity, which the display shows in its own colour. Sources of one wavelength share a
group and interfere as usual. Every engine holds every source (the other wavelengths' silenced), so source cells
act on passing waves alike, the sum is the full run, and all groups share one intensity time constant."""

from __future__ import annotations

import numpy as np

from .base import WaveEngine


class SplitEngine(WaveEngine):
    """Duck-types ``WaveEngine`` over ``engines`` (one per entry of ``levels``, the group's frequency level)."""

    def __init__(self, engines: list[WaveEngine], levels: list[float]):
        first = engines[0]
        self.engines, self.levels = engines, levels
        self.options, self.scene = first.options, first.scene
        self.h, self.w, self.stride, self.rim = first.h, first.w, first.stride, first.rim
        self.canvas_w, self.canvas_h = first.canvas_w, first.canvas_h
        self.open_edges, self.substeps = first.open_edges, first.substeps
        self.name = first.name
        self.time = 0.0
        self._sampler = None
        self._series = []
        self._xp, self._dev, self._host = first._xp, first._dev, first._host

    @property
    def groups(self) -> int:
        return len(self.engines)

    @property
    def description(self) -> str:
        return f"{self.engines[0].description}, {self.groups} wavelength(s) run separately"

    # ------------------------------------------------------------ stepping
    def step(self, n: int = 1) -> None:
        for e in self.engines:                 # each engine samples its own probes
            e.step(n)
        self.time = self.engines[0].time

    def reset(self) -> None:
        for e in self.engines:
            e.reset()
        self.time = 0.0

    # ------------------------------------------------------------ probes: every engine samples, the sums are the
    # probe values, each engine's share is kept for the per-wavelength curves (``group_samples``, ``group_profiles``)
    def set_probes(self, sampler) -> None:
        self._sampler = sampler if sampler is not None and sampler.count else None
        for e in self.engines:
            e.set_probes(sampler)
        self.group_samples = None
        self.group_profiles = None

    def take_probe_samples(self):
        parts = [e.take_probe_samples() for e in self.engines]
        if any(p is None for p in parts):
            self.group_samples = None
            return None
        self.group_samples = parts
        return np.sum(parts, axis=0, dtype=np.float32)

    def read_probe_profiles(self) -> list:
        parts = [e.read_probe_profiles() for e in self.engines]
        self.group_profiles = parts
        return [None if col[0] is None else np.sum(col, axis=0, dtype=np.float32) for col in zip(*parts)] \
            if parts and parts[0] else []

    def _advance(self, source_values) -> None:  # pragma: no cover - step() is overridden
        raise NotImplementedError

    def _clear(self) -> None:  # pragma: no cover - reset() is overridden
        raise NotImplementedError

    def sync(self) -> None:
        for e in self.engines:
            e.sync()

    # ------------------------------------------------------------ probes: sums over the groups
    def _flat_field(self):
        return sum(e._flat_field() for e in self.engines[1:]) + self.engines[0]._flat_field()

    def _flat_intensity(self):
        return sum(e._flat_intensity() for e in self.engines[1:]) + self.engines[0]._flat_intensity()

    def _gather_intensity(self, idx):
        return sum(e._gather_intensity(idx) for e in self.engines[1:]) + self.engines[0]._gather_intensity(idx)

    # ------------------------------------------------------------ reading
    def field(self) -> np.ndarray:
        return np.sum([e.field() for e in self.engines], axis=0, dtype=np.float32)

    def intensities(self) -> list[np.ndarray | None]:
        return [e.intensity() for e in self.engines]

    def intensity(self) -> np.ndarray | None:
        planes = self.intensities()
        return None if planes[0] is None else np.sum(planes, axis=0, dtype=np.float32)

    def preview_start(self, step: int = 1):
        finishes = [e.preview_start(step) for e in self.engines]

        def finish():
            parts = [f() for f in finishes]
            field = np.sum([p[0] for p in parts], axis=0, dtype=np.float32).astype(np.float16)
            if parts[0][1] is None:
                return field, None, None
            groups = np.stack([p[1] for p in parts])
            return field, groups.sum(axis=0, dtype=np.float32).astype(np.float16), groups
        return finish

    def preview(self, step: int = 1):
        return self.preview_start(step)()
