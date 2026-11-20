# SPDX-License-Identifier: GPL-3.0-or-later
# PyWaveDude: Python port of WaveDude Next (C) Alexander Graef, after Jeroen Vleggaar (Huygens Optics) and 0x23.
"""Object sources and their per-step drive (port of ``Emitter.cs`` and ``WaveSimulator.ApplySources``)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..rng import Pcg32, stable_hash

SOURCE_OPACITY = 0.9          # u = u * 0.9 + drive * 0.1 at every source cell
DETACH_RAMP_STEPS = 30
_CYCLE_ADVANCE_GUARD = 1000
_EDGE_WIDTH = 0.35


class SourceMode(str, Enum):
    CONTINUOUS = "Continuous"
    PULSED = "Pulsed"


class PulseShape(str, Enum):
    GAUSSIAN = "Gaussian"
    SQUARE = "SmoothedSquare"


@dataclass
class PointSource:
    """A source as in the project tree (``Source.cs``): one emitter, or a composite grid of them.

    A composite is ``size_x`` × ``size_y`` emitters ``spacing_x`` / ``spacing_y`` apart, centred on (x, y) and turned
    by ``array_angle`` (counter-clockwise on the screen); each element may be jittered in position, frequency,
    phase and off-width by up to ± the ``random_*`` amounts, drawn from the project seed and the source's name.
    """

    x: int
    y: int
    name: str = "Source"
    enabled: bool = True
    level: float = 100.0          # frequency level 0.01 .. 255
    amplitude: float = 100.0      # percent
    phase: float = 0.0            # degrees
    mode: SourceMode = SourceMode.CONTINUOUS
    pulse_shape: PulseShape = PulseShape.GAUSSIAN
    pulse_width: int = 10         # on-span, cycles
    off_width: int = 10           # off-span, cycles
    detach_after: int = -1        # step; -1 never
    composite: bool = False
    size_x: int = 1
    size_y: int = 1
    spacing_x: int = 10
    spacing_y: int = 10
    array_angle: float = 0.0
    random_frequency: int = 0
    random_phase: int = 0
    random_off_width: int = 0
    random_position: int = 0

    def normalize(self) -> None:
        self.level = min(max(self.level, 0.01), 255.0) if math.isfinite(self.level) else 100.0
        self.amplitude = min(max(self.amplitude, 0.0), 100.0) if math.isfinite(self.amplitude) else 100.0
        self.phase = min(max(self.phase, -180.0), 180.0) if math.isfinite(self.phase) else 0.0
        self.pulse_width = max(1, self.pulse_width)
        self.off_width = max(1, self.off_width)
        if self.detach_after < 0:
            self.detach_after = -1
        self.size_x, self.size_y = max(1, self.size_x), max(1, self.size_y)
        a = self.array_angle if math.isfinite(self.array_angle) else 0.0
        if a <= -180.0 or a > 180.0:
            a -= 360.0 * math.ceil((a - 180.0) / 360.0)
        self.array_angle = a
        for f in ("random_frequency", "random_phase", "random_off_width", "random_position"):
            setattr(self, f, max(0, getattr(self, f)))

    def elements(self, project_seed: int = 0) -> list[tuple[int, int, float, float, int]]:
        """(x, y, level, phase, off width) of every emitter (port of ``Source.GenerateElements``)."""
        self.normalize()
        rng = Pcg32(project_seed & 0xFFFFFFFF, stable_hash(self.name))
        jit = lambda amount: 0 if amount <= 0 else rng.next_int(-amount, amount)  # noqa: E731
        if not self.composite:
            return [(self.x, self.y, self.level, self.phase, self.off_width)]
        r = math.radians(self.array_angle)
        c, s = math.cos(r), math.sin(r)
        out = []
        for row in range(self.size_y):
            for col in range(self.size_x):
                dx = (col - (self.size_x - 1) / 2.0) * self.spacing_x
                dy = (row - (self.size_y - 1) / 2.0) * self.spacing_y
                x = self.x + round(dx * c + dy * s)
                y = self.y + round(dy * c - dx * s)
                ex = x + jit(self.random_position)
                ey = y + jit(self.random_position)
                level = min(max(self.level + jit(self.random_frequency), 0.01), 255.0)
                phase = min(max(self.phase + jit(self.random_phase), -180.0), 180.0)
                off = max(1, self.off_width + jit(self.random_off_width))
                out.append((ex, ey, level, phase, off))
        return out


def xorshift32(x: int) -> int:
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= x >> 17
    x ^= (x << 5) & 0xFFFFFFFF
    return x & 0xFFFFFFFF


@dataclass
class Emitter:
    x: int
    y: int
    level: float
    amplitude: float
    phase_radians: float
    mode: SourceMode
    pulse_shape: PulseShape
    on_width: int
    off_width: int
    off_width_base: int
    randomize_off_width: int
    detach_after: int
    rng_seed: int

    @classmethod
    def from_source(cls, s: PointSource, project_seed: int = 0) -> list["Emitter"]:
        """Every emitter of a source, each with its own gap seed (port of ``Emitter.FromSource``)."""
        if not s.enabled:
            return []
        rng = Pcg32(project_seed & 0xFFFFFFFF, stable_hash(s.name) ^ 0x47415053)
        out = []
        for x, y, level, phase, off in s.elements(project_seed):
            seed = rng.next_uint32()
            out.append(cls(x, y, level, s.amplitude / 100.0, math.radians(phase), s.mode, s.pulse_shape, s.pulse_width,
                           off, s.off_width, s.random_off_width if s.composite else 0, s.detach_after, seed or 1))
        return out

    @classmethod
    def from_point(cls, s: PointSource, seed: int = 1) -> "Emitter":
        return cls(s.x, s.y, s.level, s.amplitude / 100.0, math.radians(s.phase), s.mode, s.pulse_shape,
                   s.pulse_width, s.off_width, s.off_width, 0, s.detach_after, seed or 1)


def pulse_envelope(el: Emitter, pos: float, off_width: int, first_cycle: bool) -> float:
    on = el.on_width
    period = on + off_width
    if period <= 0:
        return 1.0
    on_pos = pos - off_width
    if el.pulse_shape == PulseShape.GAUSSIAN:
        if on_pos < 0.0:
            return 0.0
        center = on / 2.0
        sigma = max(on / 6.0, 1e-6)
        z = (on_pos - center) / sigma
        return math.exp(-0.5 * z * z)
    if on_pos < -off_width / 2.0 and not first_cycle:
        on_pos += period
    rise = 0.5 + math.atan(on_pos / _EDGE_WIDTH) / math.pi
    fall = 0.5 - math.atan((on_pos - on) / _EDGE_WIDTH) / math.pi
    return rise * fall


class SourceDriver:
    """Host-side drive of all sources: computes the value each source cell is pulled toward at time ``t``.

    Image sources (green pixels) and object emitters are merged into one index/value pair so every engine applies
    them with the same ``u[idx] = u[idx] * 0.9 + v * 0.1`` scatter.
    """

    def __init__(self, padded_index_image: np.ndarray, image_levels: np.ndarray, emitters: list[Emitter],
                 emitter_index: np.ndarray, frequency_scale: float, image_amplitude: float):
        self.scale = frequency_scale
        self.image_amp = image_amplitude
        self.image_freq = image_levels.astype(np.float64) / 255.0 * 0.5 * frequency_scale
        self.emitters = emitters
        self.index = np.concatenate([padded_index_image, emitter_index]).astype(np.int64)
        self.n_image = len(padded_index_image)
        self.values = np.zeros(len(self.index), np.float32)
        self.gain: np.ndarray | None = None      # per entry (image sources, then emitters): 0 silences one
        self.reset()

    def reset(self) -> None:
        self._anchor = [0.0] * len(self.emitters)
        self._cur_off = [e.off_width for e in self.emitters]
        self._rng = [e.rng_seed for e in self.emitters]

    def level_frequency(self, level: float) -> float:
        return float(np.float32(level / 255.0 * 0.5 * self.scale))

    def __len__(self) -> int:
        return len(self.index)

    def compute(self, t: float) -> np.ndarray:
        """Drive values at time ``t``; NaN marks a detached emitter that must not be touched."""
        v = self.values
        if self.n_image:
            v[:self.n_image] = np.sin(self.image_freq * t) * self.image_amp
        for i, e in enumerate(self.emitters):
            fade = 1.0
            if e.detach_after >= 0:
                since = t - e.detach_after
                if since >= DETACH_RAMP_STEPS:
                    v[self.n_image + i] = np.nan
                    continue
                if since > 0.0:
                    fade = 1.0 - since / DETACH_RAMP_STEPS
            freq = self.level_frequency(e.level)
            env = 1.0
            if e.mode != SourceMode.CONTINUOUS:
                pos, off, first = self._pulse_position(i, e, freq, t)
                env = pulse_envelope(e, pos, off, first)
            v[self.n_image + i] = math.sin(e.phase_radians + freq * t) * e.amplitude * env * fade
        if self.gain is not None:
            v *= self.gain          # a silenced source still holds its cell (u * 0.9), as in the full run
        return v

    def _pulse_position(self, i: int, e: Emitter, freq: float, t: float):
        clock = freq * t / (2.0 * math.pi)
        if e.randomize_off_width <= 0:
            off = e.off_width
            period = e.on_width + off
            if period <= 0:
                return 0.0, off, True
            adj = clock + off
            k = math.floor(adj / period)
            return adj - period * k, off, k == 0
        off = self._cur_off[i]
        period = e.on_width + off
        if period <= 0:
            return 0.0, off, True
        since = clock - self._anchor[i]
        guard = 0
        while since >= period and guard < _CYCLE_ADVANCE_GUARD:
            self._anchor[i] += period
            since -= period
            self._rng[i] = state = xorshift32(self._rng[i])
            amt = e.randomize_off_width
            off = max(1, e.off_width_base + int(state % (2 * amt + 1)) - amt)
            self._cur_off[i] = off
            period = e.on_width + off
            guard += 1
        return max(0.0, since), off, self._anchor[i] == 0.0


def wavelength(level: float, frequency_scale: float = 2.0, wave_speed: float = 1.0) -> float:
    cps = level / 255.0 * 0.5 * frequency_scale / (2.0 * math.pi)
    return wave_speed / cps if cps > 0 else math.nan
