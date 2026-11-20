# SPDX-License-Identifier: GPL-3.0-or-later
# PyWaveDude: Python port of WaveDude Next (C) Alexander Graef, after Jeroen Vleggaar (Huygens Optics) and 0x23.
"""Per-cell wave speed and damping maps of a scene (port of ``SceneData.cs``)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MIRROR_REFRACTIVE_INDEX = 1000.0


@dataclass
class SceneData:
    """c² (= 1/n²) and damping maps on the simulated grid (canvas plus rim), plus the image sources.

    Arrays are float32 of shape ``(height, width)`` of the grid; the canvas sits ``rim`` cells inside it.
    """

    c2: np.ndarray
    d: np.ndarray
    rim: int
    open_edges: bool
    source_x: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    source_y: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    source_level: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint8))

    @property
    def height(self) -> int:
        return self.c2.shape[0]

    @property
    def width(self) -> int:
        return self.c2.shape[1]

    @property
    def canvas_width(self) -> int:
        return self.width - 2 * self.rim

    @property
    def canvas_height(self) -> int:
        return self.height - 2 * self.rim

    # ------------------------------------------------------------------ builders
    @classmethod
    def from_rgb(cls, rgb: np.ndarray, use_sources: bool = True, damping_edge: int = -32,
                 open_edges: bool = True) -> "SceneData":
        """Scene from an ``(h, w, 3+)`` uint8 image: red = index × 100 (0 = mirror), green = source level,
        blue = absorption."""
        red = rgb[..., 0].astype(np.float64)
        green = rgb[..., 1]
        blue = rgb[..., 2].astype(np.float64)
        n = np.where(red == 0, MIRROR_REFRACTIVE_INDEX, np.clip(red / 100.0, 1.0, 10.0))
        c2 = (1.0 / n) ** 2
        d = 1.0 - blue / 255.0
        if use_sources:
            sy, sx = np.nonzero(green)
            level = green[sy, sx].astype(np.uint8)
        else:
            sy = sx = np.zeros(0, np.int64)
            level = np.zeros(0, np.uint8)
        return cls._with_rim(c2.astype(np.float32), d.astype(np.float32), damping_edge, open_edges,
                             sx.astype(np.int64), sy.astype(np.int64), level)

    @classmethod
    def empty(cls, width: int, height: int, damping_edge: int = -32, refractive_index: float = 1.0,
              open_edges: bool = True) -> "SceneData":
        speed = 1.0 / float(np.clip(refractive_index, 1.0, 10.0))
        c2 = np.full((height, width), speed * speed, np.float32)
        d = np.ones((height, width), np.float32)
        z = np.zeros(0, np.int64)
        return cls._with_rim(c2, d, damping_edge, open_edges, z, z, np.zeros(0, np.uint8))

    @classmethod
    def _with_rim(cls, c2, d, damping_edge, open_edges, sx, sy, level) -> "SceneData":
        rim = max(0, -damping_edge)
        if rim == 0:
            _apply_border(d, damping_edge)
            return cls(c2, d, 0, open_edges, sx, sy, level)
        # the rim continues the outermost canvas pixels (edge padding) and damps quadratically outwards
        c2 = np.pad(c2, rim, mode="edge")
        d = np.pad(d, rim, mode="edge")
        _apply_rim(d, rim)
        return cls(c2, d, rim, open_edges, sx + rim, sy + rim, level)


def _apply_rim(d: np.ndarray, rim: int) -> None:
    h, w = d.shape
    for i in range(rim):
        into = (rim - i) / rim
        f = np.float32(1.0 - 0.5 * into * into)
        d[i, i:w - i] = f
        d[h - 1 - i, i:w - i] = f
        d[i:h - i, i] = f
        d[i:h - i, w - 1 - i] = f


def _apply_border(d: np.ndarray, thickness: int) -> None:
    h, w = d.shape
    border = min(thickness, min(w, h) // 2)
    for i in range(max(0, border)):
        f = np.float32(np.sqrt(i / border))
        d[i, i:w - i] = f
        d[h - 1 - i, i:w - i] = f
        d[i:h - i, i] = f
        d[i:h - i, w - 1 - i] = f
