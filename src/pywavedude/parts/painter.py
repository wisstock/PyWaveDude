# SPDX-License-Identifier: GPL-3.0-or-later
# PyWaveDude: Python port of WaveDude Next (C) Alexander Graef, after Jeroen Vleggaar (Huygens Optics) and 0x23.
"""Paints parts into a scene image (port of ``Parts/PartPainter.cs``).

The image is RGB as the scene images are: red = index × 100 (0 = mirror), green = sources, blue = absorption.
Pixels well inside or outside a part take the centre sample; pixels within reach of an edge (the outline, an
inner edge or the coating band's inner edge) average a 4 × 4 supersample, outside samples counting as the index
the pixel had before.
"""

from __future__ import annotations

import numpy as np

from . import model as m
from .shapes import PartShape, pieces_of

SUPERSAMPLE = 4
SAMPLE_REACH = 0.6


def blank_scene(width: int, height: int, background_index: float = 1.0) -> np.ndarray:
    rgb = np.zeros((height, width, 3), np.uint8)
    rgb[..., 0] = int(np.clip(round(background_index * 100.0), 0, 255))
    return rgb


def paint_parts(rgb: np.ndarray | None, parts: list[m.OpticalPart], width: int = 0, height: int = 0,
                background_index: float = 1.0) -> np.ndarray:
    """A copy of ``rgb`` (or a blank canvas) with every enabled part painted in, in list order."""
    out = blank_scene(width, height, background_index) if rgb is None else np.array(rgb[..., :3], np.uint8)
    for part in parts:
        if part.enabled:
            paint_part(out, part)
    return out


def paint_part(rgb: np.ndarray, part: m.OpticalPart) -> None:
    h, w = rgb.shape[:2]
    for shape in pieces_of(part):
        x0, y0, x1, y1 = shape.bounds(w, h)
        if x0 > x1 or y0 > y1:
            continue
        ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1].astype(np.float64)
        fx, fy = shape.to_frame(xs + 0.5, ys + 0.5)
        if shape.absorption > 0:
            _paint_absorption(rgb[y0:y1 + 1, x0:x1 + 1, 2], shape, fx, fy)
        if shape.paints_index:
            _paint_index(rgb[y0:y1 + 1, x0:x1 + 1, 0], shape, xs, ys, fx, fy)


def _paint_index(red: np.ndarray, shape: PartShape, xs, ys, fx, fy) -> None:
    if shape.is_mirror:
        red[shape.depth(fx, fy) >= 0.0] = 0
        return
    depth, inner = shape.depths(fx, fy)
    existing = np.where(red == 0, 1.0, red / 100.0)
    coat = shape.coating
    clear = (depth > SAMPLE_REACH) & (inner > SAMPLE_REACH)
    if coat:
        clear &= np.abs(depth - coat) > SAMPLE_REACH
    edge = ~clear & (depth >= -SAMPLE_REACH)
    new = red.copy()
    if clear.any():
        n = shape.index(fx[clear], fy[clear], existing[clear])
        ok = ~np.isnan(n)
        vals = np.clip(np.round(n[ok] * 100.0), 1, 255).astype(np.uint8)
        idx = np.flatnonzero(clear)[ok]
        new.flat[idx] = vals
    if edge.any():
        ex, ey, old = xs[edge], ys[edge], existing[edge]
        total = np.zeros(ex.shape)
        inside = np.zeros(ex.shape, np.int32)
        step = 1.0 / SUPERSAMPLE
        offs = step / 2 + step * np.arange(SUPERSAMPLE)
        # all 16 sub-samples of all edge pixels in one call
        sx = (ex[:, None, None] + offs[None, None, :]).repeat(SUPERSAMPLE, 1)
        sy = (ey[:, None, None] + offs[None, :, None]).repeat(SUPERSAMPLE, 2)
        sfx, sfy = shape.to_frame(sx, sy)
        n = shape.index(sfx, sfy, old[:, None, None])
        hit = ~np.isnan(n)
        total = np.where(hit, n, old[:, None, None]).sum(axis=(1, 2))
        inside = hit.sum(axis=(1, 2))
        vals = np.clip(np.round(total / SUPERSAMPLE ** 2 * 100.0), 1, 255).astype(np.uint8)
        idx = np.flatnonzero(edge)
        keep = inside > 0
        new.flat[idx[keep]] = vals[keep]
    red[...] = new


def _paint_absorption(blue: np.ndarray, shape: PartShape, fx, fy) -> None:
    depth = shape.depth(fx, fy)
    deepest = max(0.0, float(np.max(depth)))
    ramp = min(shape.absorption_ramp, deepest)
    inside = depth >= 0.0
    if not inside.any():
        return
    share = shape.absorption_share(fx[inside], fy[inside], depth[inside], ramp)
    val = np.round(shape.absorption * np.asarray(share)).astype(np.int64)
    cur = blue[inside].astype(np.int64)
    blue[inside] = np.maximum(cur, np.clip(val, 0, 255)).astype(np.uint8)
