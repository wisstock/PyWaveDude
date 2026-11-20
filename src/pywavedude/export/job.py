# SPDX-License-Identifier: GPL-3.0-or-later
# PyWaveDude: Python port of WaveDude Next (C) Alexander Graef, after Jeroen Vleggaar (Huygens Optics) and 0x23.
"""Export runs (role of ``Export/ExportJob.cs``): always full resolution and full precision.

The original renders from a 16-bit frame store, recomputing a reduced preview first. Here the simulation is simply
run again from step 0 on a fresh engine (the engines are deterministic) and every frame in the export range is
coloured and handed to the sinks as it is computed, so nothing is stored and the samples are the engine's float32.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..render import colorize, colour_mix, group_colours
from .writers import DATA, ExportFormat, open_sink

EDGE_SCALE = 0.25


def glowing_edges(rgb: np.ndarray) -> np.ndarray:
    """Per-channel Sobel magnitude × 0.25 of the scene image (port of ``GlowingEdges.Compute``), RGB uint8."""
    a = rgb.astype(np.int32)
    out = np.zeros_like(rgb, np.uint8)
    if a.shape[0] < 3 or a.shape[1] < 3:
        return out
    up, mid, down = a[:-2], a[1:-1], a[2:]
    col = up + 2 * mid + down
    diff = down - up
    gx = col[:, 2:] - col[:, :-2]
    gy = diff[:, :-2] + 2 * diff[:, 1:-1] + diff[:, 2:]
    out[1:-1, 1:-1] = np.minimum(255.0, np.floor(np.sqrt(gx * gx + gy * gy, dtype=np.float64) * EDGE_SCALE))
    return out


def blend_edges(rgb: np.ndarray, edges: np.ndarray | None, gain: float) -> np.ndarray:
    if edges is None or gain <= 0:
        return rgb
    return np.minimum(255, rgb.astype(np.int32) + (edges * gain).astype(np.int32)).astype(np.uint8)


def to_rgb8(rgba: np.ndarray) -> np.ndarray:
    return np.round(rgba[..., :3] * 255.0).astype(np.uint8)


@dataclass
class ExportLayer:
    intensity: bool
    lut: np.ndarray
    brightness: float
    edge_gain: float = 0.0
    true_colour: bool = False          # split runs: each wavelength's intensity in its own colour
    nm_per_px: float = 0.0
    frequency_scale: float = 2.0
    black: float = 0.0


@dataclass
class ExportSpec:
    fmt: ExportFormat
    path: Path
    layers: list[ExportLayer]          # one output per layer; several get _field / _intensity suffixes
    first_step: int
    last_step: int
    every: int
    fps: int = 60
    quality: int = 85
    still: bool = False
    decor: dict | None = None       # probes, numbers, graphs, markers, sources, seed, total_steps


@dataclass
class ExportProgress:
    frames_done: int = 0
    frames_total: int = 0
    steps_done: int = 0
    finished: bool = False
    cancelled: bool = False
    error: str = ""
    outputs: list[str] = field(default_factory=list)
    seconds: float = 0.0


def output_paths(spec: ExportSpec) -> list[Path]:
    path = Path(spec.path)
    if len(spec.layers) == 1:
        return [path]
    return [path.with_name(f"{path.stem}_{'intensity' if l.intensity else 'field'}{path.suffix}") for l in spec.layers]


class ExportJob:
    """Runs ``make_engine()`` from step 0 to ``last_step`` on a worker thread, writing the frames in range."""

    def __init__(self, make_engine: Callable[[], object], spec: ExportSpec, edges: np.ndarray | None = None):
        self.make_engine = make_engine
        self.spec = spec
        self.edges = edges
        e = max(1, spec.every)
        first = -(-spec.first_step // e) * e
        self.steps = list(range(first, spec.last_step + 1, e)) if not spec.still else [spec.first_step]
        # the collection points: every frame from step 0, as in the preview run, so detectors count alike
        self.collect_at = list(range(e, (self.steps[-1] if self.steps else 0) + 1, e))
        if spec.still and self.steps and self.steps[0] not in self.collect_at and self.steps[0] > 0:
            self.collect_at = sorted(set(self.collect_at + self.steps))
        self.progress = ExportProgress(frames_total=len(self.steps))
        self._cancel = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="export")

    def start(self) -> None:
        self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()

    def join(self, timeout=None) -> None:
        self._thread.join(timeout)

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        sp, pr = self.spec, self.progress
        t0 = time.perf_counter()
        sinks = []
        try:
            eng = self.make_engine()
            w, h = eng.canvas_w, eng.canvas_h
            for layer, path in zip(sp.layers, output_paths(sp)):
                sinks.append(open_sink(sp.fmt, path, w, h, sp.fps, sp.quality, len(self.steps), sp.still,
                                       first=self.steps[0] // max(1, sp.every) if self.steps else 0))
                pr.outputs.append(str(path))
            self._setup_probes(eng)
            wanted = set(self.steps)
            if 0 in wanted:
                self._collect(eng)
                self._write(eng, sinks)
                pr.frames_done += 1
            for target in self.collect_at:
                while eng.time < target:
                    if self._cancel.is_set():
                        pr.cancelled = True
                        return
                    eng.step(min(16, int(target - eng.time)))
                    pr.steps_done = int(eng.time)
                self._collect(eng)
                if target in wanted:
                    self._write(eng, sinks)
                    pr.frames_done += 1
        except Exception as exc:  # reported in the dialog
            pr.error = f"{type(exc).__name__}: {exc}"
        finally:
            for s in sinks:
                try:
                    s.close()
                except Exception as exc:
                    pr.error = pr.error or f"{type(exc).__name__}: {exc}"
            pr.seconds = time.perf_counter() - t0
            pr.finished = True

    def _setup_probes(self, eng) -> None:
        self.collector, self.profiles = None, None
        dec = self.spec.decor or {}
        probes = dec.get("probes") or []
        if probes and (dec.get("graphs") or dec.get("markers")):
            from ..probes import ProbeCollector, ProbeSampler
            sampler = ProbeSampler.build(probes, eng.canvas_w, eng.canvas_h, eng.rim, dec.get("seed", 0))
            self.collector = ProbeCollector(sampler)
            eng.set_probes(sampler)

    def _collect(self, eng) -> None:
        if self.collector is not None:
            steps = int(eng.time)
            self.collector.collect(eng, steps)
            self.profiles = self.collector.collect_profiles(eng, steps)

    def _decorate(self, rgb: np.ndarray, step: int) -> np.ndarray:
        dec = self.spec.decor
        if not dec:
            return rgb
        from ..probe_graph import decorate
        traces = self.collector.traces if self.collector is not None else []
        return decorate(rgb, probes=dec.get("probes") or [], numbers=dec.get("numbers") or [], traces=traces,
                        profiles=self.profiles, step=step, total_steps=dec.get("total_steps", 0),
                        graphs=bool(dec.get("graphs")) and self.collector is not None,
                        probe_markers=bool(dec.get("markers")), sources=dec.get("sources") or [])

    @staticmethod
    def _colours(eng, layer: ExportLayer) -> np.ndarray:
        from ..engine import wavelength
        return group_colours([wavelength(lv, layer.frequency_scale) for lv in eng.levels], layer.nm_per_px)

    def _write(self, eng, sinks) -> None:
        field_plane = intensity_plane = None
        for layer, sink in zip(self.spec.layers, sinks):
            if layer.intensity:
                intensity_plane = eng.intensity() if intensity_plane is None else intensity_plane
                values = intensity_plane
            else:
                field_plane = eng.field() if field_plane is None else field_plane
                values = field_plane
            if sink.wants_values:
                if layer.intensity and self.spec.fmt == ExportFormat.EXR_ROOT:
                    values = np.sqrt(np.maximum(values, 0.0))
                sink.add(values.astype(np.float16))
                continue
            if layer.intensity and layer.true_colour and hasattr(eng, "intensities"):
                rgb = to_rgb8(colour_mix(np.stack(eng.intensities()), self._colours(eng, layer), layer.brightness,
                                         layer.black))
            else:
                rgb = to_rgb8(colorize(values, layer.lut, layer.brightness, layer.intensity))
            sink.add(self._decorate(blend_edges(rgb, self.edges, layer.edge_gain), int(eng.time)))


__all__ = ["DATA", "ExportJob", "ExportLayer", "ExportProgress", "ExportSpec", "glowing_edges", "output_paths"]
