# SPDX-License-Identifier: GPL-3.0-or-later
# PyWaveDude: Python port of WaveDude Next (C) Alexander Graef, after Jeroen Vleggaar (Huygens Optics) and 0x23.
"""Wave engines: ``create_engine`` picks the Metal GPU (MLX) on Apple Silicon, else the multithreaded CPU."""

from __future__ import annotations

from .base import EngineOptions, WaveEngine
from .scene import SceneData
from .sources import Emitter, PointSource, PulseShape, SourceMode, wavelength

BACKENDS = ("auto", "cpu", "mlx")


def available_backends() -> list[str]:
    from . import gpu_mlx
    return ["cpu", "mlx"] if gpu_mlx.available() else ["cpu"]


def create_engine(scene: SceneData, options: EngineOptions | None = None, emitters=None,
                  backend: str = "auto") -> WaveEngine:
    options = options or EngineOptions()
    if backend == "auto":
        backend = "mlx" if "mlx" in available_backends() else "cpu"
    if backend == "mlx":
        from .gpu_mlx import MlxEngine
        return MlxEngine(scene, options, emitters)
    from .cpu import CpuEngine
    return CpuEngine(scene, options, emitters)


__all__ = ["BACKENDS", "EngineOptions", "WaveEngine", "SceneData", "Emitter", "PointSource", "PulseShape",
           "SourceMode", "wavelength", "available_backends", "create_engine"]
