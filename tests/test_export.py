# SPDX-License-Identifier: GPL-3.0-or-later
"""Export: every format writes its frames; parallel kernels from two threads do not abort."""

import threading

import numpy as np

from pywavedude.engine import EngineOptions, Emitter, PointSource, SceneData, create_engine
from pywavedude.export import EXTENSIONS, ExportFormat, ExportJob, ExportLayer, ExportSpec, ffmpeg_path
from pywavedude.render import builtin_schemes, colorize

W, H = 97, 61                       # odd sizes on purpose


def _make():
    return create_engine(SceneData.empty(W, H, damping_edge=-8), EngineOptions(fourth_order=False),
                         [Emitter.from_point(PointSource(30, 30))], "cpu")


def test_every_format(tmp_path):
    lut = builtin_schemes()["Default field"].lut()
    for fmt in ExportFormat:
        if fmt in (ExportFormat.MP4, ExportFormat.MPEG1) and ffmpeg_path() is None:
            continue
        spec = ExportSpec(fmt, tmp_path / ("o" + EXTENSIONS[fmt]), [ExportLayer(False, lut, 20.0)], 10, 100, 10,
                          fps=30)
        job = ExportJob(_make, spec)
        job.start()
        job.join()
        assert job.progress.error == "" and job.progress.frames_done == 10, fmt


def test_parallel_kernels_from_two_threads():
    """Numba's workqueue layer aborts on concurrent parallel launches; the shared lock must prevent it."""
    lut = builtin_schemes()["Default field"].lut()
    t = threading.Thread(target=lambda: _make().step(300))
    t.start()
    for _ in range(200):
        colorize(np.random.rand(H, W).astype(np.float32), lut, 1.0, False)
    t.join()
