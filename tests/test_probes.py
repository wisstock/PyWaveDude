# SPDX-License-Identifier: GPL-3.0-or-later
"""Probe sampling during a run: points, lines, integration, detectors, CSV."""

import numpy as np

from pywavedude.engine import EngineOptions, Emitter, PointSource, SceneData, create_engine
from pywavedude.probes import (Probe, ProbeDetector, ProbeKind, ProbeLayer, ProbeMeasurement, ProbeSampler,
                               line_pixels, write_csv)
from pywavedude.runner import SimulationRun

W, H = 160, 100
PROBES = [Probe("field", ax=100, ay=50),
          Probe("line", kind=ProbeKind.LINE, layer=ProbeLayer.INTENSITY, ax=120, ay=10, bx=120, by=90),
          Probe("lineint", kind=ProbeKind.LINE, layer=ProbeLayer.INTENSITY,
                measurement=ProbeMeasurement.INTEGRATED_VALUE, ax=130, ay=10, bx=130, by=90),
          Probe("count", detector=ProbeDetector.COUNTING, ax=140, ay=10, bx=140, by=90, sensitivity=50)]


def _run(steps):
    scene = SceneData.empty(W, H, damping_edge=-8)
    eng = create_engine(scene, EngineOptions(fourth_order=False), [Emitter.from_point(PointSource(40, 50))], "cpu")
    run = SimulationRun(eng, steps, 10, sampler=ProbeSampler.build(PROBES, W, H, scene.rim, seed=7))
    run._loop()
    return run, eng


def test_points_lines_and_detectors():
    run, eng = _run(300)
    tr = run.collector.traces
    v = tr[0].values()
    assert len(v) == 300 and abs(v[-1] - eng.field()[50, 100]) < 1e-7          # every step
    assert len(tr[1].profiles) == 31 and np.allclose(tr[1].profiles[-1], eng.intensity()[10:91, 120], rtol=1e-6)
    assert tr[2].profiles[-1].sum() > tr[1].profiles[-1].sum()                 # integral over the steps
    counts = tr[3].values()
    assert counts.sum() > 0 and np.all(counts >= 0)


def test_line_pixels_and_csv(tmp_path):
    assert line_pixels(0, 0, 3, 1) == [(0, 0), (1, 0), (2, 1), (3, 1)]
    run, _ = _run(50)
    assert write_csv(tmp_path / "p.csv", PROBES[0], run.collector.traces[0]) == 50
    assert (tmp_path / "p.csv").read_text().splitlines()[0] == "step;amplitude"
