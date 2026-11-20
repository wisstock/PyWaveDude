# SPDX-License-Identifier: GPL-3.0-or-later
"""Separate wavelengths: the summed field equals the full run; true colour mixing."""

import numpy as np

from pywavedude import project as pj
from pywavedude.engine import PointSource
from pywavedude.render import colour_mix, group_colours, wavelength_nm


def test_split_field_is_the_sum():
    def project(split):
        p = pj.Project(seed=3)
        p.scene.width, p.scene.height = 96, 64
        p.sim.split_wavelengths = split
        p.sources += [PointSource(30, 30, name="A", level=60), PointSource(60, 34, name="B", level=90),
                      PointSource(40, 50, name="C", level=60)]
        return p
    one, split = pj.build_engine(project(False), "cpu"), pj.build_engine(project(True), "cpu")
    assert split.levels == [60.0, 90.0]
    one.step(120)
    split.step(120)
    np.testing.assert_allclose(split.field(), one.field(), atol=1e-5)    # the medium is linear
    assert split.preview(2)[2].shape == (2, 32, 48)
    from pywavedude.probes import Probe, ProbeLayer, ProbeMeasurement, ProbeSampler
    from pywavedude.runner import SimulationRun
    eng = pj.build_engine(project(True), "cpu")
    probe = Probe("I", layer=ProbeLayer.INTENSITY, measurement=ProbeMeasurement.VALUE, ax=50, ay=30)
    run = SimulationRun(eng, 60, 10, sampler=ProbeSampler.build([probe], 96, 64, eng.rim, 1))
    run._loop()
    tr = run.collector.traces[0]                                          # a curve per colour, adding up to the total
    np.testing.assert_allclose(tr.group_values(0) + tr.group_values(1), tr.values(), rtol=1e-6)


def test_colours_and_mixing():
    assert wavelength_nm([20.0, 40.0]) == [450.0, 650.0] and wavelength_nm([20.0], 25.0) == [500.0]
    cols = group_colours([10.0, 30.0])                                    # short blue, long red
    g = np.zeros((2, 1, 3), np.float32)
    g[0, 0, 0] = g[1, 0, 1] = g[0, 0, 2] = g[1, 0, 2] = 1.0
    out = colour_mix(g, cols, 1.0)
    assert out[0, 0, 2] > 0.9 > out[0, 0, 0] and out[0, 1, 0] > 0.9 > out[0, 1, 2]
    assert out[0, 2, 0] > 0.9 and out[0, 2, 2] > 0.9                      # both add up
