# SPDX-License-Identifier: GPL-3.0-or-later
"""Probe graph ticks, ranges and layout."""

import numpy as np

from pywavedude.probe_graph import GraphData, format_tick, layout, log_ticks, nice_ticks, ranges
from pywavedude.probes import Probe, ProbeScaleMode


def test_ticks_and_labels():
    assert np.allclose(nice_ticks(0, 1, 5), [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    assert log_ticks(1e-6, 1, 4) == [1e-06, 0.0001, 0.01, 1.0]
    assert format_tick(0) == "0" and format_tick(0.0472) == "0.0472" and format_tick(1234) == "1.23E+03"


def test_ranges_and_layout():
    v = np.sin(np.arange(1000) / 10).astype(np.float32) * 0.5
    xmin, xmax, ymin, ymax = ranges(Probe(scale_mode=ProbeScaleMode.RUNNING, window=100), GraphData(v, 1000), 5000)
    assert (xmin, xmax) == (900, 1000) and ymax == -ymin                 # running window, symmetric amplitude
    long = np.random.default_rng(0).normal(size=20000).astype(np.float32)
    g = layout(Probe(name="P", scale_mode=ProbeScaleMode.FULL), GraphData(long, 20000), 20000, 400, 200)
    assert g.has_plot and len(g.curve) == 2 * int(g.plot_w)               # min and max per pixel column
    assert g.curve[:, 1].min() >= g.plot_y - 1 and g.curve[:, 1].max() <= g.plot_y + g.plot_h + 1
