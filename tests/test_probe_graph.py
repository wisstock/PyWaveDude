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


def test_exported_frame_has_colour_curves_and_labels():
    from pywavedude.probe_graph import decorate
    from pywavedude.probes import ProbeBoxSize, ProbeTrace
    v = np.linspace(0, 1, 200, dtype=np.float32)
    trace = ProbeTrace(series=[v], steps_sampled=200, group_series=[[v * 0.3], [v * 0.7]])
    p = Probe(name="P", ax=10, ay=10, box_size=ProbeBoxSize.CUSTOM, box_x=0, box_y=0, box_width=300, box_height=150)
    blank = np.zeros((200, 400, 3), np.uint8)
    out = decorate(blank, probes=[p], numbers=[1], traces=[trace], step=200, total_steps=200, probe_markers=False,
                   labels=[(320, 170, "Lens 1")], group_colours=[(255, 0, 0, 255), (0, 0, 255, 255)])
    box = out[:150, :300].reshape(-1, 3).astype(int)
    assert ((box[:, 0] > 200) & (box[:, 1] < 60)).any() and ((box[:, 2] > 200) & (box[:, 0] < 60)).any()
    assert out[170:200, 320:400].max() == 255                                # the label's white text
