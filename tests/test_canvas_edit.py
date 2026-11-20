# SPDX-License-Identifier: GPL-3.0-or-later
"""Picking under the mouse, centres and grid snapping."""

from pywavedude.canvas_edit import part_center, pick, snap_point
from pywavedude.engine import PointSource
from pywavedude.parts import model as pm
from pywavedude.probes import Probe


def test_pick_and_snap():
    assert snap_point(103, 248, 50, 5) == (100, 250) and snap_point(103, 230, 50, 5) == (100, 230)
    rect = pm.RectanglePart(x=200, y=100)
    assert part_center(rect) == (200, 100)
    src, probe = PointSource(205, 100), Probe(ax=400, ay=400)
    assert pick(206, 101, [src], [probe], [rect], 4) == ("source", 0)   # a marker wins over the part it sits in
    assert pick(195, 100, [src], [probe], [rect], 4) == ("part", 0)
    assert pick(402, 399, [src], [probe], [rect], 4) == ("probe", 0)
    assert pick(900, 900, [src], [probe], [rect], 4) is None
