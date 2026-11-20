# SPDX-License-Identifier: GPL-3.0-or-later
"""The GUI's model helpers without a window: undo history, handles, objects as text, field metadata."""

import dataclasses

import pytest

from pywavedude import handles as hd
from pywavedude import project as pj
from pywavedude.engine import PointSource
from pywavedude.history import UndoHistory
from pywavedude.parts import PART_TYPES
from pywavedude.parts import model as m
from pywavedude.probes import Probe
from pywavedude.ui_meta import field_meta, kind_description


def _project():
    p = pj.Project(seed=5)
    p.sources.append(PointSource(100, 100, name="S"))
    p.parts.append(m.LensPart(name="L", x=300, y=200))
    return p


def test_undo_redo_steps():
    p = _project()
    h = UndoHistory(p)
    assert not h.can_undo and not h.dirty
    p.parts[0].x = 350
    h.edit_made()
    assert h.end_step(p) and h.dirty
    h.edit_made()
    assert not h.end_step(p)                       # nothing changed: no step
    q = h.undo(p)
    assert q.parts[0].x == 300 and h.can_redo
    r = h.redo(q)
    assert r.parts[0].x == 350
    h.mark_saved()
    assert not h.dirty


def test_handles():
    lens = m.LensPart(x=500, y=300, r1=300.0, r2=-300.0, thickness=40, diameter=200)
    assert {"rotate", "size", "r1", "r2"} <= {k for k, *_ in hd.part_handles(lens, 20)}
    hd.drag_part_handle(lens, "size", 500 + 30, 300 + 150)
    assert (lens.thickness, lens.diameter) == (60, 300)
    hd.drag_part_handle(lens, "rotate", 500, 200)          # straight up on the screen
    assert lens.angle == pytest.approx(90.0)
    pr = Probe(ax=10, ay=10, box_x=10, box_y=10)
    hd.drag_probe_handle(pr, "a", 20, 30, 800, 600)
    assert (pr.box_x, pr.box_y) == (20, 30)                # a box left on point A follows it


def test_objects_as_text_round_trip():
    p = _project()
    p.probes.append(Probe(name="P", ax=5, ay=6))
    text = pj.items_to_xml([p.sources[0], p.parts[0], p.probes[0]])
    items = pj.items_from_xml(text)
    assert items["sources"][0] == p.sources[0] and items["probes"][0].name == "P"
    assert items["parts"][0].x == 300 and type(items["parts"][0]) is m.LensPart
    with pytest.raises(Exception):
        pj.items_from_xml("plain words")


def test_every_field_has_a_label():
    objs = [PointSource(0, 0), Probe(), pj.SceneSettings(), pj.SimSettings(), pj.DisplaySettings(),
            *(cls() for cls in PART_TYPES.values())]
    for o in objs:
        for f in dataclasses.fields(o):
            assert field_meta(o, f.name).label
    assert "lens" in kind_description(m.LensPart()).lower()
    assert field_meta(m.LensPart(), "r1").hint                    # the original's description
    assert field_meta(PointSource(0, 0), "level").label == "Frequency"
    from pywavedude.ui_props import computed_fields                   # wavelength in px beside every frequency
    src, sim = PointSource(0, 0), pj.SimSettings()
    computed_fields(src, 2.0)[0].set(src, 25.0)
    computed_fields(sim, 2.0)[0].set(sim, 20.0)
    assert computed_fields(src, 2.0)[0].get(src) == pytest.approx(25.0, rel=1e-4)
    assert computed_fields(sim, 2.0)[0].get(sim) == pytest.approx(20.0, rel=1e-4)
