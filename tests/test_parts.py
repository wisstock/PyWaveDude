# SPDX-License-Identifier: GPL-3.0-or-later
"""Parts: painted index, coating, mirror and absorption; every kind paints; designs and optics."""

import math

import numpy as np
import pytest

from pywavedude.parts import PART_TYPES, paint_parts
from pywavedude.parts import model as m


def test_rectangle_coating_mirror_absorption():
    rgb = paint_parts(None, [m.RectanglePart(x=200, y=150, length=100, width=60, coating=5),
                             m.EllipsePart(x=60, y=60, length=40, width=40, mirror=True),
                             m.RectanglePart(x=330, y=150, length=100, width=100, absorption=200,
                                             absorption_ramp=20)], 400, 300)
    red = rgb[..., 0]
    assert red[150, 200] == 150 and red[150, 152] == 122 and red[150, 140] == 100   # body, √n band, outside
    assert red[60, 60] == 0                                                           # mirror
    blue = rgb[150, 280:381, 2].astype(int)
    assert blue[0] < blue[10] < blue[25] == 200                                       # absorption ramp


def test_every_part_type_paints():
    for stem, cls in PART_TYPES.items():
        part = cls(x=300, y=300)
        if stem == "Zone plate":
            part.design_frequency, part.diameter = 200.0, 400
        rgb = paint_parts(None, [part], 600, 600)
        assert ((rgb[..., 0] != 100) | (rgb[..., 2] != 0)).any(), stem


def test_gate_design_matches_original_example():
    d = m.XorGatePart().design                 # Example 18 drives its gates with sources 60 px apart
    assert d.port_spacing == 60 and d.core_width == 6 and d.phase_shift == 90.0


def test_phase_step_plate_flips_one_half():
    """Two zones, 180°: behind the plate the upper and lower halves of a plane wave oscillate in antiphase."""
    from pywavedude import project as pj
    from pywavedude.engine import PointSource

    def corr(with_plate):
        p = pj.Project(seed=1)
        p.scene.width, p.scene.height = 200, 160
        p.sources.append(PointSource(20, 80, name="A", level=50, composite=True, size_x=1, size_y=64, spacing_y=2))
        if with_plate:
            p.parts.append(m.PhaseStepPlatePart(x=70, y=80, aperture=150, refractive_index=2.0))
        eng = pj.build_engine(p, "cpu")
        eng.step(300)
        a = []
        for _ in range(40):
            eng.step(1)
            f = eng.field()
            a.append((f[60, 100], f[100, 100]))
        return np.corrcoef(np.array(a).T)[0, 1]

    assert corr(False) > 0.95 and corr(True) < -0.9


def test_lens_focal_length():
    thin = m.LensPart(r1=1000.0, r2=-1000.0, thickness=3, diameter=100)   # sags fit: radii as given
    assert thin.focal_lengths()[0] == pytest.approx(1000.0, rel=1e-3)          # R / (2 (n - 1))
    assert thin.focal_lengths(1.5)[0] == math.inf                                   # index matched: no power
    assert m.LensPart(mirror=True).focal_lengths() is None


def test_thin_film_quarter_wave_stack():
    from pywavedude.parts import thinfilm as tf
    lam, nh, nl = 30.0, 2.3, 1.45
    stack = [(tf.quarter_wave(nh, lam, 1.0, 0.0), nh), (tf.quarter_wave(nl, lam, 1.0, 0.0), nl)] * 8
    assert stack[0][0] == pytest.approx(lam / 4 / nh)
    assert tf.reflectance([], lam, 1.0, 1.5, 0.0) == pytest.approx(0.04)                 # bare glass, (n-1)²/(n+1)²
    assert tf.reflectance(stack, lam, 1.0, 1.5, 0.0) > 0.99                              # in the band
    assert tf.reflectance(stack[:2], lam / 2, 1.0, 1.5, 0.0) == pytest.approx(0.04, abs=1e-6)  # half waves: absent
    assert tf.quarter_wave(nh, lam, 1.0, 45.0) > stack[0][0]                             # oblique: thicker


@pytest.mark.parametrize("mode", list(m.DichroicMode))
def test_dichroic_mirror_band_at_the_cutoff(mode):
    d = m.DichroicMirrorPart(mode=mode, design_frequency=50.0)
    a, b = d.band
    edge = {m.DichroicMode.LONGPASS: b, m.DichroicMode.SHORTPASS: a}.get(mode, d.centre)
    assert edge == pytest.approx(d.cutoff)
    assert d.reflectance(d.centre) > 0.99
    passband = {m.DichroicMode.LONGPASS: np.linspace(b * 1.1, b * 1.6, 30),
                m.DichroicMode.SHORTPASS: np.linspace(a * 0.6, a * 0.9, 30)}.get(mode)
    if passband is not None:
        assert np.mean([d.reflectance(lam) for lam in passband]) < 0.15
    assert d.stack_thickness == pytest.approx(sum(l.thickness for l in d.layers))
