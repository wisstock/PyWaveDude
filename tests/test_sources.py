# SPDX-License-Identifier: GPL-3.0-or-later
"""Composite sources and the random numbers they share with the original (bit for bit)."""

from pywavedude.engine import PointSource
from pywavedude.rng import Pcg32, stable_hash


def test_grid_centred_and_rotated():
    s = PointSource(100, 50, name="S", composite=True, size_x=3, size_y=2, spacing_x=20, spacing_y=10)
    assert [(x, y) for x, y, *_ in s.elements()] == [(80, 45), (100, 45), (120, 45), (80, 55), (100, 55), (120, 55)]
    s.array_angle = 90.0                         # counter-clockwise on the screen: +x goes up
    pts = [(x, y) for x, y, *_ in s.elements()]
    assert pts[2] == (95, 30) and pts[0] == (95, 70)


def test_random_numbers_and_seeded_jitter():
    r = Pcg32(42, 54)                            # the reference pcg32_srandom_r's first outputs
    assert [r.next_uint32() for _ in range(3)] == [0xa15c02b7, 0x7b47f409, 0xba1d3330]
    assert stable_hash("") == 2166136261 and stable_hash("a") == 0xe40c292c
    kw = dict(composite=True, size_x=5, size_y=5, random_phase=30, random_position=3, random_frequency=5)
    a = PointSource(100, 100, name="A", **kw)
    assert a.elements(1) == a.elements(1) != a.elements(2)
    assert a.elements(1) != PointSource(100, 100, name="B", **kw).elements(1)
