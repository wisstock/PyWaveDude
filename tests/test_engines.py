# SPDX-License-Identifier: GPL-3.0-or-later
"""The GPU engine computes what the CPU engine computes, for both stencils and fused blocks of any length."""

import numpy as np
import pytest

from pywavedude.engine import (EngineOptions, Emitter, PointSource, PulseShape, SceneData, SourceMode,
                               available_backends, create_engine)


@pytest.mark.skipif("mlx" not in available_backends(), reason="no Metal GPU")
def test_cpu_matches_gpu():
    rgb = np.zeros((120, 200, 3), np.uint8)
    rgb[..., 0] = 100
    rgb[40:80, 100:130, 0] = 150          # glass
    rgb[10:12, :, 0] = 0                  # mirror
    rgb[60, 20, 1] = 60                   # image source
    rgb[90:110, 150:170, 2] = 40          # absorber
    sc = SceneData.from_rgb(rgb, damping_edge=-16)
    for fourth in (False, True):
        em = lambda: [Emitter.from_point(PointSource(30, 30, level=80, mode=SourceMode.PULSED, pulse_width=3,  # noqa
                                                     off_width=4, detach_after=150)),
                      Emitter.from_point(PointSource(50, 90, level=120, pulse_shape=PulseShape.SQUARE,
                                                     mode=SourceMode.PULSED))]
        a = create_engine(sc, EngineOptions(fourth_order=fourth), em(), "cpu")
        b = create_engine(sc, EngineOptions(fourth_order=fourth), em(), "mlx")
        for n in (7, 5, 1, 3, 2, 13, 200):            # fused launches and their remainders
            a.step(n)
            b.step(n)
        fa, ia = a.field(), a.intensity()
        assert np.abs(fa).max() > 1e-3
        assert np.abs(fa - b.field()).max() < 1e-4 * np.abs(fa).max() + 1e-6
        assert np.abs(ia - b.intensity()).max() < 1e-4 * ia.max() + 1e-9
