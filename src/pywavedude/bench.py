# SPDX-License-Identifier: GPL-3.0-or-later
"""Engine benchmark: ``python -m pywavedude.bench [width height steps]``."""

import sys
import time

from .engine import EngineOptions, Emitter, PointSource, SceneData, available_backends, create_engine


def main() -> None:
    w, h, steps = (int(a) for a in sys.argv[1:4]) if len(sys.argv) >= 4 else (1920, 1080, 300)
    scene = SceneData.empty(w, h)
    em = [Emitter.from_point(PointSource(w // 4, h // 2))]
    for fourth in (False, True):
        for backend in available_backends():
            eng = create_engine(scene, EngineOptions(fourth_order=fourth), em, backend)
            eng.step(10)
            eng.sync()
            t0 = time.perf_counter()
            eng.step(steps)
            eng.sync()
            dt = time.perf_counter() - t0
            print(f"{eng.description:60s} {steps / dt:8.1f} steps/s  {w * h * steps / dt / 1e9:6.2f} Gcell/s")


if __name__ == "__main__":
    main()
