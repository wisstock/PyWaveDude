# SPDX-License-Identifier: GPL-3.0-or-later
# PyWaveDude: Python port of WaveDude Next (C) Alexander Graef, after Jeroen Vleggaar (Huygens Optics) and 0x23.
"""Scratch frames: "frame k of the project as it is now", computed from step 0 on a worker thread while you edit.

One request computes at a time, to its end; the newest request made meanwhile follows it, the ones between are
dropped. A result carries the request's ``key`` so the app shows it only for the position it was asked for."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class ScratchResult:
    key: object
    step: int
    field: np.ndarray
    intensity: np.ndarray | None
    groups: np.ndarray | None = None
    levels: tuple | None = None
    profiles: list | None = None
    error: str = ""
    seconds: float = 0.0
    edges: object = None             # the scene's glowing edges, for the outline


class ScratchRunner:
    def __init__(self):
        self._lock = threading.Condition()
        self._next: tuple | None = None
        self._busy = False
        self.result: ScratchResult | None = None
        self.version = 0                 # counts results
        self._thread = threading.Thread(target=self._loop, daemon=True, name="scratch")
        self._stop = False
        self._thread.start()

    def request(self, key, make_engine: Callable[[], object], steps: int, preview_step: int) -> None:
        """Compute ``steps`` steps on ``make_engine()`` (called on the worker) and keep the preview of the end."""
        with self._lock:
            self._next = (key, make_engine, max(0, steps), max(1, preview_step))
            self._lock.notify()

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy or self._next is not None

    def close(self) -> None:
        with self._lock:
            self._stop = True
            self._lock.notify()

    def _loop(self) -> None:
        import time
        while True:
            with self._lock:
                while self._next is None and not self._stop:
                    self._lock.wait()
                if self._stop:
                    return
                key, make, steps, ps = self._next
                self._next = None
                self._busy = True
            t0 = time.perf_counter()
            try:
                eng = make()
                done = 0
                while done < steps:              # finished even when a newer request waits: it follows
                    n = min(256, steps - done)
                    eng.step(n)
                    done += n
                f, i, *rest = eng.preview(ps)
                res = ScratchResult(key, int(eng.time), f, i, rest[0] if rest else None,
                                    tuple(getattr(eng, "levels", ())) or None,
                                    edges=getattr(eng, "scene_edges", None))
            except Exception as exc:              # shown in the status bar
                res = ScratchResult(key, 0, np.zeros((1, 1), np.float16), None, error=f"{type(exc).__name__}: {exc}")
            res.seconds = time.perf_counter() - t0
            with self._lock:
                self.result = res
                self.version += 1
                self._busy = False
