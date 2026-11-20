# SPDX-License-Identifier: GPL-3.0-or-later
# PyWaveDude: Python port of WaveDude Next (C) Alexander Graef, after Jeroen Vleggaar (Huygens Optics) and 0x23.
"""Background simulation run: steps the engine on a worker thread and keeps the frames for playback."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .engine import WaveEngine


@dataclass
class Frame:
    step: int
    field: np.ndarray                 # float16, preview resolution
    intensity: np.ndarray | None
    profiles: list | None = None      # per probe: this frame's line profile (or None)
    groups: np.ndarray | None = None  # (wavelengths, h, w) float16 intensity per wavelength (split runs)
    levels: tuple | None = None       # the groups' frequency levels

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in (self.field, self.intensity, self.groups) if a is not None)


@dataclass
class RunStats:
    steps_done: int = 0
    total_steps: int = 0
    steps_per_second: float = 0.0
    finished: bool = False
    error: str = ""
    frames_dropped: bool = False


class SimulationRun:
    """Owns one engine; ``start`` runs ``total_steps`` steps, capturing a frame every ``every`` steps."""

    def __init__(self, engine: WaveEngine, total_steps: int, every: int, preview_step: int = 1,
                 memory_budget: int = 2 << 30, sampler=None):
        from .probes import ProbeCollector
        self.engine = engine
        self.sampler = sampler
        self.collector = ProbeCollector(sampler) if sampler is not None and sampler.count else None
        if self.collector is not None:
            engine.set_probes(sampler)
        self.every = max(1, every)
        self.preview_step = max(1, preview_step)
        self.stats = RunStats(total_steps=total_steps)
        self.frames: list[Frame] = []
        self.memory_budget = memory_budget
        self._bytes = 0
        self._pause = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="simulation")

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._pause.clear()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def toggle_pause(self) -> None:
        if self._pause.is_set():
            self._pause.clear()
        else:
            self._pause.set()

    @property
    def paused(self) -> bool:
        return self._pause.is_set()

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def frame(self, index: int = -1) -> Frame | None:
        with self._lock:
            if not self.frames:
                return None
            return self.frames[max(-len(self.frames), min(index, len(self.frames) - 1))]

    def frame_count(self) -> int:
        with self._lock:
            return len(self.frames)

    # ------------------------------------------------------------------ worker
    def _capture_start(self):
        """Probe data now; the pictures are fetched by ``_capture_finish`` after the next steps were queued."""
        finish = self.engine.preview_start(self.preview_step)
        prof = None
        if self.collector is not None:
            steps = int(self.engine.time)
            self.collector.collect(self.engine, steps)
            prof = self.collector.collect_profiles(self.engine, steps)
        return int(self.engine.time), finish, prof

    def _capture(self) -> None:
        self._capture_finish(self._capture_start())

    def _capture_finish(self, pending) -> None:
        step, finish, prof = pending
        f, i, *rest = finish()
        groups = rest[0] if rest else None
        fr = Frame(step, f, i, prof, groups, tuple(getattr(self.engine, "levels", ())) or None)
        size = fr.nbytes
        with self._lock:
            if self._bytes + size > self.memory_budget:
                # keep the latest frame for the live view, drop the history beyond the budget
                self.stats.frames_dropped = True
                if self.frames:
                    self._bytes -= self.frames[-1].nbytes
                    self.frames[-1] = fr
                else:
                    self.frames.append(fr)
            else:
                self.frames.append(fr)
            self._bytes += size

    def _loop(self) -> None:
        eng, st = self.engine, self.stats
        pending = None
        try:
            pending = self._capture_start()
            t0, n0 = time.perf_counter(), 0
            while st.steps_done < st.total_steps and not self._stop.is_set():
                if self._pause.is_set():
                    if pending is not None:
                        self._capture_finish(pending)
                        pending = None
                    time.sleep(0.02)
                    t0, n0 = time.perf_counter(), st.steps_done
                    continue
                n = min(self.every, st.total_steps - st.steps_done)
                eng.step(n)
                st.steps_done += n
                if pending is not None:
                    self._capture_finish(pending)
                pending = self._capture_start()
                dt = time.perf_counter() - t0
                if dt > 0.25:
                    st.steps_per_second = (st.steps_done - n0) / dt
                    t0, n0 = time.perf_counter(), st.steps_done
            if pending is not None:
                self._capture_finish(pending)
        except Exception as exc:  # surfaced in the status bar
            st.error = f"{type(exc).__name__}: {exc}"
        finally:
            st.finished = True
