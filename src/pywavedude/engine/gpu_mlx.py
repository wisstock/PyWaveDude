# SPDX-License-Identifier: GPL-3.0-or-later
# PyWaveDude: Python port of WaveDude Next (C) Alexander Graef, after Jeroen Vleggaar (Huygens Optics) and 0x23.
"""Apple Silicon GPU engine on MLX (Metal): the CPU engine's scheme in one fused, temporally blocked kernel.

Each threadgroup owns a ``TW × TH`` tile of the padded grid and loads it with a halo of ``reach × sub-steps ×
steps`` cells into threadgroup memory. It then advances that block through ``K`` whole steps (each: the sources,
then the sub-steps, each followed by the Mur open edge) without touching device memory, and writes back only its
own cells. The field, previous field and the two intensity stages are read and written once per launch instead
of once per sub-step, which is what bounds a stencil this simple. The halo cells are computed redundantly and
are only right as far in as the block's shrinking valid region, which still covers the owned tile at the end.

The intensity is kept as its two filter stages; the blended, scaled value the CPU engine stores every step is
formed only when read (frames, probes), saving a write per cell per step.
"""

from __future__ import annotations

import math

import numpy as np

from . import base
from .base import EngineOptions, WaveEngine
from .scene import SceneData
from .sources import Emitter

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - not on Apple Silicon
    mx = None

THREADS = 256
TILE_W, TILE_H = 32, 32
MAX_STEPS_PER_LAUNCH = {False: 4, True: 1}     # by stencil order: (9-point, 4th-order with 3 sub-steps)


def available() -> bool:
    if mx is None:
        return False
    try:
        return mx.metal.is_available()
    except Exception:
        return False


def _f(x: float) -> str:
    return f"{float(np.float32(x))!r}f"


def _kernel_source(halo: int, steps: int, substeps: int, fourth: bool, open_edges: bool, track: bool) -> tuple[str, str]:
    sw, sh = TILE_W + 2 * halo, TILE_H + 2 * halo
    ncell = sw * sh
    maxc = -(-ncell // THREADS)
    reach = 2 if fourth else 1
    header = f"""
#define TW {TILE_W}
#define TH {TILE_H}
#define HALO {halo}
#define SW {sw}
#define SH {sh}
#define NCELL {ncell}
#define MAXC {maxc}
#define NTHREADS {THREADS}
#define KSTEPS {steps}
#define SUBSTEPS {substeps}
#define REACH {reach}
#define USE4TH {int(fourth)}
#define OPEN {int(open_edges)}
#define TRACK {int(track)}
constant float CORNER = {_f(base.KERNEL_CORNER)};
constant float EDGEK = {_f(base.KERNEL_EDGE)};
constant float NEAR = {_f(base.FOURTH_NEAR)};
constant float FAR = {_f(base.FOURTH_FAR)};
constant float CENTER4 = {_f(base.FOURTH_CENTER)};
constant float BLEND = {_f(base.INTENSITY_BLEND)};
constant float EDGE_SPEED = {_f(base.EDGE_SPEED_FACTOR)};
"""
    source = """
    const uint tid = thread_index_in_threadgroup;
    const uint3 tg = threadgroup_position_in_grid;
    const int W = pi[0];
    const int H = pi[1];
    const int GW = W + 2;
    const int GH = H + 2;
    const int tile = int(tg.y) * pi[2] + int(tg.x);
    const int ox = int(tg.x) * TW - HALO;
    const int oy = int(tg.y) * TH - HALO;
    const float mult = pf[0];
    const float alpha = pf[1];
    const float scale = pf[2];
    const int nsrc = pi[3];

    threadgroup float bufA[NCELL];
    threadgroup float bufB[NCELL];
    float rc2[MAXC];
    float rd[MAXC];
#if TRACK
    float r1[MAXC];
    float r2[MAXC];
#endif
    // load the block: field, previous field, c^2 (times the sub-step factor), damping, intensity stages
    for (int c = 0; c < MAXC; c++) {
        int idx = int(tid) + c * NTHREADS;
        if (idx >= NCELL) break;
        int gx = ox + idx % SW;
        int gy = oy + idx / SW;
        bool inside = gx >= 0 && gx < GW && gy >= 0 && gy < GH;
        int g = gy * GW + gx;
        bufA[idx] = inside ? u[g] : 0.0f;
        bufB[idx] = inside ? up[g] : 0.0f;
        rc2[c] = inside ? c2[g] * mult : 0.0f;
        rd[c] = inside ? d[g] : 0.0f;
#if TRACK
        r1[c] = inside ? s1[g] : 0.0f;
        r2[c] = inside ? s2[g] : 0.0f;
#endif
    }
    threadgroup float* cur = bufA;
    threadgroup float* prv = bufB;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k = 0; k < KSTEPS; k++) {
        // the sources pull their cells toward the drive: in order, by one thread (a cell may be listed twice)
        if (tid == 0) {
            for (int j = toff[tile]; j < toff[tile + 1]; j++) {
                float v = vals[k * nsrc + tslot[j]];
                if (!isnan(v)) {
                    int l = tloc[j];
                    cur[l] = cur[l] * 0.9f + v * (1.0f - 0.9f);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int s = 1; s <= SUBSTEPS; s++) {
            for (int c = 0; c < MAXC; c++) {
                int idx = int(tid) + c * NTHREADS;
                if (idx >= NCELL) break;
                int lx = idx % SW;
                int ly = idx / SW;
                if (lx < REACH || lx >= SW - REACH || ly < REACH || ly >= SH - REACH) continue;
                int gx = ox + lx;
                int gy = oy + ly;
                if (gx < 1 || gx > W || gy < 1 || gy > H) continue;
                float cc = cur[idx];
                float lap;
                if (USE4TH && gy >= 2 && gy <= H - 1 && gx >= 2 && gx <= W - 1) {
                    lap = FAR * (cur[idx - 2] + cur[idx + 2] + cur[idx - 2 * SW] + cur[idx + 2 * SW])
                        + NEAR * (cur[idx - 1] + cur[idx + 1] + cur[idx - SW] + cur[idx + SW]) + CENTER4 * cc;
                } else {
                    lap = CORNER * (cur[idx - SW - 1] + cur[idx - SW + 1] + cur[idx + SW - 1] + cur[idx + SW + 1])
                        + EDGEK * (cur[idx - 1] + cur[idx + 1] + cur[idx - SW] + cur[idx + SW]) - cc;
                }
                float n = cc + (cc - prv[idx]) * rd[c] + lap * rc2[c];
                prv[idx] = n;
#if TRACK
                if (s == SUBSTEPS) {
                    float a1 = r1[c] * alpha + n * n * (1.0f - alpha);
                    r2[c] = r2[c] * alpha + a1 * (1.0f - alpha);
                    r1[c] = a1;
                }
#endif
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
#if OPEN
            // Mur's first-order absorbing edge on the ring around the grid, from the cell just inside it
            for (int c = 0; c < MAXC; c++) {
                int idx = int(tid) + c * NTHREADS;
                if (idx >= NCELL) break;
                int lx = idx % SW;
                int ly = idx / SW;
                int gx = ox + lx;
                int gy = oy + ly;
                if (gx < 0 || gx >= GW || gy < 0 || gy >= GH) continue;
                int ix, iy;
                if (gy == 0) { iy = 1; ix = clamp(gx, 1, W); }
                else if (gy == H + 1) { iy = H; ix = clamp(gx, 1, W); }
                else if (gx == 0) { iy = gy; ix = 1; }
                else if (gx == W + 1) { iy = gy; ix = W; }
                else continue;
                int ilx = ix - ox;
                int ily = iy - oy;
                if (ilx < REACH || ilx >= SW - REACH || ily < REACH || ily >= SH - REACH) continue;
                int in = ily * SW + ilx;
                float sp = sqrt(c2[iy * GW + ix] * mult) * EDGE_SPEED;
                float kk = (sp - 1.0f) / (sp + 1.0f);
                prv[idx] = cur[in] + kk * (prv[in] - cur[idx]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
            threadgroup float* t = cur;
            cur = prv;
            prv = t;
        }
    }
    // write back the tile this group owns
    for (int c = 0; c < MAXC; c++) {
        int idx = int(tid) + c * NTHREADS;
        if (idx >= NCELL) break;
        int lx = idx % SW;
        int ly = idx / SW;
        if (lx < HALO || lx >= HALO + TW || ly < HALO || ly >= HALO + TH) continue;
        int gx = ox + lx;
        int gy = oy + ly;
        if (gx >= GW || gy >= GH) continue;
        int g = gy * GW + gx;
        u_out[g] = cur[idx];
        up_out[g] = prv[idx];
#if TRACK
        s1_out[g] = r1[c];
        s2_out[g] = r2[c];
#endif
    }
"""
    return header, source


class MlxEngine(WaveEngine):
    name = "GPU (MLX/Metal)"

    def __init__(self, scene: SceneData, options: EngineOptions, emitters: list[Emitter] | None = None):
        if not available():
            raise RuntimeError("MLX with Metal is not available on this machine")
        self.track = options.track_intensity
        super().__init__(scene, options, emitters)
        self.c2 = mx.array(self.c2_host.reshape(-1))
        self.d = mx.array(self.d_host.reshape(-1))
        self.tiles_x = -(-(self.w + 2) // TILE_W)
        self.tiles_y = -(-(self.h + 2) // TILE_H)
        self._kernels: dict = {}
        self._lists: dict = {}
        self._clear()
        self._steps_since_eval = 0

    # ------------------------------------------------------------ set-up
    def _upload_sources(self) -> None:
        self._lists = {}                     # tile lists depend on the halo; rebuilt lazily
        self._pf = mx.array([float(self.c2_mult), float(self.alpha), float(self.options.intensity_scale)], mx.float32)

    def _halo(self, steps: int) -> int:
        return (2 if self.options.fourth_order else 1) * self.substeps * steps

    def _kernel(self, steps: int, track: bool):
        key = (steps, track)
        if key not in self._kernels:
            header, source = _kernel_source(self._halo(steps), steps, self.substeps, self.options.fourth_order,
                                            self.open_edges, track)
            ins = ["u", "up", "c2", "d"] + (["s1", "s2"] if track else []) + ["toff", "tloc", "tslot", "vals", "pi", "pf"]
            outs = ["u_out", "up_out"] + (["s1_out", "s2_out"] if track else [])
            self._kernels[key] = mx.fast.metal_kernel(name=f"wave_block_{steps}_{int(track)}_{self.substeps}"
                                                      f"_{int(self.options.fourth_order)}_{int(self.open_edges)}",
                                                      input_names=ins, output_names=outs, header=header, source=source,
                                                      ensure_row_contiguous=True)
        return self._kernels[key]

    def _tile_lists(self, steps: int):
        """Per tile: the source cells within the tile and its halo, as (local index, value slot), in driver order."""
        if steps in self._lists:
            return self._lists[steps]
        halo = self._halo(steps)
        sw = TILE_W + 2 * halo
        gw = self.w + 2
        idx = self.driver.index
        gx, gy = idx % gw, idx // gw
        per_tile: dict[int, list[tuple[int, int]]] = {}
        for slot in range(len(idx)):
            x, y = int(gx[slot]), int(gy[slot])
            for ty in range(max(0, (y - halo) // TILE_H - 1), min(self.tiles_y, (y + halo) // TILE_H + 2)):
                for tx in range(max(0, (x - halo) // TILE_W - 1), min(self.tiles_x, (x + halo) // TILE_W + 2)):
                    lx, ly = x - (tx * TILE_W - halo), y - (ty * TILE_H - halo)
                    if 0 <= lx < sw and 0 <= ly < TILE_H + 2 * halo:
                        per_tile.setdefault(ty * self.tiles_x + tx, []).append((ly * sw + lx, slot))
        n = self.tiles_x * self.tiles_y
        off = np.zeros(n + 1, np.int32)
        loc, slots = [], []
        for t in range(n):
            entries = per_tile.get(t, [])
            off[t + 1] = off[t] + len(entries)
            loc += [e[0] for e in entries]
            slots += [e[1] for e in entries]
        lists = (mx.array(off), mx.array(np.array(loc or [0], np.int32)), mx.array(np.array(slots or [0], np.int32)))
        self._lists[steps] = lists
        return lists

    def _clear(self) -> None:
        n = (self.h + 2) * (self.w + 2)
        self.u = mx.zeros((n,), mx.float32)
        self.up = mx.zeros((n,), mx.float32)
        m = n if self.track else 1
        self.s1, self.s2 = mx.zeros((m,), mx.float32), mx.zeros((m,), mx.float32)
        mx.eval(self.u, self.up, self.s1, self.s2)

    # ------------------------------------------------------------ stepping
    def step(self, n: int = 1) -> None:
        kmax = 1 if self._sampler is not None else MAX_STEPS_PER_LAUNCH[bool(self.options.fourth_order)]
        done = 0
        while done < n:
            k = min(kmax, n - done)
            self._launch(k)
            done += k
            if self._sampler is not None:
                self._sample_probes()

    def _advance(self, source_values) -> None:  # pragma: no cover - step() is overridden
        raise NotImplementedError

    def _launch(self, k: int) -> None:
        ns = len(self.driver)
        if ns:
            vals = np.stack([self.driver.compute(self.time + j).copy() for j in range(k)]).astype(np.float32)
        else:
            vals = np.zeros((k, 1), np.float32)
        toff, tloc, tslot = self._tile_lists(k)
        pi = mx.array([self.w, self.h, self.tiles_x, max(ns, 1)], mx.int32)
        n = self.u.shape[0]
        ins = [self.u, self.up, self.c2, self.d] + ([self.s1, self.s2] if self.track else []) + \
              [toff, tloc, tslot, mx.array(vals.reshape(-1)), pi, self._pf]
        outs = self._kernel(k, self.track)(
            inputs=ins, output_shapes=[(n,)] * (4 if self.track else 2),
            output_dtypes=[mx.float32] * (4 if self.track else 2),
            grid=(self.tiles_x * THREADS, self.tiles_y, 1), threadgroup=(THREADS, 1, 1))
        if self.track:
            self.u, self.up, self.s1, self.s2 = outs
        else:
            self.u, self.up = outs
        self.time += k
        self._steps_since_eval += k
        if self._steps_since_eval >= 8:  # bound the lazy graph
            mx.eval(self.u, self.up, self.s1, self.s2, *self._probe_pending())
            self._steps_since_eval = 0

    def sync(self) -> None:
        mx.eval(self.u, self.up, self.s1, self.s2)
        mx.synchronize()

    # ------------------------------------------------------------ reading
    _xp = mx

    def _dev(self, a: np.ndarray):
        return mx.array(a.astype(np.int32) if a.dtype == np.int64 else a)

    def _host(self, a) -> np.ndarray:
        mx.eval(a)
        return np.array(a)

    def _flat_field(self):
        return self.u

    def _blend(self, s1, s2):
        b = base.INTENSITY_BLEND
        return (s1 * (1.0 - b) + s2 * b) * float(self.options.intensity_scale)

    def _flat_intensity(self):
        return self._blend(self.s1, self.s2) if self.track else self.u

    def _gather_intensity(self, idx):
        return self._blend(self.s1[idx], self.s2[idx]) if self.track else self.u[idx]

    def _grid(self, a) -> np.ndarray:
        return self._host(a).reshape(self.h + 2, self.w + 2)

    def preview_start(self, step: int = 1):
        """Subsampled and converted on the GPU, evaluated asynchronously: only the preview's bytes cross to the
        host, and the next steps are queued behind them instead of waiting."""
        r = 1 + self.rim
        sl = (slice(r, r + self.canvas_h, step), slice(r, r + self.canvas_w, step))
        grid = (self.h + 2, self.w + 2)
        f = self.u.reshape(grid)[sl].astype(mx.float16)
        i = self._flat_intensity().reshape(grid)[sl].astype(mx.float16) if self.track else None
        mx.async_eval(f, *([i] if i is not None else []))
        return lambda: (np.array(f), np.array(i) if i is not None else None)

    def preview(self, step: int = 1):
        return self.preview_start(step)()

    def field(self) -> np.ndarray:
        return self._interior(self._grid(self.u)).copy()

    def intensity(self) -> np.ndarray | None:
        return self._interior(self._grid(self._flat_intensity())).copy() if self.track else None
