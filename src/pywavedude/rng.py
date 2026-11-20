# SPDX-License-Identifier: GPL-3.0-or-later
# PyWaveDude: Python port of WaveDude Next (C) Alexander Graef, after Jeroen Vleggaar (Huygens Optics) and 0x23.
"""PCG32 (XSH RR) and the FNV-1a name hash, bit-identical to ``Pcg32.cs`` so seeded designs match."""

_M64 = (1 << 64) - 1
_M32 = (1 << 32) - 1
_MULT = 6364136223846793005


class Pcg32:
    def __init__(self, seed: int, stream: int = 0):
        self._inc = ((stream << 1) | 1) & _M64
        self._state = 0
        self.next_uint32()
        self._state = (self._state + seed) & _M64
        self.next_uint32()

    def next_uint32(self) -> int:
        old = self._state
        self._state = (old * _MULT + self._inc) & _M64
        xs = (((old >> 18) ^ old) >> 27) & _M32
        rot = old >> 59
        return ((xs >> rot) | (xs << ((-rot) & 31))) & _M32

    def next_int(self, lo: int, hi: int) -> int:
        """Uniform in [lo, hi] by rejection, as ``Pcg32.Next``."""
        rng = hi - lo + 1
        limit = (1 << 32) - ((1 << 32) % rng)
        while True:
            v = self.next_uint32()
            if v < limit:
                return lo + v % rng

    def next_double(self) -> float:
        return (self.next_uint32() >> 8) * (1.0 / (1 << 24))


def stable_hash(text: str) -> int:
    h = 2166136261
    data = text.encode("utf-16-le")  # UTF-16 code units, as .NET's char
    for i in range(0, len(data), 2):
        h = ((h ^ (data[i] | data[i + 1] << 8)) * 16777619) & _M32
    return h
