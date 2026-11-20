# SPDX-License-Identifier: GPL-3.0-or-later
# PyWaveDude: Python port of WaveDude Next (C) Alexander Graef, after Jeroen Vleggaar (Huygens Optics) and 0x23.
"""Frame sinks (port of ``Export/*Writer.cs``): uncompressed OpenDML AVI, QuickTime Animation/PNG, PNG/JPEG/EXR
files and sequences in-house; MP4 (H.264) and MPEG-1 through a system ``ffmpeg`` when one is installed.

A picture sink takes ``(h, w, 3)`` uint8 RGB frames; a data sink (EXR) takes ``(h, w)`` float planes.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
import zlib
from enum import Enum
from pathlib import Path

import numpy as np
from numba import njit


class ExportFormat(str, Enum):
    MP4 = "MP4 (H.264)"
    MPEG1 = "MPEG-1"
    AVI = "AVI (uncompressed)"
    MOV = "QuickTime (Animation)"
    MOV_PNG = "QuickTime (PNG)"
    PNG = "PNG sequence"
    JPEG = "JPEG sequence"
    EXR = "OpenEXR (I)"
    EXR_ROOT = "OpenEXR (√I)"


EXTENSIONS = {
    ExportFormat.MP4: ".mp4", ExportFormat.MPEG1: ".mpg", ExportFormat.AVI: ".avi", ExportFormat.MOV: ".mov",
    ExportFormat.MOV_PNG: ".mov", ExportFormat.PNG: ".png", ExportFormat.JPEG: ".jpg", ExportFormat.EXR: ".exr",
    ExportFormat.EXR_ROOT: ".exr",
}
SEQUENCES = {ExportFormat.PNG, ExportFormat.JPEG, ExportFormat.EXR, ExportFormat.EXR_ROOT}
DATA = {ExportFormat.EXR, ExportFormat.EXR_ROOT}
STILLS = [ExportFormat.PNG, ExportFormat.JPEG, ExportFormat.EXR, ExportFormat.EXR_ROOT]


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def available_formats() -> list[ExportFormat]:
    have = ffmpeg_path() is not None
    return [f for f in ExportFormat if have or f not in (ExportFormat.MP4, ExportFormat.MPEG1)]


# ====================================================================== PNG (in-house, unfiltered rows)
def png_bytes(rgb: np.ndarray, level: int = 6) -> bytes:
    h, w = rgb.shape[:2]
    raw = np.empty((h, w * 3 + 1), np.uint8)
    raw[:, 0] = 0  # filter: none, what players decode fastest
    raw[:, 1:] = rgb.reshape(h, w * 3)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw.tobytes(), level))
            + chunk(b"IEND", b""))


# ====================================================================== OpenEXR (half, uncompressed, one channel Y)
def exr_bytes(values: np.ndarray, comment: str = "PyWaveDude") -> bytes:
    h, w = values.shape
    half = np.ascontiguousarray(values, "<f2")

    def attr(name, kind, value: bytes) -> bytes:
        return name.encode() + b"\0" + kind.encode() + b"\0" + struct.pack("<i", len(value)) + value

    chlist = b"Y\0" + struct.pack("<i", 1) + b"\0\0\0\0" + struct.pack("<ii", 1, 1) + b"\0"
    window = struct.pack("<iiii", 0, 0, w - 1, h - 1)
    header = (attr("channels", "chlist", chlist) + attr("compression", "compression", b"\0")
              + attr("dataWindow", "box2i", window) + attr("displayWindow", "box2i", window)
              + attr("lineOrder", "lineOrder", b"\0") + attr("pixelAspectRatio", "float", struct.pack("<f", 1.0))
              + attr("screenWindowCenter", "v2f", b"\0" * 8) + attr("screenWindowWidth", "float", struct.pack("<f", 1.0))
              + attr("comments", "string", comment.encode()) + b"\0")
    row = w * 2
    first = 8 + len(header) + 8 * h
    offsets = struct.pack(f"<{h}Q", *(first + (8 + row) * y for y in range(h)))
    body = bytearray()
    for y in range(h):
        body += struct.pack("<ii", y, row) + half[y].tobytes()
    return struct.pack("<ii", 20000630, 2) + header + offsets + bytes(body)


# ====================================================================== sinks
class Sink:
    wants_values = False

    def add(self, frame: np.ndarray) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class ImageSequenceSink(Sink):
    """Numbered files in a folder named after the output (``name/name_0000.png``), or one still file."""

    def __init__(self, path: Path, fmt: ExportFormat, quality: int = 85, still: bool = False, first: int = 0,
                 planned: int = 1):
        self.fmt, self.quality, self.still = fmt, quality, still
        self.wants_values = fmt in DATA
        path = Path(path)
        if still:
            self.path = path
        else:
            self.folder = path.with_suffix("")
            self.folder.mkdir(parents=True, exist_ok=True)
            self.stem = path.stem
            self.digits = max(4, len(str(first + planned - 1)))
            self.n = first
        self.ext = EXTENSIONS[fmt]

    def _target(self) -> Path:
        if self.still:
            return self.path
        p = self.folder / f"{self.stem}_{self.n:0{self.digits}d}{self.ext}"
        self.n += 1
        return p

    def add(self, frame: np.ndarray) -> None:
        target = self._target()
        if self.fmt in DATA:
            target.write_bytes(exr_bytes(frame))
        elif self.fmt == ExportFormat.PNG:
            target.write_bytes(png_bytes(frame))
        else:
            from PIL import Image
            Image.fromarray(frame).save(target, "JPEG", quality=self.quality, subsampling=0)


class AviSink(Sink):
    """Uncompressed 24-bit DIB AVI with OpenDML segments past 1 GiB (port of ``AviWriter.cs``)."""

    SEGMENT_LIMIT = 1 << 30
    DMLH = 248

    def __init__(self, path: Path, width: int, height: int, fps: int, planned: int, segment_limit: int = SEGMENT_LIMIT):
        self.w, self.h, self.fps = width, height, max(1, fps)
        self.planned = max(1, planned)
        self.row_bytes = (width * 3 + 3) & ~3
        self.frame_bytes = self.row_bytes * height
        self.chunk = 8 + self.frame_bytes
        self.later = max(1, (segment_limit - 12 - 12 - 32) // (self.chunk + 8))
        entries = 1
        for _ in range(3):
            self.header_bytes = self._header_size(entries)
            self.first = max(1, (segment_limit - self.header_bytes - 12 - 32 - 8) // (self.chunk + 8 + 16))
            rest = max(0, self.planned - self.first)
            needed = 1 + (rest + self.later - 1) // self.later
            if needed <= entries:
                break
            entries = needed
        self.super_entries = entries
        self.f = open(path, "wb")
        self.f.truncate(self.header_bytes)
        self.f.seek(self.header_bytes)
        self.segments: list[dict] = []
        self.count = 0
        self._begin()

    def _header_size(self, e):
        strl = 12 + (8 + 56) + (8 + 40) + (8 + 24 + 16 * e)
        hdrl = 12 + (8 + 56) + strl
        return 12 + hdrl + (12 + 8 + self.DMLH)

    def _begin(self):
        pos = self.f.tell()
        if not self.segments:
            seg = dict(riff=0, movi=pos, frames=0)
            self.f.write(b"LIST\0\0\0\0movi")
        else:
            seg = dict(riff=pos, movi=pos + 12, frames=0)
            self.f.write(b"RIFF\0\0\0\0AVIXLIST\0\0\0\0movi")
        self.segments.append(seg)

    def add(self, frame: np.ndarray) -> None:
        seg = self.segments[-1]
        limit = self.first if len(self.segments) == 1 else self.later
        if seg["frames"] >= limit:
            self._end()
            self._begin()
            seg = self.segments[-1]
        bgr = np.zeros((self.h, self.row_bytes), np.uint8)
        bgr[:, :self.w * 3] = frame[::-1, :, ::-1].reshape(self.h, self.w * 3)  # bottom-up BGR
        self.f.write(b"00db" + struct.pack("<i", self.frame_bytes))
        self.f.write(bgr.tobytes())
        seg["frames"] += 1
        self.count += 1

    def _end(self):
        seg = self.segments[-1]
        n, movi = seg["frames"], seg["movi"]
        data = movi + 12
        idx = bytearray(b"ix00" + struct.pack("<iHBBi4sq", 24 + 8 * n, 2, 0, 1, n, b"00db", movi) + b"\0" * 4)
        for i in range(n):
            idx += struct.pack("<ii", data + i * self.chunk + 8 - movi, self.frame_bytes)
        seg["index"] = self.f.tell()
        self.f.write(idx)
        movi_end = self.f.tell()
        if len(self.segments) == 1:
            legacy = bytearray(b"idx1" + struct.pack("<i", 16 * n))
            for i in range(n):
                legacy += b"00db" + struct.pack("<Iii", 0x10, data + i * self.chunk - (movi + 8), self.frame_bytes)
            self.f.write(legacy)
        seg["end"] = self.f.tell()
        if len(self.segments) > 1:
            self._patch(seg["riff"] + 4, seg["end"] - seg["riff"] - 8)
        self._patch(movi + 4, movi_end - movi - 8)
        self.f.seek(seg["end"])

    def _patch(self, pos, value):
        self.f.seek(pos)
        self.f.write(struct.pack("<I", value))

    def close(self) -> None:
        if self.f.closed:
            return
        self._end()
        self.f.seek(0)
        self.f.write(self._header())
        self.f.close()

    def _header(self) -> bytes:
        e, first = self.super_entries, self.segments[0]
        strl = 12 + (8 + 56) + (8 + 40) + (8 + 24 + 16 * e)
        hdrl = 12 + (8 + 56) + strl
        avih = struct.pack("<IIIIIIIIII16x", 1000000 // self.fps, min(0xFFFFFFFF, self.chunk * self.fps), 0,
                           0x10 | 0x800, first["frames"], 0, 1, self.frame_bytes, self.w, self.h)
        strh = (b"vidsDIB " + struct.pack("<IHHIIIIIIIIhhhh", 0, 0, 0, 0, 1, self.fps, 0, self.count,
                                          self.frame_bytes, 0xFFFFFFFF, 0, 0, 0, self.w, self.h))
        strf = struct.pack("<iiiHHIIiiII", 40, self.w, self.h, 1, 24, 0, self.frame_bytes, 0, 0, 0, 0)
        indx = bytearray(struct.pack("<HBBi4s12x", 4, 0, 0, min(len(self.segments), e), b"00db"))
        for i in range(e):
            if i < len(self.segments):
                s = self.segments[i]
                indx += struct.pack("<qii", s["index"], 32 + 8 * s["frames"], s["frames"])
            else:
                indx += b"\0" * 16
        out = (b"RIFF" + struct.pack("<I", first["end"] - 8) + b"AVI "
               + b"LIST" + struct.pack("<i", hdrl - 8) + b"hdrl"
               + b"avih" + struct.pack("<i", 56) + avih
               + b"LIST" + struct.pack("<i", strl - 8) + b"strl"
               + b"strh" + struct.pack("<i", 56) + strh
               + b"strf" + struct.pack("<i", 40) + strf
               + b"indx" + struct.pack("<i", 24 + 16 * e) + bytes(indx)
               + b"LIST" + struct.pack("<i", 4 + 8 + self.DMLH) + b"odmldmlh" + struct.pack("<ii", self.DMLH, self.count))
        return out + b"\0" * (self.header_bytes - len(out))


@njit(cache=True)
def _rle_encode(px, out):
    """QuickTime Animation (``rle ``, 24-bit) of a frame of packed 0xRRGGBB ints; returns the byte count."""
    h, w = px.shape
    p = 6
    for y in range(h):
        out[p] = 1
        p += 1
        x = 0
        while x < w:
            v = px[y, x]
            run = 1
            while x + run < w and run < 128 and px[y, x + run] == v:
                run += 1
            if run >= 2:
                out[p] = (256 - run) & 0xFF
                out[p + 1] = (v >> 16) & 0xFF
                out[p + 2] = (v >> 8) & 0xFF
                out[p + 3] = v & 0xFF
                p += 4
                x += run
                continue
            start = x
            count = 0
            while x < w and count < 127:
                if x + 1 < w and px[y, x + 1] == px[y, x]:
                    break
                x += 1
                count += 1
            if count == 0:
                continue
            out[p] = count
            p += 1
            for i in range(count):
                v = px[y, start + i]
                out[p] = (v >> 16) & 0xFF
                out[p + 1] = (v >> 8) & 0xFF
                out[p + 2] = v & 0xFF
                p += 3
        out[p] = 0xFF
        p += 1
    out[0] = (p >> 24) & 0xFF
    out[1] = (p >> 16) & 0xFF
    out[2] = (p >> 8) & 0xFF
    out[3] = p & 0xFF
    out[4] = 0
    out[5] = 0
    return p


def _atom(kind: str, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + kind.encode() + payload


class MovSink(Sink):
    """QuickTime with the Animation (RLE) or PNG codec, moov written first when the planned frames fit."""

    def __init__(self, path: Path, width: int, height: int, fps: int, planned: int = 0, png: bool = False):
        self.w, self.h, self.fps, self.png = width, height, max(1, fps), png
        self.f = open(path, "wb")
        self.f.write(_atom("ftyp", b"qt  \0\0\2\0qt  "))
        self.reserved_start = self.f.tell()
        self.reserved = 1024 + 12 * planned + 8 if planned > 0 else 0
        if self.reserved:
            self.f.write(struct.pack(">I", self.reserved) + b"free")
            self.f.seek(self.reserved_start + self.reserved)
        self.mdat = self.f.tell()
        self.f.write(struct.pack(">I", 1) + b"mdat" + struct.pack(">Q", 0))
        self.offsets: list[int] = []
        self.sizes: list[int] = []
        self._buf = np.zeros(6 + height * (width * 3 + width // 127 + 4), np.uint8)

    def add(self, frame: np.ndarray) -> None:
        self.offsets.append(self.f.tell())
        if self.png:
            data = png_bytes(frame)
        else:
            px = (frame[..., 0].astype(np.int32) << 16) | (frame[..., 1].astype(np.int32) << 8) | frame[..., 2]
            n = _rle_encode(px, self._buf)
            data = self._buf[:n].tobytes()
        self.f.write(data)
        self.sizes.append(len(data))

    def close(self) -> None:
        if self.f.closed:
            return
        end = self.f.tell()
        moov = self._moov()
        if self.reserved and len(moov) + 8 <= self.reserved:
            self.f.seek(self.reserved_start)
            self.f.write(moov)
            self.f.write(struct.pack(">I", self.reserved - len(moov)) + b"free")
        else:
            self.f.seek(end)
            self.f.write(moov)
        self.f.seek(self.mdat + 8)
        self.f.write(struct.pack(">Q", end - self.mdat))
        self.f.close()

    def _moov(self) -> bytes:
        n, ts = len(self.sizes), self.fps
        matrix = struct.pack(">9I", 0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000)
        mvhd = _atom("mvhd", bytes(12) + struct.pack(">IIIH", ts, n, 0x10000, 0x100) + bytes(10) + matrix
                     + bytes(24) + struct.pack(">I", 2))
        tkhd = _atom("tkhd", b"\0\0\0\x0f" + bytes(8) + struct.pack(">I", 1) + bytes(4) + struct.pack(">I", n)
                     + bytes(16) + matrix + struct.pack(">II", self.w << 16, self.h << 16))
        mdhd = _atom("mdhd", bytes(12) + struct.pack(">IIHH", ts, n, 0x55C4, 0))

        def hdlr(t, s, name):
            return _atom("hdlr", bytes(4) + t.encode() + s.encode() + b"appl" + bytes(8) + bytes([len(name)])
                         + name.encode())

        name = b"PNG" if self.png else b"Animation"
        entry = bytearray(86)
        struct.pack_into(">I4s", entry, 0, 86, b"png " if self.png else b"rle ")
        struct.pack_into(">H", entry, 14, 1)
        entry[20:24] = b"appl"
        struct.pack_into(">IHHII", entry, 28, 0x3FF, self.w, self.h, 0x480000, 0x480000)
        struct.pack_into(">H", entry, 48, 1)
        entry[50] = len(name)
        entry[51:51 + len(name)] = name
        struct.pack_into(">Hh", entry, 82, 24, -1)
        stsd = _atom("stsd", struct.pack(">II", 0, 1) + bytes(entry))
        stts = _atom("stts", struct.pack(">IIII", 0, 1, n, 1))
        stsc = _atom("stsc", struct.pack(">IIIII", 0, 1, 1, 1, 1))
        stsz = _atom("stsz", struct.pack(f">III{n}I", 0, 0, n, *self.sizes))
        co64 = _atom("co64", struct.pack(f">II{n}Q", 0, n, *self.offsets))
        minf = _atom("minf", _atom("vmhd", b"\0\0\0\1" + bytes(8)) + hdlr("dhlr", "alis", "DataHandler")
                     + _atom("dinf", _atom("dref", struct.pack(">II", 0, 1) + _atom("alis", b"\0\0\0\1")))
                     + _atom("stbl", stsd + stts + stsc + stsz + co64))
        mdia = _atom("mdia", mdhd + hdlr("mhlr", "vide", "VideoHandler") + minf)
        return _atom("moov", mvhd + _atom("trak", tkhd + mdia))


class FfmpegSink(Sink):
    """MP4 (H.264, yuv420p) or MPEG-1 through a system ffmpeg fed raw RGB on a pipe."""

    def __init__(self, path: Path, width: int, height: int, fps: int, quality: int, fmt: ExportFormat):
        exe = ffmpeg_path()
        if exe is None:
            raise RuntimeError("MP4 and MPEG-1 export need ffmpeg on the PATH")
        q = min(max(quality, 1), 100)
        # yuv420p needs even sizes: pad by one black pixel where odd
        pad = ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
        if fmt == ExportFormat.MP4:
            codec = ["-c:v", "libx264", "-crf", str(round(35 - 0.25 * q)), "-preset", "medium", "-pix_fmt", "yuv420p",
                     "-movflags", "+faststart"]
        else:
            codec = ["-c:v", "mpeg1video", "-q:v", str(min(max(round(31 - 0.3 * q), 1), 31)), "-pix_fmt", "yuv420p",
                     "-f", "mpeg"]
        cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
               "-r", str(fps), "-i", "-"] + pad + codec + [str(path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def add(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        err = self.proc.stderr.read().decode(errors="replace") if self.proc.stderr else ""
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {err.strip()[:300]}")


def open_sink(fmt: ExportFormat, path: Path, width: int, height: int, fps: int, quality: int, planned: int,
              still: bool = False, first: int = 0) -> Sink:
    if fmt in SEQUENCES:
        return ImageSequenceSink(path, fmt, quality, still, first, planned)
    if fmt == ExportFormat.AVI:
        return AviSink(path, width, height, fps, planned)
    if fmt in (ExportFormat.MOV, ExportFormat.MOV_PNG):
        return MovSink(path, width, height, fps, planned, png=fmt == ExportFormat.MOV_PNG)
    return FfmpegSink(path, width, height, fps, quality, fmt)
