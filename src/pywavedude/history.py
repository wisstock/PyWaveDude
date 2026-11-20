# SPDX-License-Identifier: GPL-3.0-or-later
# PyWaveDude: Python port of WaveDude Next (C) Alexander Graef, after Jeroen Vleggaar (Huygens Optics) and 0x23.
"""Unlimited undo and redo: one snapshot of the project file per finished edit.

A snapshot is the project's XML without the scene image plus the image's bytes by reference (equal bytes shared).
Changes gather in a step in progress (``edit_made``) until the app says the edit has finished (``end_step``)."""

from __future__ import annotations

from dataclasses import dataclass

from . import project as project_io


@dataclass(frozen=True)
class Snapshot:
    xml: bytes
    image: bytes | None


def snapshot(p) -> Snapshot:
    return Snapshot(project_io.dumps(p, embed_image=False), p.image_bytes)


def restore(s: Snapshot):
    return project_io.loads(s.xml, image_bytes=s.image)


class UndoHistory:
    def __init__(self, p=None):
        self._steps: list[Snapshot] = []
        self._at = -1
        self._pending = False
        self.saved_at = 0
        if p is not None:
            self.start(p)

    def start(self, p) -> None:
        """A new project: its state is the first step, and the saved one."""
        self._steps = [snapshot(p)]
        self._at = 0
        self._pending = False
        self.saved_at = 0

    def edit_made(self) -> None:
        self._pending = True

    @property
    def pending(self) -> bool:
        return self._pending

    def end_step(self, p) -> bool:
        """Record the step in progress, if any and if it changed something; True when one was recorded."""
        if not self._pending:
            return False
        self._pending = False
        s = snapshot(p)
        if self._steps and s == self._steps[self._at]:
            return False
        del self._steps[self._at + 1:]
        self._steps.append(s)
        self._at += 1
        return True

    @property
    def can_undo(self) -> bool:
        return self._at > 0 or self._pending

    @property
    def can_redo(self) -> bool:
        return self._at < len(self._steps) - 1

    def undo(self, p):
        """The project one step back (a new object), or None. A step in progress is ended first."""
        self.end_step(p)
        if self._at <= 0:
            return None
        self._at -= 1
        return restore(self._steps[self._at])

    def redo(self, p):
        self.end_step(p)
        if not self.can_redo:
            return None
        self._at += 1
        return restore(self._steps[self._at])

    def mark_saved(self) -> None:
        self.saved_at = self._at

    @property
    def dirty(self) -> bool:
        return self._pending or self._at != self.saved_at

    def __len__(self) -> int:
        return len(self._steps)
