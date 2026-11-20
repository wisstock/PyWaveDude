# SPDX-License-Identifier: GPL-3.0-or-later
"""Optical parts: definitions (``model``), per-pixel geometry (``shapes``) and the scene painter (``painter``)."""

from .model import *  # noqa: F401,F403
from .model import PART_TYPES
from .painter import blank_scene, paint_parts

__all__ = ["PART_TYPES", "blank_scene", "paint_parts"]
