# SPDX-License-Identifier: GPL-3.0-or-later
# PyWaveDude: Python port of WaveDude Next (C) Alexander Graef, after Jeroen Vleggaar (Huygens Optics) and 0x23.
"""Font (Inter, SIL Open Font License, bundled) and the dark theme of the app."""

from __future__ import annotations

from pathlib import Path

import dearpygui.dearpygui as dpg

from . import config

FONT = Path(__file__).parent / "assets" / "fonts" / "Inter.ttf"
BASE_SIZE = 15

# colours (RGBA 0..255)
BG = (30, 30, 32, 255)
PANEL = (37, 37, 40, 255)
FRAME = (48, 48, 52, 255)
FRAME_HOVER = (60, 62, 68, 255)
BORDER = (62, 62, 68, 255)
TEXT = (222, 222, 226, 255)
TEXT_DIM = (150, 152, 160, 255)
ACCENT = (0, 122, 204, 255)
ACCENT_HOVER = (28, 151, 234, 255)
SELECTED = (255, 220, 0, 255)          # the selected object on the pictures
HANDLE = (215, 215, 220, 230)          # other objects' handles
HALO = (0, 0, 0, 150)
WARNING = (255, 200, 90, 255)
ERROR = (255, 110, 100, 255)

_fonts: dict = {}


def _scale_hint() -> float:
    """Pixels per point of the frame buffer (2 on a Retina screen), for a sharp font atlas."""
    try:
        import ctypes
        import sys
        if sys.platform == "darwin":
            return 2.0
        if sys.platform == "win32":
            return ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100.0
    except Exception:
        pass
    return 1.0


def load_fonts(zoom: float = 1.0) -> None:
    """Inter at the UI size and a larger one for headings; rasterised at the screen's density."""
    hint = _scale_hint()
    with dpg.font_registry() as reg:
        base = float(config.app("ui.font_size", BASE_SIZE))
        for name, size in (("body", base), ("heading", base + 5), ("small", base - 3)):
            f = dpg.add_font(str(FONT), size * zoom * hint)    # glyph ranges load on demand
            _fonts[name] = f
    _fonts["registry"] = reg
    dpg.set_global_font_scale(1.0 / hint)
    dpg.bind_font(_fonts["body"])


def font(name: str):
    return _fonts.get(name)


def set_zoom(zoom: float) -> None:
    """A new font atlas for another UI zoom."""
    old = _fonts.get("registry")
    load_fonts(zoom)
    if old is not None:
        dpg.delete_item(old)


def apply_theme() -> None:
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvAll):
            c = dpg.add_theme_color
            c(dpg.mvThemeCol_WindowBg, BG)
            c(dpg.mvThemeCol_ChildBg, PANEL)
            c(dpg.mvThemeCol_PopupBg, (40, 40, 44, 250))
            c(dpg.mvThemeCol_MenuBarBg, (45, 45, 48, 255))
            c(dpg.mvThemeCol_Border, BORDER)
            c(dpg.mvThemeCol_Text, TEXT)
            c(dpg.mvThemeCol_TextDisabled, TEXT_DIM)
            c(dpg.mvThemeCol_FrameBg, FRAME)
            c(dpg.mvThemeCol_FrameBgHovered, FRAME_HOVER)
            c(dpg.mvThemeCol_FrameBgActive, (70, 72, 80, 255))
            c(dpg.mvThemeCol_Button, (55, 56, 62, 255))
            c(dpg.mvThemeCol_ButtonHovered, (70, 72, 80, 255))
            c(dpg.mvThemeCol_ButtonActive, ACCENT)
            c(dpg.mvThemeCol_Header, (0, 90, 150, 200))
            c(dpg.mvThemeCol_HeaderHovered, (0, 110, 180, 160))
            c(dpg.mvThemeCol_HeaderActive, ACCENT)
            c(dpg.mvThemeCol_CheckMark, ACCENT_HOVER)
            c(dpg.mvThemeCol_SliderGrab, ACCENT)
            c(dpg.mvThemeCol_SliderGrabActive, ACCENT_HOVER)
            c(dpg.mvThemeCol_TitleBg, (45, 45, 48, 255))
            c(dpg.mvThemeCol_TitleBgActive, ACCENT)
            c(dpg.mvThemeCol_Separator, BORDER)
            c(dpg.mvThemeCol_ScrollbarBg, PANEL)
            c(dpg.mvThemeCol_ScrollbarGrab, (70, 72, 80, 255))
            c(dpg.mvThemeCol_Tab, (45, 45, 48, 255))
            c(dpg.mvThemeCol_TabActive, ACCENT)
            c(dpg.mvThemeCol_PlotLines, (255, 200, 60, 255))
            s = dpg.add_theme_style
            s(dpg.mvStyleVar_FrameRounding, 2)
            s(dpg.mvStyleVar_WindowRounding, 3)
            s(dpg.mvStyleVar_ChildRounding, 2)
            s(dpg.mvStyleVar_FramePadding, 6, 3)
            s(dpg.mvStyleVar_ItemSpacing, 6, 4)
            s(dpg.mvStyleVar_WindowPadding, 6, 6)
            s(dpg.mvStyleVar_IndentSpacing, 14)
            s(dpg.mvStyleVar_ScrollbarSize, 12)
    dpg.bind_theme(theme)
    with dpg.theme(tag="theme_accent_button"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, ACCENT)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, ACCENT_HOVER)
    with dpg.theme(tag="theme_small_button"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 0, 0, 0))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (70, 72, 80, 255))
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 1)
    with dpg.theme(tag="theme_status"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (0, 100, 170, 255))
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 8, 3)
