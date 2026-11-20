# SPDX-License-Identifier: GPL-3.0-or-later
# PyWaveDude: Python port of WaveDude Next (C) Alexander Graef, after Jeroen Vleggaar (Huygens Optics) and 0x23.
"""The defaults file (``defaults.yaml``): what a new project, a new object and the app start with.

Read once at start (``load``); every key must name an existing setting, so a typo is an error, not a silent
default. The dataclasses' own defaults stay the project file format's: an attribute a ``.wdproj`` leaves out always
means the format's value, whatever this file says."""

from __future__ import annotations

import dataclasses
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

DEFAULTS_FILE = Path(__file__).parent / "defaults.yaml"
_cfg: dict | None = None
_path: Path | None = None


class ConfigError(ValueError):
    pass


def load(path: str | Path | None = None) -> dict:
    """Read and check the defaults file (the bundled one when ``path`` is None)."""
    global _cfg, _path
    p = Path(path) if path else DEFAULTS_FILE
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"{p}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{p}: the top level must be a mapping")
    _check(data, str(p))
    _cfg, _path = data, p
    return data


def get() -> dict:
    return _cfg if _cfg is not None else load()


def path() -> Path:
    get()
    return _path


def app(key: str, fallback: Any = None) -> Any:
    """An app setting by its dotted path, e.g. ``app("ui.left_pane_width")``."""
    node: Any = get().get("app", {})
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return fallback
        node = node[part]
    return node


# ------------------------------------------------------------ objects
def _convert(value, like, where: str):
    if isinstance(like, bool):
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true or false, got {value!r}")
        return value
    if isinstance(like, Enum):
        for e in type(like):
            if value in (e.value, e.name):
                return e
        raise ConfigError(f"{where}: {value!r} is not one of {[e.value for e in type(like)]}")
    if isinstance(like, int) and not isinstance(value, bool) and isinstance(value, (int, float)):
        return int(value)
    if isinstance(like, float) and not isinstance(value, bool) and isinstance(value, (int, float)):
        return float(value)
    if isinstance(like, str):
        return str(value)
    raise ConfigError(f"{where}: cannot use {value!r} for a {type(like).__name__}")


def apply(obj, mapping: dict | None, where: str):
    """Set ``obj``'s fields from ``mapping`` (each key a field of its dataclass); returns ``obj``."""
    names = {f.name for f in dataclasses.fields(obj)}
    for k, v in (mapping or {}).items():
        if k not in names:
            raise ConfigError(f"{where}: unknown setting '{k}'")
        setattr(obj, k, _convert(v, getattr(obj, k), f"{where}.{k}"))
    return obj


def new_object(cls, **fields):
    """A new source, probe or part: the format's defaults, then this file's, then ``fields``."""
    from .engine import PointSource
    from .probes import Probe
    objs = get().get("new_objects", {}) or {}
    if cls is PointSource:
        obj = apply(PointSource(0, 0), objs.get("source"), "new_objects.source")
    elif cls is Probe:
        obj = apply(Probe(), objs.get("probe"), "new_objects.probe")
    else:
        obj = apply(cls(), (objs.get("parts") or {}).get(cls.__name__), f"new_objects.parts.{cls.__name__}")
    for k, v in fields.items():
        setattr(obj, k, v)
    return obj


def new_project():
    """A new project: its settings and its first sources from the file."""
    from .engine import PointSource
    from .project import Project
    c = get().get("new_project", {}) or {}
    p = Project()
    apply(p.scene, c.get("scene"), "new_project.scene")
    apply(p.sim, c.get("simulation"), "new_project.simulation")
    apply(p.display, c.get("display"), "new_project.display")
    for i, s in enumerate(c.get("sources") or []):
        p.sources.append(apply(new_object(PointSource), s, f"new_project.sources[{i}]"))
    return p


# ------------------------------------------------------------ checking
def _check(data: dict, where: str) -> None:
    from .engine import PointSource
    from .parts import PART_TYPES
    from .probes import Probe
    from .project import Project
    unknown = set(data) - {"app", "new_project", "new_objects"}
    if unknown:
        raise ConfigError(f"{where}: unknown section(s) {sorted(unknown)}")
    np_ = data.get("new_project") or {}
    p = Project()
    for key, obj in (("scene", p.scene), ("simulation", p.sim), ("display", p.display)):
        apply(obj, np_.get(key), f"new_project.{key}")
    for i, s in enumerate(np_.get("sources") or []):
        apply(PointSource(0, 0), s, f"new_project.sources[{i}]")
    objs = data.get("new_objects") or {}
    apply(PointSource(0, 0), objs.get("source"), "new_objects.source")
    apply(Probe(), objs.get("probe"), "new_objects.probe")
    kinds = {c.__name__: c for c in PART_TYPES.values()}
    for name, fields in (objs.get("parts") or {}).items():
        if name not in kinds:
            raise ConfigError(f"new_objects.parts: unknown part kind '{name}'")
        apply(kinds[name](), fields, f"new_objects.parts.{name}")
